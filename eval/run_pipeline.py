"""CLI: run the full linking pipeline (gate -> fuse -> windowed min-cost flow
-> trajectories) on an ON-DISK dataset (e.g. `data/run1`), and report the
real trajectory-level scoreboard (architecture.md §8): IDF1 and friends via
`eval.metrics.compute_trajectory_metrics`, PLUS the same metrics for
Baseline A (exact plate-string match within the same spatio-temporal gate,
chained greedily in time) -- the single comparison the whole project's
thesis rests on.

Priors are fit on a SEPARATE training dataset generated on the SAME city
(architecture.md §10 / docs/decisions.md "Day 2": a kinematic model's
per-camera-pair fit is only meaningful evaluated against candidates from
that same camera network, but the vehicle population and OCR/appearance
noise realisation must differ from what's being evaluated, or the reported
numbers would be measuring memorisation, not generalisation).

Usage:
    python -m eval.run_pipeline --data data/run1 --out data/run1/pipeline
    python -m eval.run_pipeline --data data/run1 --max-hours 2   # quick prefix check
"""

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import typer
from rich.console import Console

from engine.association.gating import Gate, gate_candidates
from engine.association.mincostflow import fit_entry_exit_costs
from engine.association.window import WindowStats, solve_windowed
from engine.calibration.fit_priors import fit_all
from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent
from engine.contracts.store import EventStore
from engine.contracts.trajectory import Trajectory
from engine.scoring.fusion import FusionModel
from eval.metrics import TrajectoryMetrics, compute_trajectory_metrics
from sim.generate import generate_dataset_with_city

app = typer.Typer(add_completion=False)
console = Console()

# The train split's vehicle population is independent of (and smaller than)
# the evaluated dataset's -- it only needs to give the kinematic/appearance/
# plate-prior models a reasonable per-(camera-pair, tod-bucket) and
# per-distance-bin sample size on the SAME city, not to match production
# scale itself (docs/decisions.md's own train/holdout splits use 1500-8000
# vehicles against much larger holdouts for exactly this reason).
DEFAULT_TRAIN_VEHICLES = 8000
DEFAULT_TRAIN_SEED_OFFSET = 9001


def _load_city(data_dir: Path) -> CityConfig:
    return CityConfig.model_validate_json((data_dir / "city.json").read_text(encoding="utf-8"))


def _load_events(data_dir: Path) -> list[DetectionEvent]:
    store = EventStore(data_dir / "events.jsonl", data_dir / "embeddings.npy")
    return store.read_all()


def _metrics_to_dict(m: TrajectoryMetrics) -> dict:
    return {
        "idf1": m.idf1,
        "id_precision": m.id_precision,
        "id_recall": m.id_recall,
        "id_switches": m.id_switches,
        "fragmentation": m.fragmentation,
        "trajectory_completeness": m.trajectory_completeness,
        "n_predicted_trajectories": m.n_predicted_trajectories,
        "n_gt_vehicles": m.n_gt_vehicles,
        "n_gt_events": m.n_gt_events,
        "n_predicted_events": m.n_predicted_events,
        "idtp": m.idtp,
        "idfp": m.idfp,
        "idfn": m.idfn,
    }


def _build_exact_match_baseline(
    events: list[DetectionEvent], gate: Gate
) -> list[Trajectory]:
    """Baseline A (architecture.md §8): "exact plate string match ... what
    the problem statement calls the standard solution". Within the SAME
    spatio-temporal gate the real pipeline uses (so the comparison isolates
    the plate-matching STRATEGY, not the candidate search), link event `j`
    to the closest-in-time gated predecessor `i` whose `plate_argmax`
    strings are EXACTLY equal, greedily, in ascending time order, forming
    chains one plate-run at a time. No scoring, no global optimisation --
    this is deliberately the naive standard the project's thesis is
    measured against."""
    events_sorted = sorted(events, key=lambda e: e.timestamp)
    events_by_id = {e.event_id: e for e in events_sorted}
    gate_result = gate_candidates(events_sorted, gate)

    used_as_predecessor: set[str] = set()
    chain_next: dict[str, str] = {}  # predecessor event_id -> successor event_id
    chain_prev: dict[str, str] = {}  # successor event_id -> predecessor event_id

    for e_j in events_sorted:
        for cand_id in gate_result.candidates.get(e_j.event_id, []):
            if cand_id in used_as_predecessor:
                continue
            if events_by_id[cand_id].plate_argmax != e_j.plate_argmax:
                continue
            chain_next[cand_id] = e_j.event_id
            chain_prev[e_j.event_id] = cand_id
            used_as_predecessor.add(cand_id)
            break

    trajectories: list[Trajectory] = []
    i = 0
    for e in events_sorted:
        if e.event_id in chain_prev:
            continue  # not a chain head
        seq = [e.event_id]
        cur = e.event_id
        while cur in chain_next:
            cur = chain_next[cur]
            seq.append(cur)
        path_events = [events_by_id[eid] for eid in seq]
        first = path_events[0]
        trajectories.append(
            Trajectory(
                trajectory_id=f"baseline_{i:06d}",
                event_ids=seq,
                decoded_plate=first.plate_argmax,
                plate_confidence=first.plate_confidence,
                start_time=path_events[0].timestamp,
                end_time=path_events[-1].timestamp,
                camera_sequence=[ev.camera_id for ev in path_events],
                links=[],
                gt_vehicle_id=first.gt_vehicle_id,
            )
        )
        i += 1
    return trajectories


@app.command()
def main(
    data: Path = typer.Option(Path("data/run1"), help="On-disk dataset directory to evaluate."),
    out: Path = typer.Option(Path("data/run1/pipeline"), help="Output directory."),
    train_vehicles: int = typer.Option(DEFAULT_TRAIN_VEHICLES, help="Train split vehicle count."),
    train_seed_offset: int = typer.Option(
        DEFAULT_TRAIN_SEED_OFFSET, help="Added to the eval dataset's seed for the train split."
    ),
    max_hours: float | None = typer.Option(
        None,
        help=(
            "Evaluate only the first N hours of events (by timestamp from the "
            "earliest event) instead of the full dataset -- for a quick "
            "before-you-commit-to-the-full-run timing check. The train split "
            "is unaffected (still generated at its own full `--hours`)."
        ),
    ),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    report_path = Path("eval/reports/trajectory_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads((data / "config.json").read_text(encoding="utf-8"))
    console.print(f"[bold]Loading[/bold] {data} (config={config})")

    t0 = time.time()
    city = _load_city(data)
    events = _load_events(data)
    if max_hours is not None:
        events_sorted_for_cut = sorted(events, key=lambda e: e.timestamp)
        cutoff = events_sorted_for_cut[0].timestamp + timedelta(hours=max_hours)
        events = [e for e in events_sorted_for_cut if e.timestamp < cutoff]
    t1 = time.time()
    console.print(f"[green]Loaded[/green] {len(events)} events in {t1 - t0:.1f}s")

    train_seed = config["seed"] + train_seed_offset
    console.print(
        f"[bold]Fitting priors[/bold] on a separate train split: "
        f"{train_vehicles} vehicles, seed={train_seed}, same city"
    )
    train_ds = generate_dataset_with_city(
        city=city,
        n_vehicles=train_vehicles,
        hours=config["hours"],
        seed=train_seed,
        clone_fraction=config.get("clone_fraction", 0.02),
    )
    t2 = time.time()
    models = fit_all(train_ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(train_ds.events, city)
    gate = Gate(model=models.kinematic_model)
    t3 = time.time()
    console.print(
        f"[green]Fitted models[/green] in {t3 - t2:.1f}s (train generation {t2 - t1:.1f}s)"
    )

    console.print("[bold]Gating[/bold] the full evaluated dataset")
    gate_result = gate_candidates(events, gate)
    t4 = time.time()
    console.print(
        f"[green]Gated[/green] {len(events)} events, "
        f"mean_candidates_per_event={gate_result.mean_candidates_per_event:.1f} in {t4 - t3:.1f}s"
    )

    console.print("[bold]Windowed solve[/bold] (gate + score + min-cost flow per window)")

    window_log: list[dict] = []

    def _on_window(window_i: int, n_windows: int, stats: WindowStats, elapsed_s: float) -> None:
        pct_kept = 100.0 * stats.n_arcs_after / stats.n_arcs_before if stats.n_arcs_before else 0.0
        print(
            f"  window {window_i + 1}/{n_windows}: events={stats.n_events} "
            f"arcs={stats.n_arcs_before}->{stats.n_arcs_after} ({pct_kept:.1f}% kept) "
            f"components={stats.n_components} largest={stats.largest_component} "
            f"{elapsed_s:.2f}s",
            flush=True,
        )
        sys.stdout.flush()
        window_log.append(
            {
                "window_index": window_i,
                "n_events": stats.n_events,
                "n_arcs_before": stats.n_arcs_before,
                "n_arcs_after": stats.n_arcs_after,
                "n_components": stats.n_components,
                "largest_component": stats.largest_component,
                "elapsed_s": elapsed_s,
            }
        )

    result = solve_windowed(events, gate, fusion_model, entry_exit, city, on_window=_on_window)
    t5 = time.time()
    console.print(
        f"[green]Solved[/green] {result.n_windows} windows -> "
        f"{len(result.trajectories)} trajectories in {t5 - t4:.1f}s"
    )

    metrics = compute_trajectory_metrics(result.trajectories, events)
    t6 = time.time()

    console.print("[bold]Baseline A[/bold] (exact plate match, same gate, greedy chaining)")
    baseline_trajectories = _build_exact_match_baseline(events, gate)
    baseline_metrics = compute_trajectory_metrics(baseline_trajectories, events)
    t7 = time.time()

    trajectories_path = out / "trajectories.jsonl"
    with trajectories_path.open("w", encoding="utf-8", newline="\n") as f:
        for traj in result.trajectories:
            f.write(traj.model_dump_json())
            f.write("\n")
    t8 = time.time()

    pipeline_wall_s = t5 - t3  # gating + scoring + windowed solve, excludes train generation/fit
    total_wall_s = t8 - t0

    is_prefix_run = max_hours is not None
    description = (
        f"PARTIAL run: first {max_hours}h of events only (--max-hours), NOT the full dataset -- "
        f"for timing projection only. Real trajectory numbers need the full run."
        if is_prefix_run
        else (
            "FULL-dataset trajectory-level result (SUTRA vs Baseline A, exact plate "
            "match). architecture.md section 8's headline scoreboard."
        )
    )

    report = {
        "description": description,
        "data_dir": str(data),
        "max_hours": max_hours,
        "config": config,
        "train": {
            "n_vehicles": train_vehicles,
            "seed": train_seed,
            "hours": config["hours"],
        },
        "n_events": len(events),
        "gating": {
            "mean_candidates_per_event": gate_result.mean_candidates_per_event,
            "n_candidate_pairs": gate_result.n_candidate_pairs,
        },
        "n_windows": result.n_windows,
        "window_log": window_log,
        "solver": {
            "total_arcs_before_pruning": sum(w["n_arcs_before"] for w in window_log),
            "total_arcs_after_pruning": sum(w["n_arcs_after"] for w in window_log),
            "mean_windowed_solve_s_per_window": (
                sum(w["elapsed_s"] for w in window_log) / len(window_log) if window_log else 0.0
            ),
        },
        "sutra": _metrics_to_dict(metrics),
        "exact_match_baseline": _metrics_to_dict(baseline_metrics),
        "timing_s": {
            "load_data": t1 - t0,
            "generate_train_split": t2 - t1,
            "fit_models": t3 - t2,
            "gating": t4 - t3,
            "windowed_solve": t5 - t4,
            "compute_metrics": t6 - t5,
            "baseline_a": t7 - t6,
            "write_trajectories": t8 - t7,
            "pipeline_(gate+fit_context_excluded)": pipeline_wall_s,
            "total_wall_s": total_wall_s,
        },
        "events_per_sec_pipeline": len(events) / pipeline_wall_s if pipeline_wall_s > 0 else 0.0,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(json.dumps(report, indent=2))
    console.print(f"[green]Wrote[/green] {trajectories_path}")
    console.print(f"[green]Wrote[/green] {report_path}")


if __name__ == "__main__":
    app()
