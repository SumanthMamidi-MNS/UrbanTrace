"""CLI: gate true-predecessor recall and mean candidates per event, broken
down BY HOUR OF DAY, on an ON-DISK dataset (e.g. `data/run1` or a
production-scale run like `C:/sutra-data/runs/run2`).

Diagnoses whether an aggregate recall number hides a rush-hour collapse:
the lead measured recall=0.997 at night but 0.68 at 08h and 0.70 at 17h on
a congested day, even though every true link that reached scoring still
scored strongly positive (engine/association/gating.py's own module
docstring, ">99% recall floor" -- `tests/test_gating.py`). This script is
the per-hour instrument for that; it does not change any gate behaviour
itself (see `--congestion-tail-factor` below, which just forwards to
`Gate`).

Reuses `eval/run_pipeline.py`'s own dataset-loading and train-split
generation helpers (`_load_city`, `_load_events`, `_congestion_from_config`)
so the train split fed to `fit_all` here is generated EXACTLY the way the
real pipeline generates it: same seed offset, same congestion regime read
off the evaluated dataset's own config.json (see run_pipeline.py's module
docstring for why train/eval must share a congestion regime -- fitting on
free-flow travel times and evaluating on BPR-congested ones would silently
bias every kinematic likelihood).

Usage:
    python -m eval.gate_recall_by_hour --data data/run1
    python -m eval.gate_recall_by_hour --data C:/sutra-data/runs/run2 \\
        --out eval/reports/gate_recall_by_hour.json
    # Reproduce the pre-fix (old) gate behaviour for comparison:
    python -m eval.gate_recall_by_hour --data C:/sutra-data/runs/run2 \\
        --congestion-tail-factor 1.0 --out eval/reports/gate_recall_by_hour_old.json
"""

import json
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console

from engine.association.gating import CONGESTION_TAIL_FACTOR, Gate, gate_candidates
from engine.calibration.fit_priors import fit_all
from engine.contracts.events import DetectionEvent
from eval.run_pipeline import (
    DEFAULT_TRAIN_SEED_OFFSET,
    DEFAULT_TRAIN_VEHICLES,
    _congestion_from_config,
    _load_city,
    _load_events,
)
from sim.generate import generate_dataset_with_city

app = typer.Typer(add_completion=False)
console = Console()


def _recall_and_candidates_by_hour(
    events: list[DetectionEvent], gate_result
) -> dict[int, dict]:
    """Group real consecutive same-ground-truth-vehicle event pairs by the
    LATER event's hour-of-day (0-23) -- mirroring the gate's own convention
    of reading time-of-day off the later/destination event
    (gating.py's `gate_candidates`, `tod_j = time_of_day_bucket(e_j.
    timestamp)`) -- and report per-hour recall plus mean candidates per
    event. An hour with zero true pairs reports `recall: None`, not a
    misleading 0/0."""
    by_vehicle: dict[str, list[DetectionEvent]] = defaultdict(list)
    for e in events:
        if e.gt_vehicle_id is not None:
            by_vehicle[e.gt_vehicle_id].append(e)

    n_true_by_hour: dict[int, int] = defaultdict(int)
    n_recalled_by_hour: dict[int, int] = defaultdict(int)
    for evs in by_vehicle.values():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        for a, b in zip(evs_sorted[:-1], evs_sorted[1:], strict=True):
            hour = b.timestamp.hour
            n_true_by_hour[hour] += 1
            if a.event_id in gate_result.candidates.get(b.event_id, []):
                n_recalled_by_hour[hour] += 1

    n_candidates_by_hour: dict[int, int] = defaultdict(int)
    n_events_by_hour: dict[int, int] = defaultdict(int)
    for e in events:
        hour = e.timestamp.hour
        n_events_by_hour[hour] += 1
        n_candidates_by_hour[hour] += len(gate_result.candidates.get(e.event_id, []))

    hours = sorted(set(n_true_by_hour) | set(n_events_by_hour))
    by_hour: dict[int, dict] = {}
    for h in hours:
        n_true = n_true_by_hour.get(h, 0)
        n_recalled = n_recalled_by_hour.get(h, 0)
        n_events_h = n_events_by_hour.get(h, 0)
        by_hour[h] = {
            "n_true_pairs": n_true,
            "n_recalled": n_recalled,
            "recall": (n_recalled / n_true) if n_true else None,
            "n_events": n_events_h,
            "mean_candidates_per_event": (
                n_candidates_by_hour.get(h, 0) / n_events_h if n_events_h else 0.0
            ),
        }
    return by_hour


@app.command()
def main(
    data: Path = typer.Option(Path("data/run1"), help="On-disk dataset directory to evaluate."),
    out: Path = typer.Option(
        Path("eval/reports/gate_recall_by_hour.json"), help="Output report path."
    ),
    train_vehicles: int = typer.Option(DEFAULT_TRAIN_VEHICLES, help="Train split vehicle count."),
    train_seed_offset: int = typer.Option(
        DEFAULT_TRAIN_SEED_OFFSET, help="Added to the eval dataset's seed for the train split."
    ),
    congestion_tail_factor: float = typer.Option(
        CONGESTION_TAIL_FACTOR,
        "--congestion-tail-factor",
        help=(
            "Gate.congestion_tail_factor to use (default: the Gate class's own "
            "default). Pass 1.0 to reproduce the pre-fix gate behaviour for "
            "comparison."
        ),
    ),
) -> None:
    config = json.loads((data / "config.json").read_text(encoding="utf-8"))
    console.print(f"[bold]Loading[/bold] {data} (config={config})")

    city = _load_city(data)
    events = _load_events(data)
    console.print(f"[green]Loaded[/green] {len(events)} events")

    train_seed = config["seed"] + train_seed_offset
    congestion, congestion_config = _congestion_from_config(config)
    console.print(
        f"[bold]Fitting priors[/bold] on a separate train split: "
        f"{train_vehicles} vehicles, seed={train_seed}, same city, "
        f"congestion={'on (BPR)' if congestion else 'off (free flow)'} (matching eval dataset)"
    )
    train_ds = generate_dataset_with_city(
        city=city,
        n_vehicles=train_vehicles,
        hours=config["hours"],
        seed=train_seed,
        clone_fraction=config.get("clone_fraction", 0.02),
        congestion=congestion,
        congestion_config=congestion_config,
    )
    models = fit_all(train_ds)
    gate = Gate(model=models.kinematic_model, congestion_tail_factor=congestion_tail_factor)

    console.print(
        f"[bold]Gating[/bold] the full evaluated dataset "
        f"(congestion_tail_factor={congestion_tail_factor})"
    )
    gate_result = gate_candidates(events, gate)
    console.print(
        f"[green]Gated[/green] {len(events)} events, "
        f"mean_candidates_per_event={gate_result.mean_candidates_per_event:.1f}"
    )

    by_hour = _recall_and_candidates_by_hour(events, gate_result)

    console.print("[bold]Recall by hour[/bold]")
    for h in sorted(by_hour):
        row = by_hour[h]
        recall_str = f"{row['recall']:.4f}" if row["recall"] is not None else "n/a"
        console.print(
            f"  {h:02d}h: recall={recall_str} n_true_pairs={row['n_true_pairs']} "
            f"mean_candidates_per_event={row['mean_candidates_per_event']:.1f}"
        )

    report = {
        "data_dir": str(data),
        "config": config,
        "train": {
            "n_vehicles": train_vehicles,
            "seed": train_seed,
            "hours": config["hours"],
            "congestion": congestion,
        },
        "congestion_tail_factor": congestion_tail_factor,
        "n_events": len(events),
        "overall_mean_candidates_per_event": gate_result.mean_candidates_per_event,
        "by_hour": {str(h): row for h, row in sorted(by_hour.items())},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out}")


if __name__ == "__main__":
    app()
