"""Channel ablation study (architecture.md §8: "plate-only -> +kinematics ->
+appearance -> +global flow") on ONE in-memory dataset. Generated entirely
in memory via `sim.generate.generate_train_holdout_datasets` -- no disk
writes for the dataset itself, only the final JSON report.

Uses `engine.scoring.fusion.FusionModel.channels` (added for exactly this
purpose) to mask out channels from the FUSED TOTAL that gating/pruning/the
min-cost-flow solver actually rank and solve on, while every channel's own
log-LR is still computed underneath -- so `engine.association.window.
solve_windowed` (the REAL production pipeline: gate -> fuse -> windowed
min-cost flow) can be run unmodified for every channel combination.

The final row runs the SAME "all three channels" evidence through a GREEDY
nearest-best chaining heuristic instead of the windowed min-cost-flow
solver, to isolate what the GLOBAL optimisation buys over a locally-greedy
assignment given identical evidence. It scores every gated (predecessor,
successor) pair ONCE in a single batch (same `score_pairs_batch_totals`
call the production solver's arc-building uses, including the corrected
per-successor prior), then -- in ascending time order of successor events --
greedily attaches each to its highest-scoring STILL-AVAILABLE predecessor,
accepting only if `total_log_odds > 0` (posterior odds > 1), mirroring the
corrected arc-cost formula's linking condition. Unlike the production
solver this does NOT additionally weigh each event's own border/interior
entry/exit cost against the alternative of standing alone -- it isolates
GREEDY-vs-GLOBALLY-OPTIMAL assignment under identical evidence, not a full
replication of the flow network's cost accounting. It is run UNWINDOWED
(the whole small eval dataset at once): windowing is a production
streaming/scalability device, orthogonal to the greedy-vs-optimal question
this row exists to answer.

Usage:
    python -m eval.ablation
"""

import json
import time
from pathlib import Path

import numpy as np

from engine.association.gating import Gate, gate_candidates
from engine.association.mincostflow import fit_entry_exit_costs
from engine.association.window import solve_windowed
from engine.calibration.fit_priors import fit_all, generate_train_holdout_datasets
from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import Trajectory
from engine.scoring.fusion import CHANNEL_NAMES, FusionModel, score_pairs_batch_totals
from eval.metrics import TrajectoryMetrics, compute_trajectory_metrics

OUT_PATH = Path("eval/reports/ablation.json")

N_CAMERAS = 20
N_VEHICLES_TRAIN = 1500
N_VEHICLES_EVAL = 2500
HOURS = 4
CLONE_FRACTION = 0.02
CITY_SEED = 5101
TRAIN_SEED = 6101
EVAL_SEED = 7101

# (row name, channel mask, method)
CONFIGS: list[tuple[str, frozenset[str], str]] = [
    ("plate_only", frozenset({"plate"}), "min_cost_flow"),
    ("plate_plus_kinematic", frozenset({"plate", "kinematic"}), "min_cost_flow"),
    ("plate_plus_appearance", frozenset({"plate", "appearance"}), "min_cost_flow"),
    ("kinematic_plus_appearance_no_plate", frozenset({"kinematic", "appearance"}), "min_cost_flow"),
    ("all_three", CHANNEL_NAMES, "min_cost_flow"),
    ("all_three_greedy_chaining", CHANNEL_NAMES, "greedy"),
]


def _greedy_chain_trajectories(
    events: list[DetectionEvent], gate: Gate, fusion_model: FusionModel
) -> list[Trajectory]:
    events_sorted = sorted(events, key=lambda e: e.timestamp)
    index_of = {e.event_id: i for i, e in enumerate(events_sorted)}
    events_by_id = {e.event_id: e for e in events_sorted}

    gate_result = gate_candidates(events_sorted, gate)

    pred_idx: list[int] = []
    succ_idx: list[int] = []
    n_cand: list[int] = []
    for succ_id, pred_ids in gate_result.candidates.items():
        n_j = len(pred_ids)
        for pred_id in pred_ids:
            pred_idx.append(index_of[pred_id])
            succ_idx.append(index_of[succ_id])
            n_cand.append(n_j)

    candidates_by_succ: dict[str, list[tuple[str, float]]] = {}
    if pred_idx:
        totals = score_pairs_batch_totals(
            events_sorted,
            np.array(pred_idx),
            np.array(succ_idx),
            fusion_model,
            n_candidates_per_pair=np.array(n_cand),
        )
        for k in range(len(pred_idx)):
            succ_id = events_sorted[succ_idx[k]].event_id
            pred_id = events_sorted[pred_idx[k]].event_id
            candidates_by_succ.setdefault(succ_id, []).append((pred_id, float(totals[k])))
        for lst in candidates_by_succ.values():
            lst.sort(key=lambda t: t[1], reverse=True)

    used_as_predecessor: set[str] = set()
    chain_next: dict[str, str] = {}
    chain_prev: dict[str, str] = {}
    for e in events_sorted:
        for pred_id, score in candidates_by_succ.get(e.event_id, []):
            if score <= 0:
                break  # sorted descending -- no remaining candidate can beat 0
            if pred_id in used_as_predecessor:
                continue
            chain_next[pred_id] = e.event_id
            chain_prev[e.event_id] = pred_id
            used_as_predecessor.add(pred_id)
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
                trajectory_id=f"greedy_{i:06d}",
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
    }


def build_report() -> dict:
    train_ds, eval_ds = generate_train_holdout_datasets(
        n_cameras=N_CAMERAS,
        n_vehicles_train=N_VEHICLES_TRAIN,
        n_vehicles_holdout=N_VEHICLES_EVAL,
        hours=HOURS,
        city_seed=CITY_SEED,
        train_seed=TRAIN_SEED,
        holdout_seed=EVAL_SEED,
        clone_fraction=CLONE_FRACTION,
    )
    models = fit_all(train_ds)
    entry_exit = fit_entry_exit_costs(train_ds.events, train_ds.city)
    gate = Gate(model=models.kinematic_model)

    rows: dict[str, dict] = {}
    for name, channels, method in CONFIGS:
        fusion_model = FusionModel(
            plate_priors=models.plate_priors,
            kinematic_model=models.kinematic_model,
            appearance_model=models.appearance_model,
            channels=channels,
        )
        t0 = time.time()
        if method == "min_cost_flow":
            result = solve_windowed(eval_ds.events, gate, fusion_model, entry_exit, train_ds.city)
            trajectories = result.trajectories
        else:
            trajectories = _greedy_chain_trajectories(eval_ds.events, gate, fusion_model)
        elapsed = time.time() - t0
        metrics = compute_trajectory_metrics(trajectories, eval_ds.events)
        rows[name] = {
            "channels": sorted(channels),
            "method": method,
            **_metrics_to_dict(metrics),
            "elapsed_s": elapsed,
        }
        print(f"  {name}: idf1={metrics.idf1:.4f} n_pred={metrics.n_predicted_trajectories} "
              f"n_gt={metrics.n_gt_vehicles} id_switches={metrics.id_switches} "
              f"fragmentation={metrics.fragmentation} ({elapsed:.1f}s)", flush=True)

    return {
        "description": (
            "Channel ablation (architecture.md §8) on ONE in-memory dataset "
            f"({N_CAMERAS} cameras, {N_VEHICLES_EVAL} eval vehicles, {HOURS}h, "
            f"clone_fraction={CLONE_FRACTION}; train split {N_VEHICLES_TRAIN} vehicles, "
            "different seed, SAME city, per architecture.md §10). Priors/kinematic/"
            "appearance models fitted on the train split only. Every 'min_cost_flow' row "
            "runs the REAL production pipeline (engine.association.window.solve_windowed, "
            "unmodified) with FusionModel.channels masking out channels from the fused "
            "total; the 'all_three_greedy_chaining' row replaces that solver with a "
            "greedy nearest-best chaining heuristic over the SAME evidence, run UNWINDOWED "
            "on the whole eval dataset at once -- see module docstring for exactly how "
            "each row was produced. Measured on the FIXED engine (per-successor prior + "
            "corrected arc cost)."
        ),
        "dataset": {
            "n_cameras": N_CAMERAS,
            "n_vehicles_train": N_VEHICLES_TRAIN,
            "n_vehicles_eval": N_VEHICLES_EVAL,
            "hours": HOURS,
            "clone_fraction": CLONE_FRACTION,
            "city_seed": CITY_SEED,
            "train_seed": TRAIN_SEED,
            "eval_seed": EVAL_SEED,
        },
        "configs": rows,
        "notes": {
            "no_plate_row": (
                "kinematic_plus_appearance_no_plate shows how far kinematics+appearance "
                "get WITHOUT the near-unique plate identifier -- the honest floor the "
                "plate channel's contribution is measured against."
            ),
            "greedy_row": (
                "all_three_greedy_chaining uses the identical fused evidence as all_three "
                "but a greedy (not globally optimal) assignment -- the gap between these "
                "two rows is what the min-cost-flow global solver is worth."
            ),
        },
    }


def main() -> None:
    report = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
