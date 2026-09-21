"""First end-to-end trajectory numbers (docs/decisions.md "Day 3a" Part 5).

Pairwise channel AUC saturated in Day 2 (1.0000 on three of four strata)
and is now a weak, misleading proxy -- it says nothing about whether the
full pipeline (gating -> fusion -> windowed min-cost flow) actually
reconstructs correct trajectories. IDF1 and its components are the real
metric from here on.

Runs the full pipeline on the SAME dataset as data/run1 (50 cameras, 20000
vehicles, 24h, seed=42 -- regenerated in-memory via `generate_dataset` for
byte-identical results per tests/test_determinism.py, rather than round-
tripping through EventStore, purely so the entry/exit-cost and appearance
model fitting can reuse `engine.calibration.fit_priors.fit_all` directly
without hand-reconstructing Vehicle metadata from ground_truth.json).

Usage:
    python -m eval.trajectory_report
"""

import json
import time
from pathlib import Path

from engine.association.gating import Gate
from engine.association.mincostflow import fit_entry_exit_costs
from engine.association.window import solve_windowed
from engine.calibration.fit_priors import fit_all
from engine.scoring.fusion import FusionModel
from eval.metrics import compute_trajectory_metrics
from sim.generate import generate_dataset

OUT_PATH = Path("eval/reports/trajectory_metrics.json")

RUN1_CONFIG = {"n_cameras": 50, "n_vehicles": 20000, "hours": 24, "seed": 42}


def build_report() -> dict:
    t0 = time.time()
    ds = generate_dataset(**RUN1_CONFIG)
    t1 = time.time()

    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)
    t2 = time.time()

    result = solve_windowed(ds.events, gate, fusion_model, entry_exit, ds.city)
    t3 = time.time()

    metrics = compute_trajectory_metrics(result.trajectories, ds.events)
    t4 = time.time()

    pipeline_wall_time_s = t3 - t0
    events_per_sec = len(ds.events) / pipeline_wall_time_s if pipeline_wall_time_s > 0 else 0.0

    return {
        "description": (
            "First end-to-end trajectory-level result (docs/decisions.md "
            "'Day 3a' Part 5). Pairwise AUC saturated and is no longer a "
            "meaningful proxy -- IDF1 and its components are the real "
            "metric from here on."
        ),
        "config": RUN1_CONFIG,
        "n_events": len(ds.events),
        "n_gt_vehicles": metrics.n_gt_vehicles,
        "n_predicted_trajectories": metrics.n_predicted_trajectories,
        "idf1": metrics.idf1,
        "id_precision": metrics.id_precision,
        "id_recall": metrics.id_recall,
        "id_switches": metrics.id_switches,
        "fragmentation": metrics.fragmentation,
        "trajectory_completeness": metrics.trajectory_completeness,
        "idtp": metrics.idtp,
        "idfp": metrics.idfp,
        "idfn": metrics.idfn,
        "n_windows": result.n_windows,
        "timing_s": {
            "generate_dataset": t1 - t0,
            "fit_models": t2 - t1,
            "windowed_solve": t3 - t2,
            "compute_metrics": t4 - t3,
            "pipeline_total_(generate+fit+solve)": pipeline_wall_time_s,
        },
        "events_per_sec": events_per_sec,
    }


def main() -> None:
    report = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
