"""Gate recall + mean-candidates-per-event at production scale
(docs/decisions.md, "Day 3a", Part 1). Run against data/run1 (50 cameras,
20000 vehicles, 24h) -- the same dataset Part 2 (blocking) and Part 5
(trajectory metrics) are evaluated on.

Usage:
    python -m eval.gating_report
"""

import json
import time
from pathlib import Path

from engine.association.gating import Gate, gate_candidates, true_predecessor_recall
from engine.contracts.city import CityConfig
from engine.contracts.store import EventStore
from engine.scoring.kinematic_lr import fit_kinematic_model

RUN_DIR = Path("data/run1")
OUT_PATH = Path("eval/reports/gating.json")


def build_report(run_dir: Path = RUN_DIR) -> dict:
    city = CityConfig.model_validate_json((run_dir / "city.json").read_text(encoding="utf-8"))
    store = EventStore(run_dir / "events.jsonl", run_dir / "embeddings.npy")

    t0 = time.time()
    events = store.read_all()
    t1 = time.time()

    model = fit_kinematic_model(events, city)
    t2 = time.time()

    gate = Gate(model=model)
    result = gate_candidates(events, gate)
    t3 = time.time()

    recall = true_predecessor_recall(events, result)
    t4 = time.time()

    naive_mean_candidates = (result.n_events - 1) / 2

    return {
        "description": (
            "Gate recall and mean candidates/event at production scale "
            "(data/run1). Gating is the primary blocker (architecture.md "
            "§3, §7) -- recall here caps what everything downstream can "
            "ever recover."
        ),
        "n_events": result.n_events,
        "mean_candidates_per_event": result.mean_candidates_per_event,
        "naive_mean_candidates_per_event": naive_mean_candidates,
        "cut_fraction_vs_naive": 1.0 - result.mean_candidates_per_event / naive_mean_candidates,
        "recall": recall.recall,
        "n_true_predecessor_pairs": recall.n_true_pairs,
        "n_recalled": recall.n_recalled,
        "timing_s": {
            "load_events": t1 - t0,
            "fit_kinematic_model": t2 - t1,
            "gate_candidates": t3 - t2,
            "recall_check": t4 - t3,
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
