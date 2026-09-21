"""Secondary blocking engagement rate + recall cost at production scale
(docs/decisions.md, "Day 3a", Part 2). Run against data/run1.

Usage:
    python -m eval.blocking_report
"""

import json
import time
from pathlib import Path

from engine.association.blocking import (
    DEFAULT_CANDIDATE_BUDGET,
    DEFAULT_MAX_EDIT_DISTANCE,
    apply_blocking,
    measure_recall_cost,
)
from engine.association.gating import Gate, gate_candidates
from engine.contracts.city import CityConfig
from engine.contracts.store import EventStore
from engine.scoring.kinematic_lr import fit_kinematic_model

RUN_DIR = Path("data/run1")
OUT_PATH = Path("eval/reports/blocking.json")


def build_report(run_dir: Path = RUN_DIR) -> dict:
    city = CityConfig.model_validate_json((run_dir / "city.json").read_text(encoding="utf-8"))
    store = EventStore(run_dir / "events.jsonl", run_dir / "embeddings.npy")
    events = store.read_all()
    events_by_id = {e.event_id: e for e in events}

    model = fit_kinematic_model(events, city)
    gate = Gate(model=model)
    gate_result = gate_candidates(events, gate)

    t0 = time.time()
    blocked_map, engage_stats = apply_blocking(
        gate_result.candidates, events_by_id, DEFAULT_CANDIDATE_BUDGET, DEFAULT_MAX_EDIT_DISTANCE
    )
    t1 = time.time()

    recall_stats = measure_recall_cost(events, gate_result.candidates, blocked_map)

    return {
        "description": (
            "Secondary (plate-based) blocking must be an overflow path, not "
            "the default one (architecture.md §7). Engagement rate is the "
            "fraction of events whose gated candidate set exceeded the "
            f"budget ({DEFAULT_CANDIDATE_BUDGET}); recall cost is measured "
            "ONLY on true predecessor pairs the gate itself already "
            "recalled, isolating blocking's own cost from the gate's."
        ),
        "config": {
            "candidate_budget": DEFAULT_CANDIDATE_BUDGET,
            "max_edit_distance": DEFAULT_MAX_EDIT_DISTANCE,
        },
        "n_events_total": engage_stats.n_events_total,
        "n_events_engaged": engage_stats.n_events_engaged,
        "engagement_rate": engage_stats.engagement_rate,
        "n_true_pairs_checked": recall_stats.n_true_pairs_checked,
        "n_true_pairs_dropped_by_blocking": recall_stats.n_true_pairs_dropped_by_blocking,
        "recall_cost": recall_stats.recall_cost,
        "blocking_wall_time_s": t1 - t0,
    }


def main() -> None:
    report = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
