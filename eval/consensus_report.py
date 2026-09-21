"""Consensus decoding accuracy report (engine/decode/consensus.py,
architecture.md §3 "Consensus decoding").

Measures whole-plate accuracy of single reads (per-event argmax) vs.
consensus decoding (per-trajectory fused decode), bucketed by trajectory
length, on the first N_EVENTS events of data/run1. Trajectories are formed
by GROUND-TRUTH grouping (events sharing the same `gt_vehicle_id`), not by
running association -- this isolates the consensus-decoding defect/fix from
association/linking errors, matching how the lead originally measured the
"one outlier read vetoes a clear majority" defect this module's EPS
(epsilon-contamination likelihood) constant fixes.

The report is computed twice: once with EPS forced to 0 (the original pure-
product likelihood) and once at the module's tuned `consensus.EPS` (chosen
on a separate-seed tuning dataset -- see consensus.py's EPS comment), so the
before/after improvement is visible directly. Only the tuned-eps table is
written to disk (eval/reports/consensus_accuracy.json); both are printed.

Usage:
    python -m eval.consensus_report
"""

import json
from collections import defaultdict
from pathlib import Path

import engine.decode.consensus as consensus
from engine.contracts.store import EventStore
from engine.decode.consensus import decode_trajectory
from engine.scoring.plate_lr import PlatePriors

RUN_DIR = Path("data/run1")
OUT_PATH = Path("eval/reports/consensus_accuracy.json")
N_EVENTS = 30_000


def _load_first_n_events_grouped_by_gt(run_dir: Path, n_events: int):
    """Read the first `n_events` lines of events.jsonl (already time-ordered
    on disk) and group them by `gt_vehicle_id`. Returns
    (trajectories, true_plate_by_vehicle) where `trajectories` is a list of
    event lists, each with >=1 event and a resolvable ground-truth plate."""
    store = EventStore(run_dir / "events.jsonl", run_dir / "embeddings.npy")
    events = []
    for i, e in enumerate(store):
        if i >= n_events:
            break
        events.append(e)

    gt = json.loads((run_dir / "ground_truth.json").read_text(encoding="utf-8"))
    true_plate_by_vehicle = {vid: v["true_plate"] for vid, v in gt["vehicles"].items()}

    by_vehicle: dict[str, list] = defaultdict(list)
    for e in events:
        if e.gt_vehicle_id is not None and e.gt_vehicle_id in true_plate_by_vehicle:
            by_vehicle[e.gt_vehicle_id].append(e)

    trajectories = list(by_vehicle.values())
    return trajectories, true_plate_by_vehicle


def _bucket_key(n: int) -> str:
    return str(n) if n <= 3 else "4+"


def _accuracy_table(trajectories, true_plate_by_vehicle, priors: PlatePriors) -> dict:
    buckets = {
        b: {"single_correct": 0, "single_n": 0, "cons_correct": 0, "cons_n": 0}
        for b in ("1", "2", "3", "4+")
    }
    for evs in trajectories:
        true_plate = true_plate_by_vehicle[evs[0].gt_vehicle_id]
        b = buckets[_bucket_key(len(evs))]
        for e in evs:
            b["single_n"] += 1
            b["single_correct"] += int(e.plate_argmax == true_plate)
        c = decode_trajectory(evs, priors)
        b["cons_n"] += 1
        b["cons_correct"] += int(c.decoded_plate == true_plate)

    table = {}
    total_single_correct = total_single_n = total_cons_correct = total_cons_n = 0
    for b, row in buckets.items():
        single_acc = row["single_correct"] / row["single_n"] if row["single_n"] else None
        cons_acc = row["cons_correct"] / row["cons_n"] if row["cons_n"] else None
        table[b] = {
            "n_trajectories": row["cons_n"],
            "n_events": row["single_n"],
            "single_read_accuracy": single_acc,
            "consensus_accuracy": cons_acc,
        }
        total_single_correct += row["single_correct"]
        total_single_n += row["single_n"]
        total_cons_correct += row["cons_correct"]
        total_cons_n += row["cons_n"]

    table["overall"] = {
        "n_trajectories": total_cons_n,
        "n_events": total_single_n,
        "single_read_accuracy": total_single_correct / total_single_n if total_single_n else None,
        "consensus_accuracy": total_cons_correct / total_cons_n if total_cons_n else None,
    }
    return table


def build_report(run_dir: Path = RUN_DIR, n_events: int = N_EVENTS) -> dict:
    trajectories, true_plate_by_vehicle = _load_first_n_events_grouped_by_gt(run_dir, n_events)
    priors = PlatePriors.uniform_default()

    original_eps = consensus.EPS
    try:
        consensus.EPS = 0.0
        table_eps0 = _accuracy_table(trajectories, true_plate_by_vehicle, priors)
        consensus.EPS = original_eps
        table_tuned = _accuracy_table(trajectories, true_plate_by_vehicle, priors)
    finally:
        consensus.EPS = original_eps

    return {
        "description": (
            "Single-read vs. consensus whole-plate accuracy by trajectory "
            "length, on the first "
            f"{n_events} events of {run_dir} (ground-truth grouping by "
            "gt_vehicle_id, association-error-free). Reported at eps=0 "
            "(original pure-product likelihood) and at the tuned "
            "epsilon-contamination eps, to show the outlier-veto fix."
        ),
        "run_dir": str(run_dir),
        "n_events_sampled": n_events,
        "n_trajectories": len(trajectories),
        "eps_tuned": original_eps,
        "eps_0": table_eps0,
        "eps_tuned_table": table_tuned,
    }


def main() -> None:
    report = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
