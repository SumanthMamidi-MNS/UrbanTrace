"""Baseline B/C (architecture.md §8: "Baseline B: Fuzzy match, Levenshtein
<=1" / "Baseline C: Fuzzy + hard time gate") on the SAME in-memory dataset
`eval/ablation.py` evaluates on -- identical seeds/config, deterministically
reproduced here (not regenerated with different parameters), so all three
baselines and SUTRA's ablation rows are directly comparable.

Baseline B: within the SAME spatio-temporal gate the real pipeline uses,
link event j to its closest-in-time gated predecessor i whose plate_argmax
strings are within Levenshtein distance <=1 (not exact-match, unlike
Baseline A), chained greedily in time -- otherwise structurally identical
to `eval.run_pipeline`'s Baseline A (no scoring, no global optimisation).

Baseline C: B, plus a KINEMATIC PLAUSIBILITY filter. Originally implemented
as the fitted kinematic model's own hard physical-impossibility check
(`KinematicLRResult.physically_impossible`: dt below the minimum possible
travel time at v_max between the two cameras) -- but that check is
provably a NO-OP downstream of `engine.association.gating`'s own gate
window, which is already tighter than the physical floor for every camera
pair actually observed here (see "Investigated: Baseline B vs C" below), so
B and C came out bit-identical. C is now instead tightened to the LEARNED
travel-time distribution's [5th, 95th] percentile window (per camera-pair +
time-of-day bucket, from the same fitted log-normal `KinematicModel` --
`params_for(cam_i, cam_j, tod)`'s mu/sigma), which the wide z=3.0-sigma gate
window does not itself enforce -- this makes C a meaningfully different,
tighter standard than B (see module-level `Z_90` below for the exact
percentile-to-z-score math). Baseline C is still "no scoring, no global
optimisation", just a less naive standard than B.

Investigated: Baseline B vs C (Day 4 performance/eval pass). Counting how
many of Baseline C's gate-admitted, edit-distance-passing candidates the
OLD hard-physical-impossibility filter actually rejected (on this exact
dataset/config): 0 of ~13,400. The gate's own window
(`engine.association.gating.GATE_Z_SCORE = 3.0` standard deviations around
the fitted log-normal mean) is, for every camera pair the simulator
produces, already narrower at its lower edge than `dist / v_max_kmh`'s hard
physical floor -- so nothing the gate ever admits can fail that check. This
is correct-by-construction given the gate's parameters, not a bug in the
old filter's logic; it just made Baseline C indistinguishable from B, which
is why C is redefined above to use the tighter learned-percentile window
instead.

Usage:
    python -m eval.baselines
"""

import json
import math
from pathlib import Path

from scipy.stats import norm

from engine.association.gating import Gate, gate_candidates
from engine.calibration.fit_priors import generate_train_holdout_datasets
from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import Trajectory
from engine.scoring.kinematic_lr import KinematicModel, fit_kinematic_model, time_of_day_bucket
from eval.ablation import (
    CITY_SEED,
    CLONE_FRACTION,
    EVAL_SEED,
    HOURS,
    N_CAMERAS,
    N_VEHICLES_EVAL,
    N_VEHICLES_TRAIN,
    TRAIN_SEED,
)
from eval.metrics import TrajectoryMetrics, compute_trajectory_metrics
from eval.run_pipeline import _build_exact_match_baseline
from eval.stratified import edit_distance

OUT_PATH = Path("eval/reports/baselines.json")

MAX_EDIT_DISTANCE_B = 1

# Baseline C's tightened kinematic filter: reject a candidate whose transit
# time falls outside the fitted log-normal's [5th, 95th] percentile window
# (`exp(mu +/- Z_90 * sigma)`), not just outside the physically-possible
# range. `norm.ppf(0.95)` is the two-sided 90%-coverage z-score (5% rejected
# on each tail).
Z_90 = norm.ppf(0.95)


def _outside_learned_percentile_window(
    kinematic_model: KinematicModel,
    cam_i: str,
    t_i,
    cam_j: str,
    t_j,
) -> bool:
    """True if (cam_i, t_i) -> (cam_j, t_j)'s transit time falls outside the
    fitted log-normal's central 90% mass for that camera-pair + time-of-day
    bucket (module docstring's tightened Baseline C filter). Time-of-day
    bucket is read off the EARLIER event, matching `KinematicModel.score`'s
    own convention."""
    dt = (t_j - t_i).total_seconds()
    if dt <= 0:
        return True
    tod = time_of_day_bucket(t_i)
    params = kinematic_model.params_for(cam_i, cam_j, tod)
    lo = math.exp(params.mu - Z_90 * params.sigma)
    hi = math.exp(params.mu + Z_90 * params.sigma)
    return not (lo <= dt <= hi)


def _build_fuzzy_baseline(
    events: list[DetectionEvent],
    gate: Gate,
    kinematic_model: KinematicModel | None,
    max_edit_distance: int = MAX_EDIT_DISTANCE_B,
) -> list[Trajectory]:
    """Baseline B (`kinematic_model=None`) / Baseline C (`kinematic_model`
    given) -- see module docstring."""
    events_sorted = sorted(events, key=lambda e: e.timestamp)
    events_by_id = {e.event_id: e for e in events_sorted}
    gate_result = gate_candidates(events_sorted, gate)

    used_as_predecessor: set[str] = set()
    chain_next: dict[str, str] = {}
    chain_prev: dict[str, str] = {}

    for e_j in events_sorted:
        for cand_id in gate_result.candidates.get(e_j.event_id, []):
            if cand_id in used_as_predecessor:
                continue
            cand = events_by_id[cand_id]
            if edit_distance(cand.plate_argmax, e_j.plate_argmax) > max_edit_distance:
                continue
            if kinematic_model is not None and _outside_learned_percentile_window(
                kinematic_model, cand.camera_id, cand.timestamp, e_j.camera_id, e_j.timestamp
            ):
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
    kinematic_model = fit_kinematic_model(train_ds.events, train_ds.city)
    gate = Gate(model=kinematic_model)

    events = eval_ds.events
    baseline_a = _build_exact_match_baseline(events, gate)
    baseline_b = _build_fuzzy_baseline(events, gate, kinematic_model=None)
    baseline_c = _build_fuzzy_baseline(events, gate, kinematic_model=kinematic_model)

    metrics_a = compute_trajectory_metrics(baseline_a, events)
    metrics_b = compute_trajectory_metrics(baseline_b, events)
    metrics_c = compute_trajectory_metrics(baseline_c, events)

    print(f"Baseline A (exact match):        idf1={metrics_a.idf1:.4f}")
    print(f"Baseline B (fuzzy <=1):           idf1={metrics_b.idf1:.4f}")
    print(f"Baseline C (fuzzy <=1 + kinematic): idf1={metrics_c.idf1:.4f}")

    return {
        "description": (
            "Baselines A/B/C (architecture.md §8) on the IDENTICAL in-memory dataset "
            "eval/ablation.py evaluates on (same seeds/config: "
            f"{N_CAMERAS} cameras, {N_VEHICLES_EVAL} eval vehicles, {HOURS}h, "
            f"clone_fraction={CLONE_FRACTION}, train {N_VEHICLES_TRAIN} vehicles different "
            "seed same city) -- directly comparable to eval/reports/ablation.json's rows. "
            "Baseline A: exact plate match, same gate, greedy time chaining "
            "(eval.run_pipeline._build_exact_match_baseline). Baseline B: plate_argmax "
            "Levenshtein <=1 within the same gate, greedy time chaining. Baseline C: B plus "
            "a kinematic-plausibility filter tightened to the fitted log-normal's [5th, 95th] "
            "percentile transit-time window per (camera-pair, time-of-day) -- NOT the original "
            "hard physical-impossibility check (KinematicLRResult.physically_impossible), which "
            "was measured (see module docstring, 'Investigated: Baseline B vs C') to reject 0 "
            "of ~13,400 gate-admitted candidates on this dataset because the gate's own window "
            "(z=3.0 sigma) is already tighter than the physical v_max floor for every observed "
            "camera pair here -- correct-by-construction, not a bug, but it made B and C "
            "bit-identical, so C was redefined to use the tighter learned window instead. "
            "All three: no scoring, no global optimisation. "
            "Measured on the FIXED engine (per-successor prior + corrected arc cost)."
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
        "baseline_a_exact_match": _metrics_to_dict(metrics_a),
        "baseline_b_fuzzy_levenshtein1": _metrics_to_dict(metrics_b),
        "baseline_c_fuzzy_plus_kinematic": _metrics_to_dict(metrics_c),
    }


def main() -> None:
    report = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
