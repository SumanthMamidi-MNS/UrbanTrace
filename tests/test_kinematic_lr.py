"""engine/scoring/kinematic_lr.py: the travel-time likelihood ratio."""

import math
from datetime import datetime, timedelta

import numpy as np

from engine.association.gating import Gate, gate_candidates
from engine.calibration.fit_priors import (
    fit_negative_kinematic_model,
    generate_train_holdout_datasets,
)
from engine.scoring.kinematic_lr import (
    KinematicModel,
    fit_kinematic_model,
    time_of_day_bucket,
)
from eval.metrics import roc_auc, sample_labeled_pairs
from sim.city import generate_city
from sim.generate import generate_dataset

T0 = datetime(2026, 1, 1, 8, 0, 0)


def _fit_before_after():
    """Shared fixture-ish helper for the kinematic-defect-fix tests below:
    p1-only ("before", the model as `fit_kinematic_model` alone produces --
    still what every OTHER caller of it gets, see `fit_negative_kinematic_
    model`'s docstring) vs. p1+empirically-fit-p0 ("after",
    `engine.calibration.fit_priors.fit_all`'s real production path)."""
    train_ds, _ = generate_train_holdout_datasets(
        n_cameras=15,
        n_vehicles_train=800,
        n_vehicles_holdout=500,
        hours=6,
        city_seed=11,
        train_seed=21,
        holdout_seed=31,
        clone_fraction=0.02,
    )
    events = train_ds.events
    model_before = fit_kinematic_model(events, train_ds.city)
    gate = Gate(model=model_before)
    gate_result = gate_candidates(events, gate)
    model_after = fit_negative_kinematic_model(model_before, events)
    return events, gate_result, model_before, model_after


def _gated_negative_log_lrs(events, gate_result, model) -> list[float]:
    """log_lr for every GATED pair the ground truth says is a DIFFERENT
    vehicle -- exactly `fit_negative_kinematic_priors`'s own negative
    sample space, scored through `model`."""
    events_by_id = {e.event_id: e for e in events}
    out = []
    for succ_id, pred_ids in gate_result.candidates.items():
        succ = events_by_id.get(succ_id)
        if succ is None or succ.gt_vehicle_id is None:
            continue
        for pred_id in pred_ids:
            pred = events_by_id.get(pred_id)
            if pred is None or pred.gt_vehicle_id is None:
                continue
            if pred.gt_vehicle_id == succ.gt_vehicle_id:
                continue
            result = model.score(pred.camera_id, pred.timestamp, succ.camera_id, succ.timestamp)
            out.append(result.log_lr)
    return out


def _consecutive_positive_log_lrs(events, model) -> list[float]:
    """log_lr for every literal consecutive (adjacent) same-vehicle passage
    -- a genuine positive link, unlike "any gated pair sharing a
    gt_vehicle_id" (which can include non-adjacent multi-hop pairs the
    kinematic channel is CORRECTLY not confident about)."""
    by_vehicle: dict[str, list] = {}
    for e in events:
        if e.gt_vehicle_id is not None:
            by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)
    out = []
    for evs in by_vehicle.values():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        for a, b in zip(evs_sorted[:-1], evs_sorted[1:], strict=True):
            if (b.timestamp - a.timestamp).total_seconds() <= 0:
                continue
            result = model.score(a.camera_id, a.timestamp, b.camera_id, b.timestamp)
            out.append(result.log_lr)
    return out


def test_negative_gated_pairs_no_longer_get_an_inflated_positive_lr():
    """THE kinematic-defect fix (docs/decisions.md): p0(dt) used to be a
    flat density over the whole gate window, which is misspecified -- a
    different vehicle admitted by the SAME spatio-temporal gate is not
    arriving at a uniformly random time, it's on the same road, so its dt
    clusters near the free-flow time too. That silently rewarded roughly
    HALF of all gated negative pairs with a positive kinematic LR (measured
    below), which is exactly what drove plate+kinematic's over-merging
    (2,347 predicted trajectories for 2,498 true vehicles in
    eval/reports/ablation.json). After empirically fitting p0 on the
    train split's own gated negatives, the negative population is clearly
    centred well below zero and only rarely positive."""
    events, gate_result, model_before, model_after = _fit_before_after()
    before = np.array(_gated_negative_log_lrs(events, gate_result, model_before))
    after = np.array(_gated_negative_log_lrs(events, gate_result, model_after))
    before_finite = before[np.isfinite(before)]
    after_finite = after[np.isfinite(after)]
    assert len(after_finite) > 1000, "too few gated negative pairs to measure reliably"

    frac_positive_before = float((before_finite > 0).mean())
    mean_after = float(after_finite.mean())
    frac_positive_after = float((after_finite > 0).mean())
    print(
        f"\nnegative gated pairs: before mean={before_finite.mean():.4f} "
        f"frac_positive={frac_positive_before:.4f} | "
        f"after mean={mean_after:.4f} frac_positive={frac_positive_after:.4f}"
    )

    # Historical defect, still reproducible via the uniform-H0 fallback
    # (`KinematicModel.neg_priors is None`): a large minority of DIFFERENT-
    # vehicle gated pairs got an outright positive kinematic endorsement.
    assert frac_positive_before > 0.3, (
        "expected the flat-H0 fallback to reproduce the historical defect "
        f"(>=30% of gated negatives falsely positive), got {frac_positive_before:.4f}"
    )

    # The fix: negatives are now clearly negative on average and only
    # rarely positive.
    assert mean_after < 0, f"mean kinematic LR over negatives is {mean_after:.4f}, expected < 0"
    assert frac_positive_after < 0.3, (
        f"{frac_positive_after:.1%} of gated negatives still score positive, expected < 30%"
    )


def test_positive_pairs_are_still_clearly_positive_after_the_fix():
    """The other half of the fix's contract: fixing the over-confident
    negative side must not zero out the channel's real signal on genuine
    (consecutive, same-vehicle) links."""
    events, _gate_result, _model_before, model_after = _fit_before_after()
    after = np.array(_consecutive_positive_log_lrs(events, model_after))
    after_finite = after[np.isfinite(after)]
    assert len(after_finite) > 100, "too few positive pairs to measure reliably"

    mean_after = float(after_finite.mean())
    frac_positive_after = float((after_finite > 0).mean())
    print(
        f"\npositive (consecutive) pairs after: mean={mean_after:.4f} "
        f"frac_positive={frac_positive_after:.4f}"
    )

    assert mean_after > 0, f"mean kinematic LR over positives is {mean_after:.4f}, expected > 0"
    assert frac_positive_after > 0.6, (
        f"only {frac_positive_after:.1%} of genuine same-vehicle links score positive, "
        f"expected a clear majority (> 60%)"
    )


def _model(n_cameras=15, seed=5) -> tuple[KinematicModel, list[str]]:
    city = generate_city(n_cameras=n_cameras, seed=seed)
    model = KinematicModel(city=city, pair_params={})
    return model, [c.camera_id for c in city.cameras]


def test_hard_gate_kills_physically_impossible_transit():
    model, cams = _model()
    shortest = model._shortest(cams[0], cams[-1])
    assert shortest is not None
    dist_m, _, _ = shortest
    # 1 second is impossible for any non-trivial distance at any realistic speed.
    result = model.score(cams[0], T0, cams[-1], T0 + timedelta(seconds=1))
    assert result.physically_impossible is True
    assert result.log_lr == float("-inf")
    assert dist_m > 0


def test_realistic_transit_is_not_gated():
    model, cams = _model()
    shortest = model._shortest(cams[0], cams[1])
    assert shortest is not None
    _, time_s, _ = shortest
    result = model.score(cams[0], T0, cams[1], T0 + timedelta(seconds=time_s * 1.2))
    assert result.physically_impossible is False
    assert math.isfinite(result.log_lr)


def test_negative_or_zero_dt_is_impossible():
    model, cams = _model()
    result = model.score(cams[0], T0, cams[1], T0 - timedelta(seconds=5))
    assert result.physically_impossible is True
    assert result.log_lr == float("-inf")


def test_unmodelled_pair_still_scores_via_road_graph_fallback():
    """A camera pair with zero training observations must never be left
    unmodelled -- it should fall back to a finite road-graph-based estimate,
    not blow up or silently skip scoring."""
    model, cams = _model()
    assert model.pair_params == {}
    shortest = model._shortest(cams[2], cams[3])
    assert shortest is not None
    _, time_s, _ = shortest
    result = model.score(cams[2], T0, cams[3], T0 + timedelta(seconds=time_s))
    assert math.isfinite(result.log_lr) or result.physically_impossible


def test_time_of_day_buckets_cover_all_hours_disjointly():
    seen = set()
    for h in range(24):
        b = time_of_day_bucket(T0.replace(hour=h))
        seen.add(b)
    assert seen == {"night", "early", "am_rush", "midday", "pm_rush", "evening"}


def test_skipped_cameras_penalty_is_applied():
    """A path through intermediate cameras should be penalised relative to
    the same channel with no penalty -- verify the log_lr difference equals
    exactly n_skipped * log(p_miss)."""
    ds = generate_dataset(n_cameras=15, n_vehicles=500, hours=24, seed=9, clone_fraction=0.0)
    model = fit_kinematic_model(ds.events, ds.city)
    # find a camera pair with at least one skipped intermediate camera
    for cam_i in model.node_by_camera:
        for cam_j in model.node_by_camera:
            if cam_i == cam_j:
                continue
            shortest = model._shortest(cam_i, cam_j)
            if shortest and len(shortest[2]) >= 1:
                result = model.score(cam_i, T0, cam_j, T0 + timedelta(seconds=shortest[1]))
                if result.physically_impossible:
                    continue
                # recompute without the skip penalty
                tod = time_of_day_bucket(T0)
                params = model.params_for(cam_i, cam_j, tod)
                dt = shortest[1]
                from engine.scoring.kinematic_lr import _lognormal_logpdf

                log_h1 = _lognormal_logpdf(dt, params.mu, params.sigma)
                log_h0 = -math.log(model.dt_max - model.dt_min)
                expected_no_penalty = log_h1 - log_h0
                expected_with_penalty = expected_no_penalty + len(shortest[2]) * math.log(
                    model.p_miss
                )
                assert abs(result.log_lr - expected_with_penalty) < 1e-9
                return
    raise AssertionError("no camera pair with a skipped intermediate camera was found")


def test_kinematic_channel_auc_better_than_chance():
    ds = generate_dataset(n_cameras=15, n_vehicles=1500, hours=24, seed=41, clone_fraction=0.0)
    model = fit_kinematic_model(ds.events, ds.city)
    labeled_pairs = sample_labeled_pairs(ds.events, seed=0, max_positive_pairs=2000)
    scores, labels = [], []
    for a, b, label in labeled_pairs:
        result = model.score(a.camera_id, a.timestamp, b.camera_id, b.timestamp)
        scores.append(result.log_lr)
        labels.append(label)
    auc = roc_auc(np.array(scores), np.array(labels))
    assert auc > 0.5, f"kinematic channel AUC {auc:.4f} is not better than chance"
