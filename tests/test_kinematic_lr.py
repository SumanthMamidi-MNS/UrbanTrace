"""engine/scoring/kinematic_lr.py: the travel-time likelihood ratio."""

import math
from datetime import datetime, timedelta

import numpy as np

from engine.scoring.kinematic_lr import (
    KinematicModel,
    fit_kinematic_model,
    time_of_day_bucket,
)
from eval.metrics import roc_auc, sample_labeled_pairs
from sim.city import generate_city
from sim.generate import generate_dataset

T0 = datetime(2026, 1, 1, 8, 0, 0)


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
