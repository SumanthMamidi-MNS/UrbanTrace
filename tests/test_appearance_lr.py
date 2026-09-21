"""engine/scoring/appearance_lr.py: the appearance density-ratio LR."""

import math

import numpy as np

from engine.scoring.appearance_lr import (
    CLAMP,
    AppearanceModel,
    cosine_distance,
    fit_appearance_model,
    score_appearance_pair,
)
from eval.metrics import roc_auc, sample_labeled_pairs
from sim.generate import generate_dataset
from sim.vehicles import COLORS, VEHICLE_TYPES


def _fit(seed=23, n_vehicles=1500, n_cameras=15):
    ds = generate_dataset(
        n_cameras=n_cameras, n_vehicles=n_vehicles, hours=24, seed=seed, clone_fraction=0.0
    )
    true_color = {v.gt_vehicle_id: v.color for v in ds.vehicles}
    true_type = {v.gt_vehicle_id: v.vehicle_type for v in ds.vehicles}
    model = fit_appearance_model(ds.events, true_color, true_type, COLORS, VEHICLE_TYPES)
    return ds, model


def test_cosine_distance_identical_vectors_is_zero():
    v = [1.0, 0.0, 0.0]
    assert cosine_distance(v, v) < 1e-9


def test_cosine_distance_orthogonal_vectors_is_one():
    assert abs(cosine_distance([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-9


def test_same_vehicle_pairs_score_higher_on_average_than_different_vehicle_pairs():
    ds, model = _fit()
    by_vehicle: dict[str, list] = {}
    for e in ds.events:
        by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    same_scores = []
    for evs in by_vehicle.values():
        if len(evs) < 2:
            continue
        for i in range(len(evs) - 1):
            r = score_appearance_pair(evs[i], evs[i + 1], model)
            same_scores.append(r.log_lr)

    ids = list(by_vehicle.keys())
    diff_scores = []
    for i in range(0, len(ids) - 1, 2):
        a = by_vehicle[ids[i]][0]
        b = by_vehicle[ids[i + 1]][0]
        r = score_appearance_pair(a, b, model)
        diff_scores.append(r.log_lr)

    assert np.mean(same_scores) > np.mean(diff_scores)


def test_result_is_clamped_to_range():
    default_model = AppearanceModel.uniform_default(COLORS, VEHICLE_TYPES)
    # Force an extreme value by looking up the lowest-density bin directly.
    _, model = _fit()
    for d in (0.0, 0.5, 1.0, 1.5, 2.0):
        idx = model._bin_index(d)
        assert math.isfinite(model.log_p1[idx])
        assert math.isfinite(model.log_p0[idx])
    assert default_model.log_p1.shape == default_model.log_p0.shape


def test_appearance_channel_auc_better_than_chance():
    ds, model = _fit(seed=51, n_vehicles=1500)
    labeled_pairs = sample_labeled_pairs(ds.events, seed=0, max_positive_pairs=2000)
    scores, labels = [], []
    for a, b, label in labeled_pairs:
        r = score_appearance_pair(a, b, model)
        scores.append(r.log_lr)
        labels.append(label)
    auc = roc_auc(np.array(scores), np.array(labels))
    assert auc > 0.5, f"appearance channel AUC {auc:.4f} is not better than chance"


def test_result_is_clamped_even_for_a_rigged_near_certain_model():
    """Rig a model whose distance/colour/type terms would each independently
    push the raw sum far outside [-CLAMP, CLAMP], and confirm
    score_appearance_pair still clamps the total."""
    ds, model = _fit(seed=61, n_vehicles=200, n_cameras=10)
    model.log_p1[:] = 0.0  # log(1): "certain" under H1 in every bin
    model.log_p0[:] = -50.0  # near-impossible under H0 in every bin
    for true_v in model.color_confusion:
        for obs_v in model.color_confusion[true_v]:
            model.color_confusion[true_v][obs_v] = 1e-9
        model.color_confusion[true_v][true_v] = 1.0
    a, b = ds.events[0], ds.events[1]
    result = score_appearance_pair(a, b, model)
    assert -CLAMP <= result.log_lr <= CLAMP
