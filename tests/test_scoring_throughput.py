"""Sanity check on scoring throughput (reported honestly in the Day 2
write-up, not gated on a specific pairs/sec number here -- hardware varies).
This just guards against an accidental quadratic blow-up or pathological
slowdown in a channel."""

import time

from engine.calibration.fit_priors import fit_all, generate_train_holdout_datasets
from engine.scoring.fusion import FusionModel, score_pair
from eval.metrics import sample_labeled_pairs

MIN_PAIRS_PER_SEC = 50.0


def test_fusion_scoring_throughput_is_reasonable():
    train_ds, holdout_ds = generate_train_holdout_datasets(
        n_cameras=15,
        n_vehicles_train=1500,
        n_vehicles_holdout=1500,
        hours=24,
        city_seed=3,
        train_seed=131,
        holdout_seed=232,
        clone_fraction=0.0,
    )
    models = fit_all(train_ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    labeled_pairs = sample_labeled_pairs(holdout_ds.events, seed=0, max_positive_pairs=1000)

    start = time.perf_counter()
    for a, b, _label in labeled_pairs:
        score_pair(a, b, fusion_model)
    elapsed = time.perf_counter() - start

    pairs_per_sec = len(labeled_pairs) / elapsed
    assert pairs_per_sec > MIN_PAIRS_PER_SEC, (
        f"scoring throughput {pairs_per_sec:.1f} pairs/sec is suspiciously slow "
        f"({len(labeled_pairs)} pairs in {elapsed:.2f}s)"
    )
