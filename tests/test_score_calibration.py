"""engine/calibration/calibrate.py: fused-score -> P(same) calibration.

Success criterion (phases.md Day 2 + task spec): expected calibration error
< 0.05, measured out-of-sample (fit on TRAIN pairs, evaluated on HELD-OUT
pairs — see docs/decisions.md, "Day 2", on why in-sample ECE is meaningless
for a flexible isotonic mapping).
"""

from pathlib import Path

import numpy as np
import pytest

from engine.calibration.calibrate import (
    evaluate,
    fit_calibration,
    load_calibration,
    save_calibration,
)
from engine.calibration.fit_priors import fit_all, generate_train_holdout_datasets
from engine.scoring.fusion import FusionModel

ECE_TARGET = 0.05


@pytest.fixture(scope="module")
def split_and_model():
    train_ds, holdout_ds = generate_train_holdout_datasets(
        n_cameras=20,
        n_vehicles_train=3000,
        n_vehicles_holdout=3000,
        hours=24,
        city_seed=2,
        train_seed=111,
        holdout_seed=222,
        clone_fraction=0.02,
    )
    models = fit_all(train_ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    return train_ds, holdout_ds, fusion_model


def test_ece_below_target_out_of_sample(split_and_model):
    train_ds, holdout_ds, fusion_model = split_and_model
    isotonic = fit_calibration(fusion_model, train_ds.events, seed=0, max_positive_pairs=3000)
    result = evaluate(fusion_model, isotonic, holdout_ds.events, seed=1, max_positive_pairs=3000)
    assert result.ece < ECE_TARGET, f"ECE {result.ece:.4f} exceeds target {ECE_TARGET}"
    assert result.n_pairs > 0


def test_calibrated_probabilities_are_monotonic_in_score(split_and_model):
    train_ds, _, fusion_model = split_and_model
    isotonic = fit_calibration(fusion_model, train_ds.events, seed=0, max_positive_pairs=2000)
    xs = np.linspace(-30, 30, 50)
    probs = isotonic.predict(xs)
    assert np.all(np.diff(probs) >= -1e-9)  # non-decreasing
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_calibration_persists_round_trip(tmp_path: Path, split_and_model):
    train_ds, _, fusion_model = split_and_model
    isotonic = fit_calibration(fusion_model, train_ds.events, seed=0, max_positive_pairs=2000)
    path = tmp_path / "calibration.json"
    save_calibration(isotonic, path)
    loaded = load_calibration(path)

    xs = np.linspace(-30, 30, 25)
    np.testing.assert_allclose(isotonic.predict(xs), loaded.predict(xs), atol=1e-6)
