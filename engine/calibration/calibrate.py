"""Calibrate the fused score into P(same vehicle) on a HELD-OUT split
(architecture.md §3, §10) and report the numbers Day 2's success criteria
are judged on: per-channel AUC, fused AUC (> 0.97 target), and expected
calibration error (< 0.05 target).

Isotonic regression (scikit-learn) maps `total_log_odds -> P(same)`: it
makes no parametric assumption about the score's distribution, unlike Platt
scaling's logistic-sigmoid assumption, and a log-odds sum of three
independently-fit channels has no particular reason to be logistic-shaped.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from engine.contracts.events import DetectionEvent
from engine.scoring.fusion import FusionModel, score_pair
from eval.metrics import expected_calibration_error, roc_auc, sample_labeled_pairs

# Scores beyond this are already "certain" either way; clipping keeps -inf
# (the kinematic hard gate) and other extreme values from breaking isotonic
# regression's fit without changing what they mean (reject / accept).
SCORE_CLIP = 50.0


@dataclass
class ChannelAUCs:
    plate: float
    appearance: float
    kinematic: float
    fused: float


@dataclass
class CalibrationResult:
    isotonic: IsotonicRegression
    aucs: ChannelAUCs
    ece: float
    n_pairs: int
    n_positive: int


def score_labeled_pairs(
    labeled_pairs: list[tuple[DetectionEvent, DetectionEvent, int]], fusion_model: FusionModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (plate_scores, appearance_scores, kinematic_scores, fused_scores, labels)."""
    plate, appearance, kinematic, fused, labels = [], [], [], [], []
    for a, b, label in labeled_pairs:
        link = score_pair(a, b, fusion_model)
        plate.append(link.plate_lr)
        appearance.append(link.appearance_lr)
        kinematic.append(link.kinematic_lr)
        fused.append(link.total_log_odds)
        labels.append(label)
    return (
        np.array(plate),
        np.array(appearance),
        np.array(kinematic),
        np.array(fused),
        np.array(labels),
    )


def fit_calibration(
    fusion_model: FusionModel,
    train_events: list[DetectionEvent],
    seed: int = 0,
    max_positive_pairs: int = 4000,
) -> IsotonicRegression:
    """Fit the isotonic fused-score -> P(same) mapping on TRAIN pairs."""
    labeled_pairs = sample_labeled_pairs(
        train_events, seed=seed, max_positive_pairs=max_positive_pairs
    )
    _, _, _, fused, labels = score_labeled_pairs(labeled_pairs, fusion_model)
    clipped_fused = np.clip(fused, -SCORE_CLIP, SCORE_CLIP)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(clipped_fused, labels)
    return iso


def evaluate(
    fusion_model: FusionModel,
    isotonic: IsotonicRegression,
    holdout_events: list[DetectionEvent],
    seed: int = 0,
    max_positive_pairs: int = 4000,
) -> CalibrationResult:
    """Sample labelled pairs from the HELD-OUT split, score them through
    every channel + fusion, and report per-channel/fused AUC plus the
    calibration ECE of the (train-fit) isotonic mapping applied here. ECE is
    only a meaningful number measured out-of-sample: isotonic regression is
    flexible enough to fit its own training labels almost perfectly, so
    scoring it against the same pairs it was fit on would trivially read as
    near-zero error regardless of whether the mapping actually generalises."""
    labeled_pairs = sample_labeled_pairs(
        holdout_events, seed=seed, max_positive_pairs=max_positive_pairs
    )
    plate, appearance, kinematic, fused, labels = score_labeled_pairs(labeled_pairs, fusion_model)

    aucs = ChannelAUCs(
        plate=roc_auc(plate, labels),
        appearance=roc_auc(appearance, labels),
        kinematic=roc_auc(kinematic, labels),
        fused=roc_auc(fused, labels),
    )

    clipped_fused = np.clip(fused, -SCORE_CLIP, SCORE_CLIP)
    probs = isotonic.predict(clipped_fused)
    ece = expected_calibration_error(probs, labels)

    return CalibrationResult(
        isotonic=isotonic, aucs=aucs, ece=ece, n_pairs=len(labels), n_positive=int(labels.sum())
    )


def fit_and_evaluate(
    fusion_model: FusionModel,
    train_events: list[DetectionEvent],
    holdout_events: list[DetectionEvent],
    seed: int = 0,
    max_positive_pairs: int = 4000,
) -> CalibrationResult:
    """Convenience wrapper: fit the calibrator on train, evaluate (AUCs +
    out-of-sample ECE) on held-out."""
    isotonic = fit_calibration(
        fusion_model, train_events, seed=seed, max_positive_pairs=max_positive_pairs
    )
    return evaluate(
        fusion_model, isotonic, holdout_events, seed=seed, max_positive_pairs=max_positive_pairs
    )


def save_calibration(iso: IsotonicRegression, path: Path) -> None:
    data = {
        "x_thresholds": iso.X_thresholds_.tolist(),
        "y_thresholds": iso.y_thresholds_.tolist(),
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def load_calibration(path: Path) -> IsotonicRegression:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    x = np.array(data["x_thresholds"])
    y = np.array(data["y_thresholds"])
    # Refitting on the calibrator's own (already-isotonic) threshold points
    # is a no-op for PAVA and reproduces the identical step function.
    iso.fit(x, y)
    return iso
