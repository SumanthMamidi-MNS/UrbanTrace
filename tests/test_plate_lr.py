"""engine/scoring/plate_lr.py: the noisy-channel plate LR."""

import numpy as np

from engine.contracts.plate import BLANK, NUMBER_ALPHABET, PLATE_SLOTS, SlotPosterior
from engine.scoring.plate_lr import (
    NUMBER_BOUNDARY,
    SERIES_BOUNDARY,
    PlatePriors,
    _score_aligned,
    _shift,
    score_plate_pair,
)
from eval.metrics import roc_auc, sample_labeled_pairs
from sim.generate import generate_dataset

TRUE_PLATE = "MH12AB1234"


def _peaked_posteriors(plate: str) -> list[SlotPosterior]:
    return [SlotPosterior.peaked(PLATE_SLOTS[i], c, 0.97) for i, c in enumerate(plate)]


def test_identical_high_confidence_plates_yield_large_positive_lr():
    a = _peaked_posteriors(TRUE_PLATE)
    b = _peaked_posteriors(TRUE_PLATE)
    result = score_plate_pair(a, b)
    assert result.log_lr > 5.0
    assert result.alignment == "none"


def test_never_returns_negative_infinity_on_a_conflicting_read():
    """Even a plate that conflicts on every single character must never
    drive the plate channel to -inf (docs/decisions.md, "Day 1c") -- an
    out-of-group OCR error is unlikely, never impossible."""
    a = _peaked_posteriors(TRUE_PLATE)
    conflicting = "KA99ZZ9999"
    b = _peaked_posteriors(conflicting)
    result = score_plate_pair(a, b)
    assert np.isfinite(result.log_lr)


def test_out_of_group_misread_from_real_corruption_never_neg_inf():
    """Regression case for the exact defect described in docs/decisions.md,
    "Day 1c" (true KL84RS7237, read KL84RJ7237, S getting zero posterior
    mass): run real corrupted events through the plate channel and confirm
    no pair's plate_lr is ever -inf."""
    ds = generate_dataset(n_cameras=15, n_vehicles=800, hours=24, seed=17, clone_fraction=0.0)
    true_plate_by_vehicle = {v.gt_vehicle_id: v.true_plate for v in ds.vehicles}
    by_vehicle: dict[str, list] = {}
    for e in ds.events:
        by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    checked = 0
    for vid, evs in by_vehicle.items():
        if len(evs) < 2:
            continue
        true_posteriors = _peaked_posteriors(true_plate_by_vehicle[vid])
        for e in evs:
            result = score_plate_pair(true_posteriors, e.plate_posterior)
            assert np.isfinite(result.log_lr), f"vehicle {vid}: got {result.log_lr}"
            checked += 1
    assert checked > 100


def test_unread_independent_slot_contributes_approximately_zero():
    a = _peaked_posteriors(TRUE_PLATE)
    b = _peaked_posteriors(TRUE_PLATE)
    # Slot 8 (a number digit) fully occluded in event B only.
    b[8] = SlotPosterior.uniform(NUMBER_ALPHABET)
    result = score_plate_pair(a, b)
    assert abs(result.group_contributions["slot8"]) < 1e-6


def test_fully_occluded_plate_contributes_approximately_zero():
    """B6: a fully-occluded plate (every slot uniform) must contribute ~0 to
    the plate LR, not be treated as evidence against a match."""
    a = _peaked_posteriors(TRUE_PLATE)
    b = [SlotPosterior.uniform(alphabet) for alphabet in PLATE_SLOTS]
    result = score_plate_pair(a, b)
    assert abs(result.log_lr) < 1e-6


def test_alignment_search_recovers_a_dropped_series_character():
    """Simulate a read that dropped one character right at the series
    boundary: `_shift(a, SERIES_BOUNDARY, +1)` is exactly "b's content from
    the boundary on is really a's content one position later, last slot
    padded uniform" -- the corruption `score_plate_pair` is meant to undo.
    The winning alignment must beat the naive "none" comparison, and should
    recover most of a perfect match's score."""
    a = _peaked_posteriors(TRUE_PLATE)
    shifted_b = _shift(a, SERIES_BOUNDARY, 1)

    none_total, _ = _score_aligned(a, shifted_b, PlatePriors.uniform_default())
    result = score_plate_pair(a, shifted_b)
    perfect = score_plate_pair(a, a)

    assert result.log_lr >= none_total
    assert result.alignment == "series-1"
    # Undoing the shift should recover most of the perfect-match evidence
    # (the padded/uniform slot still contributes ~0, so it won't be exact).
    assert result.log_lr > 0.6 * perfect.log_lr


def test_boundary_shift_helpers_cover_series_and_number():
    # Sanity: boundaries are where the docstring says they are.
    assert SERIES_BOUNDARY == 4
    assert NUMBER_BOUNDARY == 6


def test_blank_padded_slots_handled():
    plate_1_letter_series = "MH12A_1234"
    a = _peaked_posteriors(plate_1_letter_series)
    assert a[5].argmax() == BLANK
    result = score_plate_pair(a, a)
    assert result.log_lr > 5.0


def test_default_priors_are_valid_distributions():
    priors = PlatePriors.uniform_default()
    assert abs(priors.state_matrix.sum() - 1.0) < 1e-9
    assert abs(priors.rto_matrix.sum() - 1.0) < 1e-9
    assert (priors.state_matrix > 0).all()
    assert (priors.rto_matrix > 0).all()


def test_plate_channel_auc_better_than_chance():
    ds = generate_dataset(n_cameras=15, n_vehicles=1500, hours=24, seed=31, clone_fraction=0.0)
    labeled_pairs = sample_labeled_pairs(ds.events, seed=0, max_positive_pairs=2000)
    scores = []
    labels = []
    priors = PlatePriors.uniform_default()
    for a, b, label in labeled_pairs:
        result = score_plate_pair(a.plate_posterior, b.plate_posterior, priors)
        scores.append(result.log_lr)
        labels.append(label)
    auc = roc_auc(np.array(scores), np.array(labels))
    assert auc > 0.5, f"plate channel AUC {auc:.4f} is not better than chance"
