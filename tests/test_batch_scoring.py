"""The Day 4 performance rewrite's correctness contract: `score_pairs_batch`
must agree with repeated `score_pair` calls to within 1e-6, per channel and
in total, on real GATED candidate pairs (not synthetic random ones) --
gated pairs are what the batched path is actually asked to score in
production (engine.association.window), and they exercise the plate
channel's alignment search and the kinematic channel's hard gate far more
than uniformly-random pairs would.
"""

import numpy as np
import pytest

from engine.association.gating import Gate, gate_candidates
from engine.calibration.fit_priors import fit_all, generate_train_holdout_datasets
from engine.scoring.fusion import FusionModel, score_pair, score_pairs_batch

TOLERANCE = 1e-6
MIN_PAIRS = 3000


@pytest.fixture(scope="module")
def gated_pairs():
    """events, fusion_model, pred_idx, succ_idx, and a parallel
    n_candidates_per_pair array (that pair's SUCCESSOR's own gated-candidate
    count, engine.scoring.fusion module docstring's "THE PER-SUCCESSOR
    FIX") -- everything both tests below need, generated once per module."""
    train_ds, holdout_ds = generate_train_holdout_datasets(
        n_cameras=15,
        n_vehicles_train=1500,
        n_vehicles_holdout=2500,
        hours=24,
        city_seed=3,
        train_seed=131,
        holdout_seed=232,
        clone_fraction=0.02,
    )
    models = fit_all(train_ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    gate = Gate(model=models.kinematic_model)
    gate_result = gate_candidates(holdout_ds.events, gate)

    events = holdout_ds.events
    index_of = {e.event_id: i for i, e in enumerate(events)}
    pred_idx: list[int] = []
    succ_idx: list[int] = []
    n_candidates: list[int] = []
    for succ_id, cand_ids in gate_result.candidates.items():
        j = index_of[succ_id]
        n_j = len(cand_ids)
        for cand_id in cand_ids:
            pred_idx.append(index_of[cand_id])
            succ_idx.append(j)
            n_candidates.append(n_j)
            if len(pred_idx) >= MIN_PAIRS * 2:
                break
        if len(pred_idx) >= MIN_PAIRS * 2:
            break

    return (
        events,
        fusion_model,
        np.array(pred_idx),
        np.array(succ_idx),
        np.array(n_candidates),
    )


def test_batched_scoring_agrees_with_scalar_path_per_channel_and_total(gated_pairs):
    events, fusion_model, pred_idx, succ_idx, _n_candidates = gated_pairs
    assert len(pred_idx) >= MIN_PAIRS, (
        f"only {len(pred_idx)} gated pairs available, need >= {MIN_PAIRS}"
    )

    batched_links = score_pairs_batch(events, pred_idx, succ_idx, fusion_model)
    assert len(batched_links) == len(pred_idx)

    max_err = {"plate": 0.0, "appearance": 0.0, "kinematic": 0.0, "total": 0.0}
    n_impossible = 0
    for k in range(len(pred_idx)):
        scalar = score_pair(events[pred_idx[k]], events[succ_idx[k]], fusion_model)
        batched = batched_links[k]

        assert batched.from_event_id == events[pred_idx[k]].event_id
        assert batched.to_event_id == events[succ_idx[k]].event_id

        if np.isneginf(scalar.kinematic_lr):
            n_impossible += 1
            assert np.isneginf(batched.kinematic_lr)
            assert np.isneginf(scalar.total_log_odds)
            assert np.isneginf(batched.total_log_odds)
            continue

        max_err["plate"] = max(max_err["plate"], abs(scalar.plate_lr - batched.plate_lr))
        max_err["appearance"] = max(
            max_err["appearance"], abs(scalar.appearance_lr - batched.appearance_lr)
        )
        max_err["kinematic"] = max(
            max_err["kinematic"], abs(scalar.kinematic_lr - batched.kinematic_lr)
        )
        max_err["total"] = max(
            max_err["total"], abs(scalar.total_log_odds - batched.total_log_odds)
        )

    print(f"n_pairs={len(pred_idx)} n_physically_impossible={n_impossible}")
    print(f"max_err={max_err}")
    for channel, err in max_err.items():
        assert err < TOLERANCE, f"{channel} channel disagreement {err} exceeds {TOLERANCE}"


def test_batched_scoring_agrees_with_scalar_path_with_per_successor_prior(gated_pairs):
    """Same exactness guard as the test above, but with the REAL
    per-successor prior (engine.scoring.fusion module docstring's "THE
    PER-SUCCESSOR FIX") passed through both paths: `n_candidates_per_pair`
    to `score_pairs_batch`, and the matching per-pair `n_candidates` to each
    `score_pair` call. This is the exact usage engine.association.window
    exercises in production -- the batched-vs-scalar guarantee must hold
    with a real (non-constant, non-default) prior too, not just the
    fallback-constant prior the other test above uses."""
    events, fusion_model, pred_idx, succ_idx, n_candidates = gated_pairs
    assert len(pred_idx) >= MIN_PAIRS, (
        f"only {len(pred_idx)} gated pairs available, need >= {MIN_PAIRS}"
    )
    assert n_candidates.max() > 1, "need at least one successor with >1 gated candidate"

    batched_links = score_pairs_batch(
        events, pred_idx, succ_idx, fusion_model, n_candidates_per_pair=n_candidates
    )
    assert len(batched_links) == len(pred_idx)

    max_err = {"prior": 0.0, "total": 0.0}
    for k in range(len(pred_idx)):
        scalar = score_pair(
            events[pred_idx[k]],
            events[succ_idx[k]],
            fusion_model,
            n_candidates=float(n_candidates[k]),
        )
        batched = batched_links[k]
        if np.isneginf(scalar.total_log_odds):
            assert np.isneginf(batched.total_log_odds)
            continue
        max_err["prior"] = max(
            max_err["prior"], abs(scalar.prior_log_odds - batched.prior_log_odds)
        )
        max_err["total"] = max(
            max_err["total"], abs(scalar.total_log_odds - batched.total_log_odds)
        )

    print(f"per-successor-prior max_err={max_err}")
    for channel, err in max_err.items():
        assert err < TOLERANCE, f"{channel} disagreement {err} exceeds {TOLERANCE}"
