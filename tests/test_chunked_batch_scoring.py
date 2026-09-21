"""Regression coverage for the Task 1 memory fix: batched per-pair scoring
in `engine.scoring.{plate,appearance,kinematic}_lr` must process pairs in
bounded-size CHUNKS instead of gathering one row per pair over a whole wide
per-event array. Before the fix, `score_plate_pairs_batch` on a real ~5.9M
gated-pair batch died with:

    numpy._core._exceptions._ArrayMemoryError: Unable to allocate 15.0 GiB
    for an array with shape (5946536, 676) and data type float32

because `arrays.M_state[pred_idx]` (a fancy-index gather) materialises
(n_pairs, 676) floats in one shot. This module scores >=2M SYNTHETIC pairs
(a small pool of real events, indexed with a large random pred/succ array --
exactly the index-array shape a real gated candidate pool has, without
needing a multi-million-event dataset in memory) and checks:
  1. it completes at all (the original regression), and
  2. chunking produces numerically IDENTICAL results to a single unchunked
     call, on a smaller slice (each pair's math is fully row-independent,
     so partitioning into chunks must not change any value)."""

import numpy as np

import engine.scoring.appearance_lr as appearance_lr
import engine.scoring.kinematic_lr as kinematic_lr
import engine.scoring.plate_lr as plate_lr
from engine.scoring.appearance_lr import (
    AppearanceModel,
    precompute_appearance_batch_arrays,
    score_appearance_pairs_batch,
)
from engine.scoring.kinematic_lr import fit_kinematic_model, precompute_event_kinematic_arrays
from engine.scoring.plate_lr import (
    PlatePriors,
    precompute_plate_batch_arrays,
    score_plate_pairs_batch,
)
from sim.generate import generate_dataset
from sim.vehicles import COLORS, VEHICLE_TYPES

# Comfortably above the ~2M pairs the task calls for, and the same order of
# magnitude as the production blowup (5.9M pairs x 676-wide float32 = 15 GiB
# in one gather) -- large enough that the pre-fix code would have died here.
LARGE_N_PAIRS = 2_200_000
SMALL_N_PAIRS = 5_000
SMALL_TEST_CHUNK_SIZE = 777  # deliberately not a divisor of SMALL_N_PAIRS


def _small_event_pool():
    """A modest real event pool (small dataset, real plate posteriors/
    embeddings/cameras) that synthetic large index arrays will repeatedly
    re-index into -- mirrors how a real batch call re-uses a small n_events
    precomputed array across a huge n_pairs index array."""
    ds = generate_dataset(n_cameras=5, n_vehicles=60, hours=3, seed=11, clone_fraction=0.0)
    assert len(ds.events) > 20, "need a reasonable pool of distinct events to index into"
    return ds


def _synthetic_indices(n_events: int, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pred_idx = rng.integers(0, n_events, size=n_pairs)
    succ_idx = rng.integers(0, n_events, size=n_pairs)
    return pred_idx, succ_idx


# ---------------------------------------------------------------------------
# Plate channel
# ---------------------------------------------------------------------------


def test_plate_batch_handles_millions_of_pairs_without_blowing_up_memory():
    ds = _small_event_pool()
    events = ds.events
    priors = PlatePriors.uniform_default()
    arrays = precompute_plate_batch_arrays([e.plate_posterior for e in events], priors)

    pred_idx, succ_idx = _synthetic_indices(len(events), LARGE_N_PAIRS, seed=1)
    log_lr, alignment_idx = score_plate_pairs_batch(arrays, pred_idx, succ_idx)

    assert log_lr.shape == (LARGE_N_PAIRS,)
    assert alignment_idx.shape == (LARGE_N_PAIRS,)
    # The plate channel's noisy-channel LR is floored and never -inf (module
    # docstring, "Day 1c"): every one of these must come back finite.
    assert np.isfinite(log_lr).all()
    assert (alignment_idx >= 0).all() and (alignment_idx < len(plate_lr.ALIGNMENT_NAMES)).all()


def test_plate_batch_chunked_matches_unchunked_on_a_smaller_slice(monkeypatch):
    ds = _small_event_pool()
    events = ds.events
    priors = PlatePriors.uniform_default()
    arrays = precompute_plate_batch_arrays([e.plate_posterior for e in events], priors)
    pred_idx, succ_idx = _synthetic_indices(len(events), SMALL_N_PAIRS, seed=2)

    # Reference: the exact same per-chunk math, called ONCE on the whole
    # slice (no chunking at all).
    ref_log_lr, ref_alignment = plate_lr._score_plate_pairs_chunk(arrays, pred_idx, succ_idx)

    # Force many small chunks (SMALL_N_PAIRS is not a multiple of the forced
    # chunk size, so this also exercises a ragged final chunk).
    monkeypatch.setattr(plate_lr, "PAIR_CHUNK_SIZE", SMALL_TEST_CHUNK_SIZE)
    chunked_log_lr, chunked_alignment = score_plate_pairs_batch(arrays, pred_idx, succ_idx)

    np.testing.assert_array_equal(chunked_alignment, ref_alignment)
    np.testing.assert_allclose(chunked_log_lr, ref_log_lr, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Appearance channel
# ---------------------------------------------------------------------------


def test_appearance_batch_handles_millions_of_pairs_without_blowing_up_memory():
    ds = _small_event_pool()
    events = ds.events
    model = AppearanceModel.uniform_default(COLORS, VEHICLE_TYPES)
    arrays = precompute_appearance_batch_arrays(events, model)

    pred_idx, succ_idx = _synthetic_indices(len(events), LARGE_N_PAIRS, seed=3)
    total = score_appearance_pairs_batch(arrays, pred_idx, succ_idx, model)

    assert total.shape == (LARGE_N_PAIRS,)
    assert np.isfinite(total).all()
    assert (total >= -appearance_lr.CLAMP - 1e-9).all()
    assert (total <= appearance_lr.CLAMP + 1e-9).all()


def test_appearance_batch_chunked_matches_unchunked_on_a_smaller_slice(monkeypatch):
    ds = _small_event_pool()
    events = ds.events
    model = AppearanceModel.uniform_default(COLORS, VEHICLE_TYPES)
    arrays = precompute_appearance_batch_arrays(events, model)
    pred_idx, succ_idx = _synthetic_indices(len(events), SMALL_N_PAIRS, seed=4)

    ref_total = appearance_lr._score_appearance_pairs_chunk(arrays, pred_idx, succ_idx, model)

    monkeypatch.setattr(appearance_lr, "PAIR_CHUNK_SIZE", SMALL_TEST_CHUNK_SIZE)
    chunked_total = score_appearance_pairs_batch(arrays, pred_idx, succ_idx, model)

    np.testing.assert_allclose(chunked_total, ref_total, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Kinematic channel
# ---------------------------------------------------------------------------


def test_kinematic_batch_handles_millions_of_pairs_without_blowing_up_memory():
    ds = _small_event_pool()
    events = ds.events
    model = fit_kinematic_model(events, ds.city)
    cam_idx, epoch_s, tod_idx = precompute_event_kinematic_arrays(events, model)

    pred_idx, succ_idx = _synthetic_indices(len(events), LARGE_N_PAIRS, seed=5)
    result = model.score_batch(cam_idx, epoch_s, tod_idx, pred_idx, succ_idx)

    assert result.log_lr.shape == (LARGE_N_PAIRS,)
    assert result.physically_impossible.shape == (LARGE_N_PAIRS,)
    assert len(result.skipped_cameras) == LARGE_N_PAIRS
    # Wherever the hard physical gate fired, log_lr must be exactly -inf
    # (kinematic_lr.py module docstring: this is the one channel where -inf
    # is the correct answer), and finite everywhere else.
    assert np.all(np.isneginf(result.log_lr[result.physically_impossible]))
    assert np.isfinite(result.log_lr[~result.physically_impossible]).all()


def test_kinematic_batch_chunked_matches_unchunked_on_a_smaller_slice(monkeypatch):
    ds = _small_event_pool()
    events = ds.events
    model = fit_kinematic_model(events, ds.city)
    cam_idx, epoch_s, tod_idx = precompute_event_kinematic_arrays(events, model)
    pred_idx, succ_idx = _synthetic_indices(len(events), SMALL_N_PAIRS, seed=6)

    dense = model.dense_arrays()
    ref = model._score_batch_chunk(dense, cam_idx, epoch_s, tod_idx, pred_idx, succ_idx)

    monkeypatch.setattr(kinematic_lr, "PAIR_CHUNK_SIZE", SMALL_TEST_CHUNK_SIZE)
    chunked = model.score_batch(cam_idx, epoch_s, tod_idx, pred_idx, succ_idx)

    np.testing.assert_array_equal(chunked.physically_impossible, ref.physically_impossible)
    np.testing.assert_allclose(chunked.delta_t_s, ref.delta_t_s, rtol=0, atol=0)
    np.testing.assert_allclose(chunked.expected_t_s, ref.expected_t_s, rtol=0, atol=0)
    # log_lr is -inf in the same places on both sides; compare finite entries
    # only (np.testing.assert_allclose chokes on -inf vs -inf otherwise).
    finite = np.isfinite(ref.log_lr)
    assert np.array_equal(np.isfinite(chunked.log_lr), finite)
    np.testing.assert_allclose(chunked.log_lr[finite], ref.log_lr[finite], rtol=0, atol=0)
    assert chunked.skipped_cameras == ref.skipped_cameras
