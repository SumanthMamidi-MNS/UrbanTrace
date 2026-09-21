"""Plate likelihood ratio (architecture.md §3, "Plate term").

    log LR = log[ SUM_y pi(y) P(A|y) P(B|y) ] - log[ (SUM_y pi(y)P(A|y)) * (SUM_y pi(y)P(B|y)) ]

`y` (the unknown true plate) factorises into GROUPS, not 10 independent
slots: positions (0,1) are one joint "state code" unit with a real prior
over the state-code vocabulary, positions (2,3) are one joint "RTO" unit,
and positions 4-9 (series letters, number digits) are independent. Treating
state/RTO as independent per-character slots would wrongly assume MH/DL/KA
are just any two uncorrelated letters — real state codes are a small,
structured vocabulary, and that structure is exactly what should sharpen
the plate LR when a candidate pair shares (or conflicts on) a state code.

Because pi(y) and P(O|y) both factorise across groups, the big sum over the
whole plate factorises into a sum of per-group log-LR contributions:

    logLR = SUM_g [ log num_g - log den_g ]
    num_g = SUM_c pi_g(c) A_g(c) B_g(c)
    den_g = (SUM_c pi_g(c) A_g(c)) * (SUM_c pi_g(c) B_g(c))

Every probability is floored at FLOOR before use — a second line of defence
behind the corruption model (sim/corruption.py) and the on-disk codec
(engine/contracts/codec.py), both of which already guarantee no character
is ever assigned exactly zero probability (docs/decisions.md, "Day 1c"). No
finite input to this module can ever produce -inf.

An unread (fully uniform) slot contributes ~0 to its group: if either
event's per-character distribution over a group's vocabulary is exactly
uniform, num_g and den_g are algebraically identical regardless of the
prior or the other event's distribution (see tests/test_plate_lr.py). This
is why occlusion is "missing information, not evidence against a match."

Alignment: a real OCR misread can drop or insert a character right at a
segment boundary (the series or number segment), shifting everything after
it by one position relative to the fixed 10-slot canonical layout. Rather
than a full edit-HMM, `score_plate_pair` tries the small set of plausible
shifts at the series and number boundaries (0/±1 each) and returns the
max-likelihood alignment, so a single boundary insertion/deletion doesn't
look like ten independent character mismatches.
"""

import math
from dataclasses import dataclass

import numpy as np

from engine.contracts.plate import PLATE_SLOTS, RTO_ALPHABET, STATE_ALPHABET, SlotPosterior
from sim.vehicles import STATE_CODES

FLOOR = 1e-9

STATE_SLOTS = (0, 1)
RTO_SLOTS = (2, 3)
INDEPENDENT_SLOTS = (4, 5, 6, 7, 8, 9)

# Segment boundaries in the fixed 10-slot canonical layout: series starts at
# index 4 (right after the 2-digit RTO code), number starts at index 6.
SERIES_BOUNDARY = 4
NUMBER_BOUNDARY = 6

_N_STATE = len(STATE_ALPHABET)
_N_RTO = len(RTO_ALPHABET)
_STATE_IDX = {c: i for i, c in enumerate(STATE_ALPHABET)}
_RTO_IDX = {c: i for i, c in enumerate(RTO_ALPHABET)}


@dataclass
class PlatePriors:
    """pi(y), factorised: a dense (26, 26) prior over (state_letter_0,
    state_letter_1) and a dense (10, 10) prior over (rto_digit_0,
    rto_digit_1). Both must sum to 1 and carry no exact zeros. Independent
    slots (series letters, number digits) use a flat uniform prior over
    their own alphabet — architecture.md §3 describes the grammar as
    factorising per character position there; only state/RTO get a real,
    non-uniform joint prior (see module docstring)."""

    state_matrix: np.ndarray  # (26, 26), STATE_ALPHABET-indexed, sums to 1
    rto_matrix: np.ndarray  # (10, 10), RTO_ALPHABET-indexed, sums to 1

    @classmethod
    def uniform_default(cls, state_floor: float = 1e-4) -> "PlatePriors":
        """A grammar-only prior with no fitting: real (known) state codes
        share the bulk of the mass uniformly; every other syntactically
        valid 2-letter combo keeps a small floor (so an out-of-vocabulary
        state code is unlikely, never impossible). RTO codes are uniform
        over all 100 two-digit combinations — the grammar alone has no
        opinion on which RTO codes are common."""
        state_matrix = np.full((_N_STATE, _N_STATE), state_floor)
        n_known = len(STATE_CODES)
        floor_mass = state_floor * (_N_STATE * _N_STATE - n_known)
        per_known = max((1.0 - floor_mass) / n_known, state_floor)
        for code in STATE_CODES:
            i, j = _STATE_IDX[code[0]], _STATE_IDX[code[1]]
            state_matrix[i, j] = per_known
        state_matrix = state_matrix / state_matrix.sum()

        rto_matrix = np.full((_N_RTO, _N_RTO), 1.0 / (_N_RTO * _N_RTO))
        return cls(state_matrix=state_matrix, rto_matrix=rto_matrix)


def _slot_prob_vector(sp: SlotPosterior, alphabet: list[str]) -> np.ndarray:
    return np.array([max(sp.probs.get(c, 0.0), FLOOR) for c in alphabet], dtype=np.float64)


def _joint_group_contribution(
    pi_matrix: np.ndarray, a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> float:
    a_matrix = np.outer(a0, a1)
    b_matrix = np.outer(b0, b1)
    num = float(np.sum(pi_matrix * a_matrix * b_matrix))
    den = float(np.sum(pi_matrix * a_matrix)) * float(np.sum(pi_matrix * b_matrix))
    return math.log(max(num, FLOOR)) - math.log(max(den, FLOOR))


def _independent_slot_contribution(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    pi = np.full(n, 1.0 / n)
    num = float(np.sum(pi * a * b))
    den = float(np.sum(pi * a)) * float(np.sum(pi * b))
    return math.log(max(num, FLOOR)) - math.log(max(den, FLOOR))


def _shift(posteriors: list[SlotPosterior], boundary: int, delta: int) -> list[SlotPosterior]:
    """Reinterpret `posteriors` as if one character was spuriously inserted
    (delta=+1) or dropped (delta=-1) right at `boundary`, shifting every
    slot from there on by one position. Vacated positions become uniform
    (uninformative) rather than disappearing, keeping the list at the fixed
    length 10."""
    if delta == 0:
        return posteriors
    if delta == 1:
        return [
            *posteriors[:boundary],
            *posteriors[boundary + 1 :],
            SlotPosterior.uniform(PLATE_SLOTS[-1]),
        ]
    if delta == -1:
        return [
            *posteriors[:boundary],
            SlotPosterior.uniform(PLATE_SLOTS[boundary]),
            *posteriors[boundary:-1],
        ]
    raise ValueError(f"unsupported alignment shift {delta}")


_ALIGNMENTS: list[tuple[str, int | None, int]] = [
    ("none", None, 0),
    ("series+1", SERIES_BOUNDARY, 1),
    ("series-1", SERIES_BOUNDARY, -1),
    ("number+1", NUMBER_BOUNDARY, 1),
    ("number-1", NUMBER_BOUNDARY, -1),
]


@dataclass
class PlateLRResult:
    log_lr: float
    alignment: str
    group_contributions: dict[str, float]


def _score_aligned(
    a: list[SlotPosterior], b: list[SlotPosterior], priors: PlatePriors
) -> tuple[float, dict[str, float]]:
    contributions: dict[str, float] = {}

    a0 = _slot_prob_vector(a[STATE_SLOTS[0]], STATE_ALPHABET)
    a1 = _slot_prob_vector(a[STATE_SLOTS[1]], STATE_ALPHABET)
    b0 = _slot_prob_vector(b[STATE_SLOTS[0]], STATE_ALPHABET)
    b1 = _slot_prob_vector(b[STATE_SLOTS[1]], STATE_ALPHABET)
    contributions["state"] = _joint_group_contribution(priors.state_matrix, a0, a1, b0, b1)

    a2 = _slot_prob_vector(a[RTO_SLOTS[0]], RTO_ALPHABET)
    a3 = _slot_prob_vector(a[RTO_SLOTS[1]], RTO_ALPHABET)
    b2 = _slot_prob_vector(b[RTO_SLOTS[0]], RTO_ALPHABET)
    b3 = _slot_prob_vector(b[RTO_SLOTS[1]], RTO_ALPHABET)
    contributions["rto"] = _joint_group_contribution(priors.rto_matrix, a2, a3, b2, b3)

    for idx in INDEPENDENT_SLOTS:
        alphabet = PLATE_SLOTS[idx]
        av = _slot_prob_vector(a[idx], alphabet)
        bv = _slot_prob_vector(b[idx], alphabet)
        contributions[f"slot{idx}"] = _independent_slot_contribution(av, bv)

    return sum(contributions.values()), contributions


def score_plate_pair(
    posteriors_a: list[SlotPosterior],
    posteriors_b: list[SlotPosterior],
    priors: PlatePriors | None = None,
) -> PlateLRResult:
    """Score a candidate pair's plate evidence, searching the small set of
    plausible alignments and returning the max-likelihood one."""
    priors = priors or PlatePriors.uniform_default()
    best: PlateLRResult | None = None
    for name, boundary, delta in _ALIGNMENTS:
        b_shifted = posteriors_b if boundary is None else _shift(posteriors_b, boundary, delta)
        total, contributions = _score_aligned(posteriors_a, b_shifted, priors)
        if best is None or total > best.log_lr:
            best = PlateLRResult(log_lr=total, alignment=name, group_contributions=contributions)
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Batched scoring (Day 4 performance pass).
#
# The scalar path above recomputes, per PAIR, work that only actually depends
# on one EVENT: each event's outer-product joint matrices for the state/RTO
# groups, and each event's per-slot probability vectors under every
# alignment shift. `PlateBatchArrays` precomputes all of that ONCE per event
# (O(n_events)), leaving only a batched dot-product per pair (O(n_pairs)).
#
# Key algebraic facts this relies on (both straightforward from the module
# docstring's formulas):
#   1. `state`/`rto` are never touched by the alignment search (both
#      boundaries -- SERIES_BOUNDARY=4, NUMBER_BOUNDARY=6 -- sit at or after
#      index 4, and state/rto occupy indices 0-3). So each event needs only
#      ONE precomputed state/rto contribution, reused by every alignment.
#   2. Since the plate LR's group numerator is symmetric in how the prior
#      `pi` is distributed (`sum(pi*M_A*M_B) == dot(M_A, pi*M_B)`), only ONE
#      event needs to carry the `pi`-weighted copy (`W = pi * M`); the other
#      side supplies the raw joint matrix `M`. A pair's numerator is then a
#      single dot product, and its denominator is a product of two
#      precomputed per-event scalars.
# Only the 6 "independent" slots (series letters + number digits) are
# alignment-dependent (a boundary shift reshuffles which slot-vector sits in
# which position, per `_shift` above) -- their arrays are precomputed for
# all 5 alignments; state/rto need no such variants.
# ---------------------------------------------------------------------------

ALIGNMENT_NAMES: list[str] = [name for name, _, _ in _ALIGNMENTS]

_INDEP_ALPHABETS = [PLATE_SLOTS[i] for i in INDEPENDENT_SLOTS]
_INDEP_WIDTH = max(len(a) for a in _INDEP_ALPHABETS)  # 27 (series: 26 letters + blank)
_INDEP_N = np.array([len(a) for a in _INDEP_ALPHABETS], dtype=np.float64)  # native alphabet sizes
_INDEP_INVN = 1.0 / _INDEP_N  # (6,) per-slot uniform-prior weight 1/n

# Relative-shift maps for the 6 independent slots (local index 0..5 <->
# global plate index 4..9), one per non-trivial alignment: `reindex[k]` says
# "new local slot `k` copies OLD local slot `reindex[k]`", or -1 if it must
# instead become a fresh uniform fill (`fill[k]` names which uniform vector:
# "series" or "number"). Derived directly from `_shift`'s boundary/delta
# semantics -- see module docstring above.
_SHIFT_SPECS: dict[str, tuple[list[int], list[str | None]]] = {
    "none": ([0, 1, 2, 3, 4, 5], [None] * 6),
    "series+1": ([1, 2, 3, 4, 5, -1], [None, None, None, None, None, "number"]),
    "series-1": ([-1, 0, 1, 2, 3, 4], ["series", None, None, None, None, None]),
    "number+1": ([0, 1, 3, 4, 5, -1], [None, None, None, None, None, "number"]),
    "number-1": ([0, 1, -1, 2, 3, 4], [None, None, "number", None, None, None]),
}

# Two of the reindex entries above straddle the series/number TYPE boundary
# (a series-native slot's data ends up read at a position whose canonical
# alphabet is NUMBER_ALPHABET, or vice versa) -- exactly the "series+1" shift
# at local index 1 (old local 2, number-native, read as series) and the
# "series-1" shift at local index 2 (old local 1, series-native, read as
# number). `number+1`/`number-1` never cross the boundary (both stay >= 6),
# so they need no such patch (verified by construction of the reindex maps
# above -- see docs/decisions.md).
#
# The scalar path's `_slot_prob_vector(a[idx], PLATE_SLOTS[idx])` re-keys BY
# CHARACTER NAME when this happens: a number-native SlotPosterior's dict has
# no letter keys at all, so every letter position reads back as FLOOR, and
# only the shared "_" (blank) symbol -- present in both alphabets, always
# the LAST entry of each -- carries its real probability across. A naive
# raw-column copy would instead reuse column *indices*, silently reading a
# digit's probability as if it were some letter's -- wrong whenever the
# source and target alphabets differ. `(local_idx, source_local)` pairs that
# need this character-identity re-keying instead of a plain copy:
_CROSS_ALPHABET_PATCHES: dict[str, tuple[int, int]] = {
    "series+1": (1, 2),  # local1 (series-native target) <- old local2 (number-native)
    "series-1": (2, 1),  # local2 (number-native target) <- old local1 (series-native)
}


@dataclass
class PlateBatchArrays:
    """Per-event precomputed dense arrays for batched plate scoring. Index
    `i` in every array corresponds to `events[i]` (or `posteriors_list[i]`)
    passed to `precompute_plate_batch_arrays`. Stored as float32 (the
    architecture's asked-for storage dtype) -- `score_plate_pairs_batch`
    accumulates in float64 so this loses no meaningful precision (verified
    against the scalar path in tests/test_batch_scoring.py)."""

    n_events: int
    M_state: np.ndarray  # (N, 676) raw joint state matrix, flattened
    W_state: np.ndarray  # (N, 676) pi-weighted: W = pi_state.flatten() * M_state
    den_state: np.ndarray  # (N,) = W_state.sum(axis=1), float64
    M_rto: np.ndarray  # (N, 100)
    W_rto: np.ndarray  # (N, 100)
    den_rto: np.ndarray  # (N,) float64
    indep_base: np.ndarray  # (N, 6, _INDEP_WIDTH) UNSHIFTED per-slot vectors


def _uniform_indep_vector(alphabet: list[str]) -> np.ndarray:
    v = np.zeros(_INDEP_WIDTH, dtype=np.float64)
    n = len(alphabet)
    v[:n] = 1.0 / n
    return v


_UNIFORM_SERIES = _uniform_indep_vector(_INDEP_ALPHABETS[0])  # slot 4's alphabet (series)
_UNIFORM_NUMBER = _uniform_indep_vector(_INDEP_ALPHABETS[2])  # slot 6's alphabet (number)
_UNIFORM_BY_NAME = {"series": _UNIFORM_SERIES, "number": _UNIFORM_NUMBER}


def _cross_reinterpret(source_vecs: np.ndarray, source_local: int, target_local: int) -> np.ndarray:
    """Re-key a batch of already-floored native probability vectors (blank
    character always at column `native_len - 1`, padded with 0 beyond) into
    the OTHER native alphabet's space, matching what
    `_slot_prob_vector(sp, target_alphabet)` computes when `sp`'s dict keys
    come from a disjoint alphabet except the shared blank symbol "_" (see
    `_CROSS_ALPHABET_PATCHES`): every real target character has no entry in
    the source dict at all, so it floors to FLOOR; the target's own blank
    column inherits the source's blank probability by NAME, not by index."""
    source_len = int(_INDEP_N[source_local])
    target_len = int(_INDEP_N[target_local])
    n = source_vecs.shape[0]
    out = np.zeros((n, _INDEP_WIDTH))
    out[:, : target_len - 1] = FLOOR
    out[:, target_len - 1] = source_vecs[:, source_len - 1]
    return out


def precompute_plate_batch_arrays(
    posteriors_list: list[list[SlotPosterior]], priors: PlatePriors | None = None
) -> PlateBatchArrays:
    """Build `PlateBatchArrays` for a list of events' plate posteriors. This
    is the O(n_events) work that the scalar path redundantly repeated for
    every pair."""
    priors = priors or PlatePriors.uniform_default()
    n = len(posteriors_list)

    a0 = np.empty((n, _N_STATE))
    a1 = np.empty((n, _N_STATE))
    r0 = np.empty((n, _N_RTO))
    r1 = np.empty((n, _N_RTO))
    indep_base = np.zeros((n, 6, _INDEP_WIDTH))

    for i, posteriors in enumerate(posteriors_list):
        a0[i] = _slot_prob_vector(posteriors[STATE_SLOTS[0]], STATE_ALPHABET)
        a1[i] = _slot_prob_vector(posteriors[STATE_SLOTS[1]], STATE_ALPHABET)
        r0[i] = _slot_prob_vector(posteriors[RTO_SLOTS[0]], RTO_ALPHABET)
        r1[i] = _slot_prob_vector(posteriors[RTO_SLOTS[1]], RTO_ALPHABET)
        for local, (global_idx, alphabet) in enumerate(
            zip(INDEPENDENT_SLOTS, _INDEP_ALPHABETS, strict=True)
        ):
            vec = _slot_prob_vector(posteriors[global_idx], alphabet)
            indep_base[i, local, : len(alphabet)] = vec

    M_state = np.einsum("ni,nj->nij", a0, a1).reshape(n, -1)
    pi_state_flat = priors.state_matrix.reshape(-1)
    W_state = pi_state_flat[None, :] * M_state
    den_state = W_state.sum(axis=1)  # float64, kept full precision (see class docstring)

    M_rto = np.einsum("ni,nj->nij", r0, r1).reshape(n, -1)
    pi_rto_flat = priors.rto_matrix.reshape(-1)
    W_rto = pi_rto_flat[None, :] * M_rto
    den_rto = W_rto.sum(axis=1)

    return PlateBatchArrays(
        n_events=n,
        M_state=M_state.astype(np.float32),
        W_state=W_state.astype(np.float32),
        den_state=den_state,
        M_rto=M_rto.astype(np.float32),
        W_rto=W_rto.astype(np.float32),
        den_rto=den_rto,
        indep_base=indep_base.astype(np.float32),
    )


def _build_indep_variant(base_gathered: np.ndarray, name: str) -> np.ndarray:
    """Apply alignment `name`'s reindex/fill/cross-alphabet-patch to an
    ALREADY-GATHERED (P, 6, W) batch (see `_SHIFT_SPECS` /
    `_CROSS_ALPHABET_PATCHES`). Operating on the gathered P-sized array
    (instead of gathering a full N-sized precomputed variant per alignment,
    the original approach) means each alignment costs one cheap reindex
    over its small (size-6) slot axis rather than another random-access
    gather over every pair -- the dominant cost at scale."""
    if name == "none":
        return base_gathered
    reindex, fill = _SHIFT_SPECS[name]
    variant = np.empty_like(base_gathered)
    for local, (src, fill_name) in enumerate(zip(reindex, fill, strict=True)):
        if src == -1:
            variant[:, local, :] = _UNIFORM_BY_NAME[fill_name]
        else:
            variant[:, local, :] = base_gathered[:, src, :]
    patch = _CROSS_ALPHABET_PATCHES.get(name)
    if patch is not None:
        target_local, source_local = patch
        variant[:, target_local, :] = _cross_reinterpret(
            base_gathered[:, source_local, :], source_local, target_local
        )
    return variant


# Chunk size for batched pair scoring, in PAIRS. A per-pair gather like
# `arrays.M_state[pred_idx]` materialises one row per pair -- for the widest
# array here (M_state/W_state, 676-wide float32) that is
# n_pairs * 676 * 4 bytes, which blows past double-digit GiB at full-city
# pair counts (a real ~5.9M-pair run needs ~15 GiB just for that one gather --
# see docs/decisions.md). Scoring PAIR_CHUNK_SIZE pairs at a time bounds the
# widest per-chunk gather to ~500 MB regardless of total pair count, while
# accumulating results into a preallocated full-size output array so the
# returned arrays and their values are bit-identical to the unchunked path
# (chunking only changes how many pairs are processed per numpy call, never
# per-pair accumulation order or dtype).
PAIR_CHUNK_SIZE = 200_000


def _score_plate_pairs_chunk(
    arrays: PlateBatchArrays, pred_idx: np.ndarray, succ_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Core per-chunk math, unchanged from the original (pre-chunking)
    `score_plate_pairs_batch` body. Called by `score_plate_pairs_batch` on
    slices of at most `PAIR_CHUNK_SIZE` pairs."""
    p = len(pred_idx)

    # Storage is float32 (n_events, ...); accumulation is forced to float64
    # (`dtype=np.float64` below) so summing hundreds of float32 terms never
    # drifts past the 1e-6 scalar-agreement test tolerance -- natural
    # float32 accumulation was tried and measured to drift up to ~6e-6 on
    # real gated pairs, which fails that test, so this is not optional.
    state_num = np.einsum(
        "pk,pk->p", arrays.M_state[pred_idx], arrays.W_state[succ_idx], dtype=np.float64
    )
    state_den = arrays.den_state[pred_idx] * arrays.den_state[succ_idx]
    state_contrib = np.log(np.maximum(state_num, FLOOR)) - np.log(np.maximum(state_den, FLOOR))

    rto_num = np.einsum(
        "pk,pk->p", arrays.M_rto[pred_idx], arrays.W_rto[succ_idx], dtype=np.float64
    )
    rto_den = arrays.den_rto[pred_idx] * arrays.den_rto[succ_idx]
    rto_contrib = np.log(np.maximum(rto_num, FLOOR)) - np.log(np.maximum(rto_den, FLOOR))

    base = state_contrib + rto_contrib  # (P,) alignment-invariant part

    # Gather EACH side's raw per-slot vectors from the N-sized precomputed
    # array only ONCE; every alignment variant is then built cheaply from
    # these already-P-sized arrays (see `_build_indep_variant`).
    a_gathered = arrays.indep_base[pred_idx]  # pred/"A" is never shifted
    succ_gathered = arrays.indep_base[succ_idx]

    totals = np.empty((p, len(ALIGNMENT_NAMES)))
    sumA = a_gathered.sum(axis=2, dtype=np.float64) * _INDEP_INVN[None, :]  # (P, 6)
    for col, name in enumerate(ALIGNMENT_NAMES):
        b_indep = _build_indep_variant(succ_gathered, name)  # (P, 6, W)
        raw_dot = (
            np.einsum("psw,psw->ps", a_gathered, b_indep, dtype=np.float64) * _INDEP_INVN[None, :]
        )
        sumB = b_indep.sum(axis=2, dtype=np.float64) * _INDEP_INVN[None, :]
        den = sumA * sumB
        contrib = np.log(np.maximum(raw_dot, FLOOR)) - np.log(np.maximum(den, FLOOR))
        totals[:, col] = base + contrib.sum(axis=1)

    alignment_idx = np.argmax(totals, axis=1)
    log_lr = totals[np.arange(p), alignment_idx]
    return log_lr, alignment_idx


def score_plate_pairs_batch(
    arrays: PlateBatchArrays, pred_idx: np.ndarray, succ_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batched plate LR: for every (pred_idx[k], succ_idx[k]) pair, the
    max-likelihood alignment's total log-LR. Returns `(log_lr, alignment_idx)`
    where `alignment_idx` indexes `ALIGNMENT_NAMES`. Agrees with
    `score_plate_pair` to float precision (tests/test_batch_scoring.py).

    Processes pairs in chunks of `PAIR_CHUNK_SIZE` (see its docstring) so
    peak memory stays bounded regardless of how many pairs are scored in one
    call; results are numerically identical to scoring everything in one
    shot since each pair's math is fully independent of every other pair's."""
    pred_idx = np.asarray(pred_idx, dtype=np.int64)
    succ_idx = np.asarray(succ_idx, dtype=np.int64)
    p = len(pred_idx)
    if p == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)

    log_lr = np.empty(p, dtype=np.float64)
    alignment_idx = np.empty(p, dtype=np.int64)
    for start in range(0, p, PAIR_CHUNK_SIZE):
        end = min(start + PAIR_CHUNK_SIZE, p)
        log_lr[start:end], alignment_idx[start:end] = _score_plate_pairs_chunk(
            arrays, pred_idx[start:end], succ_idx[start:end]
        )
    return log_lr, alignment_idx


__all__ = [
    "ALIGNMENT_NAMES",
    "FLOOR",
    "PlateBatchArrays",
    "PlateLRResult",
    "PlatePriors",
    "precompute_plate_batch_arrays",
    "score_plate_pair",
    "score_plate_pairs_batch",
]
