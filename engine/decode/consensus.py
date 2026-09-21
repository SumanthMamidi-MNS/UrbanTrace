"""Consensus plate decoding (architecture.md §3, "Consensus decoding").

For a trajectory `T` = a time-ordered sequence of events all believed to be
the same vehicle: `P(y|T) ∝ pi(y) * PROD_m P(O_m|y)`. This is the "linking
repairs OCR" claim -- once N reads sit on one trajectory, fusing their
per-character posteriors recovers the true plate far above any single
read's accuracy, because independent OCR errors on different events rarely
agree with each other.

Factorisation mirrors `engine.scoring.plate_lr` exactly, because it is the
same generative model evaluated at a different point: state code (slots
0,1) and RTO code (slots 2,3) are joint units with a real prior over their
(26x26) / (10x10) vocabulary; the series/number slots (4-9) are independent
with a flat prior. Given the factorisation, the joint posterior over the
whole 10-slot plate is the *product* of each group's own posterior, so
"decoded plate" = concatenation of each group's own argmax, "confidence" =
product of each group's own argmax probability mass, and "entropy" = sum of
each group's own entropy (conditionally-independent-groups identity for
entropy of a product distribution).

All per-character multiplications happen in log-space and are normalised by
subtracting the running max before exponentiating (a standard log-sum-exp
softmax) so a trajectory with many events never underflows to all-zero
before normalisation -- the failure mode a naive PROD_m in raw probability
space would hit within a few dozen events.

Every probability read from an event's `SlotPosterior` is floored at FLOOR,
matching plate_lr's own defence -- no finite input can send a log-posterior
to -inf here either (docs/decisions.md, "Day 1c").

**Robust (epsilon-contamination) likelihood.** A pure product PROD_m
P(O_m|y) lets a single read veto an otherwise-clear majority: a character
that sits in two confusable groups (e.g. '8' in both {B,8,6} and {3,9,8})
collects leaked mass from every read regardless of which group triggered
it, while the true character only gets the tiny uniform background floor
from the one read that missed it entirely. In a product, that background
term acts as a veto proportional to 1/background -- exactly the
exact-match brittleness this project exists to remove, reintroduced in
miniature. Each read is instead modelled as coming from the OCR channel
with probability `(1 - EPS)` or from an unmodelled outlier process with
probability `EPS`:

    P(O_m|y) = (1 - EPS) * P_ocr(O_m|y) + EPS * U(O_m)

applied per slot group (the joint state/RTO alphabet, or a single
independent slot's alphabet) -- the same grouping the module already uses
-- where `U` is uniform over that group's alphabet. This caps any single
read's contribution to the log-product at ~log(EPS * U), so one outlier
can no longer veto a clear majority, while full agreement between reads is
unaffected (EPS=0 recovers the original pure-product likelihood exactly).
"""

from dataclasses import dataclass

import numpy as np

from engine.contracts.events import DetectionEvent
from engine.contracts.plate import (
    NUM_SLOTS,
    PLATE_SLOTS,
    RTO_ALPHABET,
    STATE_ALPHABET,
    SlotPosterior,
)
from engine.scoring.plate_lr import RTO_SLOTS, STATE_SLOTS, PlatePriors

FLOOR = 1e-9
TOP_K = 5

# Epsilon-contamination mixing weight for the robust per-read likelihood (see
# module docstring): chosen as the best of {0, 0.001, 0.005, 0.01, 0.02, 0.05,
# 0.1, 0.2} by read-weighted consensus accuracy (len>=2) on a seed=777 tuning
# dataset disjoint from data/run1 (0.9904 at eps=0 rising monotonically to
# 0.9911 at eps=0.2 -- the largest swept value won, with no per-length
# regression; see eval/consensus_report.py for the run1 before/after table).
EPS = 0.2

INDEPENDENT_SLOTS = tuple(i for i in range(NUM_SLOTS) if i not in (*STATE_SLOTS, *RTO_SLOTS))

_STATE_IDX = {c: i for i, c in enumerate(STATE_ALPHABET)}
_RTO_IDX = {c: i for i, c in enumerate(RTO_ALPHABET)}


@dataclass
class SlotConsensus:
    """One slot's fused posterior. `full_posterior` is the complete,
    normalised distribution over the slot's alphabet (used by
    `engine.decode.partial_search` for exact query-match probabilities);
    `top5` is the API-contract-shaped truncation (`PlateConsensus.per_slot`
    in docs/api-contract.md is `SlotRead[][]`, top <= 5 per slot)."""

    full_posterior: dict[str, float]
    top5: list[tuple[str, float]]

    def prob(self, char: str) -> float:
        """Probability of `char` under the full (untruncated) posterior;
        0.0 for a character outside this slot's alphabet."""
        return self.full_posterior.get(char, 0.0)

    def argmax(self) -> str:
        return max(self.full_posterior.items(), key=lambda kv: kv[1])[0]


@dataclass
class Consensus:
    decoded_plate: str
    confidence: float
    per_slot: list[SlotConsensus]  # length 10, index-aligned with PLATE_SLOTS
    entropy_bits: float
    single_read_plates: list[str]  # plate_argmax per event, time order


def _probs(sp: SlotPosterior, alphabet: list[str]) -> np.ndarray:
    return np.array([max(sp.probs.get(c, 0.0), FLOOR) for c in alphabet], dtype=np.float64)


def _softmax_normalise(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - np.max(log_values)
    probs = np.exp(shifted)
    return probs / probs.sum()


def _entropy_bits(probs: np.ndarray) -> float:
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log2(nz)))


def _independent_slot_consensus(events: list[DetectionEvent], slot_idx: int) -> SlotConsensus:
    alphabet = PLATE_SLOTS[slot_idx]
    n = len(alphabet)
    uniform_mass = 1.0 / n
    log_post = np.zeros(n)
    for e in events:
        p_ocr = _probs(e.plate_posterior[slot_idx], alphabet)
        mixed = (1.0 - EPS) * p_ocr + EPS * uniform_mass
        log_post += np.log(mixed)
    post = _softmax_normalise(log_post)
    full = dict(zip(alphabet, post.tolist(), strict=True))
    ranked = sorted(full.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
    return SlotConsensus(full_posterior=full, top5=ranked)


def _joint_group_consensus(
    events: list[DetectionEvent],
    slot_a: int,
    slot_b: int,
    alphabet_a: list[str],
    alphabet_b: list[str],
    prior_matrix: np.ndarray,
) -> tuple[SlotConsensus, SlotConsensus, np.ndarray, float]:
    """Returns (consensus_a, consensus_b, normalised joint matrix, entropy
    of the joint in bits). The joint matrix is `len(alphabet_a) x
    len(alphabet_b)`; per-slot posteriors are its marginals."""
    uniform_joint_mass = 1.0 / (len(alphabet_a) * len(alphabet_b))
    log_joint = np.log(np.maximum(prior_matrix, FLOOR))
    for e in events:
        p_a = _probs(e.plate_posterior[slot_a], alphabet_a)
        p_b = _probs(e.plate_posterior[slot_b], alphabet_b)
        joint_ocr = np.outer(p_a, p_b)
        mixed = (1.0 - EPS) * joint_ocr + EPS * uniform_joint_mass
        log_joint = log_joint + np.log(mixed)

    shifted = log_joint - np.max(log_joint)
    joint = np.exp(shifted)
    joint = joint / joint.sum()

    marg_a = joint.sum(axis=1)
    marg_b = joint.sum(axis=0)

    full_a = dict(zip(alphabet_a, marg_a.tolist(), strict=True))
    full_b = dict(zip(alphabet_b, marg_b.tolist(), strict=True))
    top_a = sorted(full_a.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
    top_b = sorted(full_b.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]

    entropy = _entropy_bits(joint.ravel())
    return (
        SlotConsensus(full_posterior=full_a, top5=top_a),
        SlotConsensus(full_posterior=full_b, top5=top_b),
        joint,
        entropy,
    )


def decode_trajectory(
    events: list[DetectionEvent], priors: PlatePriors | None = None
) -> Consensus:
    """Fuse every event's per-character plate posterior into one consensus
    read for the trajectory they (are believed to) belong to. `events` need
    not be pre-sorted -- they are sorted by timestamp here so
    `single_read_plates` is reported in time order."""
    if not events:
        raise ValueError("decode_trajectory requires at least one event")
    priors = priors or PlatePriors.uniform_default()
    events_sorted = sorted(events, key=lambda e: e.timestamp)

    per_slot: list[SlotConsensus | None] = [None] * NUM_SLOTS
    confidence = 1.0
    entropy_bits = 0.0
    decoded_chars: list[str] = [""] * NUM_SLOTS

    cons_s0, cons_s1, joint_state, ent_state = _joint_group_consensus(
        events_sorted, STATE_SLOTS[0], STATE_SLOTS[1], STATE_ALPHABET, STATE_ALPHABET,
        priors.state_matrix,
    )
    per_slot[STATE_SLOTS[0]] = cons_s0
    per_slot[STATE_SLOTS[1]] = cons_s1
    i0, i1 = np.unravel_index(np.argmax(joint_state), joint_state.shape)
    decoded_chars[STATE_SLOTS[0]] = STATE_ALPHABET[i0]
    decoded_chars[STATE_SLOTS[1]] = STATE_ALPHABET[i1]
    confidence *= float(joint_state[i0, i1])
    entropy_bits += ent_state

    cons_r0, cons_r1, joint_rto, ent_rto = _joint_group_consensus(
        events_sorted, RTO_SLOTS[0], RTO_SLOTS[1], RTO_ALPHABET, RTO_ALPHABET,
        priors.rto_matrix,
    )
    per_slot[RTO_SLOTS[0]] = cons_r0
    per_slot[RTO_SLOTS[1]] = cons_r1
    j0, j1 = np.unravel_index(np.argmax(joint_rto), joint_rto.shape)
    decoded_chars[RTO_SLOTS[0]] = RTO_ALPHABET[j0]
    decoded_chars[RTO_SLOTS[1]] = RTO_ALPHABET[j1]
    confidence *= float(joint_rto[j0, j1])
    entropy_bits += ent_rto

    for idx in INDEPENDENT_SLOTS:
        cons = _independent_slot_consensus(events_sorted, idx)
        per_slot[idx] = cons
        char = cons.argmax()
        decoded_chars[idx] = char
        confidence *= cons.prob(char)
        entropy_bits += _entropy_bits(np.array(list(cons.full_posterior.values())))

    assert all(c is not None for c in per_slot)

    return Consensus(
        decoded_plate="".join(decoded_chars),
        confidence=confidence,
        per_slot=per_slot,  # type: ignore[arg-type]
        entropy_bits=entropy_bits,
        single_read_plates=[e.plate_argmax for e in events_sorted],
    )


__all__ = [
    "FLOOR",
    "TOP_K",
    "Consensus",
    "SlotConsensus",
    "decode_trajectory",
]
