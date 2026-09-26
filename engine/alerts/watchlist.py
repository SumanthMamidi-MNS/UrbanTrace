"""Watchlist probabilistic plate matching (PRD component 2: "flag
blacklisted vehicles ... in real time"; docs/api-contract.md Contract v2
additions, "Watchlist matching is probabilistic").

Reuses `engine.decode.consensus.decode_trajectory` (the SAME fused-posterior
machinery ordinary trajectory consensus uses) for the GROWING TRAJECTORY
CONSENSUS check, and `engine.decode.partial_search.match_probability` /
`normalise_query` (the SAME per-slot query scoring `/api/search` uses) for
BOTH checks -- rather than duplicating either.

**Single reads deliberately do NOT go through `decode_trajectory`.** Its
epsilon-contamination mixing (`engine.decode.consensus.EPS`) exists to keep
one outlier read from vetoing a majority when fusing MULTIPLE reads; applied
to a single read it does the opposite of what a single-read check needs --
it caps how confident that ONE read is allowed to look. Empirically, at
`EPS=0.2`, a maximally confident single read run through `decode_trajectory`
tops out around P~0.19 for a fully-pinned 10-slot query (each independent
slot's mixed posterior is capped at `(1-EPS) + EPS/alphabet_size`, and the
product of ten such capped terms lands well under `MATCH_THRESHOLD` no
matter how certain the underlying read actually was) -- i.e. single-read
watchlist matching would never fire at all. Single reads therefore score
directly off the read's own reconstructed per-slot posterior (still scored
by the shared `match_probability`), while the growing trajectory consensus
-- which genuinely is fusing multiple, potentially-disagreeing reads -- uses
the full `decode_trajectory` machinery as-is.

**Why not the DB's own precomputed `TrajectoryRow.consensus`?** That blob is
the FINAL consensus over ALL of a trajectory's events, computed once at
ingest time -- exactly the thing the contract says NOT to use here ("the
trajectory consensus AS THE TRAJECTORY GROWS"). Live replay needs the
consensus recomputed on only the events revealed so far, so this module
takes a list of `EventRow`s (any prefix of a trajectory) and reconstructs a
full per-slot posterior for each -- `EventRow.posterior` only stores the
TOP-5 characters per slot (matching `EventDetail.plate_posterior`'s wire
shape), not the complete distribution `TrajectoryRow.consensus` keeps for
whole trajectories. `_reconstruct_slot_posterior` rebuilds a valid
`SlotPosterior` from that top-5 by spreading the remaining probability mass
uniformly over the rest of the slot's alphabet -- an approximation for the
discarded tail (characters ranked 6+), never for the top-5 characters
themselves, which keep their real probabilities. Documented, not hidden
(mirrors `engine.decode.partial_search`'s own documented joint-slot
approximation).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from engine.contracts.plate import PLATE_SLOTS, SlotPosterior
from engine.decode.consensus import Consensus, SlotConsensus, decode_trajectory
from engine.decode.partial_search import match_probability, normalise_query

# P(plate ∈ pattern | evidence) >= MATCH_THRESHOLD counts as a match
# (docs/api-contract.md Contract v2, "Semantics": "a read matches when
# P(plate ∈ pattern | evidence) >= 0.5 (server constant, documented)").
MATCH_THRESHOLD = 0.5


@dataclass
class LiteEvent:
    """Minimal duck-typed stand-in for `engine.contracts.events.DetectionEvent`
    carrying only the three attributes `engine.decode.consensus.decode_trajectory`
    actually reads (`.timestamp` for sort order, `.plate_posterior` for the
    per-slot fusion, `.plate_argmax` for `Consensus.single_read_plates`).
    Building a real (pydantic-validated) `DetectionEvent` would additionally
    require a fake 128-d L2-normalised embedding and vehicle attributes
    `decode_trajectory` never touches -- `decode_trajectory` only type-hints
    `list[DetectionEvent]`, it does not isinstance-check, so this lighter
    object works identically at runtime."""

    plate_posterior: list[SlotPosterior]
    plate_argmax: str
    timestamp: datetime


def reconstruct_slot_posterior(top: list[dict], slot_idx: int) -> SlotPosterior:
    """Rebuild a full `SlotPosterior` for one slot from its stored top-5
    (see module docstring for the approximation this makes)."""
    alphabet = PLATE_SLOTS[slot_idx]
    known = {t["char"]: t["prob"] for t in top if t["char"] in alphabet}
    remaining_chars = [c for c in alphabet if c not in known]
    leftover = max(0.0, 1.0 - sum(known.values()))
    if remaining_chars:
        share = leftover / len(remaining_chars)
        for c in remaining_chars:
            known[c] = share
    elif leftover > 0 and known:
        # Every alphabet character was already in the top-5 (small
        # alphabets only) -- rescale so probabilities still sum to 1.
        scale = 1.0 / sum(known.values())
        known = {c: p * scale for c, p in known.items()}
    return SlotPosterior(probs=known)


def event_row_to_lite_event(row) -> LiteEvent:
    """`row`: an `api.db.EventRow` (or anything with `.posterior`,
    `.plate_argmax`, `.timestamp` in the same shapes)."""
    posterior = [
        reconstruct_slot_posterior(slot["top"], idx) for idx, slot in enumerate(row.posterior)
    ]
    return LiteEvent(
        plate_posterior=posterior, plate_argmax=row.plate_argmax, timestamp=row.timestamp
    )


@dataclass
class _RawSingleReadConsensus:
    """Duck-typed like `Consensus` for `match_probability`'s purposes (it
    only ever reads `.per_slot[i].prob(char)`) -- but built directly from
    one read's own reconstructed posterior, with NO epsilon-contamination
    mixing (see module docstring for why that matters for single reads)."""

    per_slot: list[SlotConsensus]


def best_match_probability(consensus: Consensus, canonical_patterns: list[str]) -> float:
    """Max match probability over every grammar-consistent canonical form a
    watchlist pattern expanded to (mirrors `/api/search`'s own "merge hits by
    max probability" rule for ambiguous `?`-straddled patterns,
    `api/routers/core.py::search`)."""
    best = 0.0
    for canonical in canonical_patterns:
        prob = match_probability(consensus, normalise_query(canonical))
        if prob > best:
            best = prob
    return best


def single_read_match_probability(row, canonical_patterns: list[str]) -> float:
    """`P(plate ∈ pattern | this one read)` -- docs/api-contract.md Contract
    v2, evaluation (a). Uses the read's own reconstructed per-slot posterior
    directly (module docstring explains why `decode_trajectory`'s
    multi-read-fusion robustness is the wrong tool here)."""
    per_slot = [
        SlotConsensus(full_posterior=reconstruct_slot_posterior(slot["top"], idx).probs, top5=[])
        for idx, slot in enumerate(row.posterior)
    ]
    consensus = _RawSingleReadConsensus(per_slot=per_slot)
    return best_match_probability(consensus, canonical_patterns)


def trajectory_consensus_match_probability(rows, canonical_patterns: list[str]) -> float:
    """`P(plate ∈ pattern | trajectory consensus so far)` -- Contract v2
    evaluation (b), the product differentiator: `rows` is any prefix of a
    trajectory's events (the ones REVEALED so far during replay), and this
    can match even when `single_read_match_probability` on the latest event
    alone does not, because the earlier correctly-read events outvote one
    misread in the fused posterior."""
    lite_events = [event_row_to_lite_event(r) for r in rows]
    consensus = decode_trajectory(lite_events)
    return best_match_probability(consensus, canonical_patterns)


__all__ = [
    "MATCH_THRESHOLD",
    "LiteEvent",
    "best_match_probability",
    "event_row_to_lite_event",
    "reconstruct_slot_posterior",
    "single_read_match_probability",
    "trajectory_consensus_match_probability",
]
