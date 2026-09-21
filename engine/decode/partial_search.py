"""Partial-plate search (architecture.md §3, "Partial-plate search works").

`MH12??1234` is just a constraint on the fused plate posterior: every
trajectory can be ranked against it without needing an exact string match
anywhere. This module normalises a query string into the fixed 10-slot
canonical form and scores every candidate trajectory by
`P(y matches query | T)`, read directly off that trajectory's
`engine.decode.consensus.Consensus`.

**Query format (docs/decisions.md's canonical-form rule).** A query must be
exactly `NUM_SLOTS` (10) characters, one per canonical slot, matching how a
true plate is laid out on generation (`sim.vehicles._sample_true_plate`):
state (2) + RTO (2) + series letters left-aligned into 2 slots + number
digits right-aligned into 4 slots, `_` for a slot that is legitimately
blank (a 1-letter series, a <4-digit number). `?` means "unknown, any
character in this slot's alphabet". A shorter, human-typed plate like
`MH12A1234` is genuinely ambiguous without knowing how many series letters
vs number digits it has (`?` cannot resolve that ambiguity for you either)
-- this module deliberately does not guess a split; a caller with a
variable-length query must pad/align it to 10 slots itself before calling
`search` (this is a known limitation, not an oversight -- see the Day 3
decode report).

**Match probability.** Under the same conditionally-independent-groups
factorisation `consensus.py` uses, the probability that a trajectory's true
plate matches a *fully specified* query is the product, over every slot the
query pins down, of that slot's fused marginal probability for the pinned
character (a `?` slot contributes exactly 1.0 -- it is marginalised out,
imposing no constraint). This is an approximation for the two joint groups
(state, RTO) when a query pins only one of the pair's two slots: the exact
answer would need the joint table, but `Consensus.per_slot` only exposes
marginals (matching the api-contract.md `PlateConsensus` shape, which is
per-slot, not per-group) -- summing a joint table over the free slot to get
exactly the marginal means this is only an approximation when *both* slots
of a joint pair are pinned simultaneously (their true joint co-occurrence
probability can differ from the product of marginals if the two slots are
correlated by the fused posterior). Good enough for ranking search
candidates; documented, not hidden.
"""

from dataclasses import dataclass

from engine.contracts.plate import BLANK, NUM_SLOTS, PLATE_SLOTS
from engine.contracts.trajectory import Trajectory
from engine.decode.consensus import Consensus

WILDCARD = "?"


@dataclass
class SearchFilters:
    """Engine-layer search filters -- the subset of `/api/search`'s query
    params (docs/api-contract.md) answerable from `Trajectory` alone.
    `color`/`vehicle_type` need a join against `DetectionEvent.attributes`,
    which belongs to the API layer (it already has to fetch events to build
    `TrajectorySummary`); duplicating that join here would mean either
    threading `events_by_id` through every call or silently only ever
    filtering on the first event, so it is left to the caller."""

    time_from: object | None = None  # datetime, compared to trajectory.end_time
    time_to: object | None = None  # datetime, compared to trajectory.start_time
    camera_id: str | None = None


@dataclass
class SearchHit:
    trajectory: Trajectory
    probability: float
    matched_plate: str


def normalise_query(query: str) -> list[str]:
    """Validate and split a canonical-form query string into a list of
    `NUM_SLOTS` per-slot query characters (each one of: a real alphabet
    character, `BLANK`, or `WILDCARD`)."""
    q = query.strip().upper()
    if len(q) != NUM_SLOTS:
        raise ValueError(
            f"query must be exactly {NUM_SLOTS} canonical-slot characters "
            f"(got {len(q)!r} from {query!r}); pad/align shorter input to the "
            f"10-slot canonical form before calling search (see module docstring)"
        )
    slots = list(q)
    for idx, ch in enumerate(slots):
        if ch in (WILDCARD, BLANK):
            continue
        if ch not in PLATE_SLOTS[idx]:
            raise ValueError(f"{ch!r} is not valid in slot {idx} (alphabet {PLATE_SLOTS[idx]!r})")
    return slots


def _slot_match_probability(consensus: Consensus, slot_idx: int, query_char: str) -> float:
    if query_char == WILDCARD:
        return 1.0
    return consensus.per_slot[slot_idx].prob(query_char)


def match_probability(consensus: Consensus, query_slots: list[str]) -> float:
    """`P(y matches query | T)` under the fused posterior (module
    docstring). Multiplicative across slots -- a single pinned slot with
    near-zero fused mass on the queried character correctly drives the
    whole trajectory's match probability toward zero."""
    prob = 1.0
    for idx, q_char in enumerate(query_slots):
        prob *= _slot_match_probability(consensus, idx, q_char)
    return prob


def _most_likely_completion(consensus: Consensus, query_slots: list[str]) -> str:
    """The most probable full plate consistent with the query: pinned slots
    keep the query's character, wildcard slots take the fused posterior's
    own argmax -- i.e. "what did the linking engine actually see there,
    given what you asked for.\""""
    chars = []
    for idx, q_char in enumerate(query_slots):
        chars.append(q_char if q_char != WILDCARD else consensus.per_slot[idx].argmax())
    return "".join(chars)


def _passes_filters(traj: Trajectory, filters: SearchFilters | None) -> bool:
    if filters is None:
        return True
    if filters.time_from is not None and traj.end_time < filters.time_from:
        return False
    if filters.time_to is not None and traj.start_time > filters.time_to:
        return False
    if filters.camera_id is not None and filters.camera_id not in traj.camera_sequence:
        return False
    return True


def search(
    query: str,
    trajectories_with_consensus: list[tuple[Trajectory, Consensus]],
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
    """Rank every (filter-passing) trajectory by `P(y matches query | T)`,
    descending. Zero-probability hits are dropped (a `?`-free slot pinned to
    a character with no support at all under the fused posterior is not a
    plausible candidate, and keeping every trajectory at prob=0 would just
    be noise in the result list)."""
    query_slots = normalise_query(query)
    hits: list[SearchHit] = []
    for traj, consensus in trajectories_with_consensus:
        if not _passes_filters(traj, filters):
            continue
        prob = match_probability(consensus, query_slots)
        if prob <= 0.0:
            continue
        matched_plate = _most_likely_completion(consensus, query_slots)
        hits.append(SearchHit(trajectory=traj, probability=prob, matched_plate=matched_plate))
    hits.sort(key=lambda h: h.probability, reverse=True)
    return hits


__all__ = [
    "WILDCARD",
    "SearchFilters",
    "SearchHit",
    "match_probability",
    "normalise_query",
    "search",
]
