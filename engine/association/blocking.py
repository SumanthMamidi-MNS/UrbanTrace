"""Secondary, overflow-only candidate blocking (architecture.md §3, §5, §7).

"The spatio-temporal gate is the primary and sufficient blocker -- plate-
based blocking is not" (architecture.md §7). Using a plate index as the
PRIMARY filter would silently reintroduce the exact brittleness this whole
project exists to attack: a badly-misread plate would never even become a
candidate, and no amount of downstream scoring cleverness could recover it.

This module exists purely as an overflow valve: when gating.py's window
leaves MORE candidates for one event than a fixed budget (default 500,
matching architecture.md §3's stated per-window scoring capacity), we need
some way to cut the set down before scoring rather than silently scoring an
unbounded number of pairs. A `rapidfuzz`-backed deletion-neighbourhood index
over `plate_argmax` ranks the overflowing candidates by edit distance to the
destination event's own (possibly-misread) plate and keeps the closest
`budget` of them.

This is a lossy step -- a true predecessor whose plate was badly misread in
BOTH directions could rank outside the kept budget -- which is exactly why
architecture.md and docs/decisions.md are explicit that it must engage
RARELY (an overflow path, not the default path) and its recall cost must be
measured and reported, never assumed away (tests/test_blocking.py, and
eval/blocking_report.py at production scale)."""

from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

from engine.contracts.events import DetectionEvent

# Must sit ABOVE the gate's normal output so blocking stays an overflow valve
# (architecture.md §7: plate-based filtering must never be the default path,
# or a badly misread plate would never become a candidate). The gate's
# congestion tail floor (engine.association.gating.CONGESTION_TAIL_FACTOR)
# raised normal candidates/read from ~218 to ~1,000 (peak-hour mean ~1,700),
# at which the old budget of 500 engaged on ~74% of reads. Note: blocking is
# not in the production solve path (window.py scores every gated candidate);
# this module is kept as a documented overflow mechanism.
DEFAULT_CANDIDATE_BUDGET = 4000
DEFAULT_MAX_EDIT_DISTANCE = 3
# Above this many overflowing candidates, use the deletion-index fast path
# to pre-filter before ranking; below it, an exhaustive rapidfuzz scan is
# both simpler and, at this scale (an overflow event's candidate count is
# already bounded by the gate to a few hundred/thousand), faster than
# building a deletion index wide enough not to cost recall (a deletion
# index only guarantees no false negatives within `max_deletions` of edit
# distance, and OCR-corrupted plates can exceed that -- widening the index
# radius enough to cover them makes index construction combinatorially
# expensive, see docs/decisions.md "Day 3a").
DELETION_INDEX_THRESHOLD = 4000


def _deletion_variants(s: str, max_deletions: int) -> set[str]:
    """Every string reachable from `s` by deleting up to `max_deletions`
    characters (the "deletion neighbourhood", SymSpell-style) -- the index
    key space that lets an approximate match be found by exact dict lookup
    instead of a pairwise edit-distance scan over every candidate."""
    variants = {s}
    frontier = {s}
    for _ in range(max_deletions):
        next_frontier: set[str] = set()
        for w in frontier:
            for i in range(len(w)):
                next_frontier.add(w[:i] + w[i + 1 :])
        variants |= next_frontier
        frontier = next_frontier
    return variants


@dataclass
class DeletionIndex:
    """Deletion-neighbourhood index over a set of (event_id, plate_argmax)
    pairs: maps every deletion-variant of every plate to the event_ids whose
    plate produces it, so a query plate's own deletion neighbourhood can be
    intersected against it in O(variants) rather than O(n) candidates."""

    max_deletions: int
    _index: dict[str, list[str]] = field(default_factory=dict, init=False)
    _plate_by_event: dict[str, str] = field(default_factory=dict, init=False)

    @classmethod
    def build(cls, events: list[DetectionEvent], max_deletions: int) -> "DeletionIndex":
        idx = cls(max_deletions=max_deletions)
        for e in events:
            idx._plate_by_event[e.event_id] = e.plate_argmax
            for variant in _deletion_variants(e.plate_argmax, max_deletions):
                idx._index.setdefault(variant, []).append(e.event_id)
        return idx

    def approximate_matches(self, query_plate: str) -> set[str]:
        """event_ids whose plate shares at least one deletion-variant with
        `query_plate` -- a superset of everything within edit distance
        `2 * max_deletions` (deletions on both sides), refined by an exact
        edit-distance check in `block_candidates` below."""
        matches: set[str] = set()
        for variant in _deletion_variants(query_plate, self.max_deletions):
            matches.update(self._index.get(variant, ()))
        return matches


@dataclass
class BlockingStats:
    n_events_total: int = 0
    n_events_engaged: int = 0
    n_true_pairs_checked: int = 0
    n_true_pairs_dropped_by_blocking: int = 0

    @property
    def engagement_rate(self) -> float:
        return self.n_events_engaged / self.n_events_total if self.n_events_total else 0.0

    @property
    def recall_cost(self) -> float:
        """Fraction of checked true predecessor pairs that blocking itself
        removed from an already-gated candidate set (i.e. pairs the GATE
        found, that blocking's plate-similarity ranking then dropped)."""
        return (
            self.n_true_pairs_dropped_by_blocking / self.n_true_pairs_checked
            if self.n_true_pairs_checked
            else 0.0
        )


def block_candidates(
    target_event: DetectionEvent,
    candidate_events: list[DetectionEvent],
    budget: int = DEFAULT_CANDIDATE_BUDGET,
    max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
) -> list[DetectionEvent]:
    """If `candidate_events` (already gate-filtered) fits within `budget`,
    return it UNCHANGED -- blocking must never touch the default path. Only
    when the gate itself overflows the budget does this rank candidates by
    plate edit-distance to `target_event` and keep the closest `budget`.

    Ranking never hard-rejects a candidate for exceeding `max_edit_distance`
    -- that threshold only picks which of two ranking strategies to use
    (see `DELETION_INDEX_THRESHOLD`). Rejecting outright would silently
    drop a true predecessor whose plate was badly OCR-corrupted on BOTH
    sides just because it didn't happen to also rank inside a fixed
    distance -- exactly the brittleness this module must not reintroduce
    (architecture.md §7). The budget alone decides how many survive."""
    if len(candidate_events) <= budget:
        return candidate_events

    query = target_event.plate_argmax
    if len(candidate_events) > DELETION_INDEX_THRESHOLD:
        index = DeletionIndex.build(candidate_events, max_edit_distance)
        matched_ids = index.approximate_matches(query)
        by_id = {e.event_id: e for e in candidate_events}
        pool = [by_id[eid] for eid in matched_ids] or candidate_events
    else:
        pool = candidate_events

    scored = [(Levenshtein.distance(cand.plate_argmax, query), cand) for cand in pool]
    scored.sort(key=lambda pair: pair[0])
    return [cand for _, cand in scored[:budget]]


def apply_blocking(
    candidates: dict[str, list[str]],
    events_by_id: dict[str, DetectionEvent],
    budget: int = DEFAULT_CANDIDATE_BUDGET,
    max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
) -> tuple[dict[str, list[str]], BlockingStats]:
    """Apply overflow-only blocking to a gating.gate_candidates() result.
    Returns the (possibly-narrowed) candidate map plus engagement stats."""
    stats = BlockingStats(n_events_total=len(candidates))
    out: dict[str, list[str]] = {}

    for event_id, cand_ids in candidates.items():
        if len(cand_ids) <= budget:
            out[event_id] = cand_ids
            continue
        stats.n_events_engaged += 1
        target = events_by_id[event_id]
        cand_events = [events_by_id[cid] for cid in cand_ids]
        blocked = block_candidates(target, cand_events, budget, max_edit_distance)
        out[event_id] = [e.event_id for e in blocked]

    return out, stats


def measure_recall_cost(
    events: list[DetectionEvent],
    gated_candidates: dict[str, list[str]],
    blocked_candidates_map: dict[str, list[str]],
) -> BlockingStats:
    """How much true-predecessor recall blocking cost, isolated from the
    gate's own recall: only checks true pairs the GATE already recalled,
    and counts how many blocking then dropped."""
    stats = BlockingStats(n_events_total=len(gated_candidates))
    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in events:
        if e.gt_vehicle_id is not None:
            by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    engaged_events = {
        eid for eid, cands in gated_candidates.items() if len(cands) > DEFAULT_CANDIDATE_BUDGET
    }
    stats.n_events_engaged = len(engaged_events)

    for evs in by_vehicle.values():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        for a, b in zip(evs_sorted[:-1], evs_sorted[1:], strict=True):
            if b.event_id not in engaged_events:
                continue
            if a.event_id not in gated_candidates.get(b.event_id, []):
                continue  # gate itself missed it -- not blocking's fault
            stats.n_true_pairs_checked += 1
            if a.event_id not in blocked_candidates_map.get(b.event_id, []):
                stats.n_true_pairs_dropped_by_blocking += 1

    return stats


__all__ = [
    "DEFAULT_CANDIDATE_BUDGET",
    "DEFAULT_MAX_EDIT_DISTANCE",
    "BlockingStats",
    "DeletionIndex",
    "apply_blocking",
    "block_candidates",
    "measure_recall_cost",
]
