"""Stratified pairwise evaluation (docs/decisions.md, "Day 2b").

A plate is a near-unique 10-character identifier. Sampling negatives
uniformly (or even "hard-in-time-and-space") makes the plate channel look
artificially perfect, because two unrelated vehicles almost always have
completely different plates -- there is no headroom left for appearance or
kinematics to demonstrate any value, and reported as one scalar AUC this
would argue the three-channel architecture is unnecessary. Worse, the one
case that actually matters -- a cloned or near-miss plate -- is invisible
in an aggregate number.

Four named strata, each with its own positive/negative construction:

  S1 "routine"       -- hard in time/space (closest-in-time different
                         vehicle at the same destination camera), trivial on
                         plate. The everyday case.
  S2 "clone"         -- negatives are pairs of DIFFERENT vehicles sharing
                         the exact same true plate (the deliberately
                         injected clone population, plus any incidental
                         collisions). Positives are genuine consecutive
                         passages of those same clone-plate vehicles.
  S3 "degraded"      -- positives where the combined unread + misread slot
                         count across BOTH events is >= 4. Sampled from a
                         deliberately higher-noise dataset variant (the
                         default corruption rate makes this stratum too
                         rare to measure reliably otherwise -- see
                         docs/decisions.md, "Day 2b").
  S4 "plate-similar" -- negatives are pairs of DIFFERENT vehicles whose true
                         plates are within edit distance <= 2 (a genuine
                         near-miss registration, not a contrived one -- see
                         `sim.vehicles._inject_near_miss_plates`).

Every stratum returns the same shape: a list of (event_a, event_b, label)
triples, scoreable through `engine.calibration.calibrate.score_labeled_pairs`
exactly like `eval.metrics.sample_labeled_pairs`.
"""

import random
from dataclasses import dataclass

from engine.contracts.events import DetectionEvent
from sim.generate import GeneratedDataset
from sim.vehicles import Vehicle

LabeledPair = tuple[DetectionEvent, DetectionEvent, int]

MIN_DEGRADED_SLOTS = 4
MAX_PLATE_SIMILAR_EDIT_DISTANCE = 2


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Plates are 10 characters, so plain O(n*m) DP is
    more than fast enough -- no need for a library dependency."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _events_by_vehicle(events: list[DetectionEvent]) -> dict[str, list[DetectionEvent]]:
    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in events:
        if e.gt_vehicle_id is not None:
            by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)
    for evs in by_vehicle.values():
        evs.sort(key=lambda e: e.timestamp)
    return by_vehicle


def _events_by_camera(events: list[DetectionEvent]) -> dict[str, list[DetectionEvent]]:
    by_camera: dict[str, list[DetectionEvent]] = {}
    for e in events:
        by_camera.setdefault(e.camera_id, []).append(e)
    return by_camera


def _closest_in_time(
    candidates: list[DetectionEvent], ref: DetectionEvent
) -> DetectionEvent | None:
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e.timestamp - ref.timestamp).total_seconds()))


def _degraded_slot_count(event: DetectionEvent, true_plate: str) -> int:
    """Number of this event's plate slots that are either unread (uniform
    posterior) or read to the wrong character."""
    count = 0
    for i, sp in enumerate(event.plate_posterior):
        if sp.is_uninformative() or sp.argmax() != true_plate[i]:
            count += 1
    return count


def _plate_to_vehicle_ids(vehicles: list[Vehicle]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for v in vehicles:
        groups.setdefault(v.true_plate, []).append(v.gt_vehicle_id)
    return groups


def find_clone_groups(vehicles: list[Vehicle]) -> dict[str, list[str]]:
    """plate -> [vehicle_id, ...] for every true plate shared by >= 2
    distinct vehicles (the injected clone population, plus any astronomically
    rare incidental collision -- both are genuinely "same plate, different
    vehicle" and belong in this stratum either way)."""
    return {plate: ids for plate, ids in _plate_to_vehicle_ids(vehicles).items() if len(ids) >= 2}


def find_plate_similar_pairs(
    vehicles: list[Vehicle], max_edit_distance: int = MAX_PLATE_SIMILAR_EDIT_DISTANCE
) -> list[tuple[str, str]]:
    """(vehicle_id, vehicle_id) pairs of DIFFERENT vehicles whose true
    plates are within `max_edit_distance` -- excludes exact matches (that's
    stratum S2's job). Grouped by the (state, RTO, series) 6-character
    prefix first: two plates differing by more than a couple of characters
    outside that prefix cannot be within edit distance 2, so this avoids an
    O(n^2) full scan over the whole vehicle population."""
    by_prefix: dict[str, list[Vehicle]] = {}
    for v in vehicles:
        by_prefix.setdefault(v.true_plate[:6], []).append(v)

    pairs: list[tuple[str, str]] = []
    for group in by_prefix.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.true_plate == b.true_plate:
                    continue
                if edit_distance(a.true_plate, b.true_plate) <= max_edit_distance:
                    pairs.append((a.gt_vehicle_id, b.gt_vehicle_id))
    return pairs


def sample_routine_pairs(
    dataset: GeneratedDataset, seed: int = 0, max_positive_pairs: int = 4000
) -> list[LabeledPair]:
    """S1: the everyday case. Same construction as eval.metrics.sample_
    labeled_pairs, but explicitly excludes any negative candidate that
    happens to share the anchor's true plate, so a clone/near-miss vehicle
    picked as a "hard negative" here can't quietly leak S2/S4's signal into
    the routine baseline."""
    events = dataset.events
    true_plate = {v.gt_vehicle_id: v.true_plate for v in dataset.vehicles}
    by_vehicle = _events_by_vehicle(events)
    by_camera = _events_by_camera(events)
    rng = random.Random(seed)

    positives: list[tuple[DetectionEvent, DetectionEvent]] = []
    for evs in by_vehicle.values():
        for a, b in zip(evs[:-1], evs[1:], strict=True):
            positives.append((a, b))
    rng.shuffle(positives)
    positives = positives[:max_positive_pairs]

    labeled: list[LabeledPair] = []
    for a, b in positives:
        labeled.append((a, b, 1))
        a_plate = true_plate.get(a.gt_vehicle_id)
        candidates = [
            e
            for e in by_camera[b.camera_id]
            if e.gt_vehicle_id != a.gt_vehicle_id and true_plate.get(e.gt_vehicle_id) != a_plate
        ]
        neg = _closest_in_time(candidates, b)
        if neg is not None:
            labeled.append((a, neg, 0))
    rng.shuffle(labeled)
    return labeled


def _sample_group_pairs(
    dataset: GeneratedDataset,
    groups: dict[str, list[str]] | list[tuple[str, str]],
    seed: int,
    max_pairs: int,
) -> list[LabeledPair]:
    """Shared machinery for S2 (clone) and S4 (plate-similar): positives are
    each involved vehicle's own consecutive passages; negatives are
    closest-in-time cross-vehicle event pairs drawn from the pooled events
    of vehicles that share a group (same plate, or a near-miss plate)."""
    by_vehicle = _events_by_vehicle(dataset.events)
    rng = random.Random(seed)

    if isinstance(groups, dict):
        member_lists = list(groups.values())
    else:
        member_lists = [[a, b] for a, b in groups]

    positives: list[tuple[DetectionEvent, DetectionEvent]] = []
    negatives: list[tuple[DetectionEvent, DetectionEvent]] = []
    seen_vehicle_ids: set[str] = set()

    for members in member_lists:
        for vid in members:
            if vid in seen_vehicle_ids:
                continue
            seen_vehicle_ids.add(vid)
            evs = by_vehicle.get(vid, [])
            for a, b in zip(evs[:-1], evs[1:], strict=True):
                positives.append((a, b))

        group_events = sorted(
            (e for vid in members for e in by_vehicle.get(vid, [])), key=lambda e: e.timestamp
        )
        for x, y in zip(group_events[:-1], group_events[1:], strict=True):
            if x.gt_vehicle_id != y.gt_vehicle_id:
                negatives.append((x, y))

    rng.shuffle(positives)
    rng.shuffle(negatives)
    n = min(len(positives), len(negatives), max_pairs)
    labeled = [(a, b, 1) for a, b in positives[:n]] + [(a, b, 0) for a, b in negatives[:n]]
    rng.shuffle(labeled)
    return labeled


def sample_clone_pairs(
    dataset: GeneratedDataset, seed: int = 0, max_pairs: int = 2000
) -> list[LabeledPair]:
    """S2: the case the project's thesis rests on. Negatives are two
    DIFFERENT vehicles sharing the exact same true plate -- the plate
    channel is structurally blind here."""
    groups = find_clone_groups(dataset.vehicles)
    return _sample_group_pairs(dataset, groups, seed, max_pairs)


def find_clone_pairs_by_overlap(
    vehicles: list[Vehicle],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split every (clone, source) relationship into disjoint-route vs
    overlapping-route groups (docs/decisions.md, "Day 3a"). Unlike
    `find_clone_groups` (which groups by plate value and can merge several
    clones sharing one plate into a single group), this returns one
    (clone_id, source_id) pair per clone, tagged by its own
    `Vehicle.route_overlap` flag -- the property that actually determines
    whether the pair is the easy (disjoint) or hard (overlapping) case."""
    disjoint: list[tuple[str, str]] = []
    overlap: list[tuple[str, str]] = []
    for v in vehicles:
        if not v.is_clone or v.clone_of is None:
            continue
        pair = (v.gt_vehicle_id, v.clone_of)
        (overlap if v.route_overlap else disjoint).append(pair)
    return disjoint, overlap


def sample_clone_pairs_by_overlap(
    dataset: GeneratedDataset, seed: int = 0, max_pairs: int = 2000
) -> dict[str, list[LabeledPair]]:
    """S2, split into the easy (disjoint-route) and hard (overlapping-route)
    cases (docs/decisions.md, "Day 3a" -- Part 0). A clone whose route never
    goes near its source's is separable by kinematics alone; a clone sharing
    its source's neighbourhood is the realistic hard case where only
    appearance can be expected to carry the stratum."""
    disjoint_pairs, overlap_pairs = find_clone_pairs_by_overlap(dataset.vehicles)
    return {
        "disjoint": _sample_group_pairs(dataset, disjoint_pairs, seed, max_pairs),
        "overlap": _sample_group_pairs(dataset, overlap_pairs, seed + 1, max_pairs),
    }


def sample_plate_similar_pairs(
    dataset: GeneratedDataset,
    seed: int = 0,
    max_pairs: int = 2000,
    max_edit_distance: int = MAX_PLATE_SIMILAR_EDIT_DISTANCE,
) -> list[LabeledPair]:
    """S4: negatives are two different vehicles whose true plates are a
    near-miss of each other (edit distance <= max_edit_distance)."""
    pairs = find_plate_similar_pairs(dataset.vehicles, max_edit_distance)
    return _sample_group_pairs(dataset, pairs, seed, max_pairs)


def sample_degraded_pairs(
    dataset: GeneratedDataset,
    seed: int = 0,
    max_pairs: int = 2000,
    min_degraded_slots: int = MIN_DEGRADED_SLOTS,
) -> list[LabeledPair]:
    """S3: positives are genuine same-vehicle consecutive pairs where the
    combined unread+misread slot count across both events is >= 4. Intended
    to be called on a deliberately higher-noise dataset (see
    docs/decisions.md, "Day 2b") -- the default corruption rate makes this
    stratum too sparse to measure reliably (n=68 observed against the
    default p_occlude=0.03)."""
    events = dataset.events
    true_plate = {v.gt_vehicle_id: v.true_plate for v in dataset.vehicles}
    by_vehicle = _events_by_vehicle(events)
    by_camera = _events_by_camera(events)
    rng = random.Random(seed)

    positives: list[tuple[DetectionEvent, DetectionEvent]] = []
    for evs in by_vehicle.values():
        for a, b in zip(evs[:-1], evs[1:], strict=True):
            tp = true_plate.get(a.gt_vehicle_id)
            if tp is None:
                continue
            if _degraded_slot_count(a, tp) + _degraded_slot_count(b, tp) >= min_degraded_slots:
                positives.append((a, b))
    rng.shuffle(positives)
    positives = positives[:max_pairs]

    labeled: list[LabeledPair] = []
    for a, b in positives:
        labeled.append((a, b, 1))
        candidates = [e for e in by_camera[b.camera_id] if e.gt_vehicle_id != a.gt_vehicle_id]
        neg = _closest_in_time(candidates, b)
        if neg is not None:
            labeled.append((a, neg, 0))
    rng.shuffle(labeled)
    return labeled


def count_degraded_pairs(
    dataset: GeneratedDataset, min_degraded_slots: int = MIN_DEGRADED_SLOTS
) -> int:
    """How many real same-vehicle consecutive pairs in `dataset` meet the
    S3 threshold -- used to measure S3's NATURAL frequency (under whatever
    corruption config `dataset` was generated with), separately from the
    boosted sample used to actually measure S3's AUC."""
    true_plate = {v.gt_vehicle_id: v.true_plate for v in dataset.vehicles}
    by_vehicle = _events_by_vehicle(dataset.events)
    count = 0
    for evs in by_vehicle.values():
        for a, b in zip(evs[:-1], evs[1:], strict=True):
            tp = true_plate.get(a.gt_vehicle_id)
            if tp is None:
                continue
            if _degraded_slot_count(a, tp) + _degraded_slot_count(b, tp) >= min_degraded_slots:
                count += 1
    return count


@dataclass
class StratumFrequencies:
    """Disjoint classification of every real same-vehicle consecutive pair
    in a dataset into exactly one stratum (clone takes precedence over
    plate-similar over degraded over routine), so the frequencies sum to 1
    and can weight a single composite figure honestly (docs/decisions.md,
    "Day 2b", B3)."""

    n_total_pairs: int
    n_clone: int
    n_plate_similar: int
    n_degraded: int
    n_routine: int

    @property
    def freq_clone(self) -> float:
        return self.n_clone / self.n_total_pairs if self.n_total_pairs else 0.0

    @property
    def freq_plate_similar(self) -> float:
        return self.n_plate_similar / self.n_total_pairs if self.n_total_pairs else 0.0

    @property
    def freq_degraded(self) -> float:
        return self.n_degraded / self.n_total_pairs if self.n_total_pairs else 0.0

    @property
    def freq_routine(self) -> float:
        return self.n_routine / self.n_total_pairs if self.n_total_pairs else 0.0


def compute_stratum_frequencies(
    dataset: GeneratedDataset, min_degraded_slots: int = MIN_DEGRADED_SLOTS
) -> StratumFrequencies:
    """Classify every real same-vehicle consecutive pair in `dataset` by
    which stratum it naturally falls into, under precedence
    clone > plate-similar > degraded > routine (a pair is "clone" if either
    vehicle is part of a same-plate group, "plate-similar" if either vehicle
    is part of a near-miss group, "degraded" if it crosses the S3 slot
    threshold, else "routine"). This is NOT the sample used to measure each
    stratum's AUC (S3 in particular is measured on a deliberately boosted
    dataset) -- it's purely for an honest frequency-weighted headline."""
    vehicles_in_clone_groups = {
        vid for ids in find_clone_groups(dataset.vehicles).values() for vid in ids
    }
    vehicles_in_similar_pairs = {
        vid for pair in find_plate_similar_pairs(dataset.vehicles) for vid in pair
    }
    true_plate = {v.gt_vehicle_id: v.true_plate for v in dataset.vehicles}
    by_vehicle = _events_by_vehicle(dataset.events)

    n_total = n_clone = n_similar = n_degraded = n_routine = 0
    for vid, evs in by_vehicle.items():
        tp = true_plate.get(vid)
        for a, b in zip(evs[:-1], evs[1:], strict=True):
            n_total += 1
            if vid in vehicles_in_clone_groups:
                n_clone += 1
            elif vid in vehicles_in_similar_pairs:
                n_similar += 1
            elif tp is not None and _degraded_slot_count(a, tp) + _degraded_slot_count(b, tp) >= (
                min_degraded_slots
            ):
                n_degraded += 1
            else:
                n_routine += 1

    return StratumFrequencies(
        n_total_pairs=n_total,
        n_clone=n_clone,
        n_plate_similar=n_similar,
        n_degraded=n_degraded,
        n_routine=n_routine,
    )
