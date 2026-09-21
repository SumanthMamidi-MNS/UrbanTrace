"""Partial-plate search (engine/decode/partial_search.py, architecture.md §3
"Partial-plate search works") -- unit tests plus the headline recall@1/5/10
measurement for queries built from true plates with 2, 3 and 4 characters
masked (task spec, item 2)."""

import random
from datetime import datetime, timedelta

import pytest

from engine.contracts.events import DetectionEvent, VehicleAttributes
from engine.contracts.plate import PLATE_SLOTS, SlotPosterior
from engine.contracts.trajectory import Trajectory
from engine.decode.consensus import decode_trajectory
from engine.decode.partial_search import (
    WILDCARD,
    SearchFilters,
    match_probability,
    normalise_query,
    search,
)
from sim.city import generate_city
from sim.generate import generate_dataset_with_city

TRUE_PLATE = "MH12AB1234"


def _embedding(seed: int) -> list[float]:
    r = random.Random(seed)
    v = [r.gauss(0, 1) for _ in range(128)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def _event(event_id: str, plate_read: str, mass: float, t: datetime) -> DetectionEvent:
    posteriors = [SlotPosterior.peaked(PLATE_SLOTS[i], plate_read[i], mass) for i in range(10)]
    return DetectionEvent(
        event_id=event_id,
        camera_id="cam_0",
        timestamp=t,
        plate_posterior=posteriors,
        plate_argmax=plate_read,
        plate_confidence=mass,
        embedding=_embedding(hash(event_id) % 1000),
        attributes=VehicleAttributes(
            color="white", vehicle_type="car", color_confidence=0.9, type_confidence=0.9
        ),
        source="sim",
        gt_vehicle_id="veh_test",
    )


def _trajectory_and_consensus(plate: str, traj_id: str) -> tuple[Trajectory, object]:
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    events = [_event(f"{traj_id}_e{i}", plate, 0.9, t0 + timedelta(minutes=i)) for i in range(3)]
    traj = Trajectory(
        trajectory_id=traj_id,
        event_ids=[e.event_id for e in events],
        decoded_plate="",
        plate_confidence=0.0,
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
        camera_sequence=[e.camera_id for e in events],
    )
    return traj, decode_trajectory(events)


def test_normalise_query_requires_exactly_ten_characters():
    with pytest.raises(ValueError):
        normalise_query("MH12AB123")
    with pytest.raises(ValueError):
        normalise_query("MH12AB12345")


def test_normalise_query_rejects_invalid_character_for_slot():
    with pytest.raises(ValueError):
        normalise_query("1H12AB1234")  # digit in a state-letter slot (slot 0)


def test_normalise_query_accepts_wildcards_and_blanks():
    slots = normalise_query("MH12A_?23_")
    assert slots[0] == "M"
    assert slots[5] == "_"
    assert slots[6] == WILDCARD


def test_fully_specified_query_matches_only_the_right_plate():
    """A fully-specified query on a different plate is not literally
    impossible for the "wrong" trajectory -- posteriors always retain
    background mass on every character (docs/decisions.md, "Day 1c": no
    character is ever assigned exactly zero probability), so its match
    probability is vanishingly small, not exactly zero."""
    traj_true, cons_true = _trajectory_and_consensus(TRUE_PLATE, "t_true")
    traj_other, cons_other = _trajectory_and_consensus("KA05CD5678", "t_other")
    hits = search(TRUE_PLATE, [(traj_true, cons_true), (traj_other, cons_other)])
    assert hits[0].trajectory.trajectory_id == "t_true"
    assert hits[0].matched_plate == TRUE_PLATE
    assert hits[0].probability > 0.99
    other_hit = next(h for h in hits if h.trajectory.trajectory_id == "t_other")
    assert other_hit.probability < 1e-30


def test_wildcard_query_ranks_true_plate_first():
    traj_true, cons_true = _trajectory_and_consensus(TRUE_PLATE, "t_true")
    traj_other, cons_other = _trajectory_and_consensus("KA05CD5678", "t_other")
    query = "MH12??1234"
    hits = search(query, [(traj_true, cons_true), (traj_other, cons_other)])
    assert hits[0].trajectory.trajectory_id == "t_true"
    assert hits[0].probability > 0.5


def test_match_probability_of_all_wildcards_is_one():
    _, cons = _trajectory_and_consensus(TRUE_PLATE, "t")
    prob = match_probability(cons, [WILDCARD] * 10)
    assert abs(prob - 1.0) < 1e-9


def test_time_filter_excludes_out_of_range_trajectories():
    traj, cons = _trajectory_and_consensus(TRUE_PLATE, "t")
    filters = SearchFilters(time_from=traj.end_time + timedelta(hours=1))
    hits = search(TRUE_PLATE, [(traj, cons)], filters)
    assert hits == []


# ---------------------------------------------------------------------------
# Headline: recall@1/5/10 for queries built from true plates with 2, 3, 4
# characters masked (task spec, item 2). Small dataset, single-event
# "trajectories" (one per vehicle's first event) so consensus reduces to
# that event's own posterior -- search ranking quality is what's under test
# here, not consensus fusion (already covered by test_consensus.py).
# ---------------------------------------------------------------------------


def _build_search_corpus(seed: int, n_vehicles: int = 300):
    city = generate_city(n_cameras=6, seed=1)
    ds = generate_dataset_with_city(
        city, n_vehicles=n_vehicles, hours=2, seed=seed, clone_fraction=0.0
    )
    by_vehicle: dict[str, list] = {}
    for e in ds.events:
        by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    trajs_cons = []
    true_plate_by_traj: dict[str, str] = {}
    for vid, evs in by_vehicle.items():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        traj = Trajectory(
            trajectory_id=vid,
            event_ids=[e.event_id for e in evs_sorted],
            decoded_plate="",
            plate_confidence=0.0,
            start_time=evs_sorted[0].timestamp,
            end_time=evs_sorted[-1].timestamp,
            camera_sequence=[e.camera_id for e in evs_sorted],
        )
        cons = decode_trajectory(evs_sorted)
        trajs_cons.append((traj, cons))
        true_plate_by_traj[vid] = next(
            v.true_plate for v in ds.vehicles if v.gt_vehicle_id == vid
        )
    return trajs_cons, true_plate_by_traj


def _mask_query(true_plate: str, n_mask: int, rng: random.Random) -> str:
    chars = list(true_plate)
    idxs = rng.sample(range(10), n_mask)
    for i in idxs:
        chars[i] = WILDCARD
    return "".join(chars)


def _recall_at_k(trajs_cons, true_plate_by_traj, n_mask: int, ks: tuple[int, ...], seed: int):
    rng = random.Random(seed)
    hits_at_k = dict.fromkeys(ks, 0)
    n_queries = 0
    for traj_id, true_plate in true_plate_by_traj.items():
        query = _mask_query(true_plate, n_mask, rng)
        results = search(query, trajs_cons)
        n_queries += 1
        ranked_ids = [h.trajectory.trajectory_id for h in results]
        for k in ks:
            if traj_id in ranked_ids[:k]:
                hits_at_k[k] += 1
    return {k: hits_at_k[k] / n_queries for k in ks}, n_queries


def test_headline_recall_at_k_for_masked_plate_queries():
    trajs_cons, true_plate_by_traj = _build_search_corpus(seed=21)
    ks = (1, 5, 10)

    print(
        f"\nHeadline: partial-plate search recall@{ks} (6 cams / "
        f"{len(true_plate_by_traj)} vehicles / 2h, seed 21), by number of masked chars:"
    )
    recalls_by_mask = {}
    for n_mask in (2, 3, 4):
        recalls, n_queries = _recall_at_k(trajs_cons, true_plate_by_traj, n_mask, ks, seed=7)
        recalls_by_mask[n_mask] = recalls
        row = " ".join(f"recall@{k}={recalls[k]:.4f}" for k in ks)
        print(f"  masked={n_mask} (n_queries={n_queries}): {row}")

    # Monotonicity within a query: recall@10 >= recall@5 >= recall@1 always
    # holds by construction (top-k is nested); the real claim under test is
    # that recall degrades gracefully, not collapsing, as more characters
    # are masked -- and that recall@10 stays high even at 4 masked chars.
    for recalls in recalls_by_mask.values():
        assert recalls[1] <= recalls[5] <= recalls[10]

    assert recalls_by_mask[2][1] > 0.9, "recall@1 with only 2 masked chars should be near-total"
    assert recalls_by_mask[4][10] > 0.5, (
        f"recall@10 with 4 masked chars was {recalls_by_mask[4][10]:.4f}, expected the true "
        "plate to still usually surface in the top 10"
    )
