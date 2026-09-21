"""Clone / impossible-travel detection (engine/decode/clone_detect.py,
architecture.md §3 "Clone detection falls out") -- unit tests plus the
headline precision/recall measurement split by route overlap (task spec,
item 3): overlapping clones (sim/generate.py's `clone_route_overlap`) are
the hard case and must be reported honestly, not hidden behind an aggregate
number."""

import random
from datetime import datetime, timedelta

from engine.contracts.events import DetectionEvent, VehicleAttributes
from engine.contracts.plate import PLATE_SLOTS, SlotPosterior
from engine.contracts.trajectory import Trajectory
from engine.decode.clone_detect import (
    appearance_divergence,
    detect_clones,
    find_crossings,
)
from engine.scoring.kinematic_lr import KinematicModel
from sim.city import generate_city
from sim.generate import generate_dataset_with_city

PLATE = "MH12AB1234"


def _embedding(seed: int, base: list[float] | None = None) -> list[float]:
    r = random.Random(seed)
    if base is None:
        v = [r.gauss(0, 1) for _ in range(128)]
    else:
        v = [b + r.gauss(0, 0.02) for b in base]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def _event(
    event_id: str, camera_id: str, t: datetime, plate: str, emb: list[float]
) -> DetectionEvent:
    posteriors = [SlotPosterior.peaked(PLATE_SLOTS[i], plate[i], 0.95) for i in range(10)]
    return DetectionEvent(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=t,
        plate_posterior=posteriors,
        plate_argmax=plate,
        plate_confidence=0.95,
        embedding=emb,
        attributes=VehicleAttributes(
            color="white", vehicle_type="car", color_confidence=0.9, type_confidence=0.9
        ),
        source="sim",
        gt_vehicle_id="veh_test",
    )


def _traj(traj_id: str, events: list[DetectionEvent]) -> Trajectory:
    return Trajectory(
        trajectory_id=traj_id,
        event_ids=[e.event_id for e in events],
        decoded_plate=events[0].plate_argmax,
        plate_confidence=0.95,
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
        camera_sequence=[e.camera_id for e in events],
    )


def _city_and_km(n_cameras: int = 8, seed: int = 1) -> tuple:
    city = generate_city(n_cameras=n_cameras, seed=seed)
    km = KinematicModel(city=city, pair_params={})
    return city, km


def test_impossible_travel_with_divergent_appearance_is_flagged_as_clone():
    city, km = _city_and_km()
    cams = [c.camera_id for c in city.cameras]
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    emb_a = _embedding(1)
    emb_b = _embedding(2)  # unrelated random unit vector -> very different mean
    e1 = _event("e1", cams[0], t0, PLATE, emb_a)
    e2 = _event("e2", cams[-1], t0 + timedelta(seconds=5), PLATE, emb_b)
    traj_a, traj_b = _traj("ta", [e1]), _traj("tb", [e2])
    events_by_id = {"e1": e1, "e2": e2}

    alerts = detect_clones([traj_a, traj_b], events_by_id, km, city)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.type == "clone"
    assert alert.severity == "high"
    assert alert.plate == PLATE
    assert set(alert.trajectory_ids) == {"ta", "tb"}
    assert alert.evidence["delta_t_s"] == 5.0
    assert alert.evidence["min_required_s"] > 5.0
    assert alert.evidence["appearance_distance"] > 0.5


def test_feasible_timing_same_plate_is_not_flagged():
    """Same plate, but the second sighting is well within physically
    plausible travel time -- this looks like one real vehicle's own
    (possibly fragmented) journey, not a clone, and must not alert."""
    city, km = _city_and_km()
    cams = [c.camera_id for c in city.cameras]
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    emb = _embedding(1)
    e1 = _event("e1", cams[0], t0, PLATE, emb)
    e2 = _event("e2", cams[0], t0 + timedelta(hours=1), PLATE, _embedding(1, emb))
    traj_a, traj_b = _traj("ta", [e1]), _traj("tb", [e2])
    events_by_id = {"e1": e1, "e2": e2}

    alerts = detect_clones([traj_a, traj_b], events_by_id, km, city)
    assert alerts == []


def test_impossible_travel_without_appearance_divergence_is_impossible_travel_not_clone():
    """Same impossible timing, but this time the two sightings' appearance
    is nearly identical (e.g. two very similar-looking vehicles, or a
    borderline embedding) -- flagged, but as the weaker/ambiguous
    `impossible_travel` type, not a confident `clone`."""
    city, km = _city_and_km()
    cams = [c.camera_id for c in city.cameras]
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    emb = _embedding(1)
    e1 = _event("e1", cams[0], t0, PLATE, emb)
    e2 = _event("e2", cams[-1], t0 + timedelta(seconds=5), PLATE, _embedding(1, emb))
    traj_a, traj_b = _traj("ta", [e1]), _traj("tb", [e2])
    events_by_id = {"e1": e1, "e2": e2}

    alerts = detect_clones([traj_a, traj_b], events_by_id, km, city)
    assert len(alerts) == 1
    assert alerts[0].type == "impossible_travel"
    assert alerts[0].severity == "medium"


def test_no_alert_for_distinct_plates():
    city, km = _city_and_km()
    cams = [c.camera_id for c in city.cameras]
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    e1 = _event("e1", cams[0], t0, "MH12AB1234", _embedding(1))
    e2 = _event("e2", cams[-1], t0 + timedelta(seconds=5), "KA05CD5678", _embedding(2))
    traj_a, traj_b = _traj("ta", [e1]), _traj("tb", [e2])
    events_by_id = {"e1": e1, "e2": e2}
    alerts = detect_clones([traj_a, traj_b], events_by_id, km, city, near_identical_max_distance=0)
    assert alerts == []


def test_find_crossings_covers_interleaved_observations():
    """A genuinely interleaved pair (clone_route_overlap's hard case): A, B,
    A, B in time -- every adjacent cross-trajectory pair must be examined,
    not just "last of A to first of B.\""""
    city, km = _city_and_km()
    cams = [c.camera_id for c in city.cameras]
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    events_a = [_event("a1", cams[0], t0, PLATE, _embedding(1)),
                _event("a2", cams[1], t0 + timedelta(minutes=10), PLATE, _embedding(1))]
    events_b = [_event("b1", cams[0], t0 + timedelta(minutes=5), PLATE, _embedding(2)),
                _event("b2", cams[1], t0 + timedelta(minutes=15), PLATE, _embedding(2))]
    crossings = find_crossings(events_a, events_b, km)
    # a1->b1, b1->a2, a2->b2: three adjacent cross-trajectory transitions.
    assert len(crossings) == 3


def test_appearance_divergence_is_near_zero_for_repeats_of_one_vehicle():
    emb = _embedding(1)
    events_a = [DetectionEvent.model_construct(embedding=_embedding(1, emb))]
    events_b = [DetectionEvent.model_construct(embedding=_embedding(1, emb))]
    d = appearance_divergence(events_a, events_b)
    assert d < 0.05


# ---------------------------------------------------------------------------
# Headline: precision/recall against clone ground truth, split by
# clone_route_overlap (task spec, item 3). Ground truth "is a clone pair" =
# two distinct ground-truth vehicles sharing the same true plate.
# ---------------------------------------------------------------------------


def _precision_recall(
    predicted_pairs: set[frozenset], true_pairs: set[frozenset]
) -> tuple[float, float]:
    if not predicted_pairs and not true_pairs:
        return float("nan"), float("nan")
    tp = len(predicted_pairs & true_pairs)
    precision = tp / len(predicted_pairs) if predicted_pairs else float("nan")
    recall = tp / len(true_pairs) if true_pairs else float("nan")
    return precision, recall


def test_headline_clone_precision_recall_split_by_route_overlap():
    from engine.calibration.fit_priors import fit_all

    city = generate_city(n_cameras=10, seed=4)
    ds = generate_dataset_with_city(
        city, n_vehicles=400, hours=3, seed=17, clone_fraction=0.06, clone_route_overlap=0.5
    )
    models = fit_all(ds)
    events_by_id = {e.event_id: e for e in ds.events}

    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in ds.events:
        by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    # Ground-truth trajectories (one per vehicle that actually produced any
    # events) -- isolates clone_detect's own precision/recall from
    # association quality, which tests/test_window.py already covers.
    trajectories = []
    for vid, evs in by_vehicle.items():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        trajectories.append(
            Trajectory(
                trajectory_id=vid,
                event_ids=[e.event_id for e in evs_sorted],
                decoded_plate=evs_sorted[0].plate_argmax,
                plate_confidence=evs_sorted[0].plate_confidence,
                start_time=evs_sorted[0].timestamp,
                end_time=evs_sorted[-1].timestamp,
                camera_sequence=[e.camera_id for e in evs_sorted],
            )
        )

    from engine.decode.consensus import decode_trajectory

    consensus_by_id = {}
    for traj in trajectories:
        evs = [events_by_id[eid] for eid in traj.event_ids]
        consensus_by_id[traj.trajectory_id] = decode_trajectory(evs, models.plate_priors)
    for traj in trajectories:
        traj.decoded_plate = consensus_by_id[traj.trajectory_id].decoded_plate

    alerts = detect_clones(trajectories, events_by_id, models.kinematic_model, city)
    predicted_pairs = {frozenset(a.trajectory_ids) for a in alerts}

    true_pairs_overlap: set[frozenset] = set()
    true_pairs_disjoint: set[frozenset] = set()
    for v in ds.vehicles:
        if v.is_clone and v.gt_vehicle_id in by_vehicle and v.clone_of in by_vehicle:
            pair = frozenset({v.gt_vehicle_id, v.clone_of})
            (true_pairs_overlap if v.route_overlap else true_pairs_disjoint).add(pair)

    true_pairs_all = true_pairs_overlap | true_pairs_disjoint
    precision_all, recall_all = _precision_recall(predicted_pairs, true_pairs_all)
    _, recall_overlap = _precision_recall(predicted_pairs, true_pairs_overlap)
    _, recall_disjoint = _precision_recall(predicted_pairs, true_pairs_disjoint)

    print(
        "\nHeadline: clone detection precision/recall (10 cams / 400 vehicles / 3h, "
        "seed 17, clone_fraction=0.06, clone_route_overlap=0.5):"
    )
    print(f"  overall:  n_true_pairs={len(true_pairs_all)} n_predicted={len(predicted_pairs)} "
          f"precision={precision_all:.4f} recall={recall_all:.4f}")
    print(f"  overlapping clones (hard case): n_true={len(true_pairs_overlap)} "
          f"recall={recall_overlap:.4f}")
    print(f"  disjoint clones (easy case):    n_true={len(true_pairs_disjoint)} "
          f"recall={recall_disjoint:.4f}")

    assert len(true_pairs_all) > 3, "too few realised clone pairs to measure P/R meaningfully"
    # No fabricated floor on the hard (overlapping) case -- report it
    # honestly. The easy (disjoint) case, where clone and source never go
    # near each other in time/space, should be recovered reliably.
    if true_pairs_disjoint:
        assert recall_disjoint > 0.7, (
            f"disjoint-clone recall {recall_disjoint:.4f} is surprisingly low for the easy case"
        )
