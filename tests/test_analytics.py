"""Analytics package tests (engine/analytics/*, architecture.md §4):
OD matrix zone assignment, per-camera volumes, corridor travel times, and
loop anomalies."""

from datetime import datetime, timedelta

from engine.analytics.anomalies import MIN_REVISITS, detect_loop_anomalies, find_loops
from engine.analytics.corridors import compute_corridors, free_flow_time_s
from engine.analytics.od_matrix import ZONES, assign_zones, build_od_matrix
from engine.analytics.volumes import compute_volumes
from engine.contracts.city import Camera, CityConfig, RoadEdge, RoadNode
from engine.contracts.events import DetectionEvent, VehicleAttributes
from engine.contracts.plate import PLATE_SLOTS, SlotPosterior
from engine.contracts.trajectory import Trajectory
from sim.city import generate_city
from sim.generate import generate_dataset_with_city

PLATE = "MH12AB1234"


def _plain_embedding() -> list[float]:
    v = [1.0] + [0.0] * 127
    return v


def _event(event_id: str, camera_id: str, t: datetime) -> DetectionEvent:
    posteriors = [SlotPosterior.peaked(PLATE_SLOTS[i], PLATE[i], 0.9) for i in range(10)]
    return DetectionEvent(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=t,
        plate_posterior=posteriors,
        plate_argmax=PLATE,
        plate_confidence=0.9,
        embedding=_plain_embedding(),
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
        decoded_plate=PLATE,
        plate_confidence=0.9,
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
        camera_sequence=[e.camera_id for e in events],
    )


def _cross_city() -> CityConfig:
    """A small hand-built city: one camera at each compass direction plus
    one at dead centre, so zone assignment is unambiguous."""
    nodes = [RoadNode(node_id=f"n{i}", lat=lat, lon=lon) for i, (lat, lon) in enumerate(
        [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.0, 0.0)]
    )]
    edges = [
        RoadEdge(from_node=a.node_id, to_node=b.node_id, length_m=1000.0, speed_limit_kmh=50.0)
        for a in nodes
        for b in nodes
        if a is not b
    ]
    cameras = [
        Camera(camera_id="cam_N", name="N", lat=1.0, lon=0.0, node_id="n0",
               bearing_deg=0.0, is_border=True),
        Camera(camera_id="cam_S", name="S", lat=-1.0, lon=0.0, node_id="n1",
               bearing_deg=0.0, is_border=True),
        Camera(camera_id="cam_E", name="E", lat=0.0, lon=1.0, node_id="n2",
               bearing_deg=0.0, is_border=True),
        Camera(camera_id="cam_W", name="W", lat=0.0, lon=-1.0, node_id="n3",
               bearing_deg=0.0, is_border=True),
        Camera(camera_id="cam_C", name="C", lat=0.0, lon=0.0, node_id="n4",
               bearing_deg=0.0, is_border=False),
    ]
    return CityConfig(nodes=nodes, edges=edges, cameras=cameras)


# ---------------------------------------------------------------------------
# od_matrix
# ---------------------------------------------------------------------------


def test_zone_assignment_matches_compass_layout():
    city = _cross_city()
    assignment = assign_zones(city)
    assert assignment.zone_by_camera["cam_N"] == "N"
    assert assignment.zone_by_camera["cam_S"] == "S"
    assert assignment.zone_by_camera["cam_E"] == "E"
    assert assignment.zone_by_camera["cam_W"] == "W"
    assert assignment.zone_by_camera["cam_C"] == "Central"


def test_od_matrix_counts_origin_destination_pairs():
    city = _cross_city()
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    e1 = _event("e1", "cam_N", t0)
    e2 = _event("e2", "cam_S", t0 + timedelta(minutes=5))
    traj = _traj("t1", [e1, e2])

    zones, matrix = build_od_matrix([traj], city)
    assert zones == ZONES
    n_idx, s_idx = zones.index("N"), zones.index("S")
    assert matrix[n_idx][s_idx] == 1
    assert sum(sum(row) for row in matrix) == 1


def test_od_matrix_ignores_trajectories_with_no_cameras():
    city = _cross_city()
    traj = Trajectory(
        trajectory_id="empty",
        event_ids=[],
        decoded_plate=PLATE,
        plate_confidence=0.0,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        camera_sequence=[],
    )
    zones, matrix = build_od_matrix([traj], city)
    assert sum(sum(row) for row in matrix) == 0


# ---------------------------------------------------------------------------
# volumes
# ---------------------------------------------------------------------------


def test_compute_volumes_buckets_by_camera_and_time():
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    events = [
        _event("e1", "cam_A", t0),
        _event("e2", "cam_A", t0 + timedelta(minutes=10)),
        _event("e3", "cam_A", t0 + timedelta(hours=2)),
        _event("e4", "cam_B", t0),
    ]
    rows = compute_volumes(events, bucket_minutes=60)
    by_key = {(r.camera_id, r.bucket_start): r.count for r in rows}
    assert by_key[("cam_A", t0)] == 2
    assert by_key[("cam_A", t0 + timedelta(hours=2))] == 1
    assert by_key[("cam_B", t0)] == 1


def test_compute_volumes_filters_by_camera():
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    events = [_event("e1", "cam_A", t0), _event("e2", "cam_B", t0)]
    rows = compute_volumes(events, camera_id="cam_A")
    assert len(rows) == 1
    assert rows[0].camera_id == "cam_A"


def test_compute_volumes_empty_input():
    assert compute_volumes([]) == []


# ---------------------------------------------------------------------------
# corridors
# ---------------------------------------------------------------------------


def test_free_flow_time_matches_distance_over_speed_limit():
    city = _cross_city()
    # cam_N (n0) -> cam_E (n2): direct edge, 1000m at 50 km/h.
    t = free_flow_time_s(city, "cam_N", "cam_E")
    expected = 1000.0 / (50.0 * 1000.0 / 3600.0)
    assert t is not None
    assert abs(t - expected) < 1e-6


def test_compute_corridors_median_p90_and_congestion_index():
    city = _cross_city()
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    free_flow = free_flow_time_s(city, "cam_N", "cam_E")

    # Three trips N->E with travel times 2x, 2x, 4x free-flow (congested).
    trajectories = []
    events_by_id = {}
    for i, factor in enumerate([2.0, 2.0, 4.0]):
        e1 = _event(f"a{i}", "cam_N", t0 + timedelta(hours=i))
        e2 = _event(f"b{i}", "cam_E", t0 + timedelta(hours=i, seconds=free_flow * factor))
        events_by_id[e1.event_id] = e1
        events_by_id[e2.event_id] = e2
        trajectories.append(_traj(f"t{i}", [e1, e2]))

    rows = compute_corridors(trajectories, events_by_id, city)
    row = next(r for r in rows if r.from_camera == "cam_N" and r.to_camera == "cam_E")
    assert row.n_trips == 3
    assert abs(row.median_travel_s - free_flow * 2.0) < 1e-6
    assert row.free_flow_s is not None
    assert abs(row.congestion_index - (row.median_travel_s / row.free_flow_s)) < 1e-9
    assert row.congestion_index > 1.0  # congested, as constructed


def test_compute_corridors_respects_limit_and_sorts_by_n_trips():
    city = generate_city(n_cameras=8, seed=2)
    ds = generate_dataset_with_city(city, n_vehicles=150, hours=2, seed=9, clone_fraction=0.0)
    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in ds.events:
        by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)
    trajectories = []
    for vid, evs in by_vehicle.items():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        trajectories.append(_traj(vid, evs_sorted))
    events_by_id = {e.event_id: e for e in ds.events}

    rows = compute_corridors(trajectories, events_by_id, city, limit=3)
    assert len(rows) <= 3
    n_trips = [r.n_trips for r in rows]
    assert n_trips == sorted(n_trips, reverse=True)


# ---------------------------------------------------------------------------
# anomalies
# ---------------------------------------------------------------------------


def test_find_loops_flags_three_revisits_within_an_hour():
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    events = [
        _event("e1", "cam_A", t0),
        _event("e2", "cam_B", t0 + timedelta(minutes=10)),
        _event("e3", "cam_A", t0 + timedelta(minutes=20)),
        _event("e4", "cam_B", t0 + timedelta(minutes=30)),
        _event("e5", "cam_A", t0 + timedelta(minutes=40)),
    ]
    events_by_id = {e.event_id: e for e in events}
    traj = _traj("t1", events)
    loops = find_loops(traj, events_by_id)
    assert len(loops) == 1
    assert loops[0].camera_id == "cam_A"
    assert len(loops[0].timestamps) == MIN_REVISITS


def test_find_loops_does_not_flag_two_revisits():
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    events = [
        _event("e1", "cam_A", t0),
        _event("e2", "cam_B", t0 + timedelta(minutes=10)),
        _event("e3", "cam_A", t0 + timedelta(minutes=20)),
    ]
    events_by_id = {e.event_id: e for e in events}
    traj = _traj("t1", events)
    assert find_loops(traj, events_by_id) == []


def test_find_loops_does_not_flag_revisits_spread_across_more_than_an_hour():
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    events = [
        _event("e1", "cam_A", t0),
        _event("e2", "cam_A", t0 + timedelta(minutes=40)),
        _event("e3", "cam_A", t0 + timedelta(hours=2)),
    ]
    events_by_id = {e.event_id: e for e in events}
    traj = _traj("t1", events)
    assert find_loops(traj, events_by_id) == []


def test_detect_loop_anomalies_emits_one_alert_per_looping_trajectory():
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    looping_events = [
        _event("e1", "cam_A", t0),
        _event("e2", "cam_A", t0 + timedelta(minutes=10)),
        _event("e3", "cam_A", t0 + timedelta(minutes=20)),
    ]
    normal_events = [_event("e4", "cam_A", t0), _event("e5", "cam_B", t0 + timedelta(minutes=5))]
    events_by_id = {e.event_id: e for e in looping_events + normal_events}
    trajectories = [_traj("loop", looping_events), _traj("normal", normal_events)]

    alerts = detect_loop_anomalies(trajectories, events_by_id)
    assert len(alerts) == 1
    assert alerts[0].type == "anomaly"
    assert alerts[0].trajectory_ids == ["loop"]
