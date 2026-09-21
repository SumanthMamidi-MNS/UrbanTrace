import networkx as nx

from sim.generate import generate_dataset


def test_ground_truth_round_trips():
    ds = generate_dataset(n_cameras=15, n_vehicles=300, hours=12, seed=21)

    camera_ids = {c.camera_id for c in ds.city.cameras}
    vehicle_ids = {v.gt_vehicle_id for v in ds.vehicles}
    graph = ds.city.to_digraph()
    node_by_camera = {c.camera_id: c.node_id for c in ds.city.cameras}

    assert ds.events, "expected at least one event"

    for event in ds.events:
        assert event.camera_id in camera_ids
        assert event.gt_vehicle_id in vehicle_ids

    # per-vehicle timestamps strictly increasing, camera sequence is a valid
    # walk in the road graph
    events_by_vehicle: dict[str, list] = {}
    for event in ds.events:
        events_by_vehicle.setdefault(event.gt_vehicle_id, []).append(event)

    for events in events_by_vehicle.values():
        events.sort(key=lambda e: e.timestamp)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)

        cam_sequence = [e.camera_id for e in events]
        for a, b in zip(cam_sequence[:-1], cam_sequence[1:], strict=True):
            node_a, node_b = node_by_camera[a], node_by_camera[b]
            if node_a == node_b:
                continue
            assert nx.has_path(graph, node_a, node_b)
