from sim.city import generate_city
from sim.corruption import CorruptionConfig, Corruptor
from sim.vehicles import generate_vehicles_and_journeys


def _make_events(n_cameras=15, n_vehicles=300, hours=24, seed=11, cfg=None):
    city = generate_city(n_cameras=n_cameras, seed=seed)
    vehicles, journeys = generate_vehicles_and_journeys(city, n_vehicles, hours, seed)
    vehicles_by_id = {v.gt_vehicle_id: v for v in vehicles}
    corruptor = Corruptor(city, seed, cfg or CorruptionConfig())
    events = []
    for j in journeys:
        v = vehicles_by_id[j.gt_vehicle_id]
        for p in j.passages:
            e = corruptor.corrupt_passage(p, v)
            if e is not None:
                events.append(e)
    return city, vehicles_by_id, events


def test_every_event_has_ten_slots_summing_to_one():
    _, _, events = _make_events()
    assert events
    for e in events:
        assert len(e.plate_posterior) == 10
        for sp in e.plate_posterior:
            assert abs(sum(sp.probs.values()) - 1.0) < 1e-6


def test_every_event_references_real_camera_and_vehicle():
    city, vehicles_by_id, events = _make_events()
    camera_ids = {c.camera_id for c in city.cameras}
    for e in events:
        assert e.camera_id in camera_ids
        assert e.gt_vehicle_id in vehicles_by_id


def test_occlusion_produces_uninformative_slots():
    cfg = CorruptionConfig(p_occlude=1.0, occlude_run_len_min=3, occlude_run_len_max=3)
    _, _, events = _make_events(n_vehicles=50, cfg=cfg)
    found_uninformative = False
    for e in events:
        uninformative_count = sum(1 for sp in e.plate_posterior if sp.is_uninformative())
        if uninformative_count >= 3:
            found_uninformative = True
    assert found_uninformative


def test_missed_detection_drops_events():
    cfg_no_miss = CorruptionConfig(p_miss=0.0)
    cfg_all_miss = CorruptionConfig(p_miss=1.0)
    _, _, events_no_miss = _make_events(n_vehicles=50, cfg=cfg_no_miss)
    _, _, events_all_miss = _make_events(n_vehicles=50, cfg=cfg_all_miss)
    assert len(events_no_miss) > 0
    assert len(events_all_miss) == 0


def test_embedding_is_l2_normalised():
    _, _, events = _make_events(n_vehicles=50)
    for e in events:
        norm = sum(x * x for x in e.embedding) ** 0.5
        assert abs(norm - 1.0) < 1e-3
