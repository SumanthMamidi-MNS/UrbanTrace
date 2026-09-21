import math
import random

import networkx as nx

from sim.city import generate_city
from sim.vehicles import generate_vehicles_and_journeys


def test_plate_format():
    city = generate_city(n_cameras=10, seed=1)
    vehicles, _ = generate_vehicles_and_journeys(city, n_vehicles=200, hours=24, seed=1)
    for v in vehicles:
        assert len(v.true_plate) == 10
        assert v.true_plate[0:2].isalpha()
        assert v.true_plate[2:4].isdigit()


def test_embedding_normalised():
    city = generate_city(n_cameras=10, seed=1)
    vehicles, _ = generate_vehicles_and_journeys(city, n_vehicles=50, hours=24, seed=1)
    for v in vehicles:
        assert len(v.embedding) == 128
        norm = math.sqrt(sum(x * x for x in v.embedding))
        assert math.isclose(norm, 1.0, abs_tol=1e-6)


def test_same_class_embeddings_closer_than_random():
    """Vehicles sharing (type, colour) should on average have higher cosine
    similarity than randomly paired vehicles."""
    city = generate_city(n_cameras=10, seed=3)
    vehicles, _ = generate_vehicles_and_journeys(city, n_vehicles=400, hours=24, seed=3)

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    by_class: dict[tuple[str, str], list] = {}
    for v in vehicles:
        by_class.setdefault((v.vehicle_type, v.color), []).append(v)

    same_class_sims = []
    for members in by_class.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                same_class_sims.append(cos(members[i].embedding, members[j].embedding))

    rng = random.Random(0)
    random_sims = []
    for _ in range(len(same_class_sims) or 1):
        a, b = rng.sample(vehicles, 2)
        random_sims.append(cos(a.embedding, b.embedding))

    assert same_class_sims, "expected at least one same-class pair"
    assert sum(same_class_sims) / len(same_class_sims) > sum(random_sims) / len(random_sims)


def test_ground_truth_camera_sequence_is_valid_walk():
    city = generate_city(n_cameras=15, seed=5)
    graph = city.to_digraph()
    node_by_camera = {c.camera_id: c.node_id for c in city.cameras}
    vehicles, journeys = generate_vehicles_and_journeys(city, n_vehicles=100, hours=24, seed=5)

    for journey in journeys:
        seq = journey.camera_sequence
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            node_a, node_b = node_by_camera[a], node_by_camera[b]
            if node_a == node_b:
                continue
            assert nx.has_path(graph, node_a, node_b)


def test_vehicle_timestamps_strictly_increasing():
    city = generate_city(n_cameras=15, seed=5)
    _, journeys = generate_vehicles_and_journeys(city, n_vehicles=100, hours=24, seed=5)
    for journey in journeys:
        times = [p.timestamp for p in journey.passages]
        assert times == sorted(times)
        assert len(set(times)) == len(times)


def test_clone_injection():
    city = generate_city(n_cameras=15, seed=9)
    vehicles, _ = generate_vehicles_and_journeys(
        city, n_vehicles=500, hours=24, seed=9, clone_fraction=0.1
    )
    clones = [v for v in vehicles if v.is_clone]
    assert len(clones) > 0
    for clone in clones:
        original = next(v for v in vehicles if v.gt_vehicle_id == clone.clone_of)
        assert original.true_plate == clone.true_plate
        assert original.gt_vehicle_id != clone.gt_vehicle_id


def test_near_miss_plate_injection():
    """docs/decisions.md, "Day 2b": a near-miss vehicle keeps its own
    identity (route, appearance) but gets a plate that's a close edit of
    another vehicle's -- same state/RTO/series, a nudged number -- never an
    exact duplicate (that's what clone injection is for)."""
    from eval.stratified import edit_distance

    city = generate_city(n_cameras=15, seed=9)
    vehicles, _ = generate_vehicles_and_journeys(
        city, n_vehicles=500, hours=24, seed=9, clone_fraction=0.0, near_miss_fraction=0.1
    )
    near_misses = [v for v in vehicles if v.is_near_miss]
    assert len(near_misses) > 0
    for nm in near_misses:
        source = next(v for v in vehicles if v.gt_vehicle_id == nm.near_miss_of)
        assert source.true_plate[:6] == nm.true_plate[:6]  # same state/RTO/series
        assert source.true_plate != nm.true_plate  # never an exact duplicate
        assert edit_distance(source.true_plate, nm.true_plate) <= 2
        assert source.gt_vehicle_id != nm.gt_vehicle_id
