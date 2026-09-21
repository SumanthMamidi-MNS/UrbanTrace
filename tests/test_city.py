import networkx as nx

from sim.city import MAX_EDGE_LEN_M, MIN_EDGE_LEN_M, generate_city


def test_generate_city_deterministic():
    a = generate_city(n_cameras=20, seed=7)
    b = generate_city(n_cameras=20, seed=7)
    assert a.model_dump() == b.model_dump()


def test_generate_city_different_seed_differs():
    a = generate_city(n_cameras=20, seed=7)
    b = generate_city(n_cameras=20, seed=8)
    assert a.model_dump() != b.model_dump()


def test_camera_count_and_border_flag():
    city = generate_city(n_cameras=20, seed=42)
    assert len(city.cameras) == 20
    assert any(c.is_border for c in city.cameras)
    node_ids = {n.node_id for n in city.nodes}
    for cam in city.cameras:
        assert cam.node_id in node_ids


def test_edge_lengths_within_spec_bounds():
    city = generate_city(n_cameras=50, seed=42)
    for edge in city.edges:
        assert MIN_EDGE_LEN_M <= edge.length_m <= MAX_EDGE_LEN_M
        assert edge.speed_limit_kmh in (30.0, 50.0, 80.0)


def test_graph_is_strongly_connected():
    city = generate_city(n_cameras=50, seed=42)
    g = city.to_digraph()
    assert nx.is_strongly_connected(g)
