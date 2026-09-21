"""Synthetic city road graph + camera placement.

Topology: a grid of local streets, a handful of long arterial roads (upgraded
speed corridors plus a few long skip-chords), and a ring road around the
outer boundary. Deterministic given a seed.
"""

import math
import random

from engine.contracts.city import Camera, CityConfig, RoadEdge, RoadNode

# Roughly centred on Nagpur (geographic centre of India) — an arbitrary but
# plausible anchor for lat/lon generation.
CITY_CENTER_LAT = 21.1458
CITY_CENTER_LON = 79.0882

_METERS_PER_DEG_LAT = 111_320.0

# Edge length bounds from the spec.
MIN_EDGE_LEN_M = 200.0
MAX_EDGE_LEN_M = 3000.0

SPEED_LOCAL_SLOW = 30.0
SPEED_LOCAL_FAST = 50.0
SPEED_ARTERIAL = 80.0


def _meters_per_deg_lon(lat_deg: float) -> float:
    return _METERS_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def _offset_latlon(lat0: float, lon0: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    dlat = dy_m / _METERS_PER_DEG_LAT
    dlon = dx_m / _meters_per_deg_lon(lat0)
    return lat0 + dlat, lon0 + dlon


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _grid_size_for_cameras(n_cameras: int) -> tuple[int, int]:
    """Pick a grid large enough to hold n_cameras with headroom, min 4x4."""
    side = max(4, math.ceil(math.sqrt(n_cameras * 1.5)))
    return side, side


def generate_city(
    n_cameras: int,
    seed: int = 42,
    cell_size_m: float = 600.0,
) -> CityConfig:
    """Generate a deterministic synthetic city road graph and camera layout."""
    rng = random.Random(seed)
    grid_rows, grid_cols = _grid_size_for_cameras(n_cameras)

    nodes: list[RoadNode] = []
    node_index: dict[tuple[int, int], str] = {}
    coords: dict[str, tuple[float, float]] = {}

    for r in range(grid_rows):
        for c in range(grid_cols):
            node_id = f"n_{r}_{c}"
            jitter_x = rng.uniform(-cell_size_m * 0.08, cell_size_m * 0.08)
            jitter_y = rng.uniform(-cell_size_m * 0.08, cell_size_m * 0.08)
            dx = c * cell_size_m + jitter_x
            dy = r * cell_size_m + jitter_y
            lat, lon = _offset_latlon(CITY_CENTER_LAT, CITY_CENTER_LON, dx, dy)
            nodes.append(RoadNode(node_id=node_id, lat=lat, lon=lon))
            node_index[(r, c)] = node_id
            coords[node_id] = (lat, lon)

    def is_boundary(r: int, c: int) -> bool:
        return r == 0 or r == grid_rows - 1 or c == 0 or c == grid_cols - 1

    # Arterial lines: a few rows/cols get upgraded to arterial speed.
    arterial_rows = {0, grid_rows // 2} if grid_rows > 3 else {0}
    arterial_cols = {0, grid_cols // 2} if grid_cols > 3 else {0}

    edges: list[RoadEdge] = []
    seen_pairs: set[tuple[str, str]] = set()

    def add_edge(a: str, b: str, speed: float) -> None:
        if (a, b) in seen_pairs:
            return
        lat_a, lon_a = coords[a]
        lat_b, lon_b = coords[b]
        length_m = _haversine_m(lat_a, lon_a, lat_b, lon_b)
        length_m = min(max(length_m, MIN_EDGE_LEN_M), MAX_EDGE_LEN_M)
        edges.append(
            RoadEdge(from_node=a, to_node=b, length_m=round(length_m, 1), speed_limit_kmh=speed)
        )
        seen_pairs.add((a, b))

    def add_bidirectional(a: str, b: str, speed: float) -> None:
        add_edge(a, b, speed)
        add_edge(b, a, speed)

    # 1. Grid edges (4-neighbour local streets), speed depends on category.
    for r in range(grid_rows):
        for c in range(grid_cols):
            here = node_index[(r, c)]
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr >= grid_rows or nc >= grid_cols:
                    continue
                there = node_index[(nr, nc)]

                on_ring = is_boundary(r, c) and is_boundary(nr, nc)
                on_arterial = (dr == 0 and r in arterial_rows) or (dc == 0 and c in arterial_cols)

                if on_ring or on_arterial:
                    speed = SPEED_ARTERIAL
                elif (r + c) % 4 == 0:
                    speed = SPEED_LOCAL_SLOW
                else:
                    speed = SPEED_LOCAL_FAST

                add_bidirectional(here, there, speed)

    # 2. A few long arterial skip-chords (express shortcuts), up to ~3km.
    skip = 3
    for r in arterial_rows:
        for c in range(0, grid_cols - skip, skip):
            a = node_index[(r, c)]
            b = node_index[(r, min(c + skip, grid_cols - 1))]
            add_bidirectional(a, b, SPEED_ARTERIAL)
    for c in arterial_cols:
        for r in range(0, grid_rows - skip, skip):
            a = node_index[(r, c)]
            b = node_index[(min(r + skip, grid_rows - 1), c)]
            add_bidirectional(a, b, SPEED_ARTERIAL)

    # 3. Camera placement: deterministic sample of nodes.
    all_node_ids = [n.node_id for n in nodes]
    camera_node_ids = rng.sample(all_node_ids, k=min(n_cameras, len(all_node_ids)))

    rc_by_node = {v: k for k, v in node_index.items()}
    cameras: list[Camera] = []
    for i, node_id in enumerate(camera_node_ids):
        r, c = rc_by_node[node_id]
        lat, lon = coords[node_id]
        cameras.append(
            Camera(
                camera_id=f"cam_{i:04d}",
                name=f"Camera {i}",
                lat=lat,
                lon=lon,
                node_id=node_id,
                bearing_deg=rng.uniform(0, 360),
                is_border=is_boundary(r, c),
            )
        )

    return CityConfig(nodes=nodes, edges=edges, cameras=cameras)
