"""Corridor travel times (architecture.md §4; docs/api-contract.md
`/api/analytics/corridors`).

For every pair of consecutive cameras that actually occurs back-to-back
within a trajectory (i.e. `camera_sequence[k] -> camera_sequence[k+1]` for
some trajectory), collect the observed travel times (the two adjacent
events' timestamp difference) and compare against the road-graph free-flow
time (shortest path at the posted speed limit, independent of the fitted
kinematic model so this module doesn't depend on the concurrently-changing
`engine.scoring`/`engine.association` packages)."""

import statistics
from dataclasses import dataclass

import networkx as nx

from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import Trajectory

DEFAULT_LIMIT = 20


@dataclass
class CorridorStats:
    from_camera: str
    to_camera: str
    n_trips: int
    median_travel_s: float
    p90_travel_s: float
    free_flow_s: float | None
    congestion_index: float | None


def _percentile(sorted_values: list[float], pct: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _free_flow_time_s(
    graph: nx.DiGraph, node_by_camera: dict[str, str], from_camera: str, to_camera: str
) -> float | None:
    n_from, n_to = node_by_camera.get(from_camera), node_by_camera.get(to_camera)
    if n_from is None or n_to is None:
        return None
    try:
        return float(nx.shortest_path_length(graph, n_from, n_to, weight="weight"))
    except nx.NetworkXNoPath:
        return None


def free_flow_time_s(city: CityConfig, from_camera: str, to_camera: str) -> float | None:
    """Shortest-path travel time in seconds at the posted speed limit, or
    `None` if `to_camera` is unreachable from `from_camera` in the road
    graph. Public single-pair convenience wrapper; `compute_corridors`
    below builds the graph once and reuses it across every corridor
    instead of calling this per-pair."""
    graph = city.to_digraph()
    node_by_camera = {c.camera_id: c.node_id for c in city.cameras}
    return _free_flow_time_s(graph, node_by_camera, from_camera, to_camera)


def compute_corridors(
    trajectories: list[Trajectory],
    events_by_id: dict[str, DetectionEvent],
    city: CityConfig,
    limit: int = DEFAULT_LIMIT,
) -> list[CorridorStats]:
    """One row per observed (from_camera, to_camera) corridor, sorted by
    `n_trips` descending and truncated to `limit` -- matching
    `/api/analytics/corridors`'s `limit? (default 20)` query param."""
    graph = city.to_digraph()
    node_by_camera = {c.camera_id: c.node_id for c in city.cameras}
    travel_times: dict[tuple[str, str], list[float]] = {}

    for traj in trajectories:
        events = [events_by_id[eid] for eid in traj.event_ids]
        for e_from, e_to in zip(events[:-1], events[1:], strict=True):
            if e_from.camera_id == e_to.camera_id:
                continue
            dt = (e_to.timestamp - e_from.timestamp).total_seconds()
            if dt <= 0:
                continue
            key = (e_from.camera_id, e_to.camera_id)
            travel_times.setdefault(key, []).append(dt)

    rows: list[CorridorStats] = []
    for (from_cam, to_cam), times in travel_times.items():
        times_sorted = sorted(times)
        median = statistics.median(times_sorted)
        p90 = _percentile(times_sorted, 0.90)
        free_flow = _free_flow_time_s(graph, node_by_camera, from_cam, to_cam)
        congestion = median / free_flow if free_flow and free_flow > 0 else None
        rows.append(
            CorridorStats(
                from_camera=from_cam,
                to_camera=to_cam,
                n_trips=len(times_sorted),
                median_travel_s=median,
                p90_travel_s=p90,
                free_flow_s=free_flow,
                congestion_index=congestion,
            )
        )

    rows.sort(key=lambda r: r.n_trips, reverse=True)
    return rows[:limit]


__all__ = ["DEFAULT_LIMIT", "CorridorStats", "compute_corridors", "free_flow_time_s"]
