"""Live heatmap data (PRD component 3: "real-time heatmaps of city traffic
movement"; docs/api-contract.md Contract v2, `GET /api/analytics/heatmap`).

`density` = reads per camera in the window, normalised to [0, 1] by the
window's own busiest camera. `speed` = mean speed (km/h) of trajectory links
ARRIVING at that camera within the window -- reusing the same road-graph
shortest-path-distance / v_max-exclusion approach
`engine.analytics.corridors` uses for its own speed fields, just keeping
each individual arrival (with its own timestamp) instead of aggregating by
camera pair, since the heatmap needs to filter by an arbitrary time window
picked at request time rather than the whole dataset."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import networkx as nx

from engine.contracts.city import Camera, CityConfig
from engine.contracts.trajectory import Trajectory
from engine.scoring.kinematic_lr import DEFAULT_V_MAX_KMH


@dataclass(frozen=True)
class LinkArrival:
    to_camera: str
    arrival_ts: datetime
    speed_kmh: float


@dataclass(frozen=True)
class HeatPointRow:
    camera_id: str
    lat: float
    lon: float
    weight: float


def _distance_m(
    graph: nx.DiGraph, node_by_camera: dict[str, str], from_camera: str, to_camera: str
) -> float | None:
    n_from, n_to = node_by_camera.get(from_camera), node_by_camera.get(to_camera)
    if n_from is None or n_to is None:
        return None
    try:
        return float(nx.shortest_path_length(graph, n_from, n_to, weight="length_m"))
    except nx.NetworkXNoPath:
        return None


def compute_link_arrivals(
    trajectories: list[Trajectory],
    events_by_id: dict[str, object],  # objects with .camera_id / .timestamp
    city: CityConfig,
) -> list[LinkArrival]:
    """One row per adjacent-camera hop actually observed within a
    trajectory, each keeping its own arrival timestamp (unlike
    `engine.analytics.corridors.compute_corridors`, which aggregates by
    camera pair) so callers can filter to an arbitrary time window. Applies
    the same v_max exclusion the contract calls for: "links implying more
    than the kinematic v_max ... are clone evidence, not speed"."""
    graph = city.to_digraph()
    node_by_camera = {c.camera_id: c.node_id for c in city.cameras}
    distance_cache: dict[tuple[str, str], float | None] = {}
    out: list[LinkArrival] = []
    for traj in trajectories:
        events = [events_by_id[eid] for eid in traj.event_ids]
        for e_from, e_to in zip(events[:-1], events[1:], strict=True):
            if e_from.camera_id == e_to.camera_id:
                continue
            dt = (e_to.timestamp - e_from.timestamp).total_seconds()
            if dt <= 0:
                continue
            key = (e_from.camera_id, e_to.camera_id)
            if key not in distance_cache:
                distance_cache[key] = _distance_m(graph, node_by_camera, *key)
            distance = distance_cache[key]
            if distance is None or distance <= 0:
                continue
            speed_kmh = (distance / dt) * 3.6
            if speed_kmh > DEFAULT_V_MAX_KMH:
                continue
            out.append(
                LinkArrival(
                    to_camera=e_to.camera_id, arrival_ts=e_to.timestamp, speed_kmh=speed_kmh
                )
            )
    return out


def density_heatmap(
    cameras: list[Camera], events: list, window_start: datetime, window_end: datetime
) -> list[HeatPointRow]:
    """`events`: objects with `.camera_id` / `.timestamp`. Weight is each
    camera's read count in `[window_start, window_end]` normalised by the
    window's own busiest camera (0.0 if the window has no reads at all)."""
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        if window_start <= e.timestamp <= window_end:
            counts[e.camera_id] += 1
    max_count = max(counts.values()) if counts else 0
    return [
        HeatPointRow(
            camera_id=c.camera_id,
            lat=c.lat,
            lon=c.lon,
            weight=(counts.get(c.camera_id, 0) / max_count) if max_count > 0 else 0.0,
        )
        for c in cameras
    ]


def speed_heatmap(
    cameras: list[Camera],
    link_arrivals: list[LinkArrival],
    window_start: datetime,
    window_end: datetime,
) -> list[HeatPointRow]:
    """Weight = mean km/h of links arriving at that camera in the window
    (0.0 if no arrivals -- there is no "unknown" sentinel in `HeatPoint`)."""
    by_cam: dict[str, list[float]] = defaultdict(list)
    for link in link_arrivals:
        if window_start <= link.arrival_ts <= window_end:
            by_cam[link.to_camera].append(link.speed_kmh)
    return [
        HeatPointRow(
            camera_id=c.camera_id,
            lat=c.lat,
            lon=c.lon,
            weight=statistics.mean(by_cam[c.camera_id]) if by_cam.get(c.camera_id) else 0.0,
        )
        for c in cameras
    ]


__all__ = [
    "HeatPointRow",
    "LinkArrival",
    "compute_link_arrivals",
    "density_heatmap",
    "speed_heatmap",
]
