"""Analytics endpoints (docs/api-contract.md `/api/analytics/*`). Results
are precomputed once at startup (`api/state.py`) since "the data is static
per dataset"; volumes/heatmap/flow_trend bucket or window on the fly over
the cached lite-event / link-arrival lists (their query params vary per
request)."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from fastapi import APIRouter, Depends

from api import schemas
from api.deps import get_replay_engine, get_state
from api.replay import ReplayEngine
from api.state import AppState
from api.util import iso, parse_iso
from engine.analytics.flow_trend import compute_flow_trend
from engine.analytics.heatmap import density_heatmap, speed_heatmap
from engine.analytics.volumes import compute_volumes

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary", response_model=schemas.AnalyticsSummary)
def summary(state: AppState = Depends(get_state)) -> schemas.AnalyticsSummary:  # noqa: B008
    return schemas.AnalyticsSummary(
        n_events=state.n_events,
        n_trajectories=state.n_trajectories,
        mean_trip_duration_s=state.mean_trip_duration_s,
        mean_events_per_trajectory=state.mean_events_per_trajectory,
        plate_repair_rate=state.plate_repair_rate,
        active_alerts=state.active_alerts,
    )


@router.get("/volumes", response_model=list[schemas.VolumeBucket])
def volumes(
    bucket_minutes: int = 60,
    camera_id: str | None = None,
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[schemas.VolumeBucket]:
    events = state.lite_events
    if camera_id:
        events = [e for e in events if e.camera_id == camera_id]
    rows = compute_volumes(events, bucket_minutes=bucket_minutes, camera_id=None)
    return [
        schemas.VolumeBucket(camera_id=r.camera_id, bucket_start=iso(r.bucket_start), count=r.count)
        for r in rows
    ]


@router.get("/od_matrix", response_model=schemas.OdMatrix)
def od_matrix(state: AppState = Depends(get_state)) -> schemas.OdMatrix:  # noqa: B008
    return schemas.OdMatrix(zones=state.od_zones, matrix=state.od_matrix)


@router.get("/corridors", response_model=list[schemas.Corridor])
def corridors(limit: int = 20, state: AppState = Depends(get_state)) -> list[schemas.Corridor]:  # noqa: B008
    rows = state.corridors_cache[:limit]
    return [
        schemas.Corridor(
            from_camera=r.from_camera,
            to_camera=r.to_camera,
            n_trips=r.n_trips,
            median_travel_s=r.median_travel_s,
            p90_travel_s=r.p90_travel_s,
            free_flow_s=r.free_flow_s if r.free_flow_s is not None else 0.0,
            congestion_index=r.congestion_index if r.congestion_index is not None else 0.0,
            distance_m=r.distance_m,
            avg_speed_kmh=r.avg_speed_kmh,
            p85_speed_kmh=r.p85_speed_kmh,
            free_flow_speed_kmh=r.free_flow_speed_kmh,
        )
        for r in rows
    ]


@router.get("/heatmap", response_model=schemas.Heatmap)
def heatmap(
    at: str | None = None,
    window_minutes: int = 15,
    metric: Literal["density", "speed"] = "density",
    state: AppState = Depends(get_state),  # noqa: B008
    replay: ReplayEngine = Depends(get_replay_engine),  # noqa: B008
) -> schemas.Heatmap:
    """Contract v2: `at` defaults to the current replay sim time (if
    replay is running), else the end of the dataset."""
    if at is not None:
        at_dt = parse_iso(at)
    elif replay.running:
        at_dt = replay.sim_time
    elif state.t_max is not None:
        at_dt = state.t_max
    else:
        at_dt = replay.sim_time

    window_start = at_dt - timedelta(minutes=window_minutes)
    cameras = state.city.cameras
    if metric == "speed":
        rows = speed_heatmap(cameras, state.link_arrivals, window_start, at_dt)
    else:
        rows = density_heatmap(cameras, state.lite_events, window_start, at_dt)

    return schemas.Heatmap(
        at=iso(at_dt),
        window_minutes=window_minutes,
        metric=metric,
        points=[
            schemas.HeatPoint(camera_id=r.camera_id, lat=r.lat, lon=r.lon, weight=r.weight)
            for r in rows
        ],
    )


@router.get("/flow_trend", response_model=list[schemas.FlowBucket])
def flow_trend(
    bucket_minutes: int = 15, state: AppState = Depends(get_state)  # noqa: B008
) -> list[schemas.FlowBucket]:
    rows = compute_flow_trend(state.lite_events, state.link_arrivals, bucket_minutes=bucket_minutes)
    return [
        schemas.FlowBucket(
            bucket_start=iso(r.bucket_start),
            events=r.events,
            active_trajectories=r.active_trajectories,
            mean_speed_kmh=r.mean_speed_kmh,
        )
        for r in rows
    ]
