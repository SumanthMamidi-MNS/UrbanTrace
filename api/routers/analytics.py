"""Analytics endpoints (docs/api-contract.md `/api/analytics/*`). Results
are precomputed once at startup (`api/state.py`) since "the data is static
per dataset"; volumes bucket on the fly over the cached lite-event list
(bucket_minutes/camera_id vary per request)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api import schemas
from api.deps import get_state
from api.state import AppState
from api.util import iso
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
        )
        for r in rows
    ]
