"""App-wide cached state, built once at startup from the SQLite database
(docs/api-contract.md's analytics endpoints: "cache results in memory at
startup (the data is static per dataset)").

Engine analytics functions (`engine.analytics.*`) only ever touch
`.camera_id` / `.timestamp` on "events" and a handful of fields on
`Trajectory`, so this module reconstructs lightweight stand-ins from DB rows
rather than re-hydrating full `DetectionEvent`/`Trajectory` pydantic objects
(which would mean re-validating embeddings/posteriors for ~100k rows on
every process start for no benefit)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func

from api.db import (
    AlertRow,
    CameraRow,
    EventRow,
    MetaRow,
    TrajectoryRow,
    init_db,
    make_engine,
    make_session_factory,
)
from engine.analytics.corridors import CorridorStats, compute_corridors
from engine.analytics.heatmap import LinkArrival, compute_link_arrivals
from engine.analytics.od_matrix import build_od_matrix
from engine.contracts.city import CityConfig
from engine.contracts.trajectory import Trajectory

CORRIDORS_CACHE_LIMIT = 2000


@dataclass(frozen=True)
class LiteEvent:
    camera_id: str
    timestamp: datetime
    trajectory_id: str | None = None


@dataclass
class AppState:
    engine: object
    session_factory: object
    dataset_name: str
    city: CityConfig
    n_events: int
    n_trajectories: int
    mean_trip_duration_s: float
    mean_events_per_trajectory: float
    plate_repair_rate: float
    active_alerts: int
    od_zones: list[str]
    od_matrix: list[list[int]]
    corridors_cache: list[CorridorStats]
    camera_stats: dict[str, tuple[int, int]]  # camera_id -> (events_total, volume_last_hour)
    lite_events: list[LiteEvent]
    # Contract v2 (heatmap / flow_trend): one row per observed trajectory
    # camera-hop, and the dataset's own time span for the heatmap's `at`
    # default ("end of data if replay isn't running").
    link_arrivals: list[LinkArrival]
    t_min: datetime | None
    t_max: datetime | None


def load_app_state(db_path: str) -> AppState:
    eng = make_engine(db_path)
    init_db(eng)
    session_factory = make_session_factory(eng)

    with session_factory() as session:
        meta = {row.key: row.value for row in session.query(MetaRow).all()}
        city = CityConfig.model_validate(
            meta.get("city", {"nodes": [], "edges": [], "cameras": []})
        )
        dataset_name = meta.get("dataset_name", {}).get("name", "unknown")
        plate_repair_rate = float(meta.get("plate_repair_rate", {}).get("value", 0.0))

        lite_rows = session.query(
            EventRow.event_id, EventRow.camera_id, EventRow.timestamp, EventRow.trajectory_id
        ).all()
        events_by_id_lite = {
            r.event_id: LiteEvent(
                camera_id=r.camera_id, timestamp=r.timestamp, trajectory_id=r.trajectory_id
            )
            for r in lite_rows
        }
        lite_events = list(events_by_id_lite.values())
        n_events = len(lite_events)
        t_min = min((e.timestamp for e in lite_events), default=None)
        t_max = max((e.timestamp for e in lite_events), default=None)

        traj_rows = session.query(
            TrajectoryRow.trajectory_id,
            TrajectoryRow.decoded_plate,
            TrajectoryRow.plate_confidence,
            TrajectoryRow.start_time,
            TrajectoryRow.end_time,
            TrajectoryRow.n_events,
            TrajectoryRow.camera_sequence,
            TrajectoryRow.event_ids,
        ).all()
        n_trajectories = len(traj_rows)

        trajectories: list[Trajectory] = []
        total_duration = 0.0
        total_events_in_traj = 0
        for row in traj_rows:
            trajectories.append(
                Trajectory(
                    trajectory_id=row.trajectory_id,
                    event_ids=row.event_ids,
                    decoded_plate=row.decoded_plate,
                    plate_confidence=row.plate_confidence,
                    start_time=row.start_time,
                    end_time=row.end_time,
                    camera_sequence=row.camera_sequence,
                    links=[],
                )
            )
            total_duration += (row.end_time - row.start_time).total_seconds()
            total_events_in_traj += row.n_events

        mean_trip_duration_s = total_duration / n_trajectories if n_trajectories else 0.0
        mean_events_per_trajectory = (
            total_events_in_traj / n_trajectories if n_trajectories else 0.0
        )

        od_zones, od_mat = build_od_matrix(trajectories, city)
        corridors_cache = compute_corridors(
            trajectories, events_by_id_lite, city, limit=CORRIDORS_CACHE_LIMIT
        )
        link_arrivals = compute_link_arrivals(trajectories, events_by_id_lite, city)

        active_alerts = session.query(func.count(AlertRow.alert_id)).scalar() or 0

        cam_total_rows = (
            session.query(EventRow.camera_id, func.count(EventRow.event_id))
            .group_by(EventRow.camera_id)
            .all()
        )
        cam_totals = dict(cam_total_rows)

        t_max = session.query(func.max(EventRow.timestamp)).scalar()
        last_hour_counts: dict[str, int] = {}
        if t_max is not None:
            last_hour_start = t_max - timedelta(hours=1)
            last_hour_rows = (
                session.query(EventRow.camera_id, func.count(EventRow.event_id))
                .filter(EventRow.timestamp >= last_hour_start)
                .group_by(EventRow.camera_id)
                .all()
            )
            last_hour_counts = dict(last_hour_rows)

        camera_ids = [c.camera_id for c in session.query(CameraRow.camera_id).all()]
        camera_stats = {
            cid: (cam_totals.get(cid, 0), last_hour_counts.get(cid, 0)) for cid in camera_ids
        }

    return AppState(
        engine=eng,
        session_factory=session_factory,
        dataset_name=dataset_name,
        city=city,
        n_events=n_events,
        n_trajectories=n_trajectories,
        mean_trip_duration_s=mean_trip_duration_s,
        mean_events_per_trajectory=mean_events_per_trajectory,
        plate_repair_rate=plate_repair_rate,
        active_alerts=active_alerts,
        od_zones=od_zones,
        od_matrix=od_mat,
        corridors_cache=corridors_cache,
        camera_stats=camera_stats,
        lite_events=lite_events,
        link_arrivals=link_arrivals,
        t_min=t_min,
        t_max=t_max,
    )
