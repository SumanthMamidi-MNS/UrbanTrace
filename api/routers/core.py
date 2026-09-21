"""Health, city graph, cameras, events, trajectories, search
(docs/api-contract.md)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api import schemas
from api.builders import (
    event_row_to_detail,
    event_row_to_summary,
    trajectory_row_to_detail,
    trajectory_row_to_summary,
)
from api.db import CameraRow, EventRow, TrajectoryRow
from api.deps import get_session, get_state
from api.plate_grammar import GrammarError, enumerate_canonical_forms
from api.state import AppState
from api.util import clamp_limit, parse_iso
from engine.decode.consensus import Consensus, SlotConsensus
from engine.decode.partial_search import WILDCARD, match_probability, normalise_query

router = APIRouter(prefix="/api", tags=["core"])


@router.get("/health", response_model=schemas.Health)
def health(state: AppState = Depends(get_state)) -> schemas.Health:  # noqa: B008
    return schemas.Health(
        status="ok",
        dataset=state.dataset_name,
        n_events=state.n_events,
        n_trajectories=state.n_trajectories,
    )


@router.get("/city", response_model=schemas.City)
def city(state: AppState = Depends(get_state)) -> schemas.City:  # noqa: B008
    c = state.city
    return schemas.City(
        nodes=[schemas.RoadNode(node_id=n.node_id, lat=n.lat, lon=n.lon) for n in c.nodes],
        edges=[
            schemas.RoadEdge(
                from_node=e.from_node,
                to_node=e.to_node,
                length_m=e.length_m,
                speed_limit_kmh=e.speed_limit_kmh,
            )
            for e in c.edges
        ],
        cameras=[
            schemas.Camera(
                camera_id=cam.camera_id,
                name=cam.name,
                lat=cam.lat,
                lon=cam.lon,
                node_id=cam.node_id,
                bearing_deg=cam.bearing_deg,
                is_border=cam.is_border,
            )
            for cam in c.cameras
        ],
    )


@router.get("/cameras", response_model=list[schemas.CameraStats])
def cameras(
    session: Session = Depends(get_session), state: AppState = Depends(get_state)  # noqa: B008
) -> list[schemas.CameraStats]:
    rows = session.query(CameraRow).all()
    out = []
    for r in rows:
        events_total, volume_last_hour = state.camera_stats.get(r.camera_id, (0, 0))
        out.append(
            schemas.CameraStats(
                camera_id=r.camera_id,
                name=r.name,
                lat=r.lat,
                lon=r.lon,
                node_id=r.node_id,
                bearing_deg=r.bearing_deg,
                is_border=r.is_border,
                events_total=events_total,
                volume_last_hour=volume_last_hour,
            )
        )
    return out


@router.get("/events", response_model=schemas.Page[schemas.EventSummary])
def list_events(
    camera_id: str | None = None,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_session),  # noqa: B008
) -> schemas.Page[schemas.EventSummary]:
    limit = clamp_limit(limit)
    q = session.query(EventRow)
    if camera_id:
        q = q.filter(EventRow.camera_id == camera_id)
    if from_:
        q = q.filter(EventRow.timestamp >= parse_iso(from_))
    if to:
        q = q.filter(EventRow.timestamp <= parse_iso(to))
    total = q.count()
    rows = q.order_by(EventRow.timestamp.desc()).offset(offset).limit(limit).all()
    items = [event_row_to_summary(r) for r in rows]
    return schemas.Page[schemas.EventSummary](items=items, total=total, limit=limit, offset=offset)


@router.get("/events/{event_id}", response_model=schemas.EventDetail)
def get_event(event_id: str, session: Session = Depends(get_session)) -> schemas.EventDetail:  # noqa: B008
    row = session.get(EventRow, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event {event_id!r} not found")
    return event_row_to_detail(row)


@router.get("/trajectories", response_model=schemas.Page[schemas.TrajectorySummary])
def list_trajectories(
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    plate: str | None = None,
    camera_id: str | None = None,
    min_len: int | None = None,
    has_alert: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_session),  # noqa: B008
) -> schemas.Page[schemas.TrajectorySummary]:
    limit = clamp_limit(limit)
    q = session.query(TrajectoryRow)
    if from_:
        q = q.filter(TrajectoryRow.end_time >= parse_iso(from_))
    if to:
        q = q.filter(TrajectoryRow.start_time <= parse_iso(to))
    if plate:
        q = q.filter(TrajectoryRow.decoded_plate == plate.strip().upper())
    if min_len is not None:
        q = q.filter(TrajectoryRow.n_events >= min_len)
    if has_alert is not None:
        q = q.filter(TrajectoryRow.has_alert == has_alert)
    if camera_id:
        traj_ids = [
            r[0]
            for r in session.query(EventRow.trajectory_id)
            .filter(EventRow.camera_id == camera_id, EventRow.trajectory_id.isnot(None))
            .distinct()
            .all()
        ]
        q = q.filter(TrajectoryRow.trajectory_id.in_(traj_ids))
    total = q.count()
    rows = q.order_by(TrajectoryRow.start_time.desc()).offset(offset).limit(limit).all()
    items = [trajectory_row_to_summary(r) for r in rows]
    return schemas.Page[schemas.TrajectorySummary](
        items=items, total=total, limit=limit, offset=offset
    )


@router.get("/trajectories/{trajectory_id}", response_model=schemas.TrajectoryDetail)
def get_trajectory(
    trajectory_id: str, session: Session = Depends(get_session)  # noqa: B008
) -> schemas.TrajectoryDetail:
    row = session.get(TrajectoryRow, trajectory_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"trajectory {trajectory_id!r} not found")
    event_rows = session.query(EventRow).filter(EventRow.event_id.in_(row.event_ids)).all()
    by_id = {e.event_id: e for e in event_rows}
    ordered = [by_id[eid] for eid in row.event_ids if eid in by_id]
    return trajectory_row_to_detail(row, ordered)


def _most_likely_completion(consensus: Consensus, query_slots: list[str]) -> str:
    chars = []
    for idx, q_char in enumerate(query_slots):
        chars.append(q_char if q_char != WILDCARD else consensus.per_slot[idx].argmax())
    return "".join(chars)


@router.get("/search", response_model=list[schemas.SearchHit])
def search(
    q: str,
    color: str | None = None,
    vehicle_type: str | None = None,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    limit: int = 50,
    session: Session = Depends(get_session),  # noqa: B008
) -> list[schemas.SearchHit]:
    try:
        canonical_forms = enumerate_canonical_forms(q)
    except GrammarError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc

    limit = clamp_limit(limit)
    query_rows = session.query(TrajectoryRow)
    if color:
        query_rows = query_rows.filter(TrajectoryRow.color == color)
    if vehicle_type:
        query_rows = query_rows.filter(TrajectoryRow.vehicle_type == vehicle_type)
    if from_:
        query_rows = query_rows.filter(TrajectoryRow.end_time >= parse_iso(from_))
    if to:
        query_rows = query_rows.filter(TrajectoryRow.start_time <= parse_iso(to))
    rows = query_rows.all()

    best: dict[str, tuple[float, str, TrajectoryRow]] = {}
    for canonical in canonical_forms:
        query_slots = normalise_query(canonical)
        for row in rows:
            per_slot_full = row.consensus["per_slot_full"]
            consensus = Consensus(
                decoded_plate=row.decoded_plate,
                confidence=row.plate_confidence,
                per_slot=[SlotConsensus(full_posterior=d, top5=[]) for d in per_slot_full],
                entropy_bits=row.consensus.get("entropy_bits", 0.0),
                single_read_plates=row.consensus.get("single_read_plates", []),
            )
            prob = match_probability(consensus, query_slots)
            if prob <= 0.0:
                continue
            matched_plate = _most_likely_completion(consensus, query_slots)
            prev = best.get(row.trajectory_id)
            if prev is None or prob > prev[0]:
                best[row.trajectory_id] = (prob, matched_plate, row)

    hits = [
        schemas.SearchHit(
            trajectory=trajectory_row_to_summary(row), probability=prob, matched_plate=matched
        )
        for prob, matched, row in best.values()
    ]
    hits.sort(key=lambda h: h.probability, reverse=True)
    return hits[:limit]
