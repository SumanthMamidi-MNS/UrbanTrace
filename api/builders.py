"""DB row -> API schema converters. Single source of truth used by both the
REST routers and the WebSocket replay stream, so the two can never drift."""

from __future__ import annotations

from api import schemas
from api.db import AlertRow, EventRow, TrajectoryRow
from api.util import finite_or_none, iso, top_k_slot
from engine.analytics.direction import bearing_deg, compass_label


def event_row_to_summary(row: EventRow) -> schemas.EventSummary:
    return schemas.EventSummary(
        event_id=row.event_id,
        camera_id=row.camera_id,
        timestamp=iso(row.timestamp),
        plate_argmax=row.plate_argmax,
        plate_confidence=row.plate_confidence,
        color=row.color,
        vehicle_type=row.vehicle_type,
        trajectory_id=row.trajectory_id,
    )


def event_row_to_detail(row: EventRow) -> schemas.EventDetail:
    summary = event_row_to_summary(row)
    posterior = [
        schemas.SlotPosteriorOut(
            top=[schemas.SlotRead(char=t["char"], prob=t["prob"]) for t in slot["top"]],
            unread=slot["unread"],
        )
        for slot in row.posterior
    ]
    return schemas.EventDetail(**summary.model_dump(), plate_posterior=posterior)


def trajectory_row_to_summary(row: TrajectoryRow) -> schemas.TrajectorySummary:
    return schemas.TrajectorySummary(
        trajectory_id=row.trajectory_id,
        decoded_plate=row.decoded_plate,
        plate_confidence=row.plate_confidence,
        start_time=iso(row.start_time),
        end_time=iso(row.end_time),
        n_events=row.n_events,
        camera_sequence=row.camera_sequence,
        color=row.color,
        vehicle_type=row.vehicle_type,
        has_alert=row.has_alert,
    )


def _link_dict_to_schema(link: dict) -> schemas.LinkEvidence:
    return schemas.LinkEvidence(
        from_event_id=link["from_event_id"],
        to_event_id=link["to_event_id"],
        plate_lr=finite_or_none(link["plate_lr"]),
        appearance_lr=finite_or_none(link["appearance_lr"]),
        kinematic_lr=finite_or_none(link["kinematic_lr"]),
        prior_log_odds=link["prior_log_odds"],
        total_log_odds=finite_or_none(link["total_log_odds"]),
        delta_t_s=link["delta_t_s"],
        expected_t_s=link["expected_t_s"],
        skipped_cameras=link["skipped_cameras"],
    )


def _path_with_headings(raw_path: list[dict]) -> list[schemas.PathPoint]:
    """Bearing to the NEXT point per `PathPoint` (Contract v2); null on the
    last point."""
    n = len(raw_path)
    points = []
    for i, p in enumerate(raw_path):
        heading = None
        if i < n - 1:
            nxt = raw_path[i + 1]
            heading = bearing_deg(p["lat"], p["lon"], nxt["lat"], nxt["lon"])
        points.append(
            schemas.PathPoint(
                event_id=p["event_id"],
                camera_id=p["camera_id"],
                lat=p["lat"],
                lon=p["lon"],
                timestamp=p["timestamp"],
                heading_deg=heading,
            )
        )
    return points


def _overall_direction(raw_path: list[dict]) -> tuple[float | None, str | None]:
    """Bearing (and its 8-point compass label) from the first to the last
    camera (Contract v2, `TrajectoryDetail.overall_heading_deg` /
    `direction_label`); `(None, None)` for a single-event trajectory."""
    if len(raw_path) < 2:
        return None, None
    first, last = raw_path[0], raw_path[-1]
    heading = bearing_deg(first["lat"], first["lon"], last["lat"], last["lon"])
    return heading, compass_label(heading)


def trajectory_row_to_detail(
    row: TrajectoryRow, event_rows: list[EventRow]
) -> schemas.TrajectoryDetail:
    summary = trajectory_row_to_summary(row)
    events = [event_row_to_summary(e) for e in event_rows]
    links = [_link_dict_to_schema(link) for link in row.links]
    path = _path_with_headings(row.path)
    overall_heading, direction_label = _overall_direction(row.path)
    per_slot_full = row.consensus["per_slot_full"]
    consensus = schemas.PlateConsensus(
        per_slot=[
            [schemas.SlotRead(char=c, prob=p) for c, p in top_k_pairs(slot)]
            for slot in per_slot_full
        ],
        single_read_plates=row.consensus["single_read_plates"],
        entropy_bits=row.consensus["entropy_bits"],
    )
    return schemas.TrajectoryDetail(
        **summary.model_dump(),
        events=events,
        links=links,
        path=path,
        consensus=consensus,
        overall_heading_deg=overall_heading,
        direction_label=direction_label,
    )


def top_k_pairs(full_posterior: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    items = top_k_slot(full_posterior, k)
    return [(d["char"], d["prob"]) for d in items]


def alert_row_to_schema(row: AlertRow) -> schemas.Alert:
    evidence = dict(row.evidence)
    points = evidence.get("points")
    ev = schemas.AlertEvidence(
        distance_m=finite_or_none(evidence.get("distance_m")),
        delta_t_s=finite_or_none(evidence.get("delta_t_s")),
        min_required_s=finite_or_none(evidence.get("min_required_s")),
        appearance_distance=finite_or_none(evidence.get("appearance_distance")),
        points=[schemas.PathPoint(**p) for p in points] if points else None,
        watchlist_entry_id=evidence.get("watchlist_entry_id"),
        pattern=evidence.get("pattern"),
        match_probability=finite_or_none(evidence.get("match_probability")),
        matched_on=evidence.get("matched_on"),
    )
    return schemas.Alert(
        alert_id=row.alert_id,
        type=row.type,
        severity=row.severity,
        created_at=iso(row.created_at),
        plate=row.plate,
        trajectory_ids=row.trajectory_ids,
        summary=row.summary,
        evidence=ev,
    )
