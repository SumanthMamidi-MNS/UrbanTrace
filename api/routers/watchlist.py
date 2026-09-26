"""`/api/watchlist*` (docs/api-contract.md Contract v2) -- CRUD for entries
plus recorded hits. The probabilistic matching itself happens live during
replay (`api/replay.py`'s `ReplayEngine`, built on
`engine.alerts.watchlist`); this router only manages entries and serves what
replay already wrote to the database."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api import schemas
from api.db import WatchlistEntryRow, WatchlistHitRow
from api.deps import get_session
from api.plate_grammar import GrammarError, enumerate_canonical_forms
from api.util import clamp_limit, iso

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])


def _entry_to_schema(row: WatchlistEntryRow) -> schemas.WatchlistEntry:
    return schemas.WatchlistEntry(
        entry_id=row.entry_id,
        pattern=row.pattern,
        canonical_patterns=row.canonical_patterns,
        reason=row.reason,
        created_at=iso(row.created_at),
        active=row.active,
        hits=row.hits,
    )


@router.get("", response_model=list[schemas.WatchlistEntry])
def list_watchlist(session: Session = Depends(get_session)) -> list[schemas.WatchlistEntry]:  # noqa: B008
    rows = session.query(WatchlistEntryRow).order_by(WatchlistEntryRow.created_at.desc()).all()
    return [_entry_to_schema(r) for r in rows]


@router.post("", response_model=schemas.WatchlistEntry, status_code=201)
def create_watchlist_entry(
    body: schemas.WatchlistCreate, session: Session = Depends(get_session)  # noqa: B008
) -> schemas.WatchlistEntry:
    try:
        canonical_patterns = enumerate_canonical_forms(body.pattern)
    except GrammarError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc

    row = WatchlistEntryRow(
        entry_id=f"wl_{uuid.uuid4().hex[:12]}",
        pattern=body.pattern.strip().upper(),
        canonical_patterns=canonical_patterns,
        reason=body.reason,
        created_at=datetime.now(UTC).replace(tzinfo=None),
        active=True,
        hits=0,
    )
    session.add(row)
    session.commit()
    return _entry_to_schema(row)


@router.delete("/{entry_id}")
def delete_watchlist_entry(
    entry_id: str, session: Session = Depends(get_session)  # noqa: B008
) -> dict[str, bool]:
    row = session.get(WatchlistEntryRow, entry_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"watchlist entry {entry_id!r} not found")
    session.delete(row)
    session.commit()
    return {"deleted": True}


@router.get("/hits", response_model=list[schemas.WatchlistHit])
def list_watchlist_hits(
    entry_id: str | None = None,
    limit: int = 50,
    session: Session = Depends(get_session),  # noqa: B008
) -> list[schemas.WatchlistHit]:
    limit = clamp_limit(limit)
    q = session.query(WatchlistHitRow)
    if entry_id:
        q = q.filter(WatchlistHitRow.entry_id == entry_id)
    rows = q.order_by(WatchlistHitRow.timestamp.desc()).limit(limit).all()
    return [
        schemas.WatchlistHit(
            hit_id=r.hit_id,
            entry_id=r.entry_id,
            pattern=r.pattern,
            event_id=r.event_id,
            trajectory_id=r.trajectory_id,
            camera_id=r.camera_id,
            timestamp=iso(r.timestamp),
            probability=r.probability,
            matched_on=r.matched_on,
            plate_read=r.plate_read,
        )
        for r in rows
    ]
