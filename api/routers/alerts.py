"""`/api/alerts` (docs/api-contract.md)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api import schemas
from api.builders import alert_row_to_schema
from api.db import AlertRow
from api.deps import get_session
from api.util import clamp_limit

router = APIRouter(prefix="/api", tags=["alerts"])


@router.get("/alerts", response_model=list[schemas.Alert])
def list_alerts(
    type: str | None = None,  # noqa: A002 - matches the contract's own query param name
    limit: int = 50,
    session: Session = Depends(get_session),  # noqa: B008
) -> list[schemas.Alert]:
    limit = clamp_limit(limit)
    q = session.query(AlertRow)
    if type:
        q = q.filter(AlertRow.type == type)
    rows = q.order_by(AlertRow.created_at.desc()).limit(limit).all()
    return [alert_row_to_schema(r) for r in rows]
