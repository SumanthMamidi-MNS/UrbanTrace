"""`POST /api/replay` (docs/api-contract.md) — mutates the shared
simulation clock; `/ws/live` (api/routers/ws.py) only streams it."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api import schemas
from api.deps import get_replay_engine
from api.replay import DEFAULT_SPEED, ReplayEngine

router = APIRouter(prefix="/api", tags=["replay"])


@router.post("/replay", response_model=schemas.ReplayState)
def replay_control(
    body: schemas.ReplayRequest, engine: ReplayEngine = Depends(get_replay_engine)  # noqa: B008
) -> schemas.ReplayState:
    if body.action == "start":
        engine.start(speed=body.speed if body.speed is not None else None)
    elif body.action == "pause":
        engine.pause()
    elif body.action == "reset":
        engine.reset()
        if body.speed is not None:
            engine.speed = body.speed
    return schemas.ReplayState(**engine.state_dict())


__all__ = ["DEFAULT_SPEED", "router"]
