"""`/ws/live` (docs/api-contract.md) — server-push only; the replay clock
is driven entirely by `POST /api/replay` (api/routers/replay.py)."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["ws"])


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    engine = websocket.app.state.urbantrace_replay
    await engine.manager.connect(websocket)
    try:
        while True:
            # Server-push only; drain whatever the client sends (pings etc.)
            # so `receive` doesn't error, but ignore the content.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await engine.manager.disconnect(websocket)
