"""FastAPI app entry point.

    uvicorn api.main:app --port 8000

Reads the SQLite database at `$URBANTRACE_DB_PATH` (falls back to the legacy
`$SUTRA_DB_PATH`, default `<data_dir>/urbantrace.db` -- see `engine.paths`),
populated by `python -m api.ingest`. If `web/dist` exists, the built UI is
served at `/` (with an index.html fallback for client-side routes) so one
`uvicorn` process gives the whole demo.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.replay import build_replay_engine
from api.routers import alerts as alerts_router
from api.routers import analytics as analytics_router
from api.routers import core as core_router
from api.routers import eval as eval_router
from api.routers import replay as replay_router
from api.routers import watchlist as watchlist_router
from api.routers import ws as ws_router
from api.state import load_app_state
from engine.paths import get_db_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = get_db_path()
    state = load_app_state(db_path)
    app.state.urbantrace = state
    engine = build_replay_engine(state.session_factory)
    app.state.urbantrace_replay = engine
    task = asyncio.ensure_future(engine.run_forever())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="UrbanTrace API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(core_router.router)
app.include_router(analytics_router.router)
app.include_router(alerts_router.router)
app.include_router(eval_router.router)
app.include_router(replay_router.router)
app.include_router(watchlist_router.router)
app.include_router(ws_router.router)

if DIST_DIR.exists():
    _assets_dir = DIST_DIR / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws"):
            raise HTTPException(status_code=404, detail="not found")
        candidate = DIST_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")
