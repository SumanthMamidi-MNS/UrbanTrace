"""FastAPI dependencies: a per-request DB session, and accessors for the
app-wide cached state / replay engine built at startup (`api/main.py`)."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Request
from sqlalchemy.orm import Session

from api.replay import ReplayEngine
from api.state import AppState


def get_session(request: Request) -> Generator[Session, None, None]:
    session_factory = request.app.state.sutra.session_factory
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_state(request: Request) -> AppState:
    return request.app.state.sutra


def get_replay_engine(request: Request) -> ReplayEngine:
    return request.app.state.sutra_replay
