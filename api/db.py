"""SQLAlchemy + SQLite storage for the SUTRA API (docs/api-contract.md).

One row per camera / event / trajectory / alert, plus a small `meta`
key-value table for dataset-level facts (name, city graph, precomputed
`plate_repair_rate`) that `api/ingest.py` computes once and the API just
reads back.

Design choice: `events` and `trajectories` carry the plain scalar/queryable
columns needed for SQL filtering + pagination (camera_id, timestamp,
decoded_plate, trajectory_id — all indexed per the build brief) PLUS the
richer nested shapes (plate posterior, consensus, links, path) as JSON
columns, so the API layer never has to re-run the engine (consensus
decoding, clone detection, ...) at request time — only at ingest time. The
consensus JSON stores the FULL per-slot posterior (not the top-5-truncated
API shape) because `engine.decode.partial_search` needs the full
distribution to score arbitrary queried characters; the API truncates to
top-5 only when building the `PlateConsensus` response.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class CameraRow(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    node_id: Mapped[str] = mapped_column(String)
    bearing_deg: Mapped[float] = mapped_column(Float)
    is_border: Mapped[bool] = mapped_column(Boolean)


class EventRow(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime)
    plate_argmax: Mapped[str] = mapped_column(String)
    plate_confidence: Mapped[float] = mapped_column(Float)
    color: Mapped[str] = mapped_column(String)
    vehicle_type: Mapped[str] = mapped_column(String)
    trajectory_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # list[{"top": [{"char": str, "prob": float}], "unread": bool}], len 10
    posterior: Mapped[list] = mapped_column(JSON)
    embedding_ref: Mapped[int | None] = mapped_column(Integer, nullable=True)


Index("ix_events_camera_id", EventRow.camera_id)
Index("ix_events_timestamp", EventRow.timestamp)
Index("ix_events_trajectory_id", EventRow.trajectory_id)
Index("ix_events_decoded_plate", EventRow.plate_argmax)


class TrajectoryRow(Base):
    __tablename__ = "trajectories"

    trajectory_id: Mapped[str] = mapped_column(String, primary_key=True)
    decoded_plate: Mapped[str] = mapped_column(String)
    plate_confidence: Mapped[float] = mapped_column(Float)
    start_time: Mapped[dt.datetime] = mapped_column(DateTime)
    end_time: Mapped[dt.datetime] = mapped_column(DateTime)
    n_events: Mapped[int] = mapped_column(Integer)
    camera_sequence: Mapped[list] = mapped_column(JSON)
    color: Mapped[str] = mapped_column(String)
    vehicle_type: Mapped[str] = mapped_column(String)
    has_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    event_ids: Mapped[list] = mapped_column(JSON)
    # {"per_slot_full": [dict[str,float] x10], "single_read_plates": [...], "entropy_bits": float}
    consensus: Mapped[dict] = mapped_column(JSON)
    # list[LinkEvidence-shaped dict], non-finite floats already -> None
    links: Mapped[list] = mapped_column(JSON)
    # list[PathPoint-shaped dict]
    path: Mapped[list] = mapped_column(JSON)


Index("ix_trajectories_decoded_plate", TrajectoryRow.decoded_plate)
Index("ix_trajectories_start_time", TrajectoryRow.start_time)
Index("ix_trajectories_end_time", TrajectoryRow.end_time)


class AlertRow(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(String, primary_key=True)
    type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)
    plate: Mapped[str] = mapped_column(String)
    trajectory_ids: Mapped[list] = mapped_column(JSON)
    summary: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict] = mapped_column(JSON)


Index("ix_alerts_type", AlertRow.type)
Index("ix_alerts_created_at", AlertRow.created_at)


class WatchlistEntryRow(Base):
    """A blacklisted-plate pattern (docs/api-contract.md Contract v2,
    `WatchlistEntry`). `canonical_patterns` is precomputed at creation time
    by `api.plate_grammar.enumerate_canonical_forms` (the same grammar
    `/api/search` uses) so matching never has to re-parse `pattern` live."""

    __tablename__ = "watchlist_entries"

    entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    pattern: Mapped[str] = mapped_column(String)
    canonical_patterns: Mapped[list] = mapped_column(JSON)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    hits: Mapped[int] = mapped_column(Integer, default=0)


class WatchlistHitRow(Base):
    """One probabilistic match of a watchlist entry against a read
    (docs/api-contract.md Contract v2, `WatchlistHit`). Written on EVERY
    match >= `engine.alerts.watchlist.MATCH_THRESHOLD`; the first hit for a
    given (entry_id, trajectory_id) pair also creates an `AlertRow`
    (type="watchlist") -- see `api/replay.py`."""

    __tablename__ = "watchlist_hits"

    hit_id: Mapped[str] = mapped_column(String, primary_key=True)
    entry_id: Mapped[str] = mapped_column(String)
    pattern: Mapped[str] = mapped_column(String)
    event_id: Mapped[str] = mapped_column(String)
    trajectory_id: Mapped[str | None] = mapped_column(String, nullable=True)
    camera_id: Mapped[str] = mapped_column(String)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime)
    probability: Mapped[float] = mapped_column(Float)
    matched_on: Mapped[str] = mapped_column(String)
    plate_read: Mapped[str] = mapped_column(String)


Index("ix_watchlist_hits_entry_id", WatchlistHitRow.entry_id)
Index("ix_watchlist_hits_timestamp", WatchlistHitRow.timestamp)


class MetaRow(Base):
    """Small key-value table for dataset-level facts computed once at
    ingest time: dataset name, the full city graph (nodes/edges/cameras),
    and `plate_repair_rate` (needs the consensus-vs-single-read comparison
    that only ingest.py, not the API, ever computes)."""

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)


def make_engine(db_path: str | Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    return engine


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
