"""Simulation-clock replay + `/ws/live` broadcast (docs/api-contract.md).

Honest framing (see also api/README.md): association itself is computed
window-by-window by `engine.association.window.solve_windowed`, which is
already online-capable — it does not need a live clock to run. This module
does not re-run the solver live; it streams the ALREADY-LINKED dataset
(events, trajectories, alerts, all ingested up front by `api/ingest.py`)
against a simulated clock, so the demo can show "the city as it happened"
at 60x (or any) speed. A real streaming deployment would run the windowed
solver incrementally as events actually arrive; this replay engine is a
demo/QA harness for exactly that already-computed output.

One shared clock for every connected client (not per-connection): a
`POST /api/replay` mutates `running`/`speed`/`sim_time`; a single
background asyncio task advances `sim_time` on a tick and broadcasts newly
revealed events/trajectories/alerts plus a ~1 Hz clock message. New clients
that connect mid-replay do not receive a backfill of already-passed
messages (this is a live tail, not a history API) — `GET /api/trajectories`
etc. cover backfill.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from fastapi import WebSocket

from api import schemas
from api.builders import alert_row_to_schema, event_row_to_summary
from api.db import AlertRow, EventRow, WatchlistEntryRow, WatchlistHitRow
from engine.alerts.watchlist import (
    MATCH_THRESHOLD,
    single_read_match_probability,
    trajectory_consensus_match_probability,
)

DEFAULT_SPEED = 60.0
DEFAULT_TICK_INTERVAL_S = 0.5
CLOCK_EMIT_INTERVAL_S = 1.0


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            targets = list(self._connections)
        dead = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    @property
    def n_connections(self) -> int:
        return len(self._connections)


@dataclass
class _TrajRevealState:
    ordered_events: list[tuple[str, str, datetime]]  # (event_id, camera_id, timestamp)
    decoded_plate: str
    plate_confidence: float
    color: str
    vehicle_type: str
    has_alert: bool
    revealed: int = 0


@dataclass
class _AlertEntry:
    row_id: str
    reveal_at: datetime
    schema: schemas.Alert = field(repr=False)


class ReplayEngine:
    """Owns the shared simulation clock and the reveal cursors over one
    ingested dataset. Built once at app startup from the DB (`from_session`)."""

    def __init__(
        self,
        t_min: datetime,
        t_max: datetime,
        events: list[tuple[datetime, EventRow]],
        traj_states: dict[str, _TrajRevealState],
        alert_entries: list[_AlertEntry],
        tick_interval_s: float = DEFAULT_TICK_INTERVAL_S,
        session_factory=None,
        events_by_id: dict[str, EventRow] | None = None,
        watchlist_alerted_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        self.t_min = t_min
        self.t_max = t_max if t_max is not None and t_max >= t_min else t_min
        self.speed = DEFAULT_SPEED
        self.running = False
        self.sim_time = t_min
        self._events = events  # sorted by timestamp
        self._traj_states = traj_states
        self._alert_entries = sorted(alert_entries, key=lambda a: a.reveal_at)
        self._event_idx = 0
        self._alert_idx = 0
        self._tick_interval_s = tick_interval_s
        self._last_clock_emit = -1.0
        self.manager = ConnectionManager()
        self._task: asyncio.Task | None = None
        # Watchlist (docs/api-contract.md Contract v2): a session factory to
        # read active entries / write hits+alerts live, the full EventRow
        # per event_id (posteriors needed for matching -- `_events` above
        # only needs (timestamp, row) pairs for the reveal cursor), and the
        # (entry_id, trajectory_id) pairs already alerted so each pair only
        # ever alerts once ("later matches are recorded as hits").
        self._session_factory = session_factory
        self._events_by_id = events_by_id or {}
        self._watchlist_alerted: set[tuple[str, str]] = watchlist_alerted_pairs or set()

    # -- control -----------------------------------------------------
    def start(self, speed: float | None = None) -> None:
        if speed is not None:
            self.speed = speed
        self.running = True

    def pause(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.running = False
        self.sim_time = self.t_min
        self._event_idx = 0
        self._alert_idx = 0
        for state in self._traj_states.values():
            state.revealed = 0

    def state_dict(self) -> dict:
        return {
            "running": self.running,
            "speed": self.speed,
            "sim_time": _iso(self.sim_time),
        }

    # -- background loop ----------------------------------------------
    def ensure_running_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self.run_forever())

    async def run_forever(self) -> None:
        loop = asyncio.get_event_loop()
        last = loop.time()
        while True:
            await asyncio.sleep(self._tick_interval_s)
            now = loop.time()
            dt_wall = now - last
            last = now
            if self.running:
                self.sim_time = self.sim_time + timedelta(seconds=dt_wall * self.speed)
                if self.sim_time >= self.t_max:
                    self.sim_time = self.t_max
                    self.running = False
                await self._reveal_due()
            if now - self._last_clock_emit >= CLOCK_EMIT_INTERVAL_S:
                self._last_clock_emit = now
                await self.manager.broadcast({"type": "clock", "data": self.state_dict()})

    async def _reveal_due(self) -> None:
        newly_revealed_events: list[EventRow] = []
        updated_traj_ids: set[str] = set()

        while (
            self._event_idx < len(self._events)
            and self._events[self._event_idx][0] <= self.sim_time
        ):
            _, row = self._events[self._event_idx]
            self._event_idx += 1
            summary = event_row_to_summary(row)
            await self.manager.broadcast({"type": "event", "data": summary.model_dump()})
            newly_revealed_events.append(row)
            traj_id = row.trajectory_id
            if traj_id and traj_id in self._traj_states:
                state = self._traj_states[traj_id]
                if state.revealed < len(state.ordered_events):
                    state.revealed += 1
                    updated_traj_ids.add(traj_id)
                    partial = self._partial_trajectory_summary(traj_id)
                    if partial is not None:
                        data = partial.model_dump()
                        await self.manager.broadcast({"type": "trajectory", "data": data})

        while (
            self._alert_idx < len(self._alert_entries)
            and self._alert_entries[self._alert_idx].reveal_at <= self.sim_time
        ):
            entry = self._alert_entries[self._alert_idx]
            self._alert_idx += 1
            await self.manager.broadcast({"type": "alert", "data": entry.schema.model_dump()})

        if newly_revealed_events or updated_traj_ids:
            await self._check_watchlist(newly_revealed_events, updated_traj_ids)

    # -- watchlist (docs/api-contract.md Contract v2) ------------------
    async def _check_watchlist(
        self, new_events: list[EventRow], updated_traj_ids: set[str]
    ) -> None:
        """Evaluate every ACTIVE watchlist entry against (a) each newly
        revealed single read and (b) the growing trajectory consensus of
        every trajectory that just gained a revealed event. Entries are
        re-queried on every call (not cached at startup) so a `POST
        /api/watchlist` made mid-replay "take[s] effect for subsequent
        reads" (the contract's own wording) on the very next tick."""
        if self._session_factory is None:
            return
        alerts_to_broadcast: list[schemas.Alert] = []
        with self._session_factory() as session:
            entries = (
                session.query(WatchlistEntryRow)
                .filter(WatchlistEntryRow.active.is_(True))
                .all()
            )
            if not entries:
                return

            for row in new_events:
                for wl in entries:
                    prob = single_read_match_probability(row, wl.canonical_patterns)
                    if prob >= MATCH_THRESHOLD:
                        alert = self._record_watchlist_match(
                            session,
                            wl,
                            event_id=row.event_id,
                            trajectory_id=row.trajectory_id,
                            camera_id=row.camera_id,
                            timestamp=row.timestamp,
                            probability=prob,
                            matched_on="single_read",
                            plate_read=row.plate_argmax,
                        )
                        if alert is not None:
                            alerts_to_broadcast.append(alert)

            for traj_id in updated_traj_ids:
                state = self._traj_states.get(traj_id)
                if state is None or state.revealed == 0:
                    continue
                revealed = state.ordered_events[: state.revealed]
                rows = [
                    self._events_by_id[eid] for eid, _, _ in revealed if eid in self._events_by_id
                ]
                if not rows:
                    continue
                latest_row = rows[-1]
                for wl in entries:
                    prob = trajectory_consensus_match_probability(rows, wl.canonical_patterns)
                    if prob >= MATCH_THRESHOLD:
                        alert = self._record_watchlist_match(
                            session,
                            wl,
                            event_id=latest_row.event_id,
                            trajectory_id=traj_id,
                            camera_id=latest_row.camera_id,
                            timestamp=latest_row.timestamp,
                            probability=prob,
                            matched_on="trajectory_consensus",
                            plate_read=latest_row.plate_argmax,
                        )
                        if alert is not None:
                            alerts_to_broadcast.append(alert)

            session.commit()

        for alert in alerts_to_broadcast:
            await self.manager.broadcast({"type": "alert", "data": alert.model_dump()})

    def _record_watchlist_match(
        self,
        session,
        entry: WatchlistEntryRow,
        event_id: str,
        trajectory_id: str | None,
        camera_id: str,
        timestamp: datetime,
        probability: float,
        matched_on: str,
        plate_read: str,
    ) -> schemas.Alert | None:
        """Always records a `WatchlistHitRow`. The FIRST match for a given
        (entry_id, trajectory_id) pair also creates an `AlertRow`
        (type="watchlist") and is returned for broadcast; every later match
        of that same pair is "recorded as a hit" only, per the contract."""
        session.add(
            WatchlistHitRow(
                hit_id=f"wlhit_{uuid.uuid4().hex[:12]}",
                entry_id=entry.entry_id,
                pattern=entry.pattern,
                event_id=event_id,
                trajectory_id=trajectory_id,
                camera_id=camera_id,
                timestamp=timestamp,
                probability=probability,
                matched_on=matched_on,
                plate_read=plate_read,
            )
        )
        entry.hits += 1

        if trajectory_id is None:
            return None
        pair = (entry.entry_id, trajectory_id)
        if pair in self._watchlist_alerted:
            return None
        self._watchlist_alerted.add(pair)

        # NOTE: "trajectory_consensus" means THIS event's own single read did
        # not clear MATCH_THRESHOLD by itself -- not necessarily that it was
        # a character-for-character misread. In practice it usually is one
        # (the differentiator scenario), but it can also just be a
        # correctly-argmaxed read that was too LOW-CONFIDENCE on its own to
        # clear the threshold, with the earlier reads' agreement supplying
        # the rest of the certainty -- verified against the real dataset
        # (docs/api-contract.md Contract v2 real-database check), so the
        # summary avoids overclaiming "misread" specifically.
        consensus_note = (
            "matched via the fused multi-camera trajectory consensus (this camera's own "
            "single read did not clear the match threshold by itself)"
            if matched_on == "trajectory_consensus"
            else "matched on this camera's own single read"
        )
        summary = (
            f"Watchlisted plate {plate_read} ({entry.reason}) {consensus_note} at camera "
            f"{camera_id}, probability {probability:.2f}."
        )
        alert_row = AlertRow(
            alert_id=f"alert_wl_{uuid.uuid4().hex[:12]}",
            type="watchlist",
            severity="high",
            created_at=self.sim_time,
            plate=plate_read,
            trajectory_ids=[trajectory_id],
            summary=summary,
            evidence={
                "watchlist_entry_id": entry.entry_id,
                "pattern": entry.pattern,
                "match_probability": probability,
                "matched_on": matched_on,
            },
        )
        session.add(alert_row)
        return alert_row_to_schema(alert_row)

    def _partial_trajectory_summary(self, traj_id: str) -> schemas.TrajectorySummary | None:
        state = self._traj_states.get(traj_id)
        if state is None or state.revealed == 0:
            return None
        revealed_events = state.ordered_events[: state.revealed]
        return schemas.TrajectorySummary(
            trajectory_id=traj_id,
            decoded_plate=state.decoded_plate,
            plate_confidence=state.plate_confidence,
            start_time=_iso(revealed_events[0][2]),
            end_time=_iso(revealed_events[-1][2]),
            n_events=len(revealed_events),
            camera_sequence=[cam for _, cam, _ in revealed_events],
            color=state.color,
            vehicle_type=state.vehicle_type,
            has_alert=state.has_alert,
        )


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="milliseconds") + "Z"


def build_replay_engine(session_factory) -> ReplayEngine:
    from sqlalchemy import func

    from api.db import EventRow, TrajectoryRow

    with session_factory() as session:
        event_rows = (
            session.query(EventRow)
            .order_by(EventRow.timestamp)
            .all()
        )
        t_min = session.query(func.min(EventRow.timestamp)).scalar()
        t_max = session.query(func.max(EventRow.timestamp)).scalar()
        if t_min is None:
            t_min = t_max = datetime.utcnow()

        events: list[tuple[datetime, EventRow]] = [(r.timestamp, r) for r in event_rows]

        events_by_traj: dict[str, list[tuple[str, str, datetime]]] = {}
        for e in event_rows:
            if e.trajectory_id:
                events_by_traj.setdefault(e.trajectory_id, []).append(
                    (e.event_id, e.camera_id, e.timestamp)
                )
        for lst in events_by_traj.values():
            lst.sort(key=lambda t: t[2])

        traj_rows = session.query(TrajectoryRow).all()
        traj_states: dict[str, _TrajRevealState] = {}
        for row in traj_rows:
            traj_states[row.trajectory_id] = _TrajRevealState(
                ordered_events=events_by_traj.get(row.trajectory_id, []),
                decoded_plate=row.decoded_plate,
                plate_confidence=row.plate_confidence,
                color=row.color,
                vehicle_type=row.vehicle_type,
                has_alert=row.has_alert,
            )

        alert_rows = session.query(AlertRow).all()
        alert_entries: list[_AlertEntry] = []
        traj_end_time = {row.trajectory_id: row.end_time for row in traj_rows}
        watchlist_alerted_pairs: set[tuple[str, str]] = set()
        for row in alert_rows:
            reveal_at = row.created_at
            points = (row.evidence or {}).get("points")
            if points:
                reveal_at = max(datetime.fromisoformat(p["timestamp"].rstrip("Z")) for p in points)
            else:
                ends = [traj_end_time[tid] for tid in row.trajectory_ids if tid in traj_end_time]
                if ends:
                    reveal_at = max(ends)
            alert_entries.append(
                _AlertEntry(
                    row_id=row.alert_id, reveal_at=reveal_at, schema=alert_row_to_schema(row)
                )
            )
            if row.type == "watchlist":
                wl_entry_id = (row.evidence or {}).get("watchlist_entry_id")
                if wl_entry_id and row.trajectory_ids:
                    watchlist_alerted_pairs.add((wl_entry_id, row.trajectory_ids[0]))

        events_by_id = {r.event_id: r for r in event_rows}

    return ReplayEngine(
        t_min=t_min,
        t_max=t_max,
        events=events,
        traj_states=traj_states,
        alert_entries=alert_entries,
        session_factory=session_factory,
        events_by_id=events_by_id,
        watchlist_alerted_pairs=watchlist_alerted_pairs,
    )
