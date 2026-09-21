"""Ingest a simulated dataset (+ optionally the pipeline's linked
trajectories) into the SUTRA SQLite database.

Usage:
    python -m api.ingest --data data/run1 \\
        --trajectories data/run1/pipeline/trajectories.jsonl --db data/sutra.db

    python -m api.ingest --data data/run1 --db data/sutra.db --build-demo

If `--trajectories` is omitted (or the file does not exist) and
`--build-demo` is passed, trajectories are built in-process: priors are
fit with `engine.calibration.fit_priors.fit_all` on a SEPARATE training
dataset generated on the SAME city (different vehicle-population seed),
exactly the pattern `tests/test_window.py` uses, then
`engine.association.window.solve_windowed` links the target dataset's own
events. This makes the API usable before (or independently of) a full
`eval/run_pipeline.py` run.

A `KinematicModel` is always fit on that training split — even when
trajectories come from the pipeline file — because `engine.decode.clone_detect`
needs one for its physical-feasibility check and this ingest path does not
assume access to whatever model file another pipeline run may or may not
have saved.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from api.db import (
    AlertRow,
    CameraRow,
    EventRow,
    MetaRow,
    TrajectoryRow,
    init_db,
    make_engine,
    make_session_factory,
)
from api.util import finite_or_none, iso
from engine.analytics.anomalies import detect_loop_anomalies
from engine.association.gating import Gate
from engine.association.mincostflow import fit_entry_exit_costs
from engine.association.window import solve_windowed
from engine.calibration.fit_priors import fit_all
from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent
from engine.contracts.store import EventStore
from engine.contracts.trajectory import Trajectory
from engine.decode.clone_detect import Alert as EngineAlert
from engine.decode.clone_detect import detect_clones
from engine.decode.consensus import Consensus, decode_trajectory
from engine.scoring.fusion import FusionModel
from sim.generate import generate_dataset_with_city

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_TRAIN_VEHICLES = 8000
DEFAULT_TRAIN_SEED_OFFSET = 9001


@dataclass
class IngestStats:
    source: str  # "pipeline" | "build-demo"
    n_events: int
    n_trajectories: int
    n_alerts: int
    plate_repair_rate: float
    t_load_events_s: float
    t_fit_models_s: float
    t_trajectories_s: float
    t_consensus_alerts_s: float
    t_write_db_s: float
    t_total_s: float


def load_city(data_dir: Path) -> CityConfig:
    return CityConfig.model_validate_json((data_dir / "city.json").read_text(encoding="utf-8"))


def load_events(data_dir: Path) -> list[DetectionEvent]:
    store = EventStore(data_dir / "events.jsonl", data_dir / "embeddings.npy")
    return store.read_all()


def load_trajectories_jsonl(path: Path) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(Trajectory.model_validate_json(line))
    return trajectories


def _read_config(data_dir: Path) -> dict:
    cfg_path = data_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {}


def fit_training_models(
    city: CityConfig,
    data_dir: Path,
    events: list[DetectionEvent],
    train_vehicles: int = DEFAULT_TRAIN_VEHICLES,
    train_seed_offset: int = DEFAULT_TRAIN_SEED_OFFSET,
):
    """Fit plate/kinematic/appearance priors on a SEPARATE training dataset
    sharing the target dataset's own city (architecture.md §10;
    docs/decisions.md "Day 2" — a kinematic model is only meaningful
    evaluated against the same camera network it was fit on). Mirrors
    `eval/run_pipeline.py`'s train-split pattern and `tests/test_window.py`'s
    `fit_all` usage."""
    config = _read_config(data_dir)
    data_seed = int(config.get("seed", 0))
    hours = int(config.get("hours") or _span_hours(events))
    train_seed = data_seed + train_seed_offset
    train_ds = generate_dataset_with_city(
        city, n_vehicles=train_vehicles, hours=hours, seed=train_seed, clone_fraction=0.0
    )
    return fit_all(train_ds)


def _span_hours(events: list[DetectionEvent]) -> int:
    if not events:
        return 24
    ts = [e.timestamp for e in events]
    span_s = (max(ts) - min(ts)).total_seconds()
    return max(1, int(span_s // 3600) + 1)


def build_demo_trajectories(
    events: list[DetectionEvent],
    city: CityConfig,
    fitted_models,
) -> list[Trajectory]:
    """Link `events` end-to-end with the windowed min-cost-flow solver,
    using priors already fit on a separate training split (see
    `fit_training_models`)."""
    fusion_model = FusionModel(
        plate_priors=fitted_models.plate_priors,
        kinematic_model=fitted_models.kinematic_model,
        appearance_model=fitted_models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(events, city)
    gate = Gate(model=fitted_models.kinematic_model)
    result = solve_windowed(events, gate, fusion_model, entry_exit, city)
    return result.trajectories


def _slot_posterior_out(sp, alphabet_unused=None) -> dict:
    """`EventDetail.plate_posterior` shape: {"top": [{"char","prob"} x<=5],
    "unread": bool} per slot."""
    ranked = sorted(sp.probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "top": [{"char": c, "prob": p} for c, p in ranked],
        "unread": sp.is_uninformative(),
    }


def _consensus_blob(consensus: Consensus) -> dict:
    return {
        "per_slot_full": [sc.full_posterior for sc in consensus.per_slot],
        "single_read_plates": consensus.single_read_plates,
        "entropy_bits": consensus.entropy_bits,
    }


def _link_blob(link) -> dict:
    return {
        "from_event_id": link.from_event_id,
        "to_event_id": link.to_event_id,
        "plate_lr": finite_or_none(link.plate_lr),
        "appearance_lr": finite_or_none(link.appearance_lr),
        "kinematic_lr": finite_or_none(link.kinematic_lr),
        "prior_log_odds": link.prior_log_odds,
        "total_log_odds": finite_or_none(link.total_log_odds),
        "delta_t_s": link.delta_t_s,
        "expected_t_s": link.expected_t_s,
        "skipped_cameras": link.skipped_cameras,
    }


def _path_for(
    trajectory: Trajectory,
    events_by_id: dict[str, DetectionEvent],
    cam_latlon: dict[str, tuple[float, float]],
) -> list[dict]:
    points = []
    for eid in trajectory.event_ids:
        e = events_by_id[eid]
        lat, lon = cam_latlon.get(e.camera_id, (0.0, 0.0))
        points.append(
            {
                "event_id": e.event_id,
                "camera_id": e.camera_id,
                "lat": lat,
                "lon": lon,
                "timestamp": iso(e.timestamp),
            }
        )
    return points


def _alert_to_blob(alert: EngineAlert, cam_latlon: dict[str, tuple[float, float]]) -> dict:
    evidence = dict(alert.evidence)
    points = evidence.get("points")
    if points:
        # clone_detect.py always emits lat=lon=0.0 placeholders (it has no
        # access to the city graph by design); fill in the real camera
        # position here so the map/WHY-panel gets a real point.
        evidence["points"] = [
            {
                "event_id": p.event_id,
                "camera_id": p.camera_id,
                "lat": cam_latlon.get(p.camera_id, (p.lat, p.lon))[0],
                "lon": cam_latlon.get(p.camera_id, (p.lat, p.lon))[1],
                "timestamp": iso(p.timestamp),
            }
            for p in points
        ]
    for k in ("distance_m", "delta_t_s", "min_required_s", "appearance_distance"):
        if k in evidence:
            evidence[k] = finite_or_none(evidence[k])
    return {
        "alert_id": alert.alert_id,
        "type": alert.type,
        "severity": alert.severity,
        "created_at": alert.created_at,
        "plate": alert.plate,
        "trajectory_ids": alert.trajectory_ids,
        "summary": alert.summary,
        "evidence": evidence,
    }


def run_ingest(
    data_dir: Path,
    db_path: Path,
    trajectories_path: Path | None = None,
    build_demo: bool = False,
    train_vehicles: int = DEFAULT_TRAIN_VEHICLES,
    train_seed_offset: int = DEFAULT_TRAIN_SEED_OFFSET,
    dataset_name: str | None = None,
) -> IngestStats:
    t_start = time.perf_counter()
    data_dir = Path(data_dir)
    city = load_city(data_dir)

    t0 = time.perf_counter()
    events = load_events(data_dir)
    t_load_events_s = time.perf_counter() - t0
    events_by_id = {e.event_id: e for e in events}

    t0 = time.perf_counter()
    fitted_models = fit_training_models(city, data_dir, events, train_vehicles, train_seed_offset)
    t_fit_models_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if trajectories_path is not None and Path(trajectories_path).exists():
        trajectories = load_trajectories_jsonl(Path(trajectories_path))
        source = "pipeline"
    elif build_demo:
        trajectories = build_demo_trajectories(events, city, fitted_models)
        source = "build-demo"
    else:
        raise ValueError(
            "no trajectories: pass --trajectories pointing at an existing file, "
            "or --build-demo to link events in-process"
        )
    t_trajectories_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    event_traj: dict[str, str] = {}
    for t in trajectories:
        for eid in t.event_ids:
            event_traj[eid] = t.trajectory_id

    consensus_by_traj: dict[str, Consensus] = {}
    n_repaired = 0
    n_multi = 0
    for t in trajectories:
        traj_events = [events_by_id[eid] for eid in t.event_ids]
        consensus = decode_trajectory(traj_events)
        consensus_by_traj[t.trajectory_id] = consensus
        if len(traj_events) > 1:
            n_multi += 1
            if any(sr != consensus.decoded_plate for sr in consensus.single_read_plates):
                n_repaired += 1
    plate_repair_rate = (n_repaired / n_multi) if n_multi else 0.0

    alerts = detect_clones(trajectories, events_by_id, fitted_models.kinematic_model, city)
    alerts += detect_loop_anomalies(trajectories, events_by_id)
    traj_with_alert: set[str] = set()
    for a in alerts:
        traj_with_alert.update(a.trajectory_ids)
    t_consensus_alerts_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    engine_db = make_engine(db_path)
    init_db(engine_db)
    session_factory = make_session_factory(engine_db)
    cam_latlon = {c.camera_id: (c.lat, c.lon) for c in city.cameras}

    with session_factory() as session:  # type: Session
        session.query(EventRow).delete()
        session.query(TrajectoryRow).delete()
        session.query(AlertRow).delete()
        session.query(CameraRow).delete()
        session.query(MetaRow).delete()

        for c in city.cameras:
            session.add(
                CameraRow(
                    camera_id=c.camera_id,
                    name=c.name,
                    lat=c.lat,
                    lon=c.lon,
                    node_id=c.node_id,
                    bearing_deg=c.bearing_deg,
                    is_border=c.is_border,
                )
            )

        for e in events:
            posterior = [_slot_posterior_out(sp) for sp in e.plate_posterior]
            session.add(
                EventRow(
                    event_id=e.event_id,
                    camera_id=e.camera_id,
                    timestamp=e.timestamp,
                    plate_argmax=e.plate_argmax,
                    plate_confidence=e.plate_confidence,
                    color=e.attributes.color,
                    vehicle_type=e.attributes.vehicle_type,
                    trajectory_id=event_traj.get(e.event_id),
                    posterior=posterior,
                    embedding_ref=e.embedding_ref,
                )
            )

        for t in trajectories:
            traj_events = [events_by_id[eid] for eid in t.event_ids]
            first = traj_events[0]
            consensus = consensus_by_traj[t.trajectory_id]
            session.add(
                TrajectoryRow(
                    trajectory_id=t.trajectory_id,
                    decoded_plate=consensus.decoded_plate,
                    plate_confidence=consensus.confidence,
                    start_time=t.start_time,
                    end_time=t.end_time,
                    n_events=len(t.event_ids),
                    camera_sequence=t.camera_sequence,
                    color=first.attributes.color,
                    vehicle_type=first.attributes.vehicle_type,
                    has_alert=t.trajectory_id in traj_with_alert,
                    event_ids=t.event_ids,
                    consensus=_consensus_blob(consensus),
                    links=[_link_blob(link) for link in t.links],
                    path=_path_for(t, events_by_id, cam_latlon),
                )
            )

        for a in alerts:
            blob = _alert_to_blob(a, cam_latlon)
            session.add(
                AlertRow(
                    alert_id=blob["alert_id"],
                    type=blob["type"],
                    severity=blob["severity"],
                    created_at=blob["created_at"],
                    plate=blob["plate"],
                    trajectory_ids=blob["trajectory_ids"],
                    summary=blob["summary"],
                    evidence=blob["evidence"],
                )
            )

        session.add(MetaRow(key="dataset_name", value={"name": dataset_name or data_dir.name}))
        session.add(MetaRow(key="plate_repair_rate", value={"value": plate_repair_rate}))
        session.add(MetaRow(key="city", value=city.model_dump(mode="json")))
        session.commit()
    t_write_db_s = time.perf_counter() - t0

    t_total_s = time.perf_counter() - t_start
    stats = IngestStats(
        source=source,
        n_events=len(events),
        n_trajectories=len(trajectories),
        n_alerts=len(alerts),
        plate_repair_rate=plate_repair_rate,
        t_load_events_s=t_load_events_s,
        t_fit_models_s=t_fit_models_s,
        t_trajectories_s=t_trajectories_s,
        t_consensus_alerts_s=t_consensus_alerts_s,
        t_write_db_s=t_write_db_s,
        t_total_s=t_total_s,
    )
    return stats


@app.command()
def main(
    data: Path = typer.Option(..., help="Dataset directory (city.json/events.jsonl/embeddings)"),
    db: Path = typer.Option(..., help="Output SQLite path."),
    trajectories: Path | None = typer.Option(
        None, help="Pipeline trajectories.jsonl (Trajectory contract, one per line)."
    ),
    build_demo: bool = typer.Option(
        False, "--build-demo", help="Link events in-process if --trajectories is absent."
    ),
    train_vehicles: int = typer.Option(
        DEFAULT_TRAIN_VEHICLES, help="Training-split vehicle count."
    ),
    train_seed_offset: int = typer.Option(
        DEFAULT_TRAIN_SEED_OFFSET, help="Offset added to the dataset seed for the training split."
    ),
) -> None:
    console.print(f"[bold]Ingesting[/bold] {data} -> {db}")
    stats = run_ingest(
        data_dir=data,
        db_path=db,
        trajectories_path=trajectories,
        build_demo=build_demo,
        train_vehicles=train_vehicles,
        train_seed_offset=train_seed_offset,
    )
    console.print(f"[green]Done[/green] (source={stats.source})")
    console.print(f"  events              : {stats.n_events}")
    console.print(f"  trajectories        : {stats.n_trajectories}")
    console.print(f"  alerts              : {stats.n_alerts}")
    console.print(f"  plate_repair_rate   : {stats.plate_repair_rate:.4f}")
    console.print(f"  t_load_events_s     : {stats.t_load_events_s:.2f}")
    console.print(f"  t_fit_models_s      : {stats.t_fit_models_s:.2f}")
    console.print(f"  t_trajectories_s    : {stats.t_trajectories_s:.2f}")
    console.print(f"  t_consensus_alerts_s: {stats.t_consensus_alerts_s:.2f}")
    console.print(f"  t_write_db_s        : {stats.t_write_db_s:.2f}")
    console.print(f"  t_total_s           : {stats.t_total_s:.2f}")


if __name__ == "__main__":
    app()
