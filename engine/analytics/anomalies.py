"""Loop anomalies (architecture.md §4; docs/api-contract.md `Alert` with
`type: "anomaly"`).

A trajectory that revisits the same camera >= 3 times within any rolling
one-hour window is flagged -- a vehicle genuinely circling one camera that
often in an hour is unusual (a normal trip passes each camera on its route
at most once; a *little* revisiting can be a legitimate U-turn/loop, but
three within an hour is well past that). Reuses `engine.decode.clone_detect.Alert`
rather than redefining the same shape twice within this task's owned
files."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import Trajectory
from engine.decode.clone_detect import Alert

MIN_REVISITS = 3
WINDOW_S = 3600.0


@dataclass
class LoopOccurrence:
    camera_id: str
    timestamps: list[datetime]


def find_loops(traj: Trajectory, events_by_id: dict[str, DetectionEvent]) -> list[LoopOccurrence]:
    """Every camera in `traj` visited >= MIN_REVISITS times within some
    rolling `WINDOW_S`-second window, in the order those cameras were first
    revisited. A sliding-window (two-pointer) scan over each camera's own
    sorted visit timestamps -- linear in the number of visits to that
    camera."""
    visits_by_camera: dict[str, list[datetime]] = defaultdict(list)
    for eid in traj.event_ids:
        e = events_by_id[eid]
        visits_by_camera[e.camera_id].append(e.timestamp)

    loops: list[LoopOccurrence] = []
    for camera_id, timestamps in visits_by_camera.items():
        if len(timestamps) < MIN_REVISITS:
            continue
        ts_sorted = sorted(timestamps)
        window = timedelta(seconds=WINDOW_S)
        left = 0
        found = False
        for right in range(len(ts_sorted)):
            while ts_sorted[right] - ts_sorted[left] > window:
                left += 1
            if right - left + 1 >= MIN_REVISITS:
                found = True
                break
        if found:
            loops.append(LoopOccurrence(camera_id=camera_id, timestamps=ts_sorted))

    return loops


def detect_loop_anomalies(
    trajectories: list[Trajectory],
    events_by_id: dict[str, DetectionEvent],
    reference_time: datetime | None = None,
) -> list[Alert]:
    """One `type="anomaly"` alert per trajectory that has at least one
    looping camera (module docstring); a trajectory can have more than one
    looping camera, reported as one alert per trajectory with all of them
    named in the summary (a trajectory revisiting several cameras that
    often is one anomalous journey, not several independent ones)."""
    now = reference_time or datetime.now()
    alerts: list[Alert] = []
    for traj in trajectories:
        loops = find_loops(traj, events_by_id)
        if not loops:
            continue
        camera_ids = [loop.camera_id for loop in loops]
        summary = (
            f"Trajectory {traj.trajectory_id} (plate {traj.decoded_plate}) revisited "
            f"{', '.join(camera_ids)} >= {MIN_REVISITS} times within an hour"
        )
        alerts.append(
            Alert(
                alert_id=f"alert_loop_{traj.trajectory_id}",
                type="anomaly",
                severity="low",
                created_at=now,
                plate=traj.decoded_plate,
                trajectory_ids=[traj.trajectory_id],
                summary=summary,
                evidence={"looping_cameras": camera_ids},
            )
        )
    return alerts


__all__ = ["MIN_REVISITS", "WINDOW_S", "LoopOccurrence", "detect_loop_anomalies", "find_loops"]
