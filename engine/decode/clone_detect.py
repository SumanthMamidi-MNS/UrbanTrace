"""Clone / impossible-travel detection (architecture.md §3, "Clone detection
falls out"; docs/decisions.md, "clone detection reuses the kinematic hard
gate").

Two same-(or near-identical-)plate trajectories are examined pairwise:

1. **Physical feasibility.** Merge both trajectories' events into one
   time-ordered sequence and walk every ADJACENT pair that crosses between
   the two trajectories (this naturally covers both "one ends, the other
   starts" and any interleaved observations -- a clone operating in the same
   corridor as its source, `sim.vehicles`' `clone_route_overlap`, produces
   exactly this interleaving). A crossing is physically impossible if the
   road-graph distance between the two cameras cannot be covered in `dt` at
   `v_max`. This is the *same* hard-gate arithmetic as
   `engine.scoring.kinematic_lr.KinematicModel.score`, applied at the
   trajectory level instead of the raw-candidate-pair level, and is
   deliberately reused rather than re-derived (docs/decisions.md).
2. **Appearance divergence.** Cosine distance between the two trajectories'
   mean (re-normalised) embeddings -- a genuine clone is a different vehicle
   and should look different on average, even though every individual
   embedding is noisy.

`rapidfuzz` groups trajectories by *decoded* plate (exact) or near-identical
decoded plate (edit distance <= `near_identical_max_distance`) purely to cut
down the candidate-pair space -- consistent with architecture.md §5's stated
rapidfuzz role ("candidate blocking, never final scoring"). The actual
clone/impossible-travel decision always comes from kinematics + appearance,
never from the grouping distance itself.

Alert shape mirrors docs/api-contract.md's `Alert` type; `Alert` and
`PathPoint` are defined here (not in `engine/contracts`, which is frozen
Day-1 territory outside this task's scope) and re-exported for
`engine.analytics.anomalies` to reuse for its own (non-clone) alerts.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from rapidfuzz.distance import Levenshtein

from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import Trajectory
from engine.scoring.kinematic_lr import KinematicModel

DEFAULT_V_MAX_KMH = 120.0
# Cosine distance (unit-vector embeddings, range [0, 2]) above which two
# trajectories' mean appearance is called "divergent" -- i.e. plausibly two
# different physical vehicles rather than noisy repeats of one. Chosen well
# above typical same-vehicle repeat-embedding noise (see
# sim/vehicles.py's per-observation noise sigmas, ~0.13-0.14 in a 128-d
# space -- same-vehicle mean-embedding distances land far below 0.3) and
# well below the ~1.0 expected for two unrelated appearance-class centres.
APPEARANCE_DIVERGENCE_THRESHOLD = 0.5
# Grouping-only blocking distance (rapidfuzz, see module docstring): 0 means
# only exact-decoded-plate groups are examined; the task spec calls for
# "near-identical plates" too, so the default allows a 1-character decoded
# mismatch to still land two trajectories in the same candidate group.
DEFAULT_NEAR_IDENTICAL_MAX_DISTANCE = 1


@dataclass
class PathPoint:
    event_id: str
    camera_id: str
    lat: float
    lon: float
    timestamp: datetime


@dataclass
class Alert:
    alert_id: str
    type: str  # "clone" | "impossible_travel" | "anomaly"
    severity: str  # "high" | "medium" | "low"
    created_at: datetime
    plate: str
    trajectory_ids: list[str]
    summary: str
    evidence: dict = field(default_factory=dict)


@dataclass
class CrossingCheck:
    """One examined adjacent crossing between two trajectories' merged
    timelines."""

    from_event: DetectionEvent
    to_event: DetectionEvent
    distance_m: float | None
    delta_t_s: float
    min_required_s: float | None
    physically_impossible: bool


def _mean_embedding(events: list[DetectionEvent]) -> np.ndarray:
    mat = np.array([e.embedding for e in events], dtype=np.float64)
    mean = mat.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def appearance_divergence(events_a: list[DetectionEvent], events_b: list[DetectionEvent]) -> float:
    """Cosine distance between the two event groups' mean embeddings, in
    [0, 2] (0 = identical mean appearance, 2 = diametrically opposed)."""
    ma, mb = _mean_embedding(events_a), _mean_embedding(events_b)
    cos_sim = float(np.clip(np.dot(ma, mb), -1.0, 1.0))
    return 1.0 - cos_sim


def _check_crossing(
    from_event: DetectionEvent,
    to_event: DetectionEvent,
    kinematic_model: KinematicModel,
    v_max_kmh: float,
) -> CrossingCheck:
    dt = (to_event.timestamp - from_event.timestamp).total_seconds()
    shortest = kinematic_model.shortest_path_info(from_event.camera_id, to_event.camera_id)
    if shortest is None:
        # Unreachable in the road graph (or same camera with no self-loop):
        # cannot compute a distance-based impossibility, so this crossing is
        # not evidence either way.
        return CrossingCheck(from_event, to_event, None, dt, None, False)
    distance_m = shortest[0]
    v_max_ms = v_max_kmh * 1000.0 / 3600.0
    min_required_s = distance_m / v_max_ms if v_max_ms > 0 else math.inf
    impossible = dt < min_required_s
    return CrossingCheck(from_event, to_event, distance_m, dt, min_required_s, impossible)


def find_crossings(
    events_a: list[DetectionEvent],
    events_b: list[DetectionEvent],
    kinematic_model: KinematicModel,
    v_max_kmh: float = DEFAULT_V_MAX_KMH,
) -> list[CrossingCheck]:
    """Merge two trajectories' events into one time-ordered timeline and
    return every ADJACENT crossing between an A-event and a B-event (in
    either direction). Covers both the simple "one ends before the other
    starts" case and genuinely interleaved observations (the
    `clone_route_overlap` hard case, sim/vehicles.py)."""
    tagged = [("a", e) for e in events_a] + [("b", e) for e in events_b]
    tagged.sort(key=lambda t: t[1].timestamp)
    crossings: list[CrossingCheck] = []
    for (tag_i, e_i), (tag_j, e_j) in zip(tagged[:-1], tagged[1:], strict=True):
        if tag_i == tag_j:
            continue
        crossings.append(_check_crossing(e_i, e_j, kinematic_model, v_max_kmh))
    return crossings


def _group_trajectories_by_plate(
    trajectories: list[Trajectory], near_identical_max_distance: int
) -> list[list[Trajectory]]:
    """Union-find clustering on decoded plate, using rapidfuzz Levenshtein
    distance purely as a candidate-blocking heuristic (module docstring) --
    never the basis of the actual clone decision."""
    n = len(trajectories)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    plates = [t.decoded_plate for t in trajectories]
    for i in range(n):
        for j in range(i + 1, n):
            if Levenshtein.distance(plates[i], plates[j]) <= near_identical_max_distance:
                union(i, j)

    groups: dict[int, list[Trajectory]] = {}
    for i, traj in enumerate(trajectories):
        groups.setdefault(find(i), []).append(traj)
    return [g for g in groups.values() if len(g) >= 2]


def detect_clones(
    trajectories: list[Trajectory],
    events_by_id: dict[str, DetectionEvent],
    kinematic_model: KinematicModel,
    city: CityConfig,
    v_max_kmh: float = DEFAULT_V_MAX_KMH,
    appearance_divergence_threshold: float = APPEARANCE_DIVERGENCE_THRESHOLD,
    near_identical_max_distance: int = DEFAULT_NEAR_IDENTICAL_MAX_DISTANCE,
    reference_time: datetime | None = None,
) -> list[Alert]:
    """Group trajectories by (near-identical) decoded plate and flag pairs
    with a physically-impossible crossing. `type="clone"` when appearance
    also diverges (both channels agree it's two different vehicles);
    `type="impossible_travel"` when the timing is impossible but appearance
    doesn't clearly confirm two distinct vehicles (still worth a human
    look, lower severity). A same-plate pair with no impossible crossing is
    NOT alerted -- consistent with architecture.md's own trigger condition
    ("same plate + physically impossible travel time + divergent
    appearance"); feasible-timing same-plate trajectories are far more
    likely a fragmented single vehicle than a clone, which is exactly what
    Day 3's association is supposed to have already merged, not this
    detector's job to flag."""
    del city  # kept in the signature for API symmetry with other analytics
    alerts: list[Alert] = []
    groups = _group_trajectories_by_plate(trajectories, near_identical_max_distance)
    now = reference_time or datetime.now()

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                traj_a, traj_b = group[i], group[j]
                events_a = [events_by_id[eid] for eid in traj_a.event_ids]
                events_b = [events_by_id[eid] for eid in traj_b.event_ids]

                crossings = find_crossings(events_a, events_b, kinematic_model, v_max_kmh)
                impossible = [c for c in crossings if c.physically_impossible]
                if not impossible:
                    continue

                worst = min(
                    impossible,
                    key=lambda c: (c.delta_t_s - c.min_required_s)
                    if c.min_required_s is not None
                    else 0.0,
                )
                appearance_dist = appearance_divergence(events_a, events_b)
                is_clone = appearance_dist >= appearance_divergence_threshold

                plate = traj_a.decoded_plate
                points = [
                    PathPoint(
                        event_id=worst.from_event.event_id,
                        camera_id=worst.from_event.camera_id,
                        lat=0.0,
                        lon=0.0,
                        timestamp=worst.from_event.timestamp,
                    ),
                    PathPoint(
                        event_id=worst.to_event.event_id,
                        camera_id=worst.to_event.camera_id,
                        lat=0.0,
                        lon=0.0,
                        timestamp=worst.to_event.timestamp,
                    ),
                ]
                alert_type = "clone" if is_clone else "impossible_travel"
                severity = "high" if is_clone else "medium"
                summary = (
                    f"Plate {plate}: trajectories {traj_a.trajectory_id} and "
                    f"{traj_b.trajectory_id} require crossing "
                    f"{worst.from_event.camera_id}->{worst.to_event.camera_id} in "
                    f"{worst.delta_t_s:.0f}s, needs >= {worst.min_required_s:.0f}s at "
                    f"{v_max_kmh:.0f} km/h"
                    + (
                        f"; appearance distance {appearance_dist:.3f} indicates a distinct vehicle"
                        if is_clone
                        else "; appearance evidence inconclusive"
                    )
                )
                alerts.append(
                    Alert(
                        alert_id=f"alert_{traj_a.trajectory_id}_{traj_b.trajectory_id}",
                        type=alert_type,
                        severity=severity,
                        created_at=now,
                        plate=plate,
                        trajectory_ids=[traj_a.trajectory_id, traj_b.trajectory_id],
                        summary=summary,
                        evidence={
                            "distance_m": worst.distance_m,
                            "delta_t_s": worst.delta_t_s,
                            "min_required_s": worst.min_required_s,
                            "appearance_distance": appearance_dist,
                            "points": points,
                        },
                    )
                )

    return alerts


__all__ = [
    "APPEARANCE_DIVERGENCE_THRESHOLD",
    "DEFAULT_NEAR_IDENTICAL_MAX_DISTANCE",
    "DEFAULT_V_MAX_KMH",
    "Alert",
    "CrossingCheck",
    "PathPoint",
    "appearance_divergence",
    "detect_clones",
    "find_crossings",
]
