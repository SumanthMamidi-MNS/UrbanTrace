"""Primary candidate-predecessor generation (architecture.md §3, §7).

This is THE primary blocker and it decides the recall ceiling of the whole
system: "the spatio-temporal gate is the primary and sufficient blocker --
plate-based blocking is not" (architecture.md §7). If a true predecessor is
dropped here, nothing downstream (scoring, min-cost flow, consensus
decoding) can ever recover it -- the gate's recall requirement (>99%,
tests/test_gating.py) is load-bearing, not aspirational.

For every event `j`, a candidate predecessor `i` must satisfy:
  1. `t_i < t_j` (an earlier event);
  2. camera(i) can reach camera(j) in the road graph (there is a directed
     path, possibly through cameraless intermediate nodes, possibly through
     OTHER cameras where a passage was simply never detected);
  3. `dt = t_j - t_i` falls inside a per-(camera_i, camera_j, time-of-day)
     window derived from the fitted travel-time prior
     (engine.scoring.kinematic_lr): a wide multiple of fitted standard
     deviations around the fitted (or road-graph fallback) log-normal mean,
     widened further whenever the shortest path skips one or more
     intermediate cameras (the missed-detection case: real variance is
     higher when the vehicle wasn't seen at every camera along the way).
     The time-of-day bucket is read off the LATER event `t_j` (the one
     being searched from); since gate windows are seconds-to-minutes wide
     and tod buckets are hours wide, `t_i`'s true bucket only ever differs
     from `t_j`'s within a thin sliver near a bucket boundary -- handled by
     also unioning in the bucket that `t_j - dt_max` would fall into, not by
     unioning across all six buckets (which was tried first and produced
     mean candidate counts an order of magnitude too high, because
     unioning distinct traffic regimes -- e.g. am_rush with midday -- widens
     the window far more than the boundary-sliver case actually requires).

Two numbers this module MUST report (task spec): mean candidates per event,
and true-predecessor recall (fraction of real consecutive same-vehicle
event pairs where the earlier event is actually inside the later event's
candidate set) -- asserted > 99% in tests/test_gating.py.
"""

import bisect
import math
from dataclasses import dataclass, field

from engine.contracts.events import DetectionEvent
from engine.scoring.kinematic_lr import KinematicModel, time_of_day_bucket

# Gate window half-width, in fitted-log-normal standard deviations. A
# strict [1st, 99th] percentile window (z=2.326, scipy.stats.norm.ppf(0.99))
# is, BY DEFINITION, expected to exclude ~1-2% of the true distribution's
# mass even for a perfectly-fit model (roughly 1% below the 1st percentile,
# 1% above the 99th) -- nowhere near enough headroom to clear a >99%
# EMPIRICAL recall requirement once real per-(camera-pair, tod) fits are
# imperfect on top of that. z=3.0, together with `DEFAULT_MISS_WIDEN_FACTOR`
# below, was chosen by sweeping against data/run1 (production scale: 50
# cameras, 20000 vehicles, 24h) to land recall comfortably above 99% (see
# eval/gating_report.py) while keeping the secondary blocking budget
# (blocking.py, default 500 candidates) an OVERFLOW path in practice, not a
# routinely-triggered one (docs/decisions.md, "Day 3a").
GATE_Z_SCORE = 3.0

# Extra multiplicative widening applied to the z-score whenever the
# shortest path between two cameras skips one or more intermediate cameras
# -- the missed-detection case, where real travel-time variance is higher
# because the vehicle may have taken any of several sub-paths or paused
# somewhere along a longer, unobserved stretch.
DEFAULT_MISS_WIDEN_FACTOR = 1.3

# Absolute floor under a gate window's dt_min: the fitted/road-graph window
# can legitimately touch (or, before flooring, dip below) zero for very
# close camera pairs -- never let it go negative.
MIN_DT_FLOOR_S = 0.0

# Hard multiplicative floor on the gate's upper bound, expressed as a
# multiple of the pair's FREE-FLOW travel time (from
# KinematicModel.shortest_path_info, i.e. at the posted speed limit, no
# congestion). The gate is a recall device: architecture.md §7 says it may
# exclude only what is physically impossible, and congestion slowness is
# weighed as evidence downstream by the kinematic LR, not excluded here.
# exp(mu + z*sigma) alone tracks the FITTED distribution's spread, which on
# a congested day is only ~1.4-1.6x typical travel time near a bottleneck --
# nowhere near enough. At BPR alpha=0.15, beta=4, a volume/capacity ratio of
# 2 already gives a 3.4x travel-time multiplier over free flow, and this
# project's own calibrated bottleneck links run vehicles at ~0.23-0.42x
# free-flow speed (i.e. up to ~4.3x the free-flow time) at peak load (lead's
# measurement, run2 config). CONGESTION_TAIL_FACTOR=4.0 sits just under that
# worst case with a small margin, so a legitimately-jammed true predecessor
# is never excluded by the upper bound alone. This does NOT touch the lower
# (v_max / physically-impossible) bound -- clone detection depends on that
# staying tight.
CONGESTION_TAIL_FACTOR = 4.0


@dataclass(frozen=True)
class GateWindow:
    dt_min_s: float
    dt_max_s: float


@dataclass
class GateResult:
    # event_id -> candidate predecessor event_ids, in the order discovered
    # (sorted by ascending dt from the destination event).
    candidates: dict[str, list[str]]
    n_events: int
    n_candidate_pairs: int

    @property
    def mean_candidates_per_event(self) -> float:
        return self.n_candidate_pairs / self.n_events if self.n_events else 0.0


def _bucket_window(
    model: KinematicModel, cam_i: str, cam_j: str, tod: str, effective_z: float
) -> tuple[float, float]:
    params = model.params_for(cam_i, cam_j, tod)
    return params.mu - effective_z * params.sigma, params.mu + effective_z * params.sigma


def _pair_window(
    model: KinematicModel,
    cam_i: str,
    cam_j: str,
    tod: str,
    z: float,
    miss_widen_factor: float,
    congestion_tail_factor: float = CONGESTION_TAIL_FACTOR,
) -> GateWindow | None:
    """The gate window for camera_i -> camera_j given the LATER event's
    time-of-day bucket `tod`, widened further if the shortest path skips
    any intermediate camera. Also unions in the bucket that the window's
    own lower edge (`t_j - dt_max`) would land in, so a predecessor just
    across a tod boundary is never dropped -- without paying for a full
    union across all six buckets (see module docstring). Returns None if
    cam_j is unreachable from cam_i.

    Finally, the upper bound is floored at `congestion_tail_factor` times
    the pair's free-flow travel time (`shortest_path_info`'s full cam_i ->
    cam_j path, which already covers the skipped-camera case -- there is no
    separate "skipped path" free-flow time to look up). This is a hard
    floor, not a widening on top of the fitted/tod-union bound above: the
    gate must never exclude a merely-congested true predecessor just
    because the fitted distribution's own spread happens to be narrow (see
    module-level `CONGESTION_TAIL_FACTOR` docstring)."""
    shortest = model.shortest_path_info(cam_i, cam_j)
    if shortest is None:
        return None
    skipped = shortest[2]
    free_flow_time_s = shortest[1]
    effective_z = z * miss_widen_factor if skipped else z

    log_lo, log_hi = _bucket_window(model, cam_i, cam_j, tod, effective_z)

    # Check whether the window's earliest edge actually crosses into a
    # different tod bucket than `tod` itself; if so, union that bucket's
    # window in too (one extra lookup, not five).
    dt_max_estimate = math.exp(log_hi)
    boundary_tod = time_of_day_bucket_from_offset(tod, dt_max_estimate)
    if boundary_tod != tod:
        b_lo, b_hi = _bucket_window(model, cam_i, cam_j, boundary_tod, effective_z)
        log_lo = min(log_lo, b_lo)
        log_hi = max(log_hi, b_hi)

    if free_flow_time_s > 0:
        log_hi = max(log_hi, math.log(free_flow_time_s * congestion_tail_factor))

    dt_min_s = max(math.exp(log_lo), MIN_DT_FLOOR_S)
    dt_max_s = math.exp(log_hi)
    return GateWindow(dt_min_s=dt_min_s, dt_max_s=dt_max_s)


# Approximate bucket boundary hours (see engine.scoring.kinematic_lr.
# time_of_day_bucket) used only to decide whether a window's lower edge
# might cross into an adjacent tod bucket -- a cheap heuristic, not a
# re-implementation of the bucket function itself.
_TOD_BUCKET_START_HOUR = {
    "night": 0,
    "early": 5,
    "am_rush": 7,
    "midday": 10,
    "pm_rush": 16,
    "evening": 19,
}
_TOD_ORDER = ["night", "early", "am_rush", "midday", "pm_rush", "evening"]


def time_of_day_bucket_from_offset(tod: str, seconds_before: float) -> str:
    """If a window extends `seconds_before` seconds before the start of
    bucket `tod`, which bucket does that earlier instant fall into? Only
    ever asked with `seconds_before` on the order of a gate window's width
    (seconds to low tens of minutes), so at most one bucket boundary back
    is all that's ever needed."""
    start_hour = _TOD_BUCKET_START_HOUR[tod]
    hours_before = seconds_before / 3600.0
    # Bucket boundaries repeat every 24h; walk back one bucket if the
    # window reaches before this bucket's own start.
    if hours_before <= start_hour:
        return tod
    prev_idx = (_TOD_ORDER.index(tod) - 1) % len(_TOD_ORDER)
    return _TOD_ORDER[prev_idx]


@dataclass
class Gate:
    """Precomputes and caches the per-(camera_i, camera_j, tod) gate window
    so repeated calls (one per event) don't repeatedly re-walk the road
    graph or re-touch the kinematic model's per-tod-bucket params."""

    model: KinematicModel
    z: float = GATE_Z_SCORE
    miss_widen_factor: float = DEFAULT_MISS_WIDEN_FACTOR
    congestion_tail_factor: float = CONGESTION_TAIL_FACTOR
    _window_cache: dict[tuple[str, str, str], GateWindow | None] = field(
        default_factory=dict, init=False, repr=False
    )

    def window(self, cam_i: str, cam_j: str, tod: str) -> GateWindow | None:
        key = (cam_i, cam_j, tod)
        if key not in self._window_cache:
            self._window_cache[key] = _pair_window(
                self.model,
                cam_i,
                cam_j,
                tod,
                self.z,
                self.miss_widen_factor,
                self.congestion_tail_factor,
            )
        return self._window_cache[key]


def _index_by_camera(
    events: list[DetectionEvent],
) -> dict[str, tuple[list[float], list[str]]]:
    """camera_id -> (sorted epoch-second timestamps, parallel event_ids) for
    fast bisect-based windowed lookup. Epoch floats are precomputed once
    here rather than per (event, camera) pair in the hot loop below."""
    by_camera: dict[str, list[DetectionEvent]] = {}
    for e in events:
        by_camera.setdefault(e.camera_id, []).append(e)
    indexed: dict[str, tuple[list[float], list[str]]] = {}
    for cam, evs in by_camera.items():
        evs.sort(key=lambda e: e.timestamp)
        indexed[cam] = ([e.timestamp.timestamp() for e in evs], [e.event_id for e in evs])
    return indexed


def gate_candidates(
    events: list[DetectionEvent],
    gate: Gate,
    cameras: list[str] | None = None,
) -> GateResult:
    """For every event, find candidate predecessor events: earlier events at
    any camera that can reach this one in the road graph, within that
    camera-pair's gate window. `cameras` restricts the set of upstream
    cameras considered (defaults to every camera present in `events`) --
    mostly useful for tests on a small event set drawn from a larger city."""
    by_camera = _index_by_camera(events)
    all_cameras = cameras if cameras is not None else list(by_camera.keys())

    candidates: dict[str, list[str]] = {e.event_id: [] for e in events}
    n_pairs = 0

    for e_j in events:
        cam_j = e_j.camera_id
        t_j_epoch = e_j.timestamp.timestamp()
        tod_j = time_of_day_bucket(e_j.timestamp)
        found: list[tuple[float, str]] = []
        for cam_i in all_cameras:
            if cam_i not in by_camera:
                continue
            window = gate.window(cam_i, cam_j, tod_j)
            if window is None:
                continue
            lo = t_j_epoch - window.dt_max_s
            hi = t_j_epoch - window.dt_min_s
            if hi < lo:
                continue
            epoch_ts, event_ids = by_camera[cam_i]
            left = bisect.bisect_left(epoch_ts, lo)
            right = bisect.bisect_right(epoch_ts, hi)
            for k in range(left, right):
                cand_ts = epoch_ts[k]
                if cand_ts >= t_j_epoch:
                    continue
                cand_id = event_ids[k]
                if cand_id == e_j.event_id:
                    continue  # same event (only possible if cam_i == cam_j)
                found.append((t_j_epoch - cand_ts, cand_id))

        found.sort()
        candidates[e_j.event_id] = [eid for _, eid in found]
        n_pairs += len(found)

    return GateResult(candidates=candidates, n_events=len(events), n_candidate_pairs=n_pairs)


@dataclass
class RecallResult:
    n_true_pairs: int
    n_recalled: int

    @property
    def recall(self) -> float:
        return self.n_recalled / self.n_true_pairs if self.n_true_pairs else float("nan")


def true_predecessor_recall(
    events: list[DetectionEvent], gate_result: GateResult
) -> RecallResult:
    """Fraction of real consecutive same-ground-truth-vehicle event pairs
    whose earlier event is actually present in the later event's candidate
    set. This is the number the whole system's recall ceiling rests on."""
    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in events:
        if e.gt_vehicle_id is not None:
            by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    n_true = 0
    n_recalled = 0
    for evs in by_vehicle.values():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        for a, b in zip(evs_sorted[:-1], evs_sorted[1:], strict=True):
            n_true += 1
            if a.event_id in gate_result.candidates.get(b.event_id, []):
                n_recalled += 1

    return RecallResult(n_true_pairs=n_true, n_recalled=n_recalled)


__all__ = [
    "Gate",
    "GateResult",
    "GateWindow",
    "RecallResult",
    "GATE_Z_SCORE",
    "CONGESTION_TAIL_FACTOR",
    "gate_candidates",
    "true_predecessor_recall",
]
