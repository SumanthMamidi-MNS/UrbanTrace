"""Kinematic likelihood ratio (architecture.md §3, "Kinematic term").

Per camera-pair, per time-of-day bucket, we learn p(dt) as a log-normal —
travel times are strictly positive and right-skewed (most trips take close
to the free-flow time, a long tail takes much longer). `H0` is a flat
density over the outer gate window: with no information at all, any transit
time in the window is equally (im)plausible.

    log_lr = log p(dt | H1) - log p(dt | H0) + n_skipped * log(p_miss)

Hard gate: `dt < dist / v_max` is not just improbable, it is physically
impossible — that candidate pair gets `log_lr = -inf` and a
`physically_impossible` flag (Day 3's clone detector keys off exactly this:
same plate + physically impossible transit + divergent appearance is a
clone alert). This is the one place in the whole scoring stack where -inf is
the *correct* answer, unlike the plate channel (docs/decisions.md, "Day 1c"),
because physical impossibility is not evidence under a noisy channel — it is
a hard constraint.

Sparse pairs (few or no training observations for a given camera-pair +
time-of-day bucket) are never left unmodelled: they shrink toward a
road-graph estimate (shortest-path travel time at the posted speed limit)
via a simple empirical-Bayes blend, weighted by how many real observations
exist. A pair with zero observations falls back to the road-graph estimate
entirely.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx
import numpy as np

from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent

TOD_BUCKETS = ["night", "early", "am_rush", "midday", "pm_rush", "evening"]

DEFAULT_V_MAX_KMH = 120.0
DEFAULT_P_MISS = 0.05
MIN_SIGMA = 0.05
# How many real observations a (camera-pair, tod) bucket needs before the
# road-graph prior stops mattering much (empirical-Bayes pseudo-count).
SHRINKAGE_PSEUDOCOUNT = 5.0
# Outer gate window for the H0 uniform density. Generous on purpose: the
# real spatio-temporal gate (Day 3's gating.py) is what actually excludes
# implausible pairs before they reach scoring; this is just H0's density.
DEFAULT_DT_MIN_S = 0.0
DEFAULT_DT_MAX_S = 2 * 3600.0

# Chunk size for batched pair scoring, in PAIRS -- same rationale as
# `engine.scoring.plate_lr.PAIR_CHUNK_SIZE`. Unlike the plate/appearance
# channels, `score_batch`'s own per-pair gathers (`dense.reachable[cam_a,
# cam_b]` etc.) are already O(n_pairs) scalars, not O(n_pairs * width) --
# camera-indexed dense arrays are at most `n_cameras^2 * len(TOD_BUCKETS)`
# entries (a few thousand), never wide per-pair rows. Chunking here is for
# uniformity with the other two channels and to bound the per-pair Python
# list comprehension building `skipped_cameras`, not because this channel
# alone reproduces the plate channel's multi-GiB blowup.
PAIR_CHUNK_SIZE = 200_000


def time_of_day_bucket(ts: datetime) -> str:
    """Six buckets, chosen to bracket the simulator's morning/evening demand
    peaks (sim/vehicles._demand_weight bumps at 8.5h and 18.0h) inside their
    own rush-hour bucket."""
    h = ts.hour
    if h < 5:
        return "night"
    if h < 7:
        return "early"
    if h < 10:
        return "am_rush"
    if h < 16:
        return "midday"
    if h < 19:
        return "pm_rush"
    return "evening"


@dataclass
class PairParams:
    mu: float
    sigma: float
    n_obs: int


@dataclass
class KinematicLRResult:
    log_lr: float
    physically_impossible: bool
    skipped_cameras: list[str]
    expected_t_s: float
    delta_t_s: float


def _lognormal_logpdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0:
        return float("-inf")
    return -math.log(x * sigma * math.sqrt(2 * math.pi)) - ((math.log(x) - mu) ** 2) / (
        2 * sigma * sigma
    )


@dataclass
class KinematicModel:
    city: CityConfig
    pair_params: dict[tuple[str, str, str], PairParams]
    v_max_kmh: float = DEFAULT_V_MAX_KMH
    p_miss: float = DEFAULT_P_MISS
    default_sigma: float = 0.4
    dt_min: float = DEFAULT_DT_MIN_S
    dt_max: float = DEFAULT_DT_MAX_S
    graph: nx.DiGraph = field(init=False, repr=False)
    node_by_camera: dict[str, str] = field(init=False, repr=False)
    camera_by_node: dict[str, str] = field(init=False, repr=False)
    _path_cache: dict[tuple[str, str], tuple[float, float, list[str]] | None] = field(
        init=False, repr=False, default_factory=dict
    )
    _dense: "KinematicDenseArrays | None" = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.graph = self.city.to_digraph()
        self.node_by_camera = {c.camera_id: c.node_id for c in self.city.cameras}
        self.camera_by_node = {c.node_id: c.camera_id for c in self.city.cameras}

    def _shortest(self, cam_i: str, cam_j: str) -> tuple[float, float, list[str]] | None:
        """(distance_m, travel_time_s_at_speed_limit, intermediate_camera_ids)
        for the shortest path cam_i -> cam_j, or None if unreachable."""
        key = (cam_i, cam_j)
        if key in self._path_cache:
            return self._path_cache[key]
        result = None
        n_i, n_j = self.node_by_camera.get(cam_i), self.node_by_camera.get(cam_j)
        if n_i is not None and n_j is not None:
            try:
                path = nx.shortest_path(self.graph, n_i, n_j, weight="weight")
                dist_m = nx.path_weight(self.graph, path, weight="length_m")
                time_s = nx.path_weight(self.graph, path, weight="weight")
                intermediate = [
                    self.camera_by_node[n] for n in path[1:-1] if n in self.camera_by_node
                ]
                result = (dist_m, time_s, intermediate)
            except nx.NetworkXNoPath:
                result = None
        self._path_cache[key] = result
        return result

    def _road_graph_prior(self, cam_i: str, cam_j: str) -> tuple[float, float]:
        shortest = self._shortest(cam_i, cam_j)
        if shortest is None:
            # Unreachable in the graph: still must never be "unmodelled" —
            # fall back to a wide, deliberately uninformative estimate.
            return math.log(max(self.dt_max / 4, 1.0)), self.default_sigma * 2
        _, time_s, _ = shortest
        return math.log(max(time_s, 1.0)), self.default_sigma

    def shortest_path_info(self, cam_i: str, cam_j: str) -> tuple[float, float, list[str]] | None:
        """Public wrapper around the cached shortest-path lookup: (distance_m,
        travel_time_s_at_speed_limit, intermediate_camera_ids), or None if
        cam_j is unreachable from cam_i. Used by gating.py to determine
        upstream-reachable predecessor cameras without duplicating the road-
        graph traversal / caching logic."""
        return self._shortest(cam_i, cam_j)

    def params_for(self, cam_i: str, cam_j: str, tod: str) -> PairParams:
        fitted = self.pair_params.get((cam_i, cam_j, tod))
        prior_mu, prior_sigma = self._road_graph_prior(cam_i, cam_j)
        if fitted is None:
            return PairParams(mu=prior_mu, sigma=prior_sigma, n_obs=0)
        k0 = SHRINKAGE_PSEUDOCOUNT
        n = fitted.n_obs
        mu = (n * fitted.mu + k0 * prior_mu) / (n + k0)
        sigma = (n * fitted.sigma + k0 * prior_sigma) / (n + k0)
        return PairParams(mu=mu, sigma=max(sigma, MIN_SIGMA), n_obs=n)

    def score(self, cam_i: str, t_i: datetime, cam_j: str, t_j: datetime) -> KinematicLRResult:
        dt = (t_j - t_i).total_seconds()
        shortest = self._shortest(cam_i, cam_j)
        skipped = list(shortest[2]) if shortest else []
        expected_fallback = shortest[1] if shortest else 0.0

        if shortest is not None:
            v_max_ms = self.v_max_kmh * 1000.0 / 3600.0
            min_possible_dt = shortest[0] / v_max_ms
            if dt < min_possible_dt:
                return KinematicLRResult(
                    log_lr=float("-inf"),
                    physically_impossible=True,
                    skipped_cameras=skipped,
                    expected_t_s=expected_fallback,
                    delta_t_s=dt,
                )
        if dt <= 0:
            return KinematicLRResult(
                log_lr=float("-inf"),
                physically_impossible=True,
                skipped_cameras=skipped,
                expected_t_s=expected_fallback,
                delta_t_s=dt,
            )

        tod = time_of_day_bucket(t_i)
        params = self.params_for(cam_i, cam_j, tod)

        log_h1 = _lognormal_logpdf(dt, params.mu, params.sigma)
        log_h0 = -math.log(self.dt_max - self.dt_min)
        log_lr = log_h1 - log_h0 + len(skipped) * math.log(self.p_miss)

        return KinematicLRResult(
            log_lr=log_lr,
            physically_impossible=False,
            skipped_cameras=skipped,
            expected_t_s=math.exp(params.mu),
            delta_t_s=dt,
        )

    # -----------------------------------------------------------------
    # Batched scoring (Day 4 performance pass).
    #
    # `score()`'s per-pair work is dominated by road-graph shortest-path
    # lookups and per-tod-bucket shrinkage blending -- both are actually
    # properties of a (camera, camera[, tod]) TRIPLE, of which there are at
    # most `n_cameras^2 * 6` in the whole city (<=50 cameras -> <=15000),
    # vastly fewer than the number of scored pairs. `dense_arrays()` builds
    # every such triple once as small integer-indexed numpy arrays; scoring
    # a batch of pairs then becomes a gather (fancy-index by camera-index
    # pairs) plus one vectorised log-pdf evaluation, no further shortest-path
    # or shrinkage work at all.
    # -----------------------------------------------------------------

    def dense_arrays(self) -> "KinematicDenseArrays":
        if self._dense is None:
            self._dense = self._build_dense_arrays()
        return self._dense

    def _build_dense_arrays(self) -> "KinematicDenseArrays":
        cameras = sorted(self.node_by_camera.keys())
        cam_index = {c: i for i, c in enumerate(cameras)}
        n = len(cameras)

        dist_m = np.zeros((n, n))
        shortest_time_s = np.zeros((n, n))
        n_skipped = np.zeros((n, n), dtype=np.int64)
        reachable = np.zeros((n, n), dtype=bool)
        skipped_lists: list[list[list[str]]] = [[[] for _ in range(n)] for _ in range(n)]
        mu = np.zeros((len(TOD_BUCKETS), n, n))
        sigma = np.full((len(TOD_BUCKETS), n, n), self.default_sigma)

        for i, cam_i in enumerate(cameras):
            for j, cam_j in enumerate(cameras):
                shortest = self._shortest(cam_i, cam_j)
                if shortest is not None:
                    d, t, intermediate = shortest
                    dist_m[i, j] = d
                    shortest_time_s[i, j] = t
                    n_skipped[i, j] = len(intermediate)
                    reachable[i, j] = True
                    skipped_lists[i][j] = list(intermediate)
                for k, tod in enumerate(TOD_BUCKETS):
                    params = self.params_for(cam_i, cam_j, tod)
                    mu[k, i, j] = params.mu
                    sigma[k, i, j] = params.sigma

        v_max_ms = self.v_max_kmh * 1000.0 / 3600.0
        return KinematicDenseArrays(
            cam_index=cam_index,
            dist_m=dist_m,
            shortest_time_s=shortest_time_s,
            n_skipped=n_skipped,
            reachable=reachable,
            skipped_lists=skipped_lists,
            mu=mu,
            sigma=sigma,
            v_max_ms=v_max_ms,
        )

    def _score_batch_chunk(
        self,
        dense: "KinematicDenseArrays",
        cam_idx: np.ndarray,
        epoch_s: np.ndarray,
        tod_idx: np.ndarray,
        pred_idx: np.ndarray,
        succ_idx: np.ndarray,
    ) -> "KinematicBatchResult":
        """Core per-chunk math, unchanged from the original (pre-chunking)
        `score_batch` body. Called by `score_batch` on slices of at most
        `PAIR_CHUNK_SIZE` pairs."""
        p = len(pred_idx)
        cam_a = cam_idx[pred_idx]
        cam_b = cam_idx[succ_idx]
        dt = epoch_s[succ_idx] - epoch_s[pred_idx]
        tod_a = tod_idx[pred_idx]

        reach = dense.reachable[cam_a, cam_b]
        dist = dense.dist_m[cam_a, cam_b]
        stime = dense.shortest_time_s[cam_a, cam_b]
        nskip = dense.n_skipped[cam_a, cam_b]
        mu = dense.mu[tod_a, cam_a, cam_b]
        sigma = dense.sigma[tod_a, cam_a, cam_b]

        min_possible_dt = dist / dense.v_max_ms
        physically_impossible = (dt <= 0) | (reach & (dt < min_possible_dt))

        safe_dt = np.where(dt > 0, dt, 1.0)
        log_h1 = -np.log(safe_dt * sigma * math.sqrt(2 * math.pi)) - (
            (np.log(safe_dt) - mu) ** 2
        ) / (2 * sigma * sigma)
        log_h1 = np.where(dt > 0, log_h1, -np.inf)
        log_h0 = -math.log(self.dt_max - self.dt_min)
        log_lr = log_h1 - log_h0 + nskip * math.log(self.p_miss)
        log_lr = np.where(physically_impossible, -np.inf, log_lr)

        expected_fallback = np.where(reach, stime, 0.0)
        expected_t_s = np.where(physically_impossible, expected_fallback, np.exp(mu))

        skipped_cameras = [dense.skipped_lists[int(cam_a[k])][int(cam_b[k])] for k in range(p)]

        return KinematicBatchResult(
            log_lr=log_lr,
            physically_impossible=physically_impossible,
            delta_t_s=dt,
            expected_t_s=expected_t_s,
            skipped_cameras=skipped_cameras,
        )

    def score_batch(
        self,
        cam_idx: np.ndarray,
        epoch_s: np.ndarray,
        tod_idx: np.ndarray,
        pred_idx: np.ndarray,
        succ_idx: np.ndarray,
    ) -> "KinematicBatchResult":
        """Batched kinematic LR. `cam_idx`/`epoch_s`/`tod_idx` are per-EVENT
        arrays (see `precompute_event_kinematic_arrays`); `pred_idx`/
        `succ_idx` index into them, one entry per pair. Agrees with
        `score()` to float precision (tests/test_batch_scoring.py).

        Processes pairs in chunks of `PAIR_CHUNK_SIZE` (see its docstring in
        this module) for consistency with the plate/appearance channels'
        chunking and to keep the per-pair Python-level `skipped_cameras`
        list construction from holding the whole (potentially millions-long)
        intermediate list live at once; each pair's math is independent of
        every other pair's, so results are numerically identical to scoring
        everything in one shot."""
        pred_idx = np.asarray(pred_idx, dtype=np.int64)
        succ_idx = np.asarray(succ_idx, dtype=np.int64)
        p = len(pred_idx)
        if p == 0:
            return KinematicBatchResult(
                log_lr=np.zeros(0),
                physically_impossible=np.zeros(0, dtype=bool),
                delta_t_s=np.zeros(0),
                expected_t_s=np.zeros(0),
                skipped_cameras=[],
            )

        dense = self.dense_arrays()
        log_lr = np.empty(p, dtype=np.float64)
        physically_impossible = np.empty(p, dtype=bool)
        delta_t_s = np.empty(p, dtype=np.float64)
        expected_t_s = np.empty(p, dtype=np.float64)
        skipped_cameras: list[list[str]] = [[] for _ in range(p)]

        for start in range(0, p, PAIR_CHUNK_SIZE):
            end = min(start + PAIR_CHUNK_SIZE, p)
            chunk = self._score_batch_chunk(
                dense, cam_idx, epoch_s, tod_idx, pred_idx[start:end], succ_idx[start:end]
            )
            log_lr[start:end] = chunk.log_lr
            physically_impossible[start:end] = chunk.physically_impossible
            delta_t_s[start:end] = chunk.delta_t_s
            expected_t_s[start:end] = chunk.expected_t_s
            skipped_cameras[start:end] = chunk.skipped_cameras

        return KinematicBatchResult(
            log_lr=log_lr,
            physically_impossible=physically_impossible,
            delta_t_s=delta_t_s,
            expected_t_s=expected_t_s,
            skipped_cameras=skipped_cameras,
        )


@dataclass
class KinematicDenseArrays:
    """Integer-indexed (camera x camera[ x tod]) lookup tables built once per
    `KinematicModel` and cached via `KinematicModel.dense_arrays()`."""

    cam_index: dict[str, int]
    dist_m: np.ndarray  # (n_cams, n_cams)
    shortest_time_s: np.ndarray  # (n_cams, n_cams)
    n_skipped: np.ndarray  # (n_cams, n_cams) int
    reachable: np.ndarray  # (n_cams, n_cams) bool
    skipped_lists: list[list[list[str]]]  # [i][j] -> intermediate camera ids
    mu: np.ndarray  # (n_tod, n_cams, n_cams)
    sigma: np.ndarray  # (n_tod, n_cams, n_cams)
    v_max_ms: float


@dataclass
class KinematicBatchResult:
    log_lr: np.ndarray
    physically_impossible: np.ndarray
    delta_t_s: np.ndarray
    expected_t_s: np.ndarray
    skipped_cameras: list[list[str]]


def precompute_event_kinematic_arrays(
    events: list[DetectionEvent], model: KinematicModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-event `(cam_idx, epoch_s, tod_idx)` arrays for `score_batch`, built
    against `model`'s (cached) camera index. Events at a camera the model's
    city doesn't know about are not supported by the kinematic channel and
    will raise -- gating/scoring never produces such pairs in practice since
    every event's camera comes from the same city the model was fit on.

    `epoch_s` is seconds relative to the EARLIEST event in `events`, not raw
    Unix epoch seconds: raw epoch floats (~1.7e9) leave float64 with only
    ~4e-7s of absolute precision, and differencing two such floats to get a
    pair's `dt` can then disagree with the scalar path's exact
    `(t_j - t_i).total_seconds()` (computed from Python `datetime`/
    `timedelta`, which has microsecond precision throughout) by enough to
    fail the 1e-6 log-LR agreement test. Batch dt values only ever need to
    be internally consistent across one `events` list, so shrinking the
    reference point removes the cancellation error instead of just hoping
    it stays small."""
    dense = model.dense_arrays()
    tod_pos = {t: i for i, t in enumerate(TOD_BUCKETS)}
    n = len(events)
    cam_idx = np.fromiter((dense.cam_index[e.camera_id] for e in events), dtype=np.int64, count=n)
    reference = min(e.timestamp for e in events) if events else None
    epoch_s = np.fromiter(
        ((e.timestamp - reference).total_seconds() for e in events), dtype=np.float64, count=n
    )
    tod_idx = np.fromiter(
        (tod_pos[time_of_day_bucket(e.timestamp)] for e in events), dtype=np.int64, count=n
    )
    return cam_idx, epoch_s, tod_idx


def collect_travel_time_observations(
    events: list[DetectionEvent],
) -> list[tuple[str, str, float, datetime]]:
    """Group events by ground-truth vehicle and extract consecutive
    camera-to-camera travel times, for fitting on a training split. Stands
    in for what a real deployment would learn from historical, confidently-
    linked trajectories — the simulator's ground truth plays that role here."""
    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in events:
        if e.gt_vehicle_id is None:
            continue
        by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    observations: list[tuple[str, str, float, datetime]] = []
    for evs in by_vehicle.values():
        evs = sorted(evs, key=lambda e: e.timestamp)
        for a, b in zip(evs[:-1], evs[1:], strict=True):
            dt = (b.timestamp - a.timestamp).total_seconds()
            if dt > 0:
                observations.append((a.camera_id, b.camera_id, dt, a.timestamp))
    return observations


def fit_kinematic_model(
    events: list[DetectionEvent],
    city: CityConfig,
    v_max_kmh: float = DEFAULT_V_MAX_KMH,
    p_miss: float = DEFAULT_P_MISS,
) -> KinematicModel:
    observations = collect_travel_time_observations(events)
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for cam_i, cam_j, dt, t_i in observations:
        grouped.setdefault((cam_i, cam_j, time_of_day_bucket(t_i)), []).append(dt)

    pair_params: dict[tuple[str, str, str], PairParams] = {}
    singleton_keys: list[tuple[str, str, str]] = []
    all_sigmas: list[float] = []
    for key, dts in grouped.items():
        logs = [math.log(dt) for dt in dts]
        mu = sum(logs) / len(logs)
        if len(logs) > 1:
            var = sum((x - mu) ** 2 for x in logs) / (len(logs) - 1)
            sigma = max(var**0.5, MIN_SIGMA)
            all_sigmas.append(sigma)
        else:
            sigma = MIN_SIGMA  # placeholder, backfilled with the global average below
            singleton_keys.append(key)
        pair_params[key] = PairParams(mu=mu, sigma=sigma, n_obs=len(logs))

    default_sigma = sum(all_sigmas) / len(all_sigmas) if all_sigmas else 0.4
    for key in singleton_keys:
        p = pair_params[key]
        pair_params[key] = PairParams(mu=p.mu, sigma=default_sigma, n_obs=p.n_obs)

    return KinematicModel(
        city=city,
        pair_params=pair_params,
        v_max_kmh=v_max_kmh,
        p_miss=p_miss,
        default_sigma=default_sigma,
    )
