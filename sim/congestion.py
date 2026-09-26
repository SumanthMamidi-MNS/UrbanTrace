"""Density-dependent link travel times via the standard BPR (Bureau of
Public Roads) link performance function:

    t_edge = t_free * (1 + alpha * (v / c) ** beta)

`v` is the volume entering the edge in a time bucket, expressed as an
hourly rate; `c` is the edge's capacity, set by road class (arterial/ring
vs. local grid street). See `sim/vehicles.py`'s two-pass scheme (pass 1:
route + traverse at free flow to collect per-(edge, bucket) volumes; pass
2: recompute travel times with these volumes and re-traverse) and
docs/decisions.md for how the capacities below were calibrated.

Only two passes are used, not an iterative MSA loop to convergence:
routes are chosen once (via `_route_with_noise`) before congestion is
known and are never re-optimized against congested times in this
simulator (no dynamic rerouting), so a vehicle's set of traversed edges
cannot change between passes -- a third pass would recompute the exact
same volumes and therefore the exact same congested times. Two passes is
exactly what the feedback loop needs.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx

# 5-minute buckets, matching the task spec's example bucket width.
BUCKET_MINUTES = 5

BPR_ALPHA = 0.15
BPR_BETA = 4

# Capacities in vehicles/hour, keyed by the edge's speed_limit_kmh (the
# only three values `sim/city.py` ever emits). These are single-direction
# per-edge throughputs, not per-lane -- `city.py` already models each
# direction as its own edge, so "capacity" here is that direction's whole
# carriageway.
#
# ARTERIAL capacity is calibrated empirically (see docs/decisions.md, "Sim
# congestion calibration") against this simulator's actual free-flow
# volumes at run1 scale (50 cameras / 20,000 vehicles / 24h), measured on
# a tuning seed distinct from the run1/seed-42 evaluation day (seed 777,
# never the evaluation seed). Every journey in this simulator runs
# border-camera to border-camera (`generate_vehicles_and_journeys`), and
# `sim/city.py`'s ring road makes every boundary-to-boundary edge
# arterial-speed, so the shortest-weighted-path router puts almost all
# cross-city volume onto the 80 km/h class: measured busiest-corridor
# hourly rates at this scale peak at ~288-408 veh/h in the 07:00-09:00 and
# 17:00-19:00 windows and fall to ~96-144 veh/h overnight.
#
# The first calibration (320 veh/h) targeted edge-level v/c ~0.9-1.3, but
# ANPR only observes CAMERA-TO-CAMERA travel times, and a camera pair's
# route crosses a bottleneck edge for only part of its length, so 320 gave
# a corridor-level slowdown of just 1-3% at peak (measured: worst corridor
# 0.97 of free-flow speed) -- nothing for the congestion analytics to find.
# Re-calibrated by the lead against the OBSERVABLE (camera-pair speed /
# free-flow speed) and pinned to typical Indian-metro peak behaviour
# (bottleneck corridors ~40-60% of free-flow speed, typical links ~80-90%,
# nights ~free flow), full scale (50 cams / 20,000 vehicles / 24h), tuning
# seed 777 only:
#     capacity  peak median  peak p5 (bottlenecks)  night median
#        240        0.96            0.78               1.00
#        180        0.92            0.63               1.00
#        140        0.84            0.42               0.99   <- chosen
#        110        0.71            0.23               0.99
#
# LOCAL_FAST/LOCAL_SLOW are NOT empirically tuned the same way: at this
# city scale, with journeys always sampled border-to-border and the ring
# road covering the whole boundary, measured 30/50 km/h volume was
# ~zero -- through-traffic never needs the grid interior when a
# same-speed-or-faster ring path always exists. They are sized by simple
# lane-capacity reasoning instead (a real ~2-3 effective-lane arterial
# carriageway vs. an ordinary two-way street vs. the narrowest
# single-lane-equivalent residential/grid infill), documented honestly as
# reasoned rather than measured, and kept so that a future change to trip
# generation (e.g. local/short trips, not just border-to-border) has a
# sane default to congest against without needing this constant
# rediscovered from scratch.
DEFAULT_CAPACITY_LOCAL_SLOW_VEH_PER_HOUR = 60.0
DEFAULT_CAPACITY_LOCAL_FAST_VEH_PER_HOUR = 130.0
DEFAULT_CAPACITY_ARTERIAL_VEH_PER_HOUR = 140.0

# Fallback for any speed limit that doesn't match one of the three known
# classes exactly (shouldn't happen with `sim/city.py` as written, but a
# missing-key crash would be a worse failure mode than a reasonable
# default).
DEFAULT_CAPACITY_FALLBACK_VEH_PER_HOUR = 300.0


@dataclass
class CongestionConfig:
    """Knobs for the BPR congestion model. All defaults are the calibrated
    values described above; `enabled` is the `congestion` bool knob itself,
    duplicated here so a single `CongestionConfig` instance is enough to
    describe a run (also exposed as the standalone `congestion: bool` param
    on `generate_vehicles_and_journeys` / `generate_dataset*` / the CLI, for
    callers that don't want to construct a config)."""

    enabled: bool = True
    alpha: float = BPR_ALPHA
    beta: float = BPR_BETA
    bucket_minutes: int = BUCKET_MINUTES
    capacity_by_speed_kmh: dict[float, float] = field(
        default_factory=lambda: {
            30.0: DEFAULT_CAPACITY_LOCAL_SLOW_VEH_PER_HOUR,
            50.0: DEFAULT_CAPACITY_LOCAL_FAST_VEH_PER_HOUR,
            80.0: DEFAULT_CAPACITY_ARTERIAL_VEH_PER_HOUR,
        }
    )
    capacity_fallback: float = DEFAULT_CAPACITY_FALLBACK_VEH_PER_HOUR


def congestion_config_to_dict(cfg: CongestionConfig) -> dict:
    """Serialize a `CongestionConfig` for `sim/generate.py`'s config.json, so
    a dataset's recorded config fully describes the BPR knobs it was
    generated with -- not just the `congestion` bool -- and a caller that
    needs to reproduce the SAME congestion behaviour for a companion dataset
    (e.g. `eval/run_pipeline.py`'s train split) can round-trip it via
    `congestion_config_from_dict` instead of guessing defaults. JSON object
    keys must be strings, so `capacity_by_speed_kmh`'s float keys are
    stringified here and parsed back in `congestion_config_from_dict`."""
    d = dataclasses.asdict(cfg)
    d["capacity_by_speed_kmh"] = {str(k): v for k, v in cfg.capacity_by_speed_kmh.items()}
    return d


def congestion_config_from_dict(d: dict) -> CongestionConfig:
    """Inverse of `congestion_config_to_dict`. Missing keys fall back to
    `CongestionConfig`'s own defaults (the calibrated values), so a partial
    or hand-written dict still produces a sane config."""
    defaults = CongestionConfig()
    raw_capacity = d.get("capacity_by_speed_kmh")
    capacity_by_speed_kmh = (
        {float(k): v for k, v in raw_capacity.items()}
        if raw_capacity
        else dict(defaults.capacity_by_speed_kmh)
    )
    return CongestionConfig(
        enabled=d.get("enabled", defaults.enabled),
        alpha=d.get("alpha", defaults.alpha),
        beta=d.get("beta", defaults.beta),
        bucket_minutes=d.get("bucket_minutes", defaults.bucket_minutes),
        capacity_by_speed_kmh=capacity_by_speed_kmh,
        capacity_fallback=d.get("capacity_fallback", defaults.capacity_fallback),
    )


# (from_node, to_node, bucket_index) -> count of vehicles entering that
# edge in that bucket, during the free-flow volume-collection pass.
VolumeCounts = dict[tuple[str, str, int], int]


def bucket_index(t: datetime, epoch: datetime, bucket_minutes: int = BUCKET_MINUTES) -> int:
    """Which fixed-width bucket (absolute, not time-of-day-wrapped) `t`
    falls into, counted from `epoch`."""
    return int((t - epoch).total_seconds() // (bucket_minutes * 60))


def record_volume(
    sink: VolumeCounts,
    u: str,
    v: str,
    entry_time: datetime,
    epoch: datetime,
    bucket_minutes: int = BUCKET_MINUTES,
) -> None:
    """Record one vehicle entering edge (u, v) at `entry_time`, during a
    free-flow pass-1 traversal."""
    key = (u, v, bucket_index(entry_time, epoch, bucket_minutes))
    sink[key] = sink.get(key, 0) + 1


def capacity_for_speed(speed_kmh: float, cfg: CongestionConfig) -> float:
    return cfg.capacity_by_speed_kmh.get(speed_kmh, cfg.capacity_fallback)


def bpr_multiplier(hourly_rate: float, capacity: float, cfg: CongestionConfig) -> float:
    """The BPR `1 + alpha * (v/c)**beta` factor travel time at free flow is
    multiplied by."""
    if capacity <= 0:
        return 1.0
    ratio = hourly_rate / capacity
    return 1.0 + cfg.alpha * (ratio**cfg.beta)


def make_congestion_fn(
    graph: nx.DiGraph,
    volume_counts: VolumeCounts,
    epoch: datetime,
    cfg: CongestionConfig,
) -> Callable[[str, str, datetime], float]:
    """Build the pass-2 `edge_time_fn(u, v, entry_time) -> bpr_multiplier`
    closure from pass-1's collected volumes."""
    bucket_minutes = cfg.bucket_minutes
    buckets_per_hour = 60.0 / bucket_minutes

    def edge_time_fn(u: str, v: str, entry_time: datetime) -> float:
        bucket = bucket_index(entry_time, epoch, bucket_minutes)
        count = volume_counts.get((u, v, bucket), 0)
        hourly_rate = count * buckets_per_hour
        edge_data = graph.get_edge_data(u, v)
        speed_kmh = edge_data["speed_limit_kmh"] if edge_data else 0.0
        capacity = capacity_for_speed(speed_kmh, cfg)
        return bpr_multiplier(hourly_rate, capacity, cfg)

    return edge_time_fn
