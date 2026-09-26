"""City-wide flow trend, bucketed by time (PRD component 3: "macro traffic
flow and movement analytics"; docs/api-contract.md Contract v2,
`GET /api/analytics/flow_trend`). Mirrors `engine.analytics.volumes`'s own
bucketing (buckets anchored to the earliest event, not wall-clock
midnight), extended with active-trajectory counts and mean link speed per
bucket."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_BUCKET_MINUTES = 15


@dataclass(frozen=True)
class FlowBucketRow:
    bucket_start: datetime
    events: int
    active_trajectories: int
    mean_speed_kmh: float | None


def _bucket_start(ts: datetime, bucket_minutes: int, epoch: datetime) -> datetime:
    elapsed_minutes = (ts - epoch).total_seconds() / 60.0
    bucket_index = math.floor(elapsed_minutes / bucket_minutes)
    return epoch + timedelta(minutes=bucket_index * bucket_minutes)


def compute_flow_trend(
    events: list,  # objects with .timestamp / .trajectory_id (str | None)
    link_arrivals: list,  # objects with .arrival_ts / .speed_kmh
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES,
) -> list[FlowBucketRow]:
    """One row per non-empty bucket, sorted by `bucket_start`.
    `active_trajectories` = number of DISTINCT trajectory ids with at least
    one event in the bucket. `mean_speed_kmh` is `None` (not 0.0 -- the
    contract's type is nullable) for a bucket with no link arrivals at
    all."""
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    if not events:
        return []
    epoch = min(e.timestamp for e in events)

    event_counts: dict[datetime, int] = {}
    traj_by_bucket: dict[datetime, set[str]] = {}
    for e in events:
        bucket = _bucket_start(e.timestamp, bucket_minutes, epoch)
        event_counts[bucket] = event_counts.get(bucket, 0) + 1
        if e.trajectory_id:
            traj_by_bucket.setdefault(bucket, set()).add(e.trajectory_id)

    speed_by_bucket: dict[datetime, list[float]] = {}
    for link in link_arrivals:
        bucket = _bucket_start(link.arrival_ts, bucket_minutes, epoch)
        speed_by_bucket.setdefault(bucket, []).append(link.speed_kmh)

    rows = []
    for bucket in sorted(event_counts.keys()):
        speeds = speed_by_bucket.get(bucket)
        rows.append(
            FlowBucketRow(
                bucket_start=bucket,
                events=event_counts[bucket],
                active_trajectories=len(traj_by_bucket.get(bucket, ())),
                mean_speed_kmh=statistics.mean(speeds) if speeds else None,
            )
        )
    return rows


__all__ = ["DEFAULT_BUCKET_MINUTES", "FlowBucketRow", "compute_flow_trend"]
