"""Per-camera event volumes, bucketed by time (architecture.md §4;
docs/api-contract.md `/api/analytics/volumes`)."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from engine.contracts.events import DetectionEvent

DEFAULT_BUCKET_MINUTES = 60


@dataclass(frozen=True)
class VolumeRow:
    camera_id: str
    bucket_start: datetime
    count: int


def _bucket_start(ts: datetime, bucket_minutes: int, epoch: datetime) -> datetime:
    elapsed_minutes = (ts - epoch).total_seconds() / 60.0
    bucket_index = math.floor(elapsed_minutes / bucket_minutes)
    return epoch + timedelta(minutes=bucket_index * bucket_minutes)


def compute_volumes(
    events: list[DetectionEvent],
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES,
    camera_id: str | None = None,
) -> list[VolumeRow]:
    """One row per (camera, time bucket) with a non-zero event count, sorted
    by (camera_id, bucket_start). Buckets are anchored to the earliest
    event's timestamp, not wall-clock midnight -- the simulator's own
    `SIM_EPOCH` is an arbitrary placeholder date, not a real calendar day
    boundary worth aligning to."""
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    filtered = [e for e in events if camera_id is None or e.camera_id == camera_id]
    if not filtered:
        return []
    epoch = min(e.timestamp for e in filtered)

    counts: dict[tuple[str, datetime], int] = {}
    for e in filtered:
        bucket = _bucket_start(e.timestamp, bucket_minutes, epoch)
        key = (e.camera_id, bucket)
        counts[key] = counts.get(key, 0) + 1

    rows = [
        VolumeRow(camera_id=cid, bucket_start=bucket, count=count)
        for (cid, bucket), count in counts.items()
    ]
    rows.sort(key=lambda r: (r.camera_id, r.bucket_start))
    return rows


__all__ = ["DEFAULT_BUCKET_MINUTES", "VolumeRow", "compute_volumes"]
