"""Small shared helpers: non-finite -> null (docs/api-contract.md
"Log-likelihood values ... non-finite floats serialised as null"),
ISO-8601 timestamp formatting, and top-5 slot truncation."""

from __future__ import annotations

import math
from datetime import datetime


def finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def iso(ts: datetime) -> str:
    """ISO-8601 with a trailing Z (docs/api-contract.md's own example uses
    `Z`, not `+00:00`)."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(tz=None).replace(tzinfo=None)
    return ts.isoformat(timespec="milliseconds") + "Z"


def top_k_slot(full_posterior: dict[str, float], k: int = 5) -> list[dict[str, float]]:
    ranked = sorted(full_posterior.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"char": c, "prob": p} for c, p in ranked]


def parse_iso(s: str) -> datetime:
    """Parse a `from`/`to` query-param timestamp (ISO-8601, optionally with
    a trailing `Z`) into a naive datetime comparable with the DB's own
    naive-UTC columns."""
    cleaned = s.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(tz=None).replace(tzinfo=None)
    return parsed


def clamp_limit(limit: int, default: int = 50, max_limit: int = 500) -> int:
    if limit is None:
        return default
    return min(max(limit, 1), max_limit)
