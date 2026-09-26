"""Single source of truth for where UrbanTrace's data lives on disk.

Every module that needs a default path for raw/OCR/run data imports
`get_data_dir()` (or `DATA_DIR`, resolved at import time) instead of
hard-coding an absolute path. This keeps the codebase portable across
machines: set `URBANTRACE_DATA_DIR` once and every default path in
engine/, eval/, api/, and sim/ follows it.

Resolution order for the data directory:
    1. `URBANTRACE_DATA_DIR` environment variable, if set.
    2. `<repo_root>/data` (gitignored) otherwise.

Resolution order for the SQLite DB path (`get_db_path`):
    1. `URBANTRACE_DB_PATH` environment variable, if set.
    2. `SUTRA_DB_PATH` (legacy name, kept for backward compatibility).
    3. `<data_dir>/urbantrace.db` otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_FILENAME = "urbantrace.db"


def get_data_dir() -> Path:
    """Return the configured data directory (`URBANTRACE_DATA_DIR`, or
    `<repo_root>/data` by default)."""
    raw = os.environ.get("URBANTRACE_DATA_DIR")
    return Path(raw) if raw else REPO_ROOT / "data"


def get_db_path() -> Path:
    """Return the configured SQLite DB path (`URBANTRACE_DB_PATH`, falling
    back to the legacy `SUTRA_DB_PATH`, then `<data_dir>/urbantrace.db`)."""
    raw = os.environ.get("URBANTRACE_DB_PATH") or os.environ.get("SUTRA_DB_PATH")
    if raw:
        return Path(raw)
    return get_data_dir() / DEFAULT_DB_FILENAME
