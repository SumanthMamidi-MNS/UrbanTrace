"""Write `api/openapi.json` from the live FastAPI app schema (tracked, must
agree with docs/api-contract.md).

Usage:
    python -m api.export_openapi
"""

from __future__ import annotations

import json
from pathlib import Path

from api.main import app

OUT_PATH = Path(__file__).resolve().parent / "openapi.json"


def main() -> None:
    schema = app.openapi()
    OUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
