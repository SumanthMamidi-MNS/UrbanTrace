"""`/api/eval` — raw contents of `eval/reports/*.json` keyed by file stem
(docs/api-contract.md); tolerates a missing directory."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from api import schemas

router = APIRouter(prefix="/api", tags=["eval"])

REPORTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "reports"


@router.get("/eval", response_model=schemas.EvalReports)
def eval_reports() -> schemas.EvalReports:
    reports: dict[str, object] = {}
    if REPORTS_DIR.exists():
        for f in sorted(REPORTS_DIR.glob("*.json")):
            try:
                reports[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
    return schemas.EvalReports(reports=reports)
