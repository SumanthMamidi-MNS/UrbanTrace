"""Every endpoint documented in docs/api-contract.md (v1 AND the Contract v2
additions) must exist as a real route on the FastAPI app -- parses the
contract's own `| METHOD | \\`/path\\` | ... |` table rows rather than hand
duplicating the endpoint list, so this test actually breaks if the contract
and the implementation drift apart."""

from __future__ import annotations

import re
from pathlib import Path

from api.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "docs" / "api-contract.md"

# Matches a markdown table row like "| GET | `/api/health` | ... |".
_ROW_RE = re.compile(r"^\|\s*(GET|POST|DELETE|PUT|PATCH)\s*\|\s*`([^`]+)`")


def _endpoints_from_contract() -> list[tuple[str, str]]:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    endpoints = []
    for line in text.splitlines():
        m = _ROW_RE.match(line.strip())
        if m:
            endpoints.append((m.group(1), m.group(2)))
    return endpoints


def _iter_app_routes(routes):
    """Yields (methods: set[str], path: str) for every route, recursing
    through however routers are nested/wrapped (plain FastAPI `APIRouter`s,
    or this environment's instrumented `_IncludedRouter` wrapper which
    exposes the same routes via `.original_router.routes`)."""
    for r in routes:
        sub = getattr(r, "routes", None)
        if sub is None:
            orig = getattr(r, "original_router", None)
            if orig is not None:
                sub = getattr(orig, "routes", None)
        if sub:
            yield from _iter_app_routes(sub)
        path = getattr(r, "path", None)
        if path is not None:
            yield (getattr(r, "methods", None) or set(), path)


def test_contract_has_endpoints():
    # Sanity check the parser itself found a realistic number of rows
    # (guards against a future contract reformat silently breaking this
    # test into a vacuous pass).
    endpoints = _endpoints_from_contract()
    assert len(endpoints) >= 20, endpoints


def test_every_documented_endpoint_exists_in_app():
    endpoints = _endpoints_from_contract()
    app_routes = list(_iter_app_routes(app.routes))
    app_paths_by_method: dict[str, set[str]] = {}
    for methods, path in app_routes:
        for method in methods:
            app_paths_by_method.setdefault(method, set()).add(path)

    missing = [
        (method, path)
        for method, path in endpoints
        if path not in app_paths_by_method.get(method, set())
    ]
    assert not missing, f"documented endpoints missing from the app: {missing}"


def test_ws_live_route_exists():
    app_routes = list(_iter_app_routes(app.routes))
    ws_paths = {path for _, path in app_routes}
    assert "/ws/live" in ws_paths
