"""UrbanTrace API — FastAPI REST + WebSocket layer over the linking-engine output.

Owned exclusively by this package: `api/*`, `tests/test_api*.py`. Implements
`docs/api-contract.md` exactly (frozen contract) against a SQLite database
populated by `api/ingest.py`.
"""
