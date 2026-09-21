"""End-to-end tests for the SUTRA FastAPI layer (docs/api-contract.md).

Builds a TINY dataset (8 cameras, 150 vehicles, 2h, with a clone fraction so
alerts exist), ingests it via the `--build-demo` path into a temp SQLite
database, then hits every documented endpoint with a real `TestClient` and
validates response shapes against the contract: required keys present,
correct types, no NaN/inf anywhere in the JSON.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.ingest import run_ingest
from api.plate_grammar import GrammarError, enumerate_canonical_forms
from api.routers import eval as eval_router
from sim.city import generate_city
from sim.corruption import CorruptionConfig
from sim.generate import generate_dataset_with_city, write_dataset

N_CAMERAS = 8
N_VEHICLES = 150
HOURS = 2
CLONE_FRACTION = 0.08
TRAIN_VEHICLES = 300  # small: keeps the fixture fast, independent of prod scale


def _assert_finite(obj, path: str = "$") -> None:
    """Recursively assert no float in a parsed-JSON structure is NaN/inf —
    the contract requires non-finite floats to already be serialised as
    `null` before they ever reach JSON."""
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{path}[{i}]")


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> tuple[Path, object]:
    data_dir = tmp_path_factory.mktemp("tiny_data")
    db_path = tmp_path_factory.mktemp("tiny_db") / "sutra.db"

    city = generate_city(n_cameras=N_CAMERAS, seed=1)
    cfg = CorruptionConfig(clone_fraction=CLONE_FRACTION)
    ds = generate_dataset_with_city(
        city,
        n_vehicles=N_VEHICLES,
        hours=HOURS,
        seed=321,
        clone_fraction=CLONE_FRACTION,
        corruption_config=cfg,
    )
    write_dataset(
        ds,
        data_dir,
        {
            "cameras": N_CAMERAS,
            "vehicles": N_VEHICLES,
            "hours": HOURS,
            "seed": 321,
            "clone_fraction": CLONE_FRACTION,
        },
    )

    stats = run_ingest(
        data_dir=data_dir, db_path=db_path, build_demo=True, train_vehicles=TRAIN_VEHICLES
    )
    assert stats.source == "build-demo"
    assert stats.n_events > 0
    assert stats.n_trajectories > 0
    return db_path, stats


@pytest.fixture()
def client(tiny_dataset, monkeypatch):
    db_path, _ = tiny_dataset
    monkeypatch.setenv("SUTRA_DB_PATH", str(db_path))
    from api.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------


def test_health(client: TestClient, tiny_dataset):
    _, stats = tiny_dataset
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["dataset"], str)
    assert body["n_events"] == stats.n_events
    assert body["n_trajectories"] == stats.n_trajectories
    _assert_finite(body)


def test_city(client: TestClient):
    r = client.get("/api/city")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"nodes", "edges", "cameras"}
    assert len(body["cameras"]) == N_CAMERAS
    for cam in body["cameras"]:
        assert set(cam.keys()) == {
            "camera_id",
            "name",
            "lat",
            "lon",
            "node_id",
            "bearing_deg",
            "is_border",
        }
    for edge in body["edges"]:
        assert set(edge.keys()) == {"from_node", "to_node", "length_m", "speed_limit_kmh"}
    _assert_finite(body)


def test_cameras(client: TestClient):
    r = client.get("/api/cameras")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == N_CAMERAS
    for cam in body:
        assert set(cam.keys()) == {
            "camera_id",
            "name",
            "lat",
            "lon",
            "node_id",
            "bearing_deg",
            "is_border",
            "events_total",
            "volume_last_hour",
        }
        assert isinstance(cam["events_total"], int)
    _assert_finite(body)


def test_events_pagination(client: TestClient, tiny_dataset):
    _, stats = tiny_dataset
    r = client.get("/api/events", params={"limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"items", "total", "limit", "offset"}
    assert body["total"] == stats.n_events
    assert body["limit"] == 10
    assert len(body["items"]) == min(10, stats.n_events)
    for ev in body["items"]:
        assert set(ev.keys()) == {
            "event_id",
            "camera_id",
            "timestamp",
            "plate_argmax",
            "plate_confidence",
            "color",
            "vehicle_type",
            "trajectory_id",
        }
    # newest first
    timestamps = [e["timestamp"] for e in body["items"]]
    assert timestamps == sorted(timestamps, reverse=True)
    _assert_finite(body)

    r2 = client.get("/api/events", params={"camera_id": body["items"][0]["camera_id"]})
    assert r2.status_code == 200
    assert all(e["camera_id"] == body["items"][0]["camera_id"] for e in r2.json()["items"])


def test_event_detail_and_404(client: TestClient):
    events = client.get("/api/events", params={"limit": 1}).json()["items"]
    event_id = events[0]["event_id"]
    r = client.get(f"/api/events/{event_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["event_id"] == event_id
    assert len(body["plate_posterior"]) == 10
    for slot in body["plate_posterior"]:
        assert set(slot.keys()) == {"top", "unread"}
        assert len(slot["top"]) <= 5
        for t in slot["top"]:
            assert set(t.keys()) == {"char", "prob"}
    _assert_finite(body)

    r404 = client.get("/api/events/does-not-exist")
    assert r404.status_code == 404
    assert "detail" in r404.json()


def test_trajectories_pagination_and_filters(client: TestClient, tiny_dataset):
    _, stats = tiny_dataset
    r = client.get("/api/trajectories", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == stats.n_trajectories
    assert len(body["items"]) <= 5
    for t in body["items"]:
        assert set(t.keys()) == {
            "trajectory_id",
            "decoded_plate",
            "plate_confidence",
            "start_time",
            "end_time",
            "n_events",
            "camera_sequence",
            "color",
            "vehicle_type",
            "has_alert",
        }
    starts = [t["start_time"] for t in body["items"]]
    assert starts == sorted(starts, reverse=True)
    _assert_finite(body)

    r2 = client.get("/api/trajectories", params={"min_len": 2})
    assert r2.status_code == 200
    assert all(t["n_events"] >= 2 for t in r2.json()["items"])

    r3 = client.get("/api/trajectories", params={"has_alert": True})
    assert r3.status_code == 200
    assert all(t["has_alert"] is True for t in r3.json()["items"])


def test_trajectory_detail_and_404(client: TestClient):
    trajs = client.get("/api/trajectories", params={"limit": 1}).json()["items"]
    traj_id = trajs[0]["trajectory_id"]
    r = client.get(f"/api/trajectories/{traj_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["trajectory_id"] == traj_id
    assert set(body.keys()) >= {
        "trajectory_id",
        "decoded_plate",
        "plate_confidence",
        "start_time",
        "end_time",
        "n_events",
        "camera_sequence",
        "color",
        "vehicle_type",
        "has_alert",
        "events",
        "links",
        "path",
        "consensus",
    }
    assert len(body["events"]) == body["n_events"]
    assert len(body["path"]) == body["n_events"]
    consensus = body["consensus"]
    assert set(consensus.keys()) == {"per_slot", "single_read_plates", "entropy_bits"}
    assert len(consensus["per_slot"]) == 10
    for slot in consensus["per_slot"]:
        assert len(slot) <= 5
        for entry in slot:
            assert set(entry.keys()) == {"char", "prob"}
    for link in body["links"]:
        assert set(link.keys()) == {
            "from_event_id",
            "to_event_id",
            "plate_lr",
            "appearance_lr",
            "kinematic_lr",
            "prior_log_odds",
            "total_log_odds",
            "delta_t_s",
            "expected_t_s",
            "skipped_cameras",
        }
    for point in body["path"]:
        assert set(point.keys()) == {"event_id", "camera_id", "lat", "lon", "timestamp"}
    _assert_finite(body)

    r404 = client.get("/api/trajectories/does-not-exist")
    assert r404.status_code == 404


# ---------------------------------------------------------------------------
# Search: plate grammar parsing (unit-level, exact canonical forms)
# ---------------------------------------------------------------------------


def test_grammar_unambiguous_full_length():
    # already exactly 10 chars, one grammar-consistent split
    assert enumerate_canonical_forms("MH12??1234") == ["MH12??1234"]


def test_grammar_unambiguous_single_letter_series():
    # RTO=2 digits, series=1 letter, number=4 digits is the only valid split
    assert enumerate_canonical_forms("MH12A1234") == ["MH12A_1234"]


def test_grammar_ambiguous_enumeration():
    # "DL3C?456": RTO can only be 1 digit ("3", since "C" isn't a digit),
    # but the '?' can be the 2nd series letter (number="456") or the 1st
    # digit of the number (series="C" alone) -- both grammar-consistent.
    forms = enumerate_canonical_forms("DL3C?456")
    assert set(forms) == {"DL03C_?456", "DL03C?_456"}


def test_grammar_rejects_too_short():
    with pytest.raises(GrammarError):
        enumerate_canonical_forms("ZZ")


def test_grammar_rejects_ungrammatical():
    # a 3-letter series can never fit the 2-slot canonical series -> no split
    with pytest.raises(GrammarError):
        enumerate_canonical_forms("MH12ABC1234")


def test_search_api_wildcard_query_200(client: TestClient):
    r = client.get("/api/search", params={"q": "MH12??1234"})
    assert r.status_code == 200
    hits = r.json()
    assert isinstance(hits, list)
    probs = [h["probability"] for h in hits]
    assert probs == sorted(probs, reverse=True)
    for h in hits:
        assert set(h.keys()) == {"trajectory", "probability", "matched_plate"}
        assert 0.0 < h["probability"] <= 1.0
    _assert_finite(hits)


def test_search_api_ambiguous_query_200(client: TestClient):
    r = client.get("/api/search", params={"q": "DL3C?456"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_search_api_rejects_ungrammatical_query(client: TestClient):
    r = client.get("/api/search", params={"q": "ZZ"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_search_api_limit_and_filters(client: TestClient):
    r = client.get("/api/search", params={"q": "MH12??1234", "limit": 2})
    assert r.status_code == 200
    assert len(r.json()) <= 2

    r2 = client.get("/api/search", params={"q": "MH12??1234", "vehicle_type": "car"})
    assert r2.status_code == 200
    assert all(h["trajectory"]["vehicle_type"] == "car" for h in r2.json())


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def test_analytics_summary(client: TestClient, tiny_dataset):
    _, stats = tiny_dataset
    r = client.get("/api/analytics/summary")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "n_events",
        "n_trajectories",
        "mean_trip_duration_s",
        "mean_events_per_trajectory",
        "plate_repair_rate",
        "active_alerts",
    }
    assert body["n_events"] == stats.n_events
    assert body["n_trajectories"] == stats.n_trajectories
    assert 0.0 <= body["plate_repair_rate"] <= 1.0
    _assert_finite(body)


def test_analytics_volumes(client: TestClient):
    r = client.get("/api/analytics/volumes", params={"bucket_minutes": 30})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    for row in body:
        assert set(row.keys()) == {"camera_id", "bucket_start", "count"}
    _assert_finite(body)

    if body:
        cam = body[0]["camera_id"]
        r2 = client.get("/api/analytics/volumes", params={"camera_id": cam})
        assert r2.status_code == 200
        assert all(row["camera_id"] == cam for row in r2.json())


def test_analytics_od_matrix(client: TestClient):
    r = client.get("/api/analytics/od_matrix")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"zones", "matrix"}
    assert body["zones"] == ["N", "S", "E", "W", "Central"]
    assert len(body["matrix"]) == 5
    assert all(len(row) == 5 for row in body["matrix"])
    _assert_finite(body)


def test_analytics_corridors(client: TestClient):
    r = client.get("/api/analytics/corridors", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) <= 5
    for row in body:
        assert set(row.keys()) == {
            "from_camera",
            "to_camera",
            "n_trips",
            "median_travel_s",
            "p90_travel_s",
            "free_flow_s",
            "congestion_index",
        }
    _assert_finite(body)


# ---------------------------------------------------------------------------
# Alerts / eval
# ---------------------------------------------------------------------------


def test_alerts_exist_given_clone_fraction(client: TestClient, tiny_dataset):
    _, stats = tiny_dataset
    r = client.get("/api/alerts")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == stats.n_alerts
    for a in body:
        assert set(a.keys()) == {
            "alert_id",
            "type",
            "severity",
            "created_at",
            "plate",
            "trajectory_ids",
            "summary",
            "evidence",
        }
        assert a["type"] in ("clone", "impossible_travel", "anomaly")
        assert a["severity"] in ("high", "medium", "low")
    _assert_finite(body)

    if body:
        one_type = body[0]["type"]
        r2 = client.get("/api/alerts", params={"type": one_type})
        assert all(a["type"] == one_type for a in r2.json())


def test_eval_reports_real_dir(client: TestClient):
    r = client.get("/api/eval")
    assert r.status_code == 200
    body = r.json()
    assert "reports" in body
    assert isinstance(body["reports"], dict)


def test_eval_reports_tolerates_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_router, "REPORTS_DIR", tmp_path / "does-not-exist")
    result = eval_router.eval_reports()
    assert result.reports == {}


# ---------------------------------------------------------------------------
# Replay + WebSocket
# ---------------------------------------------------------------------------


def test_replay_start_pause_reset(client: TestClient):
    r = client.post("/api/replay", json={"action": "start", "speed": 500000})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"running", "speed", "sim_time"}
    assert body["running"] is True
    assert body["speed"] == 500000

    r2 = client.post("/api/replay", json={"action": "pause"})
    assert r2.status_code == 200
    assert r2.json()["running"] is False

    r3 = client.post("/api/replay", json={"action": "reset"})
    assert r3.status_code == 200
    assert r3.json()["running"] is False


def test_ws_live_delivers_clock_and_event(client: TestClient):
    client.post("/api/replay", json={"action": "reset"})
    with client.websocket_connect("/ws/live") as ws:
        client.post("/api/replay", json={"action": "start", "speed": 1_000_000})
        seen_types: set[str] = set()
        # A speed this high reveals the whole tiny dataset in the first
        # tick (a few hundred `event`/`trajectory` messages) before the
        # single end-of-tick `clock` message -- drain generously.
        for _ in range(1000):
            msg = ws.receive_json()
            assert "type" in msg and "data" in msg
            seen_types.add(msg["type"])
            if "clock" in seen_types and "event" in seen_types:
                break
        assert "clock" in seen_types
        assert "event" in seen_types
    client.post("/api/replay", json={"action": "reset"})
