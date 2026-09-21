"""Secondary blocking must be an overflow path, never the default one
(architecture.md §7, docs/decisions.md "Day 3a" Part 2)."""

from pathlib import Path

import pytest

from engine.association.blocking import (
    apply_blocking,
    block_candidates,
    measure_recall_cost,
)
from engine.association.gating import Gate, gate_candidates
from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent, VehicleAttributes
from engine.contracts.plate import BLANK, SlotPosterior
from engine.contracts.store import EventStore
from engine.scoring.kinematic_lr import fit_kinematic_model

RUN_DIR = Path("data/run1")
ENGAGEMENT_RATE_CEILING = 0.05  # "rarely" -- production measured ~0.25%
RECALL_COST_CEILING = 0.005  # task spec: <0.5%


def _uniform_posterior() -> SlotPosterior:
    return SlotPosterior(probs={BLANK: 1.0})


def _fake_event(event_id: str, camera_id: str, timestamp, plate_argmax: str) -> DetectionEvent:
    return DetectionEvent(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=timestamp,
        plate_posterior=[_uniform_posterior() for _ in range(10)],
        plate_argmax=plate_argmax,
        plate_confidence=0.5,
        embedding=[1.0] + [0.0] * 127,
        attributes=VehicleAttributes(
            color="white", vehicle_type="car", color_confidence=0.5, type_confidence=0.5
        ),
        source="sim",
        gt_vehicle_id=None,
    )


def _events_at(n: int, base_plate: str):
    import datetime as dt

    t0 = dt.datetime(2026, 1, 1)
    return [
        _fake_event(f"evt_{i:04d}", "camA", t0 + dt.timedelta(seconds=i), base_plate)
        for i in range(n)
    ]


def test_block_candidates_is_a_noop_under_budget():
    import datetime as dt

    target = _fake_event("tgt", "camB", dt.datetime(2026, 1, 1), "MH12AB1234")
    candidates = _events_at(10, "MH12AB1234")
    result = block_candidates(target, candidates, budget=500)
    assert result == candidates


def test_block_candidates_engages_and_caps_at_budget_when_over():
    import datetime as dt

    target = _fake_event("tgt", "camB", dt.datetime(2026, 1, 1), "MH12AB1234")
    candidates = _events_at(600, "MH12AB1234")
    result = block_candidates(target, candidates, budget=500, max_edit_distance=3)
    assert len(result) == 500


def test_block_candidates_never_hard_rejects_a_far_plate_when_under_budget_cap():
    """A candidate whose plate is far (edit distance > max_edit_distance)
    from the target must still be RANKED and kept if there's room in the
    budget -- not silently discarded outright (that would reintroduce the
    exact brittleness architecture.md §7 warns against)."""
    import datetime as dt

    target = _fake_event("tgt", "camB", dt.datetime(2026, 1, 1), "MH12AB1234")
    close = _events_at(3, "MH12AB1234")
    far = [_fake_event("far_evt", "camA", dt.datetime(2026, 1, 1), "DL99ZZ9999")]
    result = block_candidates(target, close + far, budget=4, max_edit_distance=1)
    assert len(result) == 4
    assert any(e.event_id == "far_evt" for e in result)


@pytest.mark.skipif(not RUN_DIR.exists(), reason="data/run1 not generated")
def test_blocking_engages_rarely_and_costs_negligible_recall_on_run1():
    city = CityConfig.model_validate_json((RUN_DIR / "city.json").read_text(encoding="utf-8"))
    store = EventStore(RUN_DIR / "events.jsonl", RUN_DIR / "embeddings.npy")
    events = store.read_all()
    events_by_id = {e.event_id: e for e in events}

    model = fit_kinematic_model(events, city)
    gate = Gate(model=model)
    gate_result = gate_candidates(events, gate)

    blocked_map, engage_stats = apply_blocking(gate_result.candidates, events_by_id)
    recall_stats = measure_recall_cost(events, gate_result.candidates, blocked_map)

    print(f"engagement_rate={engage_stats.engagement_rate:.4%}")
    print(f"recall_cost={recall_stats.recall_cost:.4%}")

    assert engage_stats.engagement_rate < ENGAGEMENT_RATE_CEILING, (
        f"blocking engaged on {engage_stats.engagement_rate:.2%} of events -- "
        f"it must stay an overflow path, not the default one"
    )
    assert recall_stats.recall_cost < RECALL_COST_CEILING, (
        f"blocking cost {recall_stats.recall_cost:.2%} recall on the pairs it touched, "
        f"above the {RECALL_COST_CEILING:.1%} target"
    )
