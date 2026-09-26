"""Watchlist probabilistic matching (docs/api-contract.md Contract v2;
engine.alerts.watchlist). The headline test
(`test_consensus_catches_what_single_read_misses`) is the PRD's own claim
made concrete: "a watchlisted vehicle is caught even when this camera
misread its plate" -- it builds a trajectory whose LATEST single read
misreads the watchlisted plate (so the single-read match score is below
`MATCH_THRESHOLD`) and asserts the fused multi-event trajectory consensus
still matches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from api.plate_grammar import enumerate_canonical_forms
from engine.alerts.watchlist import (
    MATCH_THRESHOLD,
    reconstruct_slot_posterior,
    single_read_match_probability,
    trajectory_consensus_match_probability,
)
from engine.contracts.plate import NUM_SLOTS

PLATE = "MH12AB1234"  # already an exact 10-slot canonical form
CANONICAL_PATTERNS = enumerate_canonical_forms(PLATE)


@dataclass
class FakeEventRow:
    """Stands in for `api.db.EventRow`: only `.posterior` / `.plate_argmax`
    / `.timestamp` are read by `engine.alerts.watchlist`."""

    posterior: list[dict]
    plate_argmax: str
    timestamp: datetime


def _sharp_posterior(char: str, confidence: float) -> dict:
    """`EventRow.posterior`'s per-slot shape: top-5 `{char, prob}` pairs +
    `unread`. A single character carries `confidence`; the rest of the
    (small) top-5 list is padding well below it so the reconstruction in
    `reconstruct_slot_posterior` sees one dominant character, like a real
    confident OCR read."""
    return {
        "top": [{"char": char, "prob": confidence}],
        "unread": False,
    }


def _make_row(plate: str, wrong_slot: int | None, wrong_char: str, ts: datetime) -> FakeEventRow:
    """A row reading `plate` confidently in every slot, except
    `wrong_slot` (if given) which confidently reads `wrong_char` instead --
    i.e. a high-confidence MISREAD, not a low-confidence/uncertain one."""
    assert len(plate) == NUM_SLOTS
    slots = []
    argmax_chars = []
    for idx, true_char in enumerate(plate):
        if wrong_slot is not None and idx == wrong_slot:
            slots.append(_sharp_posterior(wrong_char, 0.9))
            argmax_chars.append(wrong_char)
        else:
            slots.append(_sharp_posterior(true_char, 0.999))
            argmax_chars.append(true_char)
    return FakeEventRow(posterior=slots, plate_argmax="".join(argmax_chars), timestamp=ts)


def test_reconstruct_slot_posterior_sums_to_one():
    sp = reconstruct_slot_posterior([{"char": "A", "prob": 0.9}], slot_idx=0)
    assert abs(sum(sp.probs.values()) - 1.0) < 1e-6
    assert sp.probs["A"] == pytest.approx(0.9)


def test_single_correct_read_matches():
    row = _make_row(PLATE, wrong_slot=None, wrong_char="", ts=datetime(2026, 1, 1))
    prob = single_read_match_probability(row, CANONICAL_PATTERNS)
    assert prob >= MATCH_THRESHOLD


def test_single_misread_falls_below_threshold():
    # Slot 5 is the 2nd series-letter slot ("B" -> misread as "X").
    row = _make_row(PLATE, wrong_slot=5, wrong_char="X", ts=datetime(2026, 1, 1))
    prob = single_read_match_probability(row, CANONICAL_PATTERNS)
    assert prob < MATCH_THRESHOLD


def test_consensus_catches_what_single_read_misses():
    """THE differentiator test (contract's own words): three earlier
    cameras all read the plate correctly; the fourth (latest, most recent)
    camera confidently MISREADS one slot. The single-read score on that
    latest event alone must be below threshold (this camera, by itself,
    would not have flagged the vehicle) -- but the trajectory consensus
    fused across all four revealed events must still match, because the
    three correct reads outvote the one misread character."""
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    rows = [
        _make_row(PLATE, wrong_slot=None, wrong_char="", ts=t0 + timedelta(minutes=0)),
        _make_row(PLATE, wrong_slot=None, wrong_char="", ts=t0 + timedelta(minutes=5)),
        _make_row(PLATE, wrong_slot=None, wrong_char="", ts=t0 + timedelta(minutes=10)),
        _make_row(PLATE, wrong_slot=5, wrong_char="X", ts=t0 + timedelta(minutes=15)),
    ]
    latest = rows[-1]

    single_prob = single_read_match_probability(latest, CANONICAL_PATTERNS)
    assert single_prob < MATCH_THRESHOLD, (
        f"test setup invalid: the latest single read should NOT match on its own "
        f"(got {single_prob})"
    )

    consensus_prob = trajectory_consensus_match_probability(rows, CANONICAL_PATTERNS)
    assert consensus_prob >= MATCH_THRESHOLD, (
        f"the fused trajectory consensus should catch the watchlisted plate even though "
        f"the latest camera misread it (got {consensus_prob})"
    )


def test_wildcard_pattern_matches_via_consensus_too():
    """A partial pattern with `?` wildcards (as `/api/search` accepts) must
    also work through the same machinery -- `MH12??1234` should match the
    same trajectory regardless of the series letters' misread."""
    patterns = enumerate_canonical_forms("MH12??1234")
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    rows = [
        _make_row(PLATE, wrong_slot=None, wrong_char="", ts=t0),
        _make_row(PLATE, wrong_slot=5, wrong_char="X", ts=t0 + timedelta(minutes=5)),
    ]
    prob = trajectory_consensus_match_probability(rows, patterns)
    assert prob >= MATCH_THRESHOLD
