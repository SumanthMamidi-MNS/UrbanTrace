import math

import pytest
from pydantic import ValidationError

from engine.contracts.plate import NUM_SLOTS, PLATE_SLOTS, SlotPosterior


def test_plate_slots_shape():
    assert len(PLATE_SLOTS) == NUM_SLOTS == 10
    # state letters
    assert set(PLATE_SLOTS[0]) == set(PLATE_SLOTS[1])
    assert "A" in PLATE_SLOTS[0] and "0" not in PLATE_SLOTS[0]
    # RTO digits
    assert set(PLATE_SLOTS[2]) == {str(d) for d in range(10)}
    # series letters + blank
    assert "_" in PLATE_SLOTS[4]
    assert "A" in PLATE_SLOTS[4]
    # number digits + blank
    assert "_" in PLATE_SLOTS[6]
    assert "0" in PLATE_SLOTS[6]


def test_slot_posterior_sums_to_one():
    sp = SlotPosterior.uniform(PLATE_SLOTS[0])
    assert math.isclose(sum(sp.probs.values()), 1.0, abs_tol=1e-9)


def test_slot_posterior_rejects_bad_sum():
    with pytest.raises(ValidationError):
        SlotPosterior(probs={"A": 0.5, "B": 0.2})


def test_slot_posterior_argmax():
    sp = SlotPosterior.peaked(PLATE_SLOTS[0], "M", 0.9)
    assert sp.argmax() == "M"


def test_slot_posterior_uninformative():
    uniform = SlotPosterior.uniform(PLATE_SLOTS[0])
    peaked = SlotPosterior.peaked(PLATE_SLOTS[0], "M", 0.9)
    assert uniform.is_uninformative()
    assert not peaked.is_uninformative()


def test_slot_posterior_entropy_ordering():
    uniform = SlotPosterior.uniform(PLATE_SLOTS[0])
    peaked = SlotPosterior.peaked(PLATE_SLOTS[0], "M", 0.9)
    assert uniform.entropy() > peaked.entropy()
