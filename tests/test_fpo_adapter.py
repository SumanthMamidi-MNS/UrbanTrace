"""Tests for engine.perception.fpo_adapter -- runnable in the MAIN venv
(numpy + pydantic + pyyaml only). NO `fast_plate_ocr` or `onnxruntime`
import anywhere in this file or in the module under test -- see
fpo_adapter.py's module docstring for why (mirrors ctc_to_slots's own
main-venv testability).

Uses the real `cct-s-v2-global-model` alphabet/pad convention
(`'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'`, pad_char `'_'`) throughout, since
that is the actual pretrained hub model this adapter targets.
"""

import numpy as np
import pytest

from engine.contracts.plate import BLANK, NUM_SLOTS, NUMBER_ALPHABET, SERIES_ALPHABET
from engine.perception.fpo_adapter import fpo_output_to_canonical_slots, load_plate_config

FPO_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
FPO_PAD = "_"
FPO_VOCAB = len(FPO_ALPHABET)


def _fpo_probs(plate: str, max_slots: int = 10, hard: bool = True) -> np.ndarray:
    """Build a (max_slots, FPO_VOCAB) per-slot softmax array the way
    fast-plate-ocr's fixed classification heads would emit one: `plate`
    left-aligned, remaining slots predicting the pad character.

    hard=True gives literal one-hot rows (exact 1.0/0.0) -- the adversarial
    case the zero-probability invariant must survive regardless.
    """
    assert len(plate) <= max_slots
    padded = plate + FPO_PAD * (max_slots - len(plate))
    probs = np.zeros((max_slots, FPO_VOCAB), dtype=np.float64)
    for i, c in enumerate(padded):
        idx = FPO_ALPHABET.index(c)
        if hard:
            probs[i, idx] = 1.0
        else:
            probs[i, :] = 0.001 / (FPO_VOCAB - 1)
            probs[i, idx] = 0.999
    return probs


def _assert_never_zero(slots) -> None:
    for slot in slots:
        for char, p in slot.probs.items():
            assert p > 0.0, f"char {char!r} got exact-zero probability"


def _argmax_str(slots) -> str:
    return "".join(s.argmax() for s in slots)


class TestZeroProbabilityInvariant:
    """Architecture invariant (docs/architecture.md sec 3): every slot
    posterior retains background mass on every character -- never exactly
    zero, even for an adversarial literal one-hot model output."""

    def test_full_plate_hard_one_hot_never_zero(self):
        probs = _fpo_probs("MH12AB1234")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert len(slots) == NUM_SLOTS
        _assert_never_zero(slots)

    def test_short_plate_hard_one_hot_never_zero(self):
        probs = _fpo_probs("MH12A234")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        _assert_never_zero(slots)

    def test_all_pad_input_never_zero(self):
        # Degenerate: the model predicts nothing but its pad character at
        # every slot (total failure to read). Every canonical slot must
        # fall back to UNIFORM, never exactly zero for any alphabet member.
        probs = _fpo_probs("")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert len(slots) == NUM_SLOTS
        _assert_never_zero(slots)
        for slot in slots:
            assert slot.is_uninformative()

    def test_soft_confident_never_zero(self):
        probs = _fpo_probs("MH12AB1234", hard=False)
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        _assert_never_zero(slots)

    def test_mid_string_pad_hole_never_zero(self):
        # Model predicts a pad character BEFORE the end (a "hole") --
        # adversarial input a well-behaved model should never produce, but
        # the adapter must still degrade safely rather than crash or zero
        # out an alphabet member.
        probs = _fpo_probs("MH12AB1234")
        probs[5, :] = 0.0
        probs[5, FPO_ALPHABET.index(FPO_PAD)] = 1.0  # force a pad in the middle
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert len(slots) == NUM_SLOTS
        _assert_never_zero(slots)


class TestGrammarAlignment:
    """Same Indian plate grammar as ctc_to_slots.py (reused, not
    reimplemented) applied to fast-plate-ocr's fixed-slot output."""

    def test_two_letter_series_four_digit_number_no_padding(self):
        probs = _fpo_probs("MH12AB1234")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert _argmax_str(slots) == "MH12AB1234"

    def test_one_letter_series_three_digit_number(self):
        probs = _fpo_probs("MH12A234")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert _argmax_str(slots) == "MH12A" + BLANK + BLANK + "234"

    def test_two_letter_series_one_digit_number(self):
        probs = _fpo_probs("MH12AB1")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert _argmax_str(slots) == "MH12AB" + BLANK * 3 + "1"

    def test_series_grammar_blank_not_uniform_when_next_char_is_digit(self):
        # Recogniser skipped straight from RTO to digits: the grammar
        # itself says "no series here" -- real evidence of an absent
        # series (a legitimate older-format plate), not missing
        # information. Slots 4-5 must be a PEAKED BLANK, not UNIFORM
        # ("couldn't read"): a uniform posterior's argmax is arbitrary (the
        # alphabet's first letter), which corrupts the plate instead of
        # leaving it honestly blank.
        probs = _fpo_probs("MH121234")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert not slots[4].is_uninformative()
        assert not slots[5].is_uninformative()
        assert slots[4].argmax() == BLANK
        assert slots[5].argmax() == BLANK
        assert "".join(s.argmax() for s in slots[6:]) == "1234"

    def test_series_uniform_when_nothing_read_past_prefix(self):
        # Contrast case: nothing decoded past the state+RTO prefix at all
        # (occlusion/truncation) -- genuinely missing information, so
        # slots 4-5 (and the number slots) stay UNIFORM.
        probs = _fpo_probs("MH12")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert slots[4].is_uninformative()
        assert slots[5].is_uninformative()
        for slot in slots[6:]:
            assert slot.is_uninformative()

    @pytest.mark.parametrize(
        "raw,canonical",
        [
            ("AR152019", "AR15" + BLANK + BLANK + "2019"),
            ("AR200692", "AR20" + BLANK + BLANK + "0692"),
            ("OD021481", "OD02" + BLANK + BLANK + "1481"),
            ("PY055353", "PY05" + BLANK + BLANK + "5353"),
            ("UK185763", "UK18" + BLANK + BLANK + "5763"),
            ("WB061061", "WB06" + BLANK + BLANK + "1061"),
            ("HR696969", "HR69" + BLANK + BLANK + "6969"),
        ],
    )
    def test_no_series_letters_legacy_plates_regression(self, raw, canonical):
        # Regression for the adapter bug: legitimate older-format plates
        # with NO series letters (e.g. AR15__2019) were decoded as
        # AR15AA2019 by fpo_adapter -- the uniform series slots' argmax
        # defaulted to 'A', the alphabet's first letter, instead of the
        # grammar-confident BLANK.
        probs = _fpo_probs(raw)
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert "".join(s.argmax() for s in slots) == canonical

    def test_extra_trailing_digit_noise_is_dropped(self):
        # 5-digit "number" (grammar max is 4): the 5th char is dropped.
        probs = _fpo_probs("MH12AB12345", max_slots=11)
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert len(slots) == NUM_SLOTS
        assert "".join(s.argmax() for s in slots[6:]) == "1234"

    def test_short_prefix_leaves_rto_series_number_uniform(self):
        probs = _fpo_probs("MH")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert slots[0].argmax() == "M"
        assert slots[1].argmax() == "H"
        for slot in slots[2:]:
            assert slot.is_uninformative()

    def test_series_alphabet_and_number_alphabet_uniform_shapes(self):
        probs = _fpo_probs("MH")
        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert set(slots[4].probs.keys()) == set(SERIES_ALPHABET)
        assert set(slots[6].probs.keys()) == set(NUMBER_ALPHABET)


class TestPadCharUsesRealModelSignal:
    """Unlike ctc_to_slots's own CRNN (no CTC class exists for BLANK at all,
    so a slot's BLANK probability is always the bare floor), fast-plate-ocr's
    pad character IS a real class in every row -- a "read" slot whose raw
    row also assigns real, elevated probability to the pad class should
    carry that signal through, not collapse it to the floor."""

    def test_read_slot_incorporates_model_pad_mass_not_just_floor(self):
        probs = _fpo_probs("MH12AB1234", hard=False)
        # Slot 4 is the first series character ('A'). Give its raw FPO row
        # deliberately elevated pad-class mass alongside the 'A' peak,
        # simulating genuine model uncertainty about whether that position
        # is really a character vs. padding.
        row = np.full(FPO_VOCAB, 0.001, dtype=np.float64)
        row[FPO_ALPHABET.index("A")] = 0.8
        row[FPO_ALPHABET.index(FPO_PAD)] = 0.15
        probs[4, :] = row / row.sum()

        slots = fpo_output_to_canonical_slots(probs, FPO_ALPHABET, FPO_PAD)
        assert slots[4].argmax() == "A"
        # Restricted+renormalised onto SERIES_ALPHABET (letters + BLANK),
        # the model's real elevated pad mass must survive as materially
        # more than ctc_to_slots's own floor-only fallback (1e-6).
        assert slots[4].probs[BLANK] > 0.01


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        fpo_output_to_canonical_slots(np.zeros(5), FPO_ALPHABET, FPO_PAD)


def test_load_plate_config(tmp_path):
    cfg_path = tmp_path / "plate_config.yaml"
    cfg_path.write_text(
        "max_plate_slots: 10\n"
        "alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'\n"
        "pad_char: '_'\n"
        "img_height: 64\n"
        "img_width: 128\n",
        encoding="utf-8",
    )
    alphabet, pad_char, max_plate_slots = load_plate_config(str(cfg_path))
    assert alphabet == FPO_ALPHABET
    assert pad_char == "_"
    assert max_plate_slots == 10
