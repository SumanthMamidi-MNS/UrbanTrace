"""Tests for engine.perception.ctc_to_slots -- runnable in the MAIN venv
(numpy + pydantic only, no torch). See that module's docstring for the
alignment algorithm this exercises.
"""

import numpy as np
import pytest

from engine.contracts.plate import BLANK, NUM_SLOTS, NUMBER_ALPHABET, SERIES_ALPHABET
from engine.perception.ctc_to_slots import (
    CHARSET,
    CTC_BLANK_IDX,
    NUM_CLASSES,
    char_to_label,
    ctc_greedy_decode,
    decode_one_line_plate,
    decode_plate_best_of_two,
    decode_two_line_plate,
    plate_log_prob,
)


def _one_hot_probs(chars: str, hard: bool = True) -> np.ndarray:
    """Build a (T=len(chars), NUM_CLASSES) softmax-like array, one row per
    character, no repeated adjacent characters assumed (so no CTC blank
    spacer is needed to keep them from collapsing together).

    hard=True gives a literal one-hot row (exact 1.0 / exact 0.0) -- the
    adversarial case this module must survive: real softmax never emits an
    exact zero, but nothing stops a poorly-calibrated model from doing so,
    and the invariant must hold regardless.
    """
    t = len(chars)
    probs = np.zeros((t, NUM_CLASSES), dtype=np.float64)
    for i, c in enumerate(chars):
        label = char_to_label(c)
        if hard:
            probs[i, label] = 1.0
        else:
            probs[i, :] = 0.001 / (NUM_CLASSES - 1)
            probs[i, label] = 0.999
    return probs


def _assert_never_zero(slots) -> None:
    for slot in slots:
        for char, p in slot.probs.items():
            assert p > 0.0, f"char {char!r} got exact-zero probability"


class TestGreedyDecode:
    def test_simple_no_repeats(self):
        probs = _one_hot_probs("MH12AB1234")
        decoded, timesteps = ctc_greedy_decode(probs)
        assert decoded == "MH12AB1234"
        assert timesteps == list(range(10))

    def test_collapses_repeats_and_drops_blank(self):
        # "AA" with a blank between two genuine A's must decode as "AA", not "A".
        chars_and_labels = ["A", "A", None, "A"]  # None = CTC blank
        t = len(chars_and_labels)
        probs = np.zeros((t, NUM_CLASSES), dtype=np.float64)
        for i, c in enumerate(chars_and_labels):
            label = CTC_BLANK_IDX if c is None else char_to_label(c)
            probs[i, label] = 1.0
        decoded, timesteps = ctc_greedy_decode(probs)
        assert decoded == "AA"
        assert timesteps == [0, 3]

    def test_all_blank_decodes_empty(self):
        t = 5
        probs = np.zeros((t, NUM_CLASSES), dtype=np.float64)
        probs[:, CTC_BLANK_IDX] = 1.0
        decoded, timesteps = ctc_greedy_decode(probs)
        assert decoded == ""
        assert timesteps == []


class TestZeroProbabilityInvariant:
    """The architecture invariant (docs/architecture.md sec 3): every slot
    posterior retains background mass on every character -- never exactly
    zero. Adversarial: literal one-hot (hard 0.0/1.0) softmax rows."""

    def test_full_plate_hard_one_hot_never_zero(self):
        probs = _one_hot_probs("MH12AB1234", hard=True)
        slots = decode_one_line_plate(probs)
        assert len(slots) == NUM_SLOTS
        _assert_never_zero(slots)

    def test_short_plate_hard_one_hot_never_zero(self):
        probs = _one_hot_probs("MH12A234", hard=True)
        slots = decode_one_line_plate(probs)
        _assert_never_zero(slots)

    def test_all_blank_input_never_zero(self):
        # Degenerate: the model emits nothing but CTC blank at every
        # timestep (total failure to read). Every slot must fall back to
        # UNIFORM, which is never exactly zero for any alphabet member.
        t = 6
        probs = np.zeros((t, NUM_CLASSES), dtype=np.float64)
        probs[:, CTC_BLANK_IDX] = 1.0
        slots = decode_one_line_plate(probs)
        assert len(slots) == NUM_SLOTS
        _assert_never_zero(slots)
        for slot in slots:
            assert slot.is_uninformative()

    def test_two_line_hard_one_hot_never_zero(self):
        probs1 = _one_hot_probs("MH12AB", hard=True)
        probs2 = _one_hot_probs("1234", hard=True)
        slots = decode_two_line_plate(probs1, probs2)
        _assert_never_zero(slots)

    def test_soft_confident_never_zero(self):
        probs = _one_hot_probs("MH12AB1234", hard=False)
        slots = decode_one_line_plate(probs)
        _assert_never_zero(slots)


class TestGrammarAlignment:
    def test_two_letter_series_four_digit_number_no_padding(self):
        probs = _one_hot_probs("MH12AB1234")
        slots = decode_one_line_plate(probs)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "MH12AB1234"

    def test_one_letter_series_three_digit_number(self):
        probs = _one_hot_probs("MH12A234")
        slots = decode_one_line_plate(probs)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "MH12A" + BLANK + BLANK + "234"

    def test_two_letter_series_one_digit_number(self):
        probs = _one_hot_probs("MH12AB1")
        slots = decode_one_line_plate(probs)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "MH12AB" + BLANK * 3 + "1"

    def test_two_letter_series_two_digit_number(self):
        probs = _one_hot_probs("DL05CD42")
        slots = decode_one_line_plate(probs)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "DL05CD" + BLANK * 2 + "42"

    def test_one_letter_series_four_digit_number(self):
        # Digits chosen with no adjacent repeats -- _one_hot_probs has no CTC
        # blank spacer, so adjacent-equal characters would wrongly collapse.
        probs = _one_hot_probs("KA01Z9876")
        slots = decode_one_line_plate(probs)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "KA01Z" + BLANK + "9876"

    def test_series_grammar_blank_not_uniform_when_next_char_is_digit(self):
        # Recogniser skipped straight from RTO to digits: the grammar itself
        # says "no series here" -- that is real evidence of an absent
        # series (a legitimate older-format plate), not missing
        # information. Slots 4-5 must be a PEAKED BLANK, like a one-letter
        # series' slot 5, not UNIFORM ("couldn't read"). A uniform
        # posterior's argmax is arbitrary (the alphabet's first letter),
        # which would corrupt the plate rather than leave it honestly
        # blank.
        probs = _one_hot_probs("MH121234")  # prefix MH12, remainder "1234" (all digits)
        slots = decode_one_line_plate(probs)
        assert not slots[4].is_uninformative()
        assert not slots[5].is_uninformative()
        assert slots[4].argmax() == BLANK
        assert slots[5].argmax() == BLANK
        # number segment (4 digits) is still read normally.
        argmax_number = "".join(s.argmax() for s in slots[6:])
        assert argmax_number == "1234"

    def test_series_uniform_when_nothing_read_past_prefix(self):
        # Contrast case: NOTHING was decoded past the state+RTO prefix at
        # all (occlusion/truncation) -- genuinely missing information, so
        # slots 4-5 (and the number slots) stay UNIFORM, not a grammar
        # guess.
        probs = _one_hot_probs("MH12")
        slots = decode_one_line_plate(probs)
        assert slots[4].is_uninformative()
        assert slots[5].is_uninformative()
        for slot in slots[6:]:
            assert slot.is_uninformative()

    @pytest.mark.parametrize(
        "raw,canonical",
        [
            # From the fpo_adapter regression cases (docs/decisions.md),
            # with adjacent-repeat digits swapped out where needed --
            # _one_hot_probs has no CTC blank spacer, so e.g. "AR200692"'s
            # adjacent "00" would wrongly collapse under greedy CTC decode
            # (a test-harness artefact, not an adapter bug; fpo_adapter's
            # own test uses the exact original strings since its fixed
            # classification heads have no such collapsing).
            ("AR152019", "AR15" + BLANK + BLANK + "2019"),
            ("AR201692", "AR20" + BLANK + BLANK + "1692"),
            ("OD021481", "OD02" + BLANK + BLANK + "1481"),
            ("PY051253", "PY05" + BLANK + BLANK + "1253"),
            ("UK185763", "UK18" + BLANK + BLANK + "5763"),
            ("WB061061", "WB06" + BLANK + BLANK + "1061"),
            ("HR696169", "HR69" + BLANK + BLANK + "6169"),
        ],
    )
    def test_no_series_letters_legacy_plates_regression(self, raw, canonical):
        # Regression for the adapter bug: legitimate older-format plates
        # with NO series letters (e.g. AR15__2019) were decoded as
        # AR15AA2019 -- the uniform series slots' argmax defaulted to 'A',
        # the alphabet's first letter, instead of the grammar-confident
        # BLANK.
        probs = _one_hot_probs(raw)
        slots = decode_one_line_plate(probs)
        assert "".join(s.argmax() for s in slots) == canonical

    def test_two_line_one_letter_series_one_digit_number(self):
        probs1 = _one_hot_probs("MH12A")
        probs2 = _one_hot_probs("7")
        slots = decode_two_line_plate(probs1, probs2)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "MH12A" + BLANK * 4 + "7"

    def test_extra_trailing_digit_noise_is_dropped(self):
        # 5-digit "number" (grammar max is 4): the 5th char is dropped, not
        # smuggled into an 11th slot.
        probs = _one_hot_probs("MH12AB12345")
        slots = decode_one_line_plate(probs)
        assert len(slots) == NUM_SLOTS
        argmax_number = "".join(s.argmax() for s in slots[6:])
        assert argmax_number == "1234"


class TestUnreadSlotsAreUniform:
    def test_short_prefix_leaves_rto_series_number_uniform(self):
        probs = _one_hot_probs("MH")  # only state read
        slots = decode_one_line_plate(probs)
        assert slots[0].argmax() == "M"
        assert slots[1].argmax() == "H"
        for slot in slots[2:]:
            assert slot.is_uninformative()

    def test_empty_decode_is_fully_uniform(self):
        t = 4
        probs = np.zeros((t, NUM_CLASSES), dtype=np.float64)
        probs[:, CTC_BLANK_IDX] = 1.0
        slots = decode_one_line_plate(probs)
        assert len(slots) == NUM_SLOTS
        for slot in slots:
            assert slot.is_uninformative()

    def test_series_alphabet_and_number_alphabet_uniform_shapes(self):
        probs = _one_hot_probs("MH")
        slots = decode_one_line_plate(probs)
        assert set(slots[4].probs.keys()) == set(SERIES_ALPHABET)
        assert set(slots[6].probs.keys()) == set(NUMBER_ALPHABET)


def test_charset_size_and_ctc_blank_index():
    assert len(CHARSET) == 36
    assert NUM_CLASSES == 37
    assert CTC_BLANK_IDX == 0


def test_ctc_greedy_decode_rejects_bad_shape():
    with pytest.raises(ValueError):
        ctc_greedy_decode(np.zeros(5))


class TestTwoRowMismatchedWrap:
    """decode_two_line_plate must not assume the physical line break falls
    at the state+RTO+series / number boundary -- real plates wrap
    inconsistently (see module docstring: WB07D5106 -> "WB07D51"/"06",
    KL01AU585 -> "KL01"/"AU585", neither at the grammar boundary)."""

    def test_wrap_mid_number_still_decodes_correctly(self):
        # "WB07D5106" wrapped as "WB07D51" / "06" -- NOT at series|number.
        probs_top = _one_hot_probs("WB07D51")
        probs_bottom = _one_hot_probs("06")
        slots = decode_two_line_plate(probs_top, probs_bottom)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "WB07D_5106"

    def test_wrap_after_rto_before_series_still_decodes_correctly(self):
        # "KL01AU585" wrapped as "KL01" / "AU585" -- NOT at series|number
        # either (series "AU" ends up entirely on the bottom row).
        probs_top = _one_hot_probs("KL01")
        probs_bottom = _one_hot_probs("AU585")
        slots = decode_two_line_plate(probs_top, probs_bottom)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "KL01AU" + BLANK + "585"

    def test_standard_wrap_still_works(self):
        # The "normal" wrap (state+RTO+series on top, number on bottom)
        # must still work -- this is the common case, not just the odd one.
        probs_top = _one_hot_probs("MH12AB")
        probs_bottom = _one_hot_probs("1234")
        slots = decode_two_line_plate(probs_top, probs_bottom)
        argmax = "".join(s.argmax() for s in slots)
        assert argmax == "MH12AB1234"


class TestBestOfTwoPicker:
    """decode_plate_best_of_two: DESIGN A -- a crop's layout is discovered
    by which reading is more confident, never trusted from a stored label."""

    def test_one_line_shaped_input_picks_one_line_reading(self):
        # A crop that reads perfectly as one coherent line should win over
        # forcing an artificial (and here, garbage) two-row split.
        probs_whole = _one_hot_probs("MH12AB1234")
        # Splitting the SAME crop in half physically would show partial,
        # incoherent glyphs to each half -- simulate that with a poor,
        # low-confidence, mismatched read on each half.
        probs_top = _one_hot_probs("MH12A", hard=False)
        probs_bottom = _one_hot_probs("Z8765", hard=False)
        slots, which = decode_plate_best_of_two(probs_whole, probs_top, probs_bottom)
        assert which == "one_line"
        assert "".join(s.argmax() for s in slots) == "MH12AB1234"

    def test_two_row_shaped_input_picks_two_row_reading(self):
        # A crop that reads perfectly as two coherent rows should win over
        # a poor one-line reading of the same (visually stacked) content.
        probs_top = _one_hot_probs("MH12AB")
        probs_bottom = _one_hot_probs("1234")
        probs_whole = _one_hot_probs("M1H2A", hard=False)  # incoherent whole-crop read
        slots, which = decode_plate_best_of_two(probs_whole, probs_top, probs_bottom)
        assert which == "two_row"
        assert "".join(s.argmax() for s in slots) == "MH12AB1234"

    def test_returns_valid_slot_list_either_way(self):
        probs_whole = _one_hot_probs("DL05CD1234")
        probs_top = _one_hot_probs("DL05CD")
        probs_bottom = _one_hot_probs("1234")
        slots, which = decode_plate_best_of_two(probs_whole, probs_top, probs_bottom)
        assert len(slots) == NUM_SLOTS
        assert which in ("one_line", "two_row")


def test_plate_log_prob_is_higher_for_more_confident_reading():
    confident = _one_hot_probs("MH12AB1234", hard=False)
    slots_confident = decode_one_line_plate(confident)
    slots_uniform = decode_one_line_plate(np.full((1, NUM_CLASSES), 1.0 / NUM_CLASSES))
    assert plate_log_prob(slots_confident) > plate_log_prob(slots_uniform)
