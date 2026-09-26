"""CTC softmax output -> canonical 10-slot plate posterior (engine.contracts.plate).

THE CRITICAL PIECE of the OCR module: this is what lets a CRNN's raw
per-timestep character distribution become the `plate_posterior` the engine
already consumes (see docs/architecture.md sec 3-4). It is deliberately
numpy-only -- no torch import anywhere in this module -- so it can be tested
end to end from the MAIN project venv (which never gains torch); the actual
CRNN (engine/perception/crnn.py) produces a plain numpy softmax array and
hands it to the functions here.

Charset and CTC decoding
-------------------------
The recogniser's output classes are the 36 characters that actually appear
printed on a plate -- A-Z, 0-9 -- plus one CTC blank. There is no printed
glyph for the canonical plate's "_" blank slot filler (engine.contracts.plate.
BLANK): that symbol exists only in the fixed 10-slot representation, never on
a physical plate, so the model never needs to predict it directly.

Grammar alignment
-----------------
The canonical plate is `SS DD LL DDDD` (state, RTO, series, number; see
engine/contracts/plate.py). A decoded string has a KNOWN, fixed-width state
segment (2 letters) and RTO segment (2 digits) -- always exactly 4
characters if read at all -- followed by a VARIABLE-width series (1-2
letters) and number (1-4 digits). The split between series and number is
recovered the same way a human reads it: series is the leading run of
letters (capped at 2) right after the RTO digits; everything after that is
the number (capped at 4 -- any characters beyond that are treated as
spurious OCR noise and dropped, a documented, deliberately conservative
choice over trying to guess which extra character is real).

  - Series is LEFT-aligned into slots 4-5: a 1-letter series fills slot 4 and
    leaves slot 5 "grammatically blank" (see below).
  - Number is RIGHT-aligned into slots 6-9: a 2-digit number fills slots 8-9
    and leaves slots 6-7 "grammatically blank".

This mirrors sim/vehicles.py's `_sample_true_plate` padding convention
exactly, so a synthetic-plate label string round-trips through this module
byte-for-byte into the same canonical form the simulator's ground truth uses.

Two kinds of "blank" -- an important distinction this module keeps sharp:

  - "Grammatically blank, with grammar-level confidence": the model read
    enough of the plate to know a slot MUST be a pad character (e.g. it read
    a 1-letter series, so slot 5 has to be the pad). This is real evidence,
    encoded as `SlotPosterior.peaked(alphabet, BLANK, mass=GRAMMAR_BLANK_MASS)`
    -- not certainty (mass < 1), because the model could always have missed a
    second series letter entirely.
  - "Unread / missing information": the model produced too few characters to
    even reach this slot (occlusion, truncation, garbage). Per the
    architecture's invariant, missing information gets the UNIFORM posterior
    (contributes exactly zero to the plate LR), never a guess.

Invariant (docs/architecture.md sec 3, "Day 1c" in docs/decisions.md): every
slot posterior keeps background mass on every character in its alphabet --
no character is ever assigned probability exactly zero. This module enforces
it in `_char_probs_from_softmax`, the single place that turns a raw softmax
row into a slot's probability dict, with an explicit floor applied before
renormalising.

Two-row plates, without trusting a layout label
------------------------------------------------
`decode_two_line_plate` does NOT assume the top row is exactly state+RTO+
series and the bottom row is exactly the number. Real two-line plates wrap
inconsistently -- inspecting the real dataset this module was built against
found genuine plates like `WB07D5106` printed as "WB07D51" / "06" (wrapped
mid-NUMBER) and `KL01AU585` as "KL01" / "AU585" (wrapped after RTO, before
the series), neither matching the state+RTO+series / number boundary a
naive implementation would assume. Instead, each row is decoded
independently (its own CTC run), the two decoded character streams are
CONCATENATED (top's characters first, then bottom's), and the exact same
grammar alignment used for a one-line plate (`_decode_slots_from_positions`)
runs on the concatenation -- so the split point in the physical image never
matters, only the characters' types (letter vs digit) do, exactly like
reading a one-line plate. Both `decode_one_line_plate` and
`decode_two_line_plate` are thin wrappers around this one shared aligner;
each decoded character's softmax row is looked up via a `(probs_array,
timestep)` "position" pair rather than a single shared timestep index,
since a two-row plate's rows come from two independent forward passes with
no shared time axis.

`decode_plate_best_of_two` (used by eval_ocr.py and video_to_events.py, see
their module docstrings for the DESIGN A rationale) decodes a crop BOTH
ways -- once as a single row, once as two stacked rows -- and picks
whichever reading has the higher plate-level log-probability
(`plate_log_prob`: sum of each slot's log peak probability). This makes the
crop's stored `layout` column purely informational: a wrongly-labelled or
unlabelled crop still gets read correctly as long as one of the two
readings is coherent.
"""

import math

import numpy as np

from engine.contracts.plate import (
    BLANK,
    NUM_SLOTS,
    NUMBER_ALPHABET,
    PLATE_SLOTS,
    RTO_ALPHABET,
    SERIES_ALPHABET,
    STATE_ALPHABET,
    SlotPosterior,
)

# --- Charset shared with engine/perception/crnn.py (which imports these
# constants rather than redefining them, so model and decoder can never
# disagree about class indices). ---
CHARSET: list[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(10)]
CTC_BLANK_IDX = 0
NUM_CLASSES = len(CHARSET) + 1  # +1 for the CTC blank class

_CHAR_TO_LABEL = {c: i + 1 for i, c in enumerate(CHARSET)}  # labels 1..36
_LABEL_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}

# Defensive floor applied to every alphabet character before renormalising a
# slot posterior -- belt-and-suspenders alongside the codec's own residual
# term and plate_lr's floor (see engine/contracts/codec.py). No character a
# slot's alphabet contains should ever reach the engine as an exact 0.0.
FLOOR = 1e-6

# Mass assigned to a "grammatically blank, model-confident" slot (e.g. slot 5
# when a 1-letter series was read). Deliberately < 1: a two-letter series
# whose second letter was entirely dropped by the recogniser is a real
# failure mode, not one this module should rule out with certainty.
GRAMMAR_BLANK_MASS = 0.98

# Series is capped at 2 letters, number at 4 digits (engine.contracts.plate
# grammar). Extra leading letters beyond 2, or extra trailing digits beyond
# 4, are treated as spurious OCR noise past the plate's true content and
# dropped -- there is no principled way to know which of the extra
# characters is the "real" one, and keeping them would silently violate the
# fixed 10-slot contract.
MAX_SERIES_LEN = 2
MAX_NUMBER_LEN = 4
STATE_RTO_LEN = 4  # always exactly 2 state letters + 2 RTO digits


def label_to_char(label: int) -> str:
    return _LABEL_TO_CHAR[label]


def char_to_label(char: str) -> int:
    return _CHAR_TO_LABEL[char]


def ctc_greedy_decode(probs: np.ndarray, blank_idx: int = CTC_BLANK_IDX) -> tuple[str, list[int]]:
    """Greedy CTC decode: argmax per timestep, collapse consecutive repeats,
    drop blanks.

    `probs` is (T, C) softmax probabilities (NOT logits, NOT log-probs).
    Returns the decoded string and, for each decoded character, the
    "representative timestep" -- the single most-confident timestep within
    that character's contiguous run -- whose full softmax row is what
    `_align_*` restricts into a slot posterior. Using the run's peak rather
    than e.g. its first timestep is a small robustness choice: CTC runs are
    often several timesteps wide with confidence ramping up in the middle.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be (T, C), got shape {probs.shape}")
    raw_labels = np.argmax(probs, axis=1)

    runs: list[tuple[int, list[int]]] = []
    for t, label in enumerate(raw_labels.tolist()):
        if runs and runs[-1][0] == label:
            runs[-1][1].append(t)
        else:
            runs.append((label, [t]))

    chars: list[str] = []
    timesteps: list[int] = []
    for label, ts_list in runs:
        if label == blank_idx:
            continue
        best_t = max(ts_list, key=lambda t: probs[t, label])
        chars.append(label_to_char(label))
        timesteps.append(best_t)
    return "".join(chars), timesteps


def _char_probs_from_softmax(
    softmax_row: np.ndarray, alphabet: list[str], floor: float = FLOOR
) -> dict[str, float]:
    """Restrict one timestep's full-charset softmax row to `alphabet` and
    renormalise, flooring every entry first so a hard-zero softmax output (or
    a character with no CRNN class at all, i.e. BLANK) can never zero out an
    alphabet character -- the architecture invariant."""
    probs: dict[str, float] = {}
    for c in alphabet:
        if c in _CHAR_TO_LABEL:
            probs[c] = float(softmax_row[_CHAR_TO_LABEL[c]])
        else:
            # BLANK ("_"): no printed glyph, so no CRNN class -- the model
            # has no direct signal about it. Give it the same floor as any
            # other unlikely-but-not-impossible character.
            probs[c] = floor
    probs = {c: max(p, floor) for c, p in probs.items()}
    total = sum(probs.values())
    return {c: p / total for c, p in probs.items()}


def _slot_from_softmax(softmax_row: np.ndarray, alphabet: list[str]) -> SlotPosterior:
    return SlotPosterior(probs=_char_probs_from_softmax(softmax_row, alphabet))


def _grammar_blank_slot(alphabet: list[str]) -> SlotPosterior:
    return SlotPosterior.peaked(alphabet, BLANK, GRAMMAR_BLANK_MASS)


# A "position" identifies one decoded character's evidence: the softmax
# array it came from plus its timestep within that array. A single-line
# decode uses the SAME array for every position; a two-row decode's
# positions point into whichever of the two independent arrays (top's or
# bottom's) that character was read from. Every `_align_*` function below
# takes positions rather than a bare timestep list + shared array, which is
# what lets one-line and two-row decoding share one implementation.
Position = tuple[np.ndarray, int]


def _row_at(position: Position) -> np.ndarray:
    probs, t = position
    return probs[t]


def _align_state_rto(decoded_prefix: str, prefix_positions: list[Position]) -> list[SlotPosterior]:
    """Slots 0-3 (state x2, RTO x2). A decoded prefix shorter than 4 means
    the recogniser never got that far -- genuinely missing information, so
    the remaining slots are UNIFORM, not a guess."""
    alphabets = [STATE_ALPHABET, STATE_ALPHABET, RTO_ALPHABET, RTO_ALPHABET]
    slots: list[SlotPosterior] = []
    for i in range(STATE_RTO_LEN):
        if i < len(decoded_prefix):
            slots.append(_slot_from_softmax(_row_at(prefix_positions[i]), alphabets[i]))
        else:
            slots.append(SlotPosterior.uniform(alphabets[i]))
    return slots


def _split_series_number(remainder: str) -> tuple[str, str]:
    """Split the characters after state+RTO into (series_part, number_part).
    Series is the leading run of letters, capped at MAX_SERIES_LEN; anything
    genuinely a digit at the front (i.e. the recogniser skipped straight to
    digits) yields an empty series_part, which is correctly treated as
    "series unread" downstream, not "series legitimately empty" (the grammar
    never allows a 0-letter series)."""
    i = 0
    while i < len(remainder) and i < MAX_SERIES_LEN and remainder[i].isalpha():
        i += 1
    series_part = remainder[:i]
    number_part = remainder[i:][:MAX_NUMBER_LEN]
    return series_part, number_part


def _uniform_series_pair() -> list[SlotPosterior]:
    """Slots 4-5 when the series segment was never read at all (occlusion/
    truncation) -- UNIFORM, not blank. Factored out since three call sites
    need this exact pair."""
    return [SlotPosterior.uniform(SERIES_ALPHABET), SlotPosterior.uniform(SERIES_ALPHABET)]


def _align_series(
    series_part: str, series_positions: list[Position], remainder_has_content: bool
) -> list[SlotPosterior]:
    """Slots 4-5, left-aligned.

    Empty series_part is ambiguous on its own and must be disambiguated by
    whether anything was read AFTER it (`remainder_has_content`, i.e. the
    full post-prefix remainder, before the series/number split, was
    non-empty):

      - remainder_has_content=False: nothing was decoded past the state+RTO
        prefix at all -- genuinely missing information (occlusion/
        truncation) -- UNIFORM, not a guess.
      - remainder_has_content=True: the recogniser confidently read
        characters immediately after the prefix and they were NOT letters
        (it skipped straight to digits) -- the grammar itself says "no
        series here". That is real evidence a series is absent, exactly
        like a one-letter series leaving slot 5 grammatically blank, so it
        gets the same GRAMMAR_BLANK_MASS-peaked BLANK, not UNIFORM. Uniform
        means "couldn't read"; this is "read, and there's no series."
    """
    n = len(series_part)
    if n == 0:
        if remainder_has_content:
            return [_grammar_blank_slot(SERIES_ALPHABET), _grammar_blank_slot(SERIES_ALPHABET)]
        return _uniform_series_pair()
    slots = [_slot_from_softmax(_row_at(series_positions[i]), SERIES_ALPHABET) for i in range(n)]
    for _ in range(n, 2):
        slots.append(_grammar_blank_slot(SERIES_ALPHABET))
    return slots


def _align_number(number_part: str, number_positions: list[Position]) -> list[SlotPosterior]:
    """Slots 6-9, right-aligned. Empty number_part means the number segment
    was never read at all -- UNIFORM for all 4 slots."""
    n = len(number_part)
    if n == 0:
        return [SlotPosterior.uniform(NUMBER_ALPHABET) for _ in range(4)]
    pad = 4 - n
    slots = [_grammar_blank_slot(NUMBER_ALPHABET) for _ in range(pad)]
    slots.extend(
        _slot_from_softmax(_row_at(number_positions[i]), NUMBER_ALPHABET) for i in range(n)
    )
    return slots


def _decode_slots_from_positions(decoded: str, positions: list[Position]) -> list[SlotPosterior]:
    """The one grammar aligner shared by `decode_one_line_plate` and
    `decode_two_line_plate` (see module docstring): state(2)+RTO(2) prefix,
    then a series/number split recovered purely from character TYPE (letter
    vs digit), never from where a physical line-break fell."""
    prefix, prefix_pos = decoded[:STATE_RTO_LEN], positions[:STATE_RTO_LEN]
    remainder, remainder_pos = decoded[STATE_RTO_LEN:], positions[STATE_RTO_LEN:]

    state_rto_slots = _align_state_rto(prefix, prefix_pos)

    # Only split series/number out of the remainder if the full prefix was
    # actually read -- a short prefix already means "unread from here on",
    # and guessing a series/number split from characters that may not even
    # be state/RTO continuation would be worse than admitting ignorance.
    if len(prefix) < STATE_RTO_LEN:
        series_slots = _uniform_series_pair()
        number_slots = [SlotPosterior.uniform(NUMBER_ALPHABET) for _ in range(4)]
    else:
        series_part, number_part = _split_series_number(remainder)
        series_pos = remainder_pos[: len(series_part)]
        number_pos = remainder_pos[len(series_part) : len(series_part) + len(number_part)]
        series_slots = _align_series(
            series_part, series_pos, remainder_has_content=len(remainder) > 0
        )
        number_slots = _align_number(number_part, number_pos)

    slots = state_rto_slots + series_slots + number_slots
    assert len(slots) == NUM_SLOTS
    return slots


def decode_one_line_plate(probs: np.ndarray, blank_idx: int = CTC_BLANK_IDX) -> list[SlotPosterior]:
    """Full pipeline for a single-line plate crop: one CRNN forward pass
    already reduced to a (T, C) softmax array -> 10 canonical SlotPosteriors.
    """
    decoded, timesteps = ctc_greedy_decode(probs, blank_idx)
    positions = [(probs, t) for t in timesteps]
    return _decode_slots_from_positions(decoded, positions)


def decode_two_line_plate(
    probs_line1: np.ndarray, probs_line2: np.ndarray, blank_idx: int = CTC_BLANK_IDX
) -> list[SlotPosterior]:
    """Two-row plate: each row is decoded independently (its own CTC run
    over its own forward pass), then the two decoded character streams are
    CONCATENATED (row 1's characters first, row 2's second) and the exact
    same grammar aligner a one-line plate uses runs on the concatenation.
    See module docstring for why this does NOT assume row 1 is exactly
    state+RTO+series and row 2 is exactly the number -- real two-line
    plates wrap inconsistently, and this makes the physical wrap point
    irrelevant to the result.
    """
    decoded1, ts1 = ctc_greedy_decode(probs_line1, blank_idx)
    decoded2, ts2 = ctc_greedy_decode(probs_line2, blank_idx)
    decoded = decoded1 + decoded2
    positions = [(probs_line1, t) for t in ts1] + [(probs_line2, t) for t in ts2]
    return _decode_slots_from_positions(decoded, positions)


def plate_log_prob(slots: list[SlotPosterior]) -> float:
    """Plate-level log-probability: sum of each slot's log peak probability
    (log of the product-of-peaks whole-plate confidence -- see
    engine/perception/eval_ocr.py's FINDING 2). Used to compare two
    candidate readings of the same crop (`decode_plate_best_of_two`) on a
    scale where higher is always better and summation (not multiplication)
    avoids underflow across 10 slots."""
    return float(sum(math.log(s.probs[s.argmax()]) for s in slots))


def decode_plate_best_of_two(
    probs_whole: np.ndarray,
    probs_top: np.ndarray,
    probs_bottom: np.ndarray,
    blank_idx: int = CTC_BLANK_IDX,
) -> tuple[list[SlotPosterior], str]:
    """Decode a crop BOTH as a single row (`probs_whole`) and as two stacked
    rows (`probs_top`/`probs_bottom`, e.g. from
    engine.perception.crnn.split_two_line_crop), and return whichever
    reading has the higher plate-level log-probability
    (`plate_log_prob`), plus which one won ("one_line" or "two_row") --
    purely informational, callers never need to branch on it. This is what
    lets eval_ocr.py and video_to_events.py stop depending on a stored
    layout label (DESIGN A): a crop's true layout is discovered by which
    reading is actually more confident, not asserted in advance."""
    slots_one = decode_one_line_plate(probs_whole, blank_idx)
    slots_two = decode_two_line_plate(probs_top, probs_bottom, blank_idx)
    if plate_log_prob(slots_two) > plate_log_prob(slots_one):
        return slots_two, "two_row"
    return slots_one, "one_line"


def canonical_string_to_slots(canonical: str) -> str:
    """Utility for tests/eval: identity-check that a canonical 10-char string
    (state+rto+series padded with BLANK+number padded with BLANK, exactly the
    form sim/vehicles.py._sample_true_plate produces) has the right shape."""
    if len(canonical) != NUM_SLOTS:
        raise ValueError(f"canonical plate must be {NUM_SLOTS} chars, got {len(canonical)!r}")
    for i, c in enumerate(canonical):
        if c not in PLATE_SLOTS[i]:
            raise ValueError(f"char {c!r} at slot {i} not in alphabet {PLATE_SLOTS[i]!r}")
    return canonical
