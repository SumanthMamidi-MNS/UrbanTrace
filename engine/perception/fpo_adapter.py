"""fast-plate-ocr per-slot softmax -> canonical 10-slot plate posterior.

`fast-plate-ocr` (github.com/ankandrew/fast-plate-ocr, MIT) is a third-party,
pretrained, global (multi-country) license-plate OCR. Unlike our own CRNN
(engine/perception/ctc_to_slots.py), it is not CTC-based: it has a FIXED
number of classification heads (`max_plate_slots`, 10 for the pretrained
`cct-s-v2-global-model`), each a per-slot softmax over its OWN alphabet
(digits + A-Z + one pad character, e.g. `'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'`
for that model) -- no repeat-collapse/blank decoding is needed, just an
argmax per slot. Plates are left-aligned; unused trailing slots predict the
pad character.

This module is DELIBERATELY numpy-only -- no `fast_plate_ocr` or
`onnxruntime` import anywhere here -- so it is testable end to end from the
MAIN project venv, exactly like `ctc_to_slots.py`. Callers in the OCR venv
(`eval_fpo.py`, `train_fpo.py`) run the actual model and hand this module a
plain `(max_plate_slots, vocab_size)` numpy softmax array plus the model's
own alphabet string and pad character (both come from the model's
`plate_config.yaml`, read with plain `pyyaml`, not the `fast_plate_ocr`
package).

Grammar reuse (per the reviewer's explicit instruction: reuse
`ctc_to_slots.py`'s Indian plate grammar, don't write a new parser)
--------------------------------------------------------------------
The canonical form is the same `SS DD LL DDDD` (state, RTO, series, number)
fixed 10-slot layout `ctc_to_slots.py` decodes into, with the same
left-aligned series / right-aligned number padding convention. Once
fast-plate-ocr's raw output is turned into (a) a decoded string (argmax per
slot, truncated at the model's own pad character -- see `_fpo_decode`) and
(b) a list of "positions" pointing back at each decoded character's slot
row, the actual segmentation rules -- state+RTO is always exactly 4
characters if read at all, series is the leading run of letters after that
capped at `MAX_SERIES_LEN`, the remainder (capped at `MAX_NUMBER_LEN`) is the
number -- are the EXACT SAME rules, reusing `ctc_to_slots`'s own
`_split_series_number` and its constants (`STATE_RTO_LEN`, `MAX_SERIES_LEN`,
`MAX_NUMBER_LEN`, `GRAMMAR_BLANK_MASS`) directly rather than
reimplementing them.

What could NOT be reused verbatim is `ctc_to_slots._char_probs_from_softmax`
and the `_align_*` functions built on it: they hardcode ctc_to_slots' OWN
CRNN charset-to-column mapping (`_CHAR_TO_LABEL`, alphabet order A-Z then
0-9, offset by +1 for the CTC blank at column 0). fast-plate-ocr's model
uses a different alphabet and column order entirely (digits first, its own
pad character, no CTC blank), so restricting/renormalising one of ITS
softmax rows onto one of our canonical slot alphabets needs its own lookup
(`_char_probs_from_fpo_row` below) keyed off the model's own alphabet
string. The three `_align_*_fpo` functions below are otherwise structurally
identical to ctc_to_slots's `_align_state_rto`/`_align_series`/
`_align_number` -- same segment lengths, same blank-vs-uniform distinction --
just wired to that lookup instead.

Invariant (docs/architecture.md sec 3): every slot posterior keeps
background mass on every character in its alphabet -- enforced the same way
as ctc_to_slots, via `FLOOR` applied before renormalising.
"""

import numpy as np

from engine.contracts.plate import (
    BLANK,
    NUM_SLOTS,
    NUMBER_ALPHABET,
    RTO_ALPHABET,
    SERIES_ALPHABET,
    STATE_ALPHABET,
    SlotPosterior,
)
from engine.perception.ctc_to_slots import (
    FLOOR,
    GRAMMAR_BLANK_MASS,
    STATE_RTO_LEN,
    _split_series_number,
)

# A "position" for the fast-plate-ocr adapter: (per_slot_probs, alphabet_str,
# slot_index). Unlike ctc_to_slots's `Position` (which shares one CRNN
# column layout across the whole plate), each fast-plate-ocr slot is its own
# independent classification head, but in practice all heads share one
# alphabet/column layout, so `alphabet_str` is carried alongside for
# self-containedness rather than assumed global state.
FpoPosition = tuple[np.ndarray, str, int]


def _row_at(position: FpoPosition) -> tuple[np.ndarray, str]:
    probs, alphabet_str, slot_idx = position
    return probs[slot_idx], alphabet_str


def _char_probs_from_fpo_row(
    row: np.ndarray, model_alphabet: str, target_alphabet: list[str], floor: float = FLOOR
) -> dict[str, float]:
    """Restrict one fast-plate-ocr slot's softmax row -- indexed by the
    MODEL's own alphabet string -- to `target_alphabet` (one of our
    canonical slot alphabets) and renormalise, flooring every entry first.
    Mirrors `ctc_to_slots._char_probs_from_softmax` exactly in spirit; the
    only difference is the column lookup is keyed by a caller-supplied
    alphabet string instead of ctc_to_slots's own fixed CHARSET, since
    fast-plate-ocr's class ordering (and pad character) differs from ours.

    Unlike ctc_to_slots's own CRNN (which has no class at all for BLANK, the
    canonical pad symbol -- it always falls back to `floor`), fast-plate-ocr
    DOES have a real class for its pad character. When `target_alphabet`
    contains BLANK ("_") and the model's pad character is also "_", this
    correctly uses the model's actual predicted pad probability instead of
    an uninformative floor.
    """
    char_to_col = {c: i for i, c in enumerate(model_alphabet)}
    probs: dict[str, float] = {}
    for c in target_alphabet:
        if c in char_to_col:
            probs[c] = float(row[char_to_col[c]])
        else:
            # The model's alphabet has no class for this canonical
            # character at all (e.g. its pad char isn't "_") -- no direct
            # signal, so treat like ctc_to_slots treats BLANK: floor only.
            probs[c] = floor
    probs = {c: max(p, floor) for c, p in probs.items()}
    total = sum(probs.values())
    return {c: p / total for c, p in probs.items()}


def _slot_from_fpo_row(position: FpoPosition, alphabet: list[str]) -> SlotPosterior:
    row, model_alphabet = _row_at(position)
    return SlotPosterior(probs=_char_probs_from_fpo_row(row, model_alphabet, alphabet))


def _grammar_blank_slot(alphabet: list[str]) -> SlotPosterior:
    return SlotPosterior.peaked(alphabet, BLANK, GRAMMAR_BLANK_MASS)


def _align_state_rto_fpo(
    decoded_prefix: str, prefix_positions: list[FpoPosition]
) -> list[SlotPosterior]:
    """Slots 0-3 (state x2, RTO x2). Same rule as
    `ctc_to_slots._align_state_rto`: a decoded prefix shorter than 4 means
    the recogniser never confidently got that far -- UNIFORM, not a guess."""
    alphabets = [STATE_ALPHABET, STATE_ALPHABET, RTO_ALPHABET, RTO_ALPHABET]
    slots: list[SlotPosterior] = []
    for i in range(STATE_RTO_LEN):
        if i < len(decoded_prefix):
            slots.append(_slot_from_fpo_row(prefix_positions[i], alphabets[i]))
        else:
            slots.append(SlotPosterior.uniform(alphabets[i]))
    return slots


def _uniform_series_pair() -> list[SlotPosterior]:
    return [SlotPosterior.uniform(SERIES_ALPHABET), SlotPosterior.uniform(SERIES_ALPHABET)]


def _align_series_fpo(
    series_part: str, series_positions: list[FpoPosition], remainder_has_content: bool
) -> list[SlotPosterior]:
    """Slots 4-5, left-aligned. Same rule as `ctc_to_slots._align_series`:
    an empty series_part is ambiguous on its own and must be disambiguated
    by whether anything was read after it (`remainder_has_content`). If the
    recogniser confidently read digits immediately after the state+RTO
    prefix, that is real grammar evidence the series is absent (e.g. older
    plates like `AR15__2019`) -- a GRAMMAR_BLANK_MASS-peaked BLANK, not
    UNIFORM. UNIFORM is reserved for "nothing was read past the prefix at
    all" (occlusion/truncation). Uniform means "couldn't read", not
    "absent"."""
    n = len(series_part)
    if n == 0:
        if remainder_has_content:
            return [_grammar_blank_slot(SERIES_ALPHABET), _grammar_blank_slot(SERIES_ALPHABET)]
        return _uniform_series_pair()
    slots = [_slot_from_fpo_row(series_positions[i], SERIES_ALPHABET) for i in range(n)]
    for _ in range(n, 2):
        slots.append(_grammar_blank_slot(SERIES_ALPHABET))
    return slots


def _align_number_fpo(number_part: str, number_positions: list[FpoPosition]) -> list[SlotPosterior]:
    """Slots 6-9, right-aligned. Same rule as `ctc_to_slots._align_number`."""
    n = len(number_part)
    if n == 0:
        return [SlotPosterior.uniform(NUMBER_ALPHABET) for _ in range(4)]
    pad = 4 - n
    slots = [_grammar_blank_slot(NUMBER_ALPHABET) for _ in range(pad)]
    slots.extend(_slot_from_fpo_row(number_positions[i], NUMBER_ALPHABET) for i in range(n))
    return slots


def _fpo_decode(
    plate_probs: np.ndarray, model_alphabet: str, pad_char: str
) -> tuple[str, list[FpoPosition]]:
    """Argmax each of fast-plate-ocr's fixed classification heads, then take
    only the prefix up to (not including) the first predicted pad
    character. Unlike a plain `str.rstrip(pad_char)` (what
    `fast_plate_ocr.core.process.postprocess_output` does for a plain
    display string), truncating at the FIRST pad char rather than stripping
    only a trailing run is the conservative choice for evidence: if the
    model ever predicts a pad character before the end (a "hole"), treating
    everything from there on as unread (uniform) is safer than feeding a
    pad symbol into a state/RTO alphabet that doesn't even contain it.

    Returns the decoded prefix string and one `FpoPosition` per decoded
    character (each just points back at this same `plate_probs` array and
    alphabet, at that character's slot index) -- the shape
    `ctc_to_slots`'s alignment functions expect from a CTC decode, so the
    same grammar segmentation applies unmodified.
    """
    if plate_probs.ndim != 2:
        raise ValueError(
            f"plate_probs must be (max_plate_slots, vocab_size), got shape {plate_probs.shape}"
        )
    alphabet_array = np.array(list(model_alphabet))
    argmax_idx = np.argmax(plate_probs, axis=-1)
    raw_chars = alphabet_array[argmax_idx].tolist()

    decoded_chars: list[str] = []
    for c in raw_chars:
        if c == pad_char:
            break
        decoded_chars.append(c)
    decoded = "".join(decoded_chars)
    positions: list[FpoPosition] = [(plate_probs, model_alphabet, i) for i in range(len(decoded))]
    return decoded, positions


def fpo_output_to_canonical_slots(
    plate_probs: np.ndarray, model_alphabet: str, pad_char: str
) -> list[SlotPosterior]:
    """Full pipeline: one fast-plate-ocr forward pass, already reduced to a
    plain `(max_plate_slots, vocab_size)` numpy softmax array (e.g. the
    "plate" output of `LicensePlateRecognizer`/the raw ONNX session output,
    reshaped), -> 10 canonical `SlotPosterior`s. Pure numpy; the caller is
    responsible for running the actual model (OCR venv only).

    `model_alphabet`/`pad_char` come from the model's own `plate_config.yaml`
    (e.g. `alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'`, `pad_char: '_'`
    for `cct-s-v2-global-model`) -- never hardcoded here, since a different
    hub model could use a different alphabet/pad character.
    """
    decoded, positions = _fpo_decode(plate_probs, model_alphabet, pad_char)

    prefix, prefix_pos = decoded[:STATE_RTO_LEN], positions[:STATE_RTO_LEN]
    remainder, remainder_pos = decoded[STATE_RTO_LEN:], positions[STATE_RTO_LEN:]

    state_rto_slots = _align_state_rto_fpo(prefix, prefix_pos)

    if len(prefix) < STATE_RTO_LEN:
        # Same rule as ctc_to_slots._decode_slots_from_positions: a short
        # prefix already means "unread from here on".
        series_slots = _uniform_series_pair()
        number_slots = [SlotPosterior.uniform(NUMBER_ALPHABET) for _ in range(4)]
    else:
        series_part, number_part = _split_series_number(remainder)
        series_pos = remainder_pos[: len(series_part)]
        number_pos = remainder_pos[len(series_part) : len(series_part) + len(number_part)]
        series_slots = _align_series_fpo(
            series_part, series_pos, remainder_has_content=len(remainder) > 0
        )
        number_slots = _align_number_fpo(number_part, number_pos)

    slots = state_rto_slots + series_slots + number_slots
    assert len(slots) == NUM_SLOTS
    return slots


def load_plate_config(plate_config_path: str) -> tuple[str, str, int]:
    """Read `max_plate_slots`, `alphabet` and `pad_char` from a
    fast-plate-ocr `plate_config.yaml` using plain `pyyaml` -- NOT the
    `fast_plate_ocr` package, so this stays importable from the main venv.
    Returns (alphabet, pad_char, max_plate_slots).
    """
    import yaml

    with open(plate_config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["alphabet"], data["pad_char"], data["max_plate_slots"]
