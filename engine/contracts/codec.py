"""Compact on-disk codec for DetectionEvent (see docs/decisions.md, "Day 1b").

The IN-MEMORY contract (DetectionEvent, SlotPosterior) does not change: a
hydrated event always has a full dense `plate_posterior` and a full 128-d
`embedding`. Only the ON-DISK representation is compact:

- A slot posterior's dense dict[str, float] is far heavier than the
  information it carries. The corruption model (sim/corruption.py) always
  builds a posterior out of at most three internally-*uniform* cohorts: the
  peak (read) character, an optional confusable-group "leak" cohort (a
  handful of characters sharing one elevated value), and a "background" rest
  cohort (every other alphabet character sharing one small value). We keep
  the peak and any leak cohort explicit (rounded to ROUND_DECIMALS — this is
  the real noisy-channel signal plate_lr needs) and fold the background
  cohort into one extra key `"*"` carrying its *residual* total mass. On
  load, the residual is spread back uniformly over every alphabet character
  not otherwise listed — which is exact, since that cohort really was
  uniform — so no character the codec drops ever comes back as an exact zero
  (see docs/decisions.md, "Day 1c": posteriors must always retain background
  mass on every character, or a single out-of-group OCR error sends the
  plate LR's log product to -inf). A magnitude threshold alone can't safely
  tell "background" from "leak" apart (an unusually common character with no
  confusable group at all can leak more per-symbol mass than a genuine but
  tight 3-member leak group), so cohorts are identified structurally: the
  lowest-valued group of >=2 identically-valued non-peak characters is the
  background. A fully uninformative (uniform/unread) slot is still
  serialised as the sentinel `1` instead of writing every alphabet symbol at
  equal weight.
- The 128-d embedding never goes into the JSONL at all; it is written to a
  companion float16 `embeddings.npy`, and the event carries `embedding_ref`
  (its row index) instead.

engine.contracts.store.EventStore is the only thing that should read this
format back into full DetectionEvent objects.
"""

from collections import defaultdict
from typing import Any

from engine.contracts.events import DetectionEvent, VehicleAttributes
from engine.contracts.plate import SlotPosterior

SPARSE_THRESHOLD = 1e-3
ROUND_DECIMALS = 4
# Attribute confidence scores aren't covered by the posterior round-trip
# tolerance requirement, so they can round more aggressively.
ATTR_DECIMALS = 2

# Sentinel written in place of a full uniform distribution.
_UNIFORM_SENTINEL = 1

# Key carrying the residual ("everything else") mass in a sparsified slot
# dict. Never collides with a real slot symbol (letters, digits, "_" blank).
RESIDUAL_KEY = "*"

# Defensive floor applied to every alphabet character on decode, regardless
# of the stored residual. Belt-and-suspenders alongside the residual term
# itself and plate_lr's own floor (docs/decisions.md, "Day 1c") — no slot
# posterior handed to the engine should ever contain a literal 0.0.
_DECODE_FLOOR = 1e-9


def _background_cohort(non_peak: dict[str, float]) -> set[str]:
    """Identify the "rest of the alphabet" cohort among a slot's non-peak
    characters: the lowest-valued group of >=2 characters that share an
    (exactly, by construction) identical probability. Returns the empty set
    if no such cohort exists (e.g. everything left is a single, or every
    remaining character has a distinct value)."""
    by_value: dict[float, list[str]] = defaultdict(list)
    for c, v in non_peak.items():
        by_value[v].append(c)
    multi_member_values = [v for v, cs in by_value.items() if len(cs) >= 2]
    if not multi_member_values:
        return set()
    return set(by_value[min(multi_member_values)])


def encode_slot_posterior(
    sp: SlotPosterior, threshold: float = SPARSE_THRESHOLD, decimals: int = ROUND_DECIMALS
) -> dict[str, float] | int:
    """Sparsify one slot's posterior for storage. Uninformative slots collapse
    to a single sentinel int. Otherwise the peak character and any distinct
    confusable-"leak" cohort are kept explicit (rounded to `decimals`); the
    uniform background cohort (identified structurally — see
    `_background_cohort`) and anything else at or below `threshold` are
    summed into one `"*"` (residual) key, so no mass is ever silently
    discarded — see module docstring."""
    if sp.is_uninformative():
        return _UNIFORM_SENTINEL

    peak_char = sp.argmax()
    rounded = {c: round(p, decimals) for c, p in sp.probs.items()}
    non_peak = {c: v for c, v in rounded.items() if c != peak_char}
    background = _background_cohort(non_peak)

    kept = {peak_char: rounded[peak_char]}
    for c, v in non_peak.items():
        if c in background:
            continue
        if v <= threshold:
            continue
        kept[c] = v

    residual = 1.0 - sum(sp.probs[c] for c in kept)
    residual = max(0.0, round(residual, decimals))
    if residual > 0:
        kept[RESIDUAL_KEY] = residual
    return kept


def decode_slot_posterior(data: dict[str, float] | int, alphabet: list[str]) -> SlotPosterior:
    """Inverse of encode_slot_posterior. `alphabet` is the slot's full
    alphabet (from PLATE_SLOTS), needed to reconstruct the uniform case and
    to spread the residual mass. The residual (if any) is distributed
    uniformly over every alphabet character not explicitly listed, and every
    character is floored at `_DECODE_FLOOR` before renormalising — so a
    character the codec dropped can be *unlikely*, but is never *impossible*
    (docs/decisions.md, "Day 1c")."""
    if data == _UNIFORM_SENTINEL:
        return SlotPosterior.uniform(alphabet)

    residual = float(data.get(RESIDUAL_KEY, 0.0))
    kept = {c: p for c, p in data.items() if c != RESIDUAL_KEY}
    unlisted = [c for c in alphabet if c not in kept]

    probs: dict[str, float] = dict(kept)
    if unlisted:
        per_char = residual / len(unlisted) if residual > 0 else 0.0
        for c in unlisted:
            probs[c] = per_char

    for c in alphabet:
        if probs.get(c, 0.0) <= 0.0:
            probs[c] = _DECODE_FLOOR

    total = sum(probs.values())
    probs = {c: p / total for c, p in probs.items()}
    return SlotPosterior(probs=probs)


# On-disk keys are short — this is a wire-format optimisation only; the
# in-memory DetectionEvent/VehicleAttributes field names are unchanged.
# None-valued fields (crop_uri, gt_vehicle_id) are omitted entirely rather
# than written as `null`.
def encode_event_compact(
    event: DetectionEvent,
    embedding_row: int,
    threshold: float = SPARSE_THRESHOLD,
    decimals: int = ROUND_DECIMALS,
) -> dict[str, Any]:
    """Build a compact, JSON-serialisable dict for one event: sparse plate
    posteriors, short keys, embedding replaced by a row reference into
    embeddings.npy."""
    out: dict[str, Any] = {
        "id": event.event_id,
        "cam": event.camera_id,
        "t": event.timestamp.isoformat(),
        "pp": [encode_slot_posterior(sp, threshold, decimals) for sp in event.plate_posterior],
        "pa": event.plate_argmax,
        "pc": round(event.plate_confidence, ATTR_DECIMALS),
        "er": embedding_row,
        "a": {
            "c": event.attributes.color,
            "vt": event.attributes.vehicle_type,
            "cc": round(event.attributes.color_confidence, ATTR_DECIMALS),
            "tc": round(event.attributes.type_confidence, ATTR_DECIMALS),
        },
        "s": event.source,
    }
    if event.crop_uri is not None:
        out["cu"] = event.crop_uri
    if event.gt_vehicle_id is not None:
        out["gt"] = event.gt_vehicle_id
    return out


def decode_event_compact(
    data: dict[str, Any], embedding: list[float], plate_slots: list[list[str]]
) -> DetectionEvent:
    """Inverse of encode_event_compact. `embedding` is the already-hydrated
    128-d vector (from the companion embeddings.npy); `plate_slots` is
    engine.contracts.plate.PLATE_SLOTS (passed in to avoid a circular import
    at module scope)."""
    posteriors = [
        decode_slot_posterior(slot_data, plate_slots[i])
        for i, slot_data in enumerate(data["pp"])
    ]
    attr = data["a"]
    return DetectionEvent(
        event_id=data["id"],
        camera_id=data["cam"],
        timestamp=data["t"],
        plate_posterior=posteriors,
        plate_argmax=data["pa"],
        plate_confidence=data["pc"],
        embedding=embedding,
        embedding_ref=data["er"],
        attributes=VehicleAttributes(
            color=attr["c"],
            vehicle_type=attr["vt"],
            color_confidence=attr["cc"],
            type_confidence=attr["tc"],
        ),
        crop_uri=data.get("cu"),
        source=data["s"],
        gt_vehicle_id=data.get("gt"),
    )
