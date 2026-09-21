"""Canonical 10-slot Indian plate representation and per-slot posteriors.

Slot layout (fixed, canonical form ``SS DD LL DDDD``):
    0, 1 -> state letters   (A-Z)
    2, 3 -> RTO digits      (0-9)
    4, 5 -> series letters  (A-Z + "_" blank, for 1-letter series)
    6, 7, 8, 9 -> number    (0-9 + "_" blank, for <4-digit numbers)

"_" is the blank symbol used when a slot legitimately has no character
(e.g. a 1-letter series, or a 3-digit number).
"""

import math
from typing import Self

from pydantic import BaseModel, model_validator

BLANK = "_"

_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_DIGITS = [str(d) for d in range(10)]

STATE_ALPHABET = list(_LETTERS)
RTO_ALPHABET = list(_DIGITS)
SERIES_ALPHABET = [*_LETTERS, BLANK]
NUMBER_ALPHABET = [*_DIGITS, BLANK]

# Per-slot alphabets, index-aligned with the 10-slot canonical plate.
PLATE_SLOTS: list[list[str]] = [
    STATE_ALPHABET,  # 0
    STATE_ALPHABET,  # 1
    RTO_ALPHABET,  # 2
    RTO_ALPHABET,  # 3
    SERIES_ALPHABET,  # 4
    SERIES_ALPHABET,  # 5
    NUMBER_ALPHABET,  # 6
    NUMBER_ALPHABET,  # 7
    NUMBER_ALPHABET,  # 8
    NUMBER_ALPHABET,  # 9
]

NUM_SLOTS = len(PLATE_SLOTS)

_SUM_TOLERANCE = 1e-6


class SlotPosterior(BaseModel):
    """A probability distribution over one plate slot's alphabet.

    Stored as a dict[char, prob] that must sum to 1.0 (within tolerance).
    """

    probs: dict[str, float]

    @model_validator(mode="after")
    def _check_sums_to_one(self) -> Self:
        total = sum(self.probs.values())
        if abs(total - 1.0) > _SUM_TOLERANCE:
            raise ValueError(f"SlotPosterior probs must sum to 1.0, got {total}")
        if any(p < 0 for p in self.probs.values()):
            raise ValueError("SlotPosterior probs must be non-negative")
        return self

    def argmax(self) -> str:
        """Return the most likely character in this slot."""
        return max(self.probs.items(), key=lambda kv: kv[1])[0]

    def entropy(self) -> float:
        """Shannon entropy (nats) of this slot's distribution."""
        return -sum(p * math.log(p) for p in self.probs.values() if p > 0)

    def is_uninformative(self) -> bool:
        """True if the distribution is ~uniform, i.e. the slot was never read."""
        n = len(self.probs)
        if n <= 1:
            return True
        uniform_entropy = math.log(n)
        if uniform_entropy == 0:
            return True
        # Within 1% of max entropy counts as uninformative.
        return self.entropy() >= 0.99 * uniform_entropy

    @classmethod
    def uniform(cls, alphabet: list[str]) -> "SlotPosterior":
        """A uniform (uninformative) distribution over the given alphabet."""
        n = len(alphabet)
        return cls(probs={c: 1.0 / n for c in alphabet})

    @classmethod
    def peaked(cls, alphabet: list[str], char: str, mass: float) -> "SlotPosterior":
        """A distribution putting `mass` on `char` and spreading the rest uniformly
        over the remaining alphabet symbols."""
        if char not in alphabet:
            raise ValueError(f"{char!r} not in alphabet {alphabet!r}")
        if not (0.0 <= mass <= 1.0):
            raise ValueError("mass must be in [0, 1]")
        remaining = [c for c in alphabet if c != char]
        probs = {char: mass}
        if remaining:
            leftover = (1.0 - mass) / len(remaining)
            for c in remaining:
                probs[c] = leftover
        else:
            probs[char] = 1.0
        return cls(probs=probs)
