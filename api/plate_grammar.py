"""Parse a human-typed Indian plate query into the engine's 10-slot
canonical form(s) (`engine.decode.partial_search`).

Grammar (docs/api-contract.md, `/api/search`): `SS` (2 state letters) + RTO
(1-2 digits) + series (0-3 letters) + number (1-4 digits); `?` = one
unknown character. The canonical form (docs/decisions.md, "Day 1 build")
is fixed-width: state(2) + RTO(2, zero-padded on the left for a 1-digit
RTO) + series(2, left-aligned, "_"-padded) + number(4, right-aligned,
"_"-padded) — canonical series only has room for 0-2 letters, so a 3-letter
series input can never find a matching split and is correctly rejected.

When the remaining (non-state) characters can be split into
(RTO_len, series_len, number_len) more than one grammar-consistent way —
which only happens because of `?` wildcards straddling a segment boundary,
e.g. `DL3C?456` (RTO=1 digit "3" leaves "C?456", where the `?` could be the
2nd series letter with a 3-digit number, or the 1st digit of a 4-digit
number with a 1-letter series) — every consistent split is returned, and
the caller searches each canonical form and merges hits by max probability
(module docstring of the api-contract; this is not a special case, it falls
out of enumerating every (RTO_len, series_len, number_len) triple that is
consistent with the literal characters actually typed).
"""

from __future__ import annotations

from engine.contracts.plate import BLANK

WILDCARD = "?"
STATE_LEN = 2
RTO_LEN_OPTIONS = (1, 2)
SERIES_LEN_OPTIONS = (0, 1, 2)  # canonical form only has 2 series slots
NUMBER_LEN_OPTIONS = (1, 2, 3, 4)
CANONICAL_LEN = 10


def _is_letter_or_wild(ch: str) -> bool:
    return ch == WILDCARD or ch.isalpha()


def _is_digit_or_wild(ch: str) -> bool:
    return ch == WILDCARD or ch.isdigit()


class GrammarError(ValueError):
    def __init__(self, query: str, detail: str):
        super().__init__(detail)
        self.query = query
        self.detail = detail


def _build_canonical(state: str, rto: str, series: str, number: str) -> str:
    rto_canon = rto.rjust(2, "0") if len(rto) == 1 else rto
    series_canon = series.ljust(2, BLANK)
    number_canon = number.rjust(4, BLANK)
    canonical = state + rto_canon + series_canon + number_canon
    assert len(canonical) == CANONICAL_LEN
    return canonical


def enumerate_canonical_forms(raw_query: str) -> list[str]:
    """Every canonical-form (10-slot) query consistent with the Indian
    plate grammar and `raw_query`'s literal characters. Raises `GrammarError`
    (with a helpful `.detail`) if no split at all is consistent — the only
    case docs/api-contract.md asks us to reject with HTTP 422."""
    q = raw_query.strip().upper().replace(" ", "").replace("-", "")
    if len(q) < STATE_LEN + 1:
        raise GrammarError(raw_query, f"query {raw_query!r} is too short for SS+RTO+number")

    state = q[:STATE_LEN]
    if not all(_is_letter_or_wild(c) for c in state):
        raise GrammarError(
            raw_query, f"query {raw_query!r}: first {STATE_LEN} characters must be state letters"
        )

    rest = q[STATE_LEN:]
    remaining_len = len(rest)

    forms: list[str] = []
    seen: set[str] = set()
    for rto_len in RTO_LEN_OPTIONS:
        for series_len in SERIES_LEN_OPTIONS:
            number_len = remaining_len - rto_len - series_len
            if number_len not in NUMBER_LEN_OPTIONS:
                continue
            rto = rest[:rto_len]
            series = rest[rto_len : rto_len + series_len]
            number = rest[rto_len + series_len :]
            if not all(_is_digit_or_wild(c) for c in rto):
                continue
            if not all(_is_letter_or_wild(c) for c in series):
                continue
            if not all(_is_digit_or_wild(c) for c in number):
                continue
            canonical = _build_canonical(state, rto, series, number)
            if canonical not in seen:
                seen.add(canonical)
                forms.append(canonical)

    if not forms:
        raise GrammarError(
            raw_query,
            f"query {raw_query!r} does not match the Indian plate grammar "
            "SS[RTO 1-2 digits][series 0-2 letters][number 1-4 digits] "
            "('?' = one unknown character) — no consistent split found",
        )
    return forms


__all__ = ["CANONICAL_LEN", "GrammarError", "WILDCARD", "enumerate_canonical_forms"]
