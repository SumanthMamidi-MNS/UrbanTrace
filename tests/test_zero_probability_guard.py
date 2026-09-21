"""Day 1c: the zero-probability defect guard.

Measured against ground truth on a 20,000+ event sample: raising
`posterior_group_leak` to 0.999 (Day 1b, to shrink the on-disk sparse
encoding) starved every out-of-group character of real probability mass.
0.44% of read slots gave the TRUE character exactly 0.0 posterior
probability. Day 2's plate LR is `PROD_k P(o_k|y_k)` — a single zero in that
product sends the whole plate log-LR to -inf and hard-rejects a correct
match. That is exact-string-matching's failure mode rebuilt inside the
probabilistic model, for 4.3% of events (one that hit a zero-probability
slot anywhere in the plate).

The fix (docs/decisions.md, "Day 1c") is two-layered:
1. `posterior_group_leak` reverted to 0.85 (sim/corruption.py) so the
   in-memory posterior's non-peak mass isn't infinitesimal.
2. The on-disk codec (engine/contracts/codec.py) carries an explicit
   residual ("*") term per slot, redistributed uniformly over every
   character not otherwise listed on decode, so sparsification itself can
   never zero out a character either.

This test is the permanent regression guard for that invariant: it must
fail loudly if either layer is ever re-broken. It exercises the exact
encode/decode round trip the on-disk format uses (not just the in-memory
posterior), since that is where the defect actually lived.
"""

from engine.contracts.codec import decode_slot_posterior, encode_slot_posterior
from engine.contracts.plate import PLATE_SLOTS
from sim.generate import generate_dataset

MIN_EVENTS = 20_000


def test_true_character_never_has_zero_probability_in_a_read_slot():
    ds = generate_dataset(n_cameras=20, n_vehicles=6000, hours=24, seed=42, clone_fraction=0.0)
    assert len(ds.events) >= MIN_EVENTS, (
        f"only {len(ds.events)} events generated, need >= {MIN_EVENTS} for a meaningful sample"
    )

    true_plate_by_vehicle = {v.gt_vehicle_id: v.true_plate for v in ds.vehicles}

    n_read_slots = 0
    n_zero = 0
    zero_examples: list[str] = []

    for event in ds.events:
        true_plate = true_plate_by_vehicle[event.gt_vehicle_id]
        for slot_idx, sp in enumerate(event.plate_posterior):
            if sp.is_uninformative():
                continue  # unread/occluded slot: excluded, contributes no evidence
            n_read_slots += 1

            true_char = true_plate[slot_idx]
            encoded = encode_slot_posterior(sp)
            decoded = decode_slot_posterior(encoded, PLATE_SLOTS[slot_idx])

            prob = decoded.probs.get(true_char, 0.0)
            if prob <= 0.0:
                n_zero += 1
                if len(zero_examples) < 5:
                    zero_examples.append(
                        f"event={event.event_id} slot={slot_idx} true={true_char!r} "
                        f"argmax={sp.argmax()!r} encoded={encoded!r}"
                    )

    assert n_read_slots > 0
    assert n_zero == 0, (
        f"{n_zero}/{n_read_slots} read slots gave the true character exactly 0.0 "
        f"probability after the on-disk round trip — the zero-probability defect is back. "
        f"Examples: {zero_examples}"
    )
