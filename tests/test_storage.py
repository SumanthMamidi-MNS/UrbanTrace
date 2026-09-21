"""Day 1b: compact on-disk format. The old dense JSONL was 7378 bytes/event
(755MB at 107k events) — see docs/decisions.md, "Day 1b". Sparse posteriors +
embeddings pulled into a companion float16 .npy must get this under 1000
bytes/event, with round-trip fidelity: posteriors within 1e-3, embeddings
within float16 precision.

The budget was raised from 800 to 1000 bytes/event in Day 1c to pay for the
codec's explicit residual ("*") key per slot (docs/decisions.md, "Day 1c") —
that key is what stops a dropped low-probability character from round-
tripping as an exact 0.0. Correctness was not traded back for size: the
posterior_group_leak revert (0.999 -> 0.85) that made the residual mass
non-trivial again is the fix, not a regression to compensate for.
"""

from pathlib import Path

import numpy as np

from engine.contracts.store import EventStore
from sim.generate import generate_dataset, write_dataset

BYTES_PER_EVENT_TARGET = 1000
# float16 has ~3 significant decimal digits; after cast + renormalisation a
# 128-d unit vector's per-component drift is comfortably under this.
EMBEDDING_ATOL = 5e-3


def _generate_and_write(tmp_path: Path, n_vehicles=300, n_cameras=15, hours=24, seed=7) -> Path:
    ds = generate_dataset(n_cameras=n_cameras, n_vehicles=n_vehicles, hours=hours, seed=seed)
    out = tmp_path / "run"
    write_dataset(ds, out, run_config={"seed": seed})
    return out, ds


def test_bytes_per_event_under_target(tmp_path: Path):
    out, ds = _generate_and_write(tmp_path)
    events_path = out / "events.jsonl"
    embeddings_path = out / "embeddings.npy"

    n_events = len(ds.events)
    assert n_events > 0

    total_bytes = events_path.stat().st_size + embeddings_path.stat().st_size
    bytes_per_event = total_bytes / n_events
    assert bytes_per_event < BYTES_PER_EVENT_TARGET, (
        f"{bytes_per_event:.1f} bytes/event exceeds the {BYTES_PER_EVENT_TARGET} target"
    )


def test_roundtrip_posteriors_within_tolerance(tmp_path: Path):
    out, ds = _generate_and_write(tmp_path)
    store = EventStore(out / "events.jsonl", out / "embeddings.npy")
    loaded = store.read_all()

    assert len(loaded) == len(ds.events)

    for original, restored in zip(ds.events, loaded, strict=True):
        assert original.event_id == restored.event_id
        for orig_sp, rest_sp in zip(
            original.plate_posterior, restored.plate_posterior, strict=True
        ):
            # Every symbol that survived sparsification must be within 1e-3
            # of its original probability; symbols the codec dropped were
            # below the sparsify threshold (<=1e-3) to begin with.
            for char, orig_p in orig_sp.probs.items():
                rest_p = rest_sp.probs.get(char, 0.0)
                assert abs(orig_p - rest_p) < 1e-3, (
                    f"slot char {char!r}: {orig_p} vs {rest_p} in event {original.event_id}"
                )


def test_roundtrip_embeddings_within_float16_precision(tmp_path: Path):
    out, ds = _generate_and_write(tmp_path)
    store = EventStore(out / "events.jsonl", out / "embeddings.npy")
    loaded = store.read_all()

    for original, restored in zip(ds.events, loaded, strict=True):
        orig = np.array(original.embedding)
        rest = np.array(restored.embedding)
        assert np.allclose(orig, rest, atol=EMBEDDING_ATOL), (
            f"max diff {np.max(np.abs(orig - rest)):.5f} for event {original.event_id}"
        )
        # restored embedding must still satisfy the contract (unit-normalised)
        assert abs(np.linalg.norm(rest) - 1.0) < 1e-3


def test_roundtrip_preserves_occlusion():
    """A slot the codec sparsified as the uniform sentinel must decode back
    to an uninformative posterior, not a peaked one."""
    from sim.city import generate_city
    from sim.corruption import CorruptionConfig, Corruptor
    from sim.vehicles import generate_vehicles_and_journeys

    seed = 3
    city = generate_city(n_cameras=10, seed=seed)
    vehicles, journeys = generate_vehicles_and_journeys(city, 30, 24, seed)
    vehicles_by_id = {v.gt_vehicle_id: v for v in vehicles}
    cfg = CorruptionConfig(p_occlude=1.0, occlude_run_len_min=3, occlude_run_len_max=3)
    corruptor = Corruptor(city, seed, cfg)

    from engine.contracts.codec import decode_slot_posterior, encode_slot_posterior
    from engine.contracts.plate import PLATE_SLOTS

    found = False
    for j in journeys:
        v = vehicles_by_id[j.gt_vehicle_id]
        for p in j.passages:
            e = corruptor.corrupt_passage(p, v)
            if e is None:
                continue
            for i, sp in enumerate(e.plate_posterior):
                if sp.is_uninformative():
                    encoded = encode_slot_posterior(sp)
                    decoded = decode_slot_posterior(encoded, PLATE_SLOTS[i])
                    assert decoded.is_uninformative()
                    found = True
    assert found, "expected at least one occluded (uninformative) slot"
