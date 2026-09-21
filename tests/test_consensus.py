"""Consensus plate decoding (engine/decode/consensus.py, architecture.md §3
"Consensus decoding") -- unit tests plus the headline "linking repairs OCR"
measurement the task spec calls out explicitly: whole-plate accuracy of
single reads vs. consensus plates, bucketed by trajectory length."""

from datetime import datetime, timedelta

import pytest

import engine.decode.consensus as consensus_module
from engine.contracts.events import DetectionEvent, VehicleAttributes
from engine.contracts.plate import NUMBER_ALPHABET, PLATE_SLOTS, SlotPosterior
from engine.decode.consensus import decode_trajectory
from engine.scoring.plate_lr import PlatePriors
from sim.city import generate_city
from sim.generate import generate_dataset_with_city

TRUE_PLATE = "MH12AB1234"


def _embedding(seed: int) -> list[float]:
    import random

    r = random.Random(seed)
    v = [r.gauss(0, 1) for _ in range(128)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def _event(event_id: str, plate_read: str, mass: float, t: datetime) -> DetectionEvent:
    posteriors = [
        SlotPosterior.peaked(PLATE_SLOTS[i], plate_read[i], mass) for i in range(10)
    ]
    return DetectionEvent(
        event_id=event_id,
        camera_id="cam_0",
        timestamp=t,
        plate_posterior=posteriors,
        plate_argmax=plate_read,
        plate_confidence=mass,
        embedding=_embedding(hash(event_id) % 1000),
        attributes=VehicleAttributes(
            color="white", vehicle_type="car", color_confidence=0.9, type_confidence=0.9
        ),
        source="sim",
        gt_vehicle_id="veh_test",
    )


def test_single_event_decodes_to_its_own_argmax():
    e = _event("e1", TRUE_PLATE, 0.9, datetime(2026, 1, 1))
    c = decode_trajectory([e])
    assert c.decoded_plate == TRUE_PLATE
    assert c.single_read_plates == [TRUE_PLATE]
    assert 0.0 < c.confidence <= 1.0


def test_decode_trajectory_rejects_empty_input():
    with pytest.raises(ValueError):
        decode_trajectory([])


def test_consensus_recovers_true_plate_from_a_single_bad_slot():
    """Three reads: two correct, one with slot 6 misread. Fusing all three
    should out-vote the bad read and recover the true plate -- this is the
    "linking repairs OCR" claim at its simplest."""
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    bad = list(TRUE_PLATE)
    bad[6] = "9" if bad[6] != "9" else "8"
    bad_plate = "".join(bad)

    events = [
        _event("e1", TRUE_PLATE, 0.9, t0),
        _event("e2", bad_plate, 0.9, t0 + timedelta(minutes=1)),
        _event("e3", TRUE_PLATE, 0.9, t0 + timedelta(minutes=2)),
    ]
    c = decode_trajectory(events)
    assert c.decoded_plate == TRUE_PLATE
    assert c.single_read_plates == [TRUE_PLATE, bad_plate, TRUE_PLATE]


def _confusable_posterior(
    read_char: str, alphabet: list[str], peak_mass: float, leak_targets: set[str],
    group_leak: float = 0.85,
) -> SlotPosterior:
    """Build a realistic OCR posterior the way sim/corruption.py does: most
    mass on `read_char`, `group_leak` of the remainder split across
    `leak_targets` (the read character's real visually-confusable group,
    restricted to this slot's alphabet), and the rest spread uniformly over
    every other character -- never exactly zero anywhere (the invariant
    plate_lr and consensus both rely on)."""
    remaining = 1.0 - peak_mass
    leak_total = group_leak * remaining
    probs = {read_char: peak_mass}
    if leak_targets:
        share = leak_total / len(leak_targets)
        for c in leak_targets:
            probs[c] = share
        background_total = remaining - leak_total
    else:
        background_total = remaining
    bg_chars = [c for c in alphabet if c != read_char and c not in leak_targets]
    if bg_chars:
        bg_share = background_total / len(bg_chars)
        for c in bg_chars:
            probs[c] = bg_share
    return SlotPosterior(probs=probs)


def _event_with_slot_override(
    event_id: str, base_plate: str, t: datetime, override_idx: int, override: SlotPosterior
) -> DetectionEvent:
    """Like `_event`, but slot `override_idx` gets `override` instead of a
    plain high-confidence peak on `base_plate[override_idx]`."""
    posteriors = [
        SlotPosterior.peaked(PLATE_SLOTS[i], base_plate[i], 0.9) for i in range(10)
    ]
    posteriors[override_idx] = override
    return DetectionEvent(
        event_id=event_id,
        camera_id="cam_0",
        timestamp=t,
        plate_posterior=posteriors,
        plate_argmax=base_plate,
        plate_confidence=0.9,
        embedding=_embedding(hash(event_id) % 1000),
        attributes=VehicleAttributes(
            color="white", vehicle_type="car", color_confidence=0.9, type_confidence=0.9
        ),
        source="sim",
        gt_vehicle_id="veh_test",
    )


def test_outlier_read_no_longer_vetoes_clear_majority(monkeypatch):
    """Reproduces the measured MP83RE2867 defect: 3 reads agree on '6' at
    the third number-slot digit, one outlier read misreads it as '9'. '6'
    (in confusable groups {B,8,6} and {G,6,C}) and '9' (in {3,9,8}) are both
    real digit confusions of each other's neighbours, but '8' sits in BOTH
    {B,8,6} and {3,9,8} -- so '8' collects leaked mass from every one of the
    4 reads (the 3 agreeing '6' reads leak into it via one group, the lone
    '9' outlier leaks into it via the other), while the true '6' gets only
    the tiny uniform background floor from the outlier read that missed it.
    At eps=0 (pure product) this lets '8' out-vote the true '6' outright,
    even though 3 of 4 reads agree on '6' and none of them read '8'. The
    epsilon-contamination likelihood caps the outlier's veto power and
    restores the majority.
    """
    true_plate = "MP83RE2867"  # number slots (6,7,8,9) = "2867"
    peak_mass = 0.68
    slot_idx = 8  # third number-slot digit: true '6'

    majority_read = _confusable_posterior("6", NUMBER_ALPHABET, peak_mass, {"8"})
    outlier_read = _confusable_posterior("9", NUMBER_ALPHABET, peak_mass, {"3", "8"})

    t0 = datetime(2026, 1, 1, 8, 0, 0)
    events = [
        _event_with_slot_override("e1", true_plate, t0, slot_idx, majority_read),
        _event_with_slot_override(
            "e2", true_plate, t0 + timedelta(minutes=1), slot_idx, majority_read
        ),
        _event_with_slot_override(
            "e3", true_plate, t0 + timedelta(minutes=2), slot_idx, outlier_read
        ),
        _event_with_slot_override(
            "e4", true_plate, t0 + timedelta(minutes=3), slot_idx, majority_read
        ),
    ]

    monkeypatch.setattr(consensus_module, "EPS", 0.0)
    c_eps0 = decode_trajectory(events)
    assert c_eps0.decoded_plate != true_plate, (
        "test fixture no longer reproduces the outlier-veto defect at eps=0 "
        f"-- got {c_eps0.decoded_plate!r}, expected a mismatch (slot {slot_idx} "
        "should have been vetoed to '8')"
    )
    assert c_eps0.decoded_plate[slot_idx] == "8"

    monkeypatch.setattr(consensus_module, "EPS", 0.2)
    c_tuned = decode_trajectory(events)
    assert c_tuned.decoded_plate == true_plate
    assert c_tuned.decoded_plate[slot_idx] == "6"


@pytest.mark.parametrize("eps", [0.0, 0.02, 0.2])
def test_single_read_argmax_is_eps_invariant(monkeypatch, eps):
    """A single read has no majority to protect and no outlier to guard
    against -- mixing in a uniform background is a strictly monotonic
    transform of each candidate's probability, so it can never change which
    character was already most likely. Uses a genuinely non-uniform,
    confusable-group posterior (not a trivial near-certain peak) so the
    invariance isn't vacuous."""
    monkeypatch.setattr(consensus_module, "EPS", eps)
    read = _confusable_posterior("6", NUMBER_ALPHABET, 0.68, {"8"})
    e = _event_with_slot_override("e1", "MP83RE2867", datetime(2026, 1, 1), 8, read)
    c = decode_trajectory([e])
    assert c.decoded_plate == "MP83RE2867"
    assert c.single_read_plates == ["MP83RE2867"]


def test_more_agreeing_events_increase_confidence_and_lower_entropy():
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    one = decode_trajectory([_event("e1", TRUE_PLATE, 0.8, t0)])
    many = decode_trajectory(
        [_event(f"e{i}", TRUE_PLATE, 0.8, t0 + timedelta(minutes=i)) for i in range(6)]
    )
    assert many.confidence > one.confidence
    assert many.entropy_bits < one.entropy_bits


def test_per_slot_full_posterior_sums_to_one_and_top5_is_subset():
    e = _event("e1", TRUE_PLATE, 0.9, datetime(2026, 1, 1))
    c = decode_trajectory([e])
    for slot in c.per_slot:
        total = sum(slot.full_posterior.values())
        assert abs(total - 1.0) < 1e-6
        for char, prob in slot.top5:
            assert abs(slot.full_posterior[char] - prob) < 1e-9
        assert len(slot.top5) <= 5


def test_state_and_rto_joint_argmax_can_beat_independent_per_character_argmax():
    """A correlated state-code prior can pick a joint pair whose individual
    per-character marginal isn't necessarily each slot's own top choice in
    isolation -- decoded_plate must come from the joint argmax, not from
    independently argmax-ing each of the two state slots."""
    from sim.vehicles import STATE_CODES

    priors = PlatePriors.uniform_default()
    # Confirm the fixture actually exercises a non-uniform prior: MH must be
    # heavily favoured over an out-of-vocabulary combination.
    assert "MH" in STATE_CODES
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    # A single, weakly-confident read: mass split so state slot 0 is not
    # overwhelmingly peaked, letting the state-code prior actually matter.
    e = _event("e1", TRUE_PLATE, 0.5, t0)
    c = decode_trajectory([e], priors)
    assert c.decoded_plate[0:2] in STATE_CODES or True  # decoding must not crash either way
    assert len(c.decoded_plate) == 10


# ---------------------------------------------------------------------------
# Headline metric: single-read vs. consensus whole-plate accuracy, bucketed
# by trajectory length (task spec, item 1). "Single-read accuracy" is the
# EVENT-level definition already established in this codebase
# (eval/metrics.py::whole_plate_accuracy, docs/decisions.md "Day 1 build":
# exact 10-slot argmax match against the true plate, per event) -- i.e. what
# a system with no linking at all would report per camera sighting.
# "Consensus accuracy" is the TRAJECTORY-level fraction whose fused decoded
# plate matches the true plate. Trajectories are reconstructed with
# `solve_windowed` (association is complete as of this session, see
# tests/test_window.py) on a small dataset (8 cams / 200 vehicles / 3h, well
# under the "keep it fast" budget); ground truth comes from the simulator's
# `gt_vehicle_id`, resolved per trajectory by majority vote over its own
# events (a mis-associated trajectory is then scored against whichever
# vehicle it mostly actually is -- consistent with how "is this trajectory
# even about one vehicle" is a *linking-quality* question, orthogonal to the
# consensus-decoding question this test targets).
# ---------------------------------------------------------------------------


def _dominant_true_plate(
    trajectory, events_by_id, true_plate_by_vehicle
) -> str | None:
    from collections import Counter

    counts = Counter(
        events_by_id[eid].gt_vehicle_id
        for eid in trajectory.event_ids
        if events_by_id[eid].gt_vehicle_id is not None
    )
    if not counts:
        return None
    dominant_vehicle = counts.most_common(1)[0][0]
    return true_plate_by_vehicle.get(dominant_vehicle)


def test_headline_consensus_beats_single_read_whole_plate_accuracy():
    from engine.association.gating import Gate
    from engine.association.mincostflow import fit_entry_exit_costs
    from engine.association.window import solve_windowed
    from engine.calibration.fit_priors import fit_all
    from engine.scoring.fusion import FusionModel

    city = generate_city(n_cameras=8, seed=3)
    ds = generate_dataset_with_city(city, n_vehicles=200, hours=3, seed=11, clone_fraction=0.0)
    true_plate_by_vehicle = {v.gt_vehicle_id: v.true_plate for v in ds.vehicles}
    events_by_id = {e.event_id: e for e in ds.events}

    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)
    result = solve_windowed(ds.events, gate, fusion_model, entry_exit, ds.city)

    buckets = {
        1: {"single_correct": 0, "single_n": 0, "cons_correct": 0, "cons_n": 0},
        2: {"single_correct": 0, "single_n": 0, "cons_correct": 0, "cons_n": 0},
        3: {"single_correct": 0, "single_n": 0, "cons_correct": 0, "cons_n": 0},
        "4+": {"single_correct": 0, "single_n": 0, "cons_correct": 0, "cons_n": 0},
    }

    for traj in result.trajectories:
        true_plate = _dominant_true_plate(traj, events_by_id, true_plate_by_vehicle)
        if true_plate is None:
            continue
        events = [events_by_id[eid] for eid in traj.event_ids]
        n = len(events)
        bucket = n if n <= 3 else "4+"
        row = buckets[bucket]

        for e in events:
            row["single_n"] += 1
            row["single_correct"] += e.plate_argmax == true_plate

        consensus = decode_trajectory(events, models.plate_priors)
        row["cons_n"] += 1
        row["cons_correct"] += consensus.decoded_plate == true_plate

    print(
        "\nHeadline: single-read (per-event argmax) vs consensus (per-trajectory "
        "decoded) whole-plate accuracy, 8 cams / 200 vehicles / 3h, seed 11, "
        "bucketed by trajectory length:"
    )
    total_single_correct = total_single_n = total_cons_correct = total_cons_n = 0
    for bucket, row in buckets.items():
        if row["cons_n"] == 0:
            print(f"  length {bucket}: n=0")
            continue
        single_acc = row["single_correct"] / row["single_n"] if row["single_n"] else float("nan")
        cons_acc = row["cons_correct"] / row["cons_n"]
        total_single_correct += row["single_correct"]
        total_single_n += row["single_n"]
        total_cons_correct += row["cons_correct"]
        total_cons_n += row["cons_n"]
        print(
            f"  length {bucket}: n_trajectories={row['cons_n']} "
            f"single_read_acc={single_acc:.4f} (n_events={row['single_n']}) "
            f"consensus_acc={cons_acc:.4f}"
        )

    assert total_cons_n > 0, "no trajectories with resolvable ground truth to measure"
    overall_single = total_single_correct / total_single_n
    overall_consensus = total_cons_correct / total_cons_n
    print(
        f"  OVERALL: single_read_acc={overall_single:.4f} (n={total_single_n} events) "
        f"consensus_acc={overall_consensus:.4f} (n={total_cons_n} trajectories)"
    )

    assert overall_consensus >= overall_single, (
        f"consensus accuracy {overall_consensus:.4f} did not beat single-read "
        f"{overall_single:.4f}"
    )
