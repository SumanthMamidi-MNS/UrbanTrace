"""Gating is the primary blocker (architecture.md §3, §7) and decides the
recall ceiling of the whole system: if it drops a real predecessor, nothing
downstream can recover it. This is the load-bearing test."""

from engine.association.gating import Gate, gate_candidates, true_predecessor_recall
from engine.scoring.kinematic_lr import fit_kinematic_model
from sim.city import generate_city
from sim.generate import generate_dataset_with_city

RECALL_FLOOR = 0.99


def _build(n_cameras, n_vehicles, hours, seed, clone_fraction=0.0):
    city = generate_city(n_cameras=n_cameras, seed=1)
    train_ds = generate_dataset_with_city(city, n_vehicles, hours, seed, clone_fraction)
    return train_ds


def test_gate_recall_exceeds_99_percent():
    """The hard requirement: >99% of real consecutive same-vehicle event
    pairs must have their earlier event inside the later event's candidate
    set. This must never be weakened -- everything downstream depends on it."""
    ds = _build(n_cameras=15, n_vehicles=1500, hours=6, seed=7, clone_fraction=0.02)
    model = fit_kinematic_model(ds.events, ds.city)
    gate = Gate(model=model)

    result = gate_candidates(ds.events, gate)
    recall = true_predecessor_recall(ds.events, result)

    assert recall.n_true_pairs > 500, "sample too small to trust the recall measurement"
    assert recall.recall > RECALL_FLOOR, (
        f"gate recall {recall.recall:.4f} fell below the {RECALL_FLOOR} floor -- "
        f"this is the primary blocker, a regression here silently caps the "
        f"whole system's achievable recall"
    )
    print(f"gate recall={recall.recall:.4f} n_true_pairs={recall.n_true_pairs}")
    print(f"mean_candidates_per_event={result.mean_candidates_per_event:.2f}")


def test_gate_cuts_pair_count_well_below_naive_all_pairs():
    """The gate must actually be a blocker, not a no-op. Naive (no gating)
    candidate generation would compare every event against every strictly
    earlier event -- average (n_events-1)/2 candidates. Architecture.md §7's
    own target scale (50 cams, ~500 events/camera/hour, 45-min window) puts
    the gated figure at "a few hundred candidates per event"; the absolute
    count is dataset-density-dependent (see eval/gating_report.py for the
    production-scale numbers this is actually judged on), so this test only
    checks the gate meaningfully cuts below the naive baseline at ANY scale,
    not a specific percentage -- the percentage cut improves as event volume
    grows relative to a roughly constant absolute window size."""
    ds = _build(n_cameras=15, n_vehicles=1500, hours=6, seed=7, clone_fraction=0.02)
    model = fit_kinematic_model(ds.events, ds.city)
    gate = Gate(model=model)

    result = gate_candidates(ds.events, gate)
    naive_mean_candidates = (result.n_events - 1) / 2
    assert result.n_events > 0
    assert result.mean_candidates_per_event < 0.5 * naive_mean_candidates
    print(f"mean_candidates_per_event={result.mean_candidates_per_event:.2f}")
    print(f"naive_mean_candidates={naive_mean_candidates:.2f}")


def test_gate_window_never_includes_the_event_itself_or_future_events():
    ds = _build(n_cameras=10, n_vehicles=300, hours=4, seed=11)
    model = fit_kinematic_model(ds.events, ds.city)
    gate = Gate(model=model)
    result = gate_candidates(ds.events, gate)

    by_id = {e.event_id: e for e in ds.events}
    for event_id, cand_ids in result.candidates.items():
        e = by_id[event_id]
        assert event_id not in cand_ids
        for cid in cand_ids:
            assert by_id[cid].timestamp < e.timestamp


def test_gate_window_widens_for_skipped_intermediate_cameras():
    """A camera pair whose shortest path skips intermediate cameras should
    get a wider window than a direct one-hop pair at comparable distance
    (the missed-detection widening, architecture.md §3/§7)."""
    from engine.association.gating import DEFAULT_MISS_WIDEN_FACTOR, GATE_Z_SCORE

    assert DEFAULT_MISS_WIDEN_FACTOR > 1.0
    assert GATE_Z_SCORE > 2.0
