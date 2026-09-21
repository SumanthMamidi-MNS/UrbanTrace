"""Fast smoke tests for eval/baselines.py's Baseline B/C builders: tiny
in-memory datasets, just checking they run and return valid metric dicts."""

from engine.association.gating import Gate
from engine.scoring.kinematic_lr import fit_kinematic_model
from eval.baselines import _build_fuzzy_baseline, _metrics_to_dict
from eval.metrics import compute_trajectory_metrics
from eval.run_pipeline import _build_exact_match_baseline
from sim.city import generate_city
from sim.generate import generate_dataset_with_city


def _tiny_dataset():
    city = generate_city(n_cameras=5, seed=11)
    return generate_dataset_with_city(
        city=city, n_vehicles=150, hours=1, seed=21, clone_fraction=0.02
    )


def test_baseline_b_runs_and_returns_valid_metrics():
    ds = _tiny_dataset()
    kinematic_model = fit_kinematic_model(ds.events, ds.city)
    gate = Gate(model=kinematic_model)

    trajectories = _build_fuzzy_baseline(ds.events, gate, kinematic_model=None)
    metrics = compute_trajectory_metrics(trajectories, ds.events)
    result = _metrics_to_dict(metrics)

    assert result["n_predicted_trajectories"] > 0
    assert result["n_gt_vehicles"] > 0
    assert 0.0 <= result["idf1"] <= 1.0
    assert result["id_switches"] >= 0
    assert result["fragmentation"] >= 0
    # Every event must land in exactly one predicted trajectory.
    assert result["n_predicted_events"] == result["n_gt_events"]


def test_baseline_c_runs_and_returns_valid_metrics():
    ds = _tiny_dataset()
    kinematic_model = fit_kinematic_model(ds.events, ds.city)
    gate = Gate(model=kinematic_model)

    trajectories = _build_fuzzy_baseline(ds.events, gate, kinematic_model=kinematic_model)
    metrics = compute_trajectory_metrics(trajectories, ds.events)
    result = _metrics_to_dict(metrics)

    assert result["n_predicted_trajectories"] > 0
    assert 0.0 <= result["idf1"] <= 1.0
    assert result["n_predicted_events"] == result["n_gt_events"]


def test_baseline_c_never_chains_more_aggressively_than_baseline_b():
    """Baseline C only ADDS a rejection filter on top of B's candidate
    selection, so it can never produce FEWER trajectories than B (it can
    only refuse a link B would have made, never make one B wouldn't)."""
    ds = _tiny_dataset()
    kinematic_model = fit_kinematic_model(ds.events, ds.city)
    gate = Gate(model=kinematic_model)

    b_trajectories = _build_fuzzy_baseline(ds.events, gate, kinematic_model=None)
    c_trajectories = _build_fuzzy_baseline(ds.events, gate, kinematic_model=kinematic_model)

    assert len(c_trajectories) >= len(b_trajectories)


def test_fuzzy_builder_degenerates_to_exact_match_at_zero_edit_distance():
    """max_edit_distance=0 means "identical strings only" -- the fuzzy
    builder should then produce EXACTLY the same trajectories as the known-
    good Baseline A exact-match builder on the same tiny dataset/gate."""
    ds = _tiny_dataset()
    kinematic_model = fit_kinematic_model(ds.events, ds.city)
    gate = Gate(model=kinematic_model)

    a_trajectories = _build_exact_match_baseline(ds.events, gate)
    b_trajectories = _build_fuzzy_baseline(
        ds.events, gate, kinematic_model=None, max_edit_distance=0
    )

    a_sets = sorted(sorted(t.event_ids) for t in a_trajectories)
    b_sets = sorted(sorted(t.event_ids) for t in b_trajectories)
    assert a_sets == b_sets
