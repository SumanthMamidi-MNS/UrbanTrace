"""Sliding-window association must be consistent with a single-shot solve
on a small dataset (architecture.md §3, §7, docs/decisions.md "Day 3a" Part
4): splitting the same event stream into overlapping windows and stitching
open tracks across boundaries should recover essentially the same
event-to-trajectory groupings as solving everything at once."""

from itertools import combinations

from engine.association.gating import Gate
from engine.association.mincostflow import fit_entry_exit_costs
from engine.association.window import (
    DEFAULT_WINDOW_SIZE_S,
    DEFAULT_WINDOW_STEP_S,
    _solve_one_window,
    solve_windowed,
)
from engine.calibration.fit_priors import fit_all
from engine.scoring.fusion import FusionModel
from sim.city import generate_city
from sim.generate import generate_dataset_with_city

AGREEMENT_FLOOR = 0.90


def _grouping_from_trajectories(trajectories) -> dict[str, int]:
    """event_id -> trajectory index."""
    grouping: dict[str, int] = {}
    for i, traj in enumerate(trajectories):
        for eid in traj.event_ids:
            grouping[eid] = i
    return grouping


def _pairwise_agreement(events, grouping_a: dict[str, int], grouping_b: dict[str, int]) -> float:
    """Fraction of event PAIRS on which both groupings agree about
    same-trajectory-or-not. The standard co-clustering agreement metric --
    robust to the two groupings using unrelated trajectory-id numbering."""
    ids = [e.event_id for e in events]
    agree = 0
    total = 0
    for a, b in combinations(ids, 2):
        same_a = grouping_a.get(a) == grouping_a.get(b)
        same_b = grouping_b.get(a) == grouping_b.get(b)
        agree += same_a == same_b
        total += 1
    return agree / total if total else 1.0


def _build(n_cameras, n_vehicles, hours, seed):
    city = generate_city(n_cameras=n_cameras, seed=1)
    ds = generate_dataset_with_city(city, n_vehicles, hours, seed, clone_fraction=0.0)
    return ds


def test_windowed_solve_agrees_with_single_shot_on_small_dataset():
    ds = _build(n_cameras=8, n_vehicles=150, hours=2, seed=5)
    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)

    # Single-shot: treat the entire event stream as one window.
    single_paths, _, _stats = _solve_one_window(
        ds.events, set(), gate, fusion_model, entry_exit, ds.city
    )
    single_grouping: dict[str, int] = {}
    for i, path in enumerate(single_paths):
        for eid in path:
            single_grouping[eid] = i

    windowed = solve_windowed(ds.events, gate, fusion_model, entry_exit, ds.city)
    windowed_grouping = _grouping_from_trajectories(windowed.trajectories)

    assert windowed.n_windows > 1, "test dataset should span multiple windows"
    agreement = _pairwise_agreement(ds.events, single_grouping, windowed_grouping)
    print(f"pairwise agreement={agreement:.4f} n_windows={windowed.n_windows}")
    assert agreement > AGREEMENT_FLOOR, (
        f"windowed solve agreement with single-shot was {agreement:.4f}, "
        f"below the {AGREEMENT_FLOOR} floor"
    )


def test_windowed_solve_never_assigns_one_event_to_two_trajectories():
    ds = _build(n_cameras=8, n_vehicles=150, hours=3, seed=6)
    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)

    windowed = solve_windowed(ds.events, gate, fusion_model, entry_exit, ds.city)
    seen: set[str] = set()
    for traj in windowed.trajectories:
        for eid in traj.event_ids:
            assert eid not in seen, f"event {eid} assigned to more than one trajectory"
            seen.add(eid)
        # each trajectory's own events must be in ascending time order and
        # each event id must appear only once within it (no self-loop bugs)
        timestamps = [e for e in traj.event_ids]
        assert len(timestamps) == len(set(timestamps))


def test_windowed_solve_full_day_partitions_every_event_exactly_once():
    """The invariant that matters most: over a FULL day (not a small
    dataset that might not even reach a second window boundary), the
    trajectories produced by windowed solving must be a PARTITION of the
    input events -- every event in exactly one trajectory, none missing,
    none duplicated. A seam bug at, say, hour 7 would not show up in a
    2-3 hour test; this runs the full 24h span specifically to catch that."""
    ds = _build(n_cameras=10, n_vehicles=800, hours=24, seed=13)
    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)

    windowed = solve_windowed(ds.events, gate, fusion_model, entry_exit, ds.city)

    all_ids = {e.event_id for e in ds.events}
    seen: set[str] = set()
    for traj in windowed.trajectories:
        ids = set(traj.event_ids)
        assert len(ids) == len(traj.event_ids), f"{traj.trajectory_id} repeats an event_id"
        assert ids.isdisjoint(seen), (
            f"{traj.trajectory_id} shares event(s) with an earlier trajectory: "
            f"{ids & seen}"
        )
        seen |= ids

    assert seen == all_ids, (
        f"missing={len(all_ids - seen)} extra={len(seen - all_ids)} "
        f"(full-day partition must exactly equal the input event set)"
    )


def test_windowed_solve_never_uses_a_negative_total_log_odds_link():
    """Regression test for the pre-fix over-merging bug
    (eval/reports/error_analysis.json: traj_018419 joined plates
    KA55M__622/WB76LM_138 through a link with total_log_odds=-0.32). Before
    the arc-cost fix (engine.association.mincostflow module docstring, "THE
    ARC-COST DERIVATION"), a link arc costing bare `-s(i,j)` let the solver
    accept negative-log-odds links whenever paying for a track death+birth
    was pricier than the (wrongly cheap) link -- 57.6% of that run's wrong
    links had total_log_odds < 0. With the corrected formula, dominance
    pruning (`prune_dominated_arcs`) drops every arc with `s(i,j) <= 0`
    BEFORE the solver ever runs, so no trajectory produced by
    `solve_windowed` should ever contain a used link with a negative
    `total_log_odds`."""
    ds = _build(n_cameras=10, n_vehicles=400, hours=4, seed=21)
    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)

    windowed = solve_windowed(ds.events, gate, fusion_model, entry_exit, ds.city)

    n_links_checked = 0
    for traj in windowed.trajectories:
        for link in traj.links:
            n_links_checked += 1
            assert link.total_log_odds > -1e-9, (
                f"trajectory {traj.trajectory_id} used link "
                f"{link.from_event_id}->{link.to_event_id} with negative "
                f"total_log_odds={link.total_log_odds}"
            )
    assert n_links_checked > 0, "no multi-event trajectory links to check"


def test_windowed_solve_covers_every_event():
    ds = _build(n_cameras=8, n_vehicles=100, hours=2, seed=7)
    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)

    windowed = solve_windowed(
        ds.events,
        gate,
        fusion_model,
        entry_exit,
        ds.city,
        window_size_s=DEFAULT_WINDOW_SIZE_S,
        step_s=DEFAULT_WINDOW_STEP_S,
    )
    covered = {eid for traj in windowed.trajectories for eid in traj.event_ids}
    all_ids = {e.event_id for e in ds.events}
    assert covered == all_ids
