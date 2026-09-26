"""Correctness guards for the Day 4 solver performance pass
(engine/association/mincostflow.py: `prune_dominated_arcs`,
`connected_components_by_arcs`, and topological-order initial potentials).
All three are claimed to be EXACT transformations, not heuristics -- these
tests are the proof-by-measurement that pruning/decomposing/topo-DP never
change the optimal answer, only how fast the solver gets there."""

import random

import networkx as nx
import numpy as np
import pytest

from engine.association.gating import Gate, gate_candidates
from engine.association.mincostflow import (
    MinCostFlowSolver,
    connected_components_by_arcs,
    default_event_topo_order,
    fit_entry_exit_costs,
    prune_dominated_arcs,
)
from engine.association.mincostflow import (
    detection_cost as real_detection_cost,
)
from engine.calibration.fit_priors import fit_all
from engine.scoring.fusion import FusionModel, score_pairs_batch_totals
from sim.city import generate_city
from sim.generate import generate_dataset_with_city

FLOAT_TOL = 1e-6
N_TRIALS = 20
N_EVENTS = 16


def _random_costs(seed: int, n_events: int = N_EVENTS):
    """A random time-ordered event-association-shaped instance: per-event
    (entry, detection, exit) costs and a random set of forward-only link
    arcs `v_i -> u_j` (i < j, mirroring "gating only links an earlier event
    to a later one"). Returns event ids, per-event cost dicts, and the arc
    list `(pred_id, succ_id, cost)`."""
    rng = random.Random(seed)
    event_ids = [f"e{i}" for i in range(n_events)]
    entry_cost = {eid: round(rng.uniform(0.1, 3.0), 4) for eid in event_ids}
    detection_cost = {eid: round(rng.uniform(0.0, 1.5), 4) for eid in event_ids}
    exit_cost = {eid: round(rng.uniform(0.1, 3.0), 4) for eid in event_ids}
    arcs: list[tuple[str, str, float]] = []
    for i in range(n_events):
        for j in range(i + 1, n_events):
            if rng.random() < 0.35:
                cost = round(rng.uniform(-6.0, 2.0), 4)
                arcs.append((event_ids[i], event_ids[j], cost))
    return event_ids, entry_cost, detection_cost, exit_cost, arcs


def _build_solver(
    event_ids: list[str],
    entry_cost: dict[str, float],
    detection_cost: dict[str, float],
    exit_cost: dict[str, float],
    arcs: list[tuple[str, str, float]],
) -> MinCostFlowSolver:
    """One monolithic solver over ALL of `event_ids`, in the standard
    SOURCE=0/SINK=1/u_i=2+2i/v_i=2+2i+1 node convention."""
    index_of = {eid: i for i, eid in enumerate(event_ids)}
    solver = MinCostFlowSolver(2 + 2 * len(event_ids))

    def u(eid: str) -> int:
        return 2 + 2 * index_of[eid]

    def v(eid: str) -> int:
        return 2 + 2 * index_of[eid] + 1

    for eid in event_ids:
        solver.add_edge(0, u(eid), 1.0, entry_cost[eid])
        solver.add_edge(u(eid), v(eid), 1.0, detection_cost[eid])
        solver.add_edge(v(eid), 1, 1.0, exit_cost[eid])
    for pred_id, succ_id, cost in arcs:
        solver.add_edge(v(pred_id), u(succ_id), 1.0, cost)
    return solver


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_pruned_and_unpruned_reach_the_same_optimal_cost(seed):
    event_ids, entry_cost, detection_cost, exit_cost, arcs = _random_costs(seed)

    unpruned_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, arcs
    ).solve(0, 1)

    pruned_arcs = prune_dominated_arcs(arcs, exit_cost, entry_cost)
    assert len(pruned_arcs) <= len(arcs)
    pruned_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, pruned_arcs
    ).solve(0, 1)

    assert abs(unpruned_cost - pruned_cost) < FLOAT_TOL, (
        f"seed={seed}: pruning changed the optimal cost "
        f"({unpruned_cost} unpruned vs {pruned_cost} pruned, "
        f"{len(arcs)}->{len(pruned_arcs)} arcs)"
    )


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_decomposed_and_monolithic_reach_the_same_total_cost(seed):
    event_ids, entry_cost, detection_cost, exit_cost, arcs = _random_costs(seed)
    pruned_arcs = prune_dominated_arcs(arcs, exit_cost, entry_cost)

    monolithic_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, pruned_arcs
    ).solve(0, 1)

    components = connected_components_by_arcs(event_ids, pruned_arcs)
    comp_of = {eid: ci for ci, comp in enumerate(components) for eid in comp}
    arcs_by_component: list[list[tuple[str, str, float]]] = [[] for _ in components]
    for pred_id, succ_id, cost in pruned_arcs:
        arcs_by_component[comp_of[pred_id]].append((pred_id, succ_id, cost))

    decomposed_total = 0.0
    for ci, comp in enumerate(components):
        if len(comp) == 1:
            continue  # a singleton component trivially contributes 0 link-cost flow
        comp_cost, _ = _build_solver(
            comp, entry_cost, detection_cost, exit_cost, arcs_by_component[ci]
        ).solve(0, 1)
        decomposed_total += comp_cost

    assert abs(monolithic_cost - decomposed_total) < FLOAT_TOL, (
        f"seed={seed}: decomposed total {decomposed_total} != "
        f"monolithic total {monolithic_cost} ({len(components)} components)"
    )


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_topo_potentials_match_bellman_ford_on_random_dags(seed):
    event_ids, entry_cost, detection_cost, exit_cost, arcs = _random_costs(seed)
    solver = _build_solver(event_ids, entry_cost, detection_cost, exit_cost, arcs)

    bellman = solver._bellman_ford(0)
    topo = solver._topo_shortest(0, default_event_topo_order(len(event_ids)))

    assert len(bellman) == len(topo)
    for b, t in zip(bellman, topo, strict=True):
        if b == float("inf") or t == float("inf"):
            assert b == t
        else:
            assert abs(b - t) < FLOAT_TOL


def test_solver_still_matches_networkx_oracle_after_pruning():
    """The existing oracle test (test_mincostflow.py) already covers the
    UNPRUNED solver; this re-runs the same style of check on the PRUNED
    graph to confirm pruning + the oracle agree too, on a small instance."""
    event_ids, entry_cost, detection_cost, exit_cost, arcs = _random_costs(seed=7)
    pruned_arcs = prune_dominated_arcs(arcs, exit_cost, entry_cost)

    solver = _build_solver(event_ids, entry_cost, detection_cost, exit_cost, pruned_arcs)
    our_cost, our_flow = solver.solve(0, 1)
    k_star = int(round(our_flow))
    assert abs(our_flow - k_star) < FLOAT_TOL

    index_of = {eid: i for i, eid in enumerate(event_ids)}
    scale = 10_000
    g = nx.DiGraph()
    for i in range(len(event_ids)):
        g.add_node(2 + 2 * i, demand=0)
        g.add_node(2 + 2 * i + 1, demand=0)
    g.add_node(0, demand=0)
    g.add_node(1, demand=0)
    for eid in event_ids:
        i = index_of[eid]
        g.add_edge(0, 2 + 2 * i, capacity=1, weight=int(round(entry_cost[eid] * scale)))
        g.add_edge(
            2 + 2 * i, 2 + 2 * i + 1, capacity=1, weight=int(round(detection_cost[eid] * scale))
        )
        g.add_edge(2 + 2 * i + 1, 1, capacity=1, weight=int(round(exit_cost[eid] * scale)))
    for pred_id, succ_id, cost in pruned_arcs:
        pi, si = index_of[pred_id], index_of[succ_id]
        g.add_edge(2 + 2 * pi + 1, 2 + 2 * si, capacity=1, weight=int(round(cost * scale)))

    if k_star == 0:
        assert our_cost == 0.0
        return

    g.nodes[0]["demand"] = -k_star
    g.nodes[1]["demand"] = k_star
    oracle_cost = nx.min_cost_flow_cost(g) / scale
    assert abs(our_cost - oracle_cost) < 1e-4


BETAS_FOR_PRUNING_TEST = [-3.0, -1.0, 0.0, 1.0, 3.0]


@pytest.mark.parametrize("seed", range(N_TRIALS))
@pytest.mark.parametrize("beta", BETAS_FOR_PRUNING_TEST)
def test_pruning_stays_exact_under_nonzero_link_bias(seed, beta):
    """`link_bias` (beta, docs/decisions.md's calibration sweep) shifts
    every link arc's cost by a constant `-beta` (engine.association.window's
    `link_bias`), so the link threshold moves from `s(i,j) > 0` to
    `s(i,j) > -beta` -- but `prune_dominated_arcs`'s bypass comparison must
    still use the UNBIASED exit/entry costs (the bypass literally routes
    through those arcs, which beta never touches). This reproduces
    `test_pruned_and_unpruned_reach_the_same_optimal_cost` but with a beta
    shift baked into the arc costs first, the same way
    `engine.association.window._solve_one_window` builds them, to prove
    dominance pruning is still lossless at every swept beta value."""
    event_ids, entry_cost, detection_cost, exit_cost, base_arcs = _random_costs(seed)
    arcs = [(pred, succ, cost - beta) for pred, succ, cost in base_arcs]

    unpruned_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, arcs
    ).solve(0, 1)

    pruned_arcs = prune_dominated_arcs(arcs, exit_cost, entry_cost)
    assert len(pruned_arcs) <= len(arcs)
    pruned_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, pruned_arcs
    ).solve(0, 1)

    assert abs(unpruned_cost - pruned_cost) < FLOAT_TOL, (
        f"seed={seed} beta={beta}: pruning changed the optimal cost "
        f"({unpruned_cost} unpruned vs {pruned_cost} pruned, "
        f"{len(arcs)}->{len(pruned_arcs)} arcs)"
    )


@pytest.mark.parametrize("beta", BETAS_FOR_PRUNING_TEST)
def test_pruning_matches_unpruned_on_a_real_small_window_with_link_bias(beta):
    """Same claim as `test_pruning_matches_unpruned_on_a_real_small_window`,
    with a nonzero beta baked into the real scored arcs the same way
    `engine.association.window._solve_one_window` does -- confirms the
    dominance-pruning exactness proof holds on real fused-score data, not
    just synthetic random costs, at every swept beta value."""
    city = generate_city(n_cameras=8, seed=1)
    ds = generate_dataset_with_city(city, n_vehicles=120, hours=1, seed=5, clone_fraction=0.0)
    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)

    events = sorted(ds.events, key=lambda e: e.timestamp)
    assert len(events) > 20, "need a real handful of events for this to mean anything"
    index_of = {e.event_id: i for i, e in enumerate(events)}
    border_cameras = {c.camera_id for c in ds.city.cameras if c.is_border}
    entry_cost = {
        e.event_id: entry_exit.entry_cost(e.camera_id in border_cameras) for e in events
    }
    exit_cost = {e.event_id: entry_exit.exit_cost(e.camera_id in border_cameras) for e in events}
    detection_cost = {e.event_id: real_detection_cost(e) for e in events}

    gate_result = gate_candidates(events, gate)
    pred_idx, succ_idx = [], []
    for succ_id, pred_ids in gate_result.candidates.items():
        for pred_id in pred_ids:
            pred_idx.append(index_of[pred_id])
            succ_idx.append(index_of[succ_id])
    assert pred_idx, "need real gated pairs for this to mean anything"

    totals = score_pairs_batch_totals(events, np.array(pred_idx), np.array(succ_idx), fusion_model)
    event_ids = [e.event_id for e in events]
    all_arcs = [
        (
            events[pred_idx[k]].event_id,
            events[succ_idx[k]].event_id,
            -float(totals[k])
            + exit_cost[events[pred_idx[k]].event_id]
            + entry_cost[events[succ_idx[k]].event_id]
            - beta,
        )
        for k in range(len(pred_idx))
    ]

    unpruned_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, all_arcs
    ).solve(0, 1)
    pruned_arcs = prune_dominated_arcs(all_arcs, exit_cost, entry_cost)
    print(f"real window beta={beta}: kept {len(pruned_arcs)}/{len(all_arcs)} arcs")
    pruned_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, pruned_arcs
    ).solve(0, 1)

    assert abs(unpruned_cost - pruned_cost) < FLOAT_TOL


def test_pruning_matches_unpruned_on_a_real_small_window():
    """Same claim as `test_pruned_and_unpruned_reach_the_same_optimal_cost`,
    but with REAL scored arcs from the actual pipeline (gating + fused
    log-odds scoring), not synthetic random costs -- the task's explicit
    "and on a real small window" requirement."""
    city = generate_city(n_cameras=8, seed=1)
    ds = generate_dataset_with_city(city, n_vehicles=120, hours=1, seed=5, clone_fraction=0.0)
    models = fit_all(ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(ds.events, ds.city)
    gate = Gate(model=models.kinematic_model)

    events = sorted(ds.events, key=lambda e: e.timestamp)
    assert len(events) > 20, "need a real handful of events for this to mean anything"
    index_of = {e.event_id: i for i, e in enumerate(events)}
    border_cameras = {c.camera_id for c in ds.city.cameras if c.is_border}
    entry_cost = {
        e.event_id: entry_exit.entry_cost(e.camera_id in border_cameras) for e in events
    }
    exit_cost = {e.event_id: entry_exit.exit_cost(e.camera_id in border_cameras) for e in events}
    detection_cost = {e.event_id: real_detection_cost(e) for e in events}

    gate_result = gate_candidates(events, gate)
    pred_idx, succ_idx = [], []
    for succ_id, pred_ids in gate_result.candidates.items():
        for pred_id in pred_ids:
            pred_idx.append(index_of[pred_id])
            succ_idx.append(index_of[succ_id])
    assert pred_idx, "need real gated pairs for this to mean anything"

    totals = score_pairs_batch_totals(events, np.array(pred_idx), np.array(succ_idx), fusion_model)
    all_arcs = [
        (events[pred_idx[k]].event_id, events[succ_idx[k]].event_id, -float(totals[k]))
        for k in range(len(pred_idx))
    ]
    event_ids = [e.event_id for e in events]

    unpruned_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, all_arcs
    ).solve(0, 1)
    pruned_arcs = prune_dominated_arcs(all_arcs, exit_cost, entry_cost)
    print(f"real window: kept {len(pruned_arcs)}/{len(all_arcs)} arcs")
    pruned_cost, _ = _build_solver(
        event_ids, entry_cost, detection_cost, exit_cost, pruned_arcs
    ).solve(0, 1)

    assert abs(unpruned_cost - pruned_cost) < FLOAT_TOL
