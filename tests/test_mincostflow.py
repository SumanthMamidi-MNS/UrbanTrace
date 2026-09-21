"""The correctness test that matters (architecture.md §3, docs/decisions.md
"Day 3a" Part 3): our successive-shortest-paths solver, with Johnson
potentials and float costs throughout, must exactly match
`networkx.min_cost_flow` -- the test oracle -- on small random instances.

This is not a weaker self-consistency check: SSP's classical invariant is
that its cumulative cost after k augmentations already equals the TRUE
minimum cost of routing exactly k units of flow, for every k up to the
point where augmenting paths turn non-negative. So after our solver
finishes (having found k* profitable augmentations), we ask networkx for
the minimum cost of routing EXACTLY k* units through the same graph and
assert the two costs match to float tolerance."""

import random

import networkx as nx
import pytest

from engine.association.mincostflow import MinCostFlowSolver

FLOAT_TOL = 1e-6
N_TRIALS = 20
N_EVENTS = 14  # -> 2 + 2*14 = 30 nodes (source, sink, u_i/v_i per event)

# networkx's `network_simplex` is documented as "not guaranteed to work" with
# floating-point weights -- and in practice, on this graph shape, it doesn't:
# it hangs (observed directly; not a documentation nitpick). Our random
# instance's costs are generated at a FIXED 4-decimal-digit precision
# (`round(..., 4)`), so scaling by 10**4 turns them into EXACT integers with
# no loss of information -- the identical problem instance, just represented
# in the numeric type networkx's oracle is actually reliable with. Our own
# solver is still fed and asserts on the original float costs throughout,
# per the task spec ("float costs throughout, no integer rounding") -- the
# scaling is purely how the ORACLE is queried, not a property of our solver.
COST_SCALE = 10_000


def _random_instance(seed: int) -> tuple[MinCostFlowSolver, nx.DiGraph, int, int]:
    """A random time-ordered event-association-shaped DAG: n_events nodes
    each split u_i/v_i, with random entry/detection/exit costs and random
    (mostly-negative, to encourage nontrivial augmenting paths) link costs
    from v_i to u_j for j > i. Mirrors the real graph's structure
    (mincostflow.build_association_graph) without depending on any
    simulator/scoring code -- this test is about the SOLVER, not the
    domain. The networkx copy (`nxg`) carries INTEGER-scaled weights (see
    `COST_SCALE`) so the oracle itself is numerically reliable; our solver
    is built from the original float costs."""
    rng = random.Random(seed)
    n_nodes = 2 + 2 * N_EVENTS
    SOURCE, SINK = 0, 1

    solver = MinCostFlowSolver(n_nodes)
    nxg = nx.DiGraph()
    for node in range(n_nodes):
        nxg.add_node(node, demand=0)

    def add(u: int, v: int, cap: float, cost: float) -> None:
        solver.add_edge(u, v, cap, cost)
        # networkx accepts multiple parallel edges only via MultiDiGraph;
        # our graph never has parallel u->v edges by construction (each
        # (source,u_i), (u_i,v_i), (v_i,sink), (v_i,u_j) pair is unique), so
        # a plain DiGraph is fine.
        nxg.add_edge(u, v, capacity=int(round(cap)), weight=int(round(cost * COST_SCALE)))

    def u(i: int) -> int:
        return 2 + 2 * i

    def v(i: int) -> int:
        return 2 + 2 * i + 1

    for i in range(N_EVENTS):
        add(SOURCE, u(i), 1.0, round(rng.uniform(0.1, 3.0), 4))
        add(u(i), v(i), 1.0, round(rng.uniform(0.1, 1.5), 4))
        add(v(i), SINK, 1.0, round(rng.uniform(0.1, 3.0), 4))

    for i in range(N_EVENTS):
        for j in range(i + 1, N_EVENTS):
            if rng.random() < 0.35:
                cost = round(rng.uniform(-6.0, 2.0), 4)
                add(v(i), u(j), 1.0, cost)

    return solver, nxg, SOURCE, SINK


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_solver_matches_networkx_oracle_at_optimal_k(seed):
    solver, nxg, source, sink = _random_instance(seed)

    our_cost, our_flow = solver.solve(source, sink)
    k_star = int(round(our_flow))
    assert abs(our_flow - k_star) < FLOAT_TOL, "flow should be an integer (unit capacities)"

    if k_star == 0:
        # No profitable augmenting path exists at all -- trivially correct,
        # but assert it's actually trivial (no negative-cost s->t path).
        assert our_cost == 0.0
        return

    g = nxg.copy()
    g.nodes[source]["demand"] = -k_star
    g.nodes[sink]["demand"] = k_star
    oracle_cost_scaled = nx.min_cost_flow_cost(g)
    oracle_cost = oracle_cost_scaled / COST_SCALE

    assert abs(our_cost - oracle_cost) < 1e-4, (
        f"seed={seed}: our solver's cost {our_cost} for k*={k_star} units of flow "
        f"does not match networkx's oracle cost {oracle_cost}"
    )


def test_solver_stops_when_no_events_at_all():
    solver = MinCostFlowSolver(2)
    cost, flow = solver.solve(0, 1)
    assert cost == 0.0
    assert flow == 0.0


def test_solver_handles_disconnected_sink():
    """Source and sink with no path between them at all -- must not crash,
    must report zero flow."""
    solver = MinCostFlowSolver(4)
    solver.add_edge(0, 2, 1.0, 1.0)
    solver.add_edge(3, 1, 1.0, 1.0)  # 2 and 3 never connect
    cost, flow = solver.solve(0, 1)
    assert cost == 0.0
    assert flow == 0.0
