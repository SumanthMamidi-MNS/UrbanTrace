"""Global association via min-cost flow over a time-ordered DAG
(architecture.md §3, §7).

One node per event, split `u_i -> v_i` (arc cost = a detection-confidence
cost). `source -> u_i` (entry) and `v_i -> sink` (exit) costs are learned
from the fraction of events at border vs interior cameras
(`fit_entry_exit_costs` below). Each unit of source->sink flow traces one
vehicle's trajectory; capacity-1 node-split arcs enforce "one event, one
vehicle" for free.

An arc `v_i -> u_j` costs `-s(i,j) + exit_cost(i) + entry_cost(j)`, where
`s(i,j)` is the fused log-odds score from `engine.scoring.fusion`, for
every gated/blocked candidate pair.

THE ARC-COST DERIVATION (post-Day-3a fix -- this is the whole defence
against over-merging, eval/reports/error_analysis.json). In ANY feasible
flow, sending a unit along the link arc `v_i -> u_j` means `i`'s own exit
arc and `j`'s own entry arc sit idle (both nodes already have their one
unit of capacity committed to the link -- see `prune_dominated_arcs`'s
proof below). So linking `i` to `j` doesn't just "cost `-s(i,j)`" in
isolation: it REPLACES "i exits, pays exit_cost(i)" and "j enters, pays
entry_cost(j)" with the single link arc. Costing the link arc at the bare
`-s(i,j)` (the pre-fix formula) silently made linking cheaper than it
should be by exactly `exit_cost(i) + entry_cost(j)` -- a "merge bonus"
that let the solver prefer wrongly linking two different vehicles over
correctly paying for a track death and a new track birth, whenever
`-s(i,j) < exit_cost(i) + entry_cost(j)`, i.e. whenever `s(i,j)` was only
MILDLY negative. `eval/reports/error_analysis.json` found 57.6% of
full-city UrbanTrace's wrong links had `total_log_odds < 0` -- accepted only
because of this missing additive term.

Charging the link arc the SAME two costs it replaces (`-s(i,j) +
exit_cost(i) + entry_cost(j)`) fixes this exactly: the solver now prefers
linking over the bypass (ending i's path, starting j's separately) if and
only if
    -s(i,j) + exit_cost(i) + entry_cost(j)  <  exit_cost(i) + entry_cost(j)
i.e. if and only if `s(i,j) > 0` -- posterior odds > 1, the honest
Bayesian threshold. Mutual exclusion (capacity-1 node splits) and global
optimality (still a min-cost flow over the same graph shape) are
unchanged by this -- only the arc weights differ. `prune_dominated_arcs`
below drops every arc for which the bypass is at least as cheap as
linking, which under this formula is now EXACTLY the arcs with `s(i,j) <=
0` -- see that function's own docstring.

Solved by our OWN successive shortest augmenting paths (SSP) implementation
with Johnson potentials -- float costs throughout, no integer rounding
(unlike OR-Tools' integer-only `SimpleMinCostFlow`, see docs/decisions.md).
Total cost is convex in flow value k, so we stop augmenting the instant the
shortest augmenting path's TRUE cost turns >= 0 -- the optimal vehicle
count k falls out of that single pass, no binary search over k needed.

`networkx.min_cost_flow` is the correctness ORACLE (tests/test_mincostflow.py):
on small random instances, our solver's cumulative cost after k*
augmentations must exactly match networkx's min-cost flow of EXACTLY k*
units (the classical SSP invariant: at every intermediate flow value, SSP's
running cost already equals the true minimum for that value)."""

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import LinkEvidence, Trajectory

INF = float("inf")
EPS = 1e-9

MIN_PLATE_CONFIDENCE = 1e-3
DETECTION_COST_SCALE = 0.5

# Additive smoothing pseudo-count for the entry/exit border-fraction fit
# (fit_entry_exit_costs) -- keeps a class with zero observed starts/ends
# from producing a literal -log(0) = +inf cost.
ENTRY_EXIT_SMOOTHING = 1.0


# ---------------------------------------------------------------------------
# Generic min-cost flow solver (successive shortest paths + Johnson potentials)
# ---------------------------------------------------------------------------


@dataclass
class _Edge:
    to: int
    cap: float  # REMAINING residual capacity, mutated in place as flow is pushed
    cost: float
    original_cap: float  # fixed at construction; used to recover flow sent


class MinCostFlowSolver:
    """Directed graph min-cost flow via successive shortest augmenting
    paths. Add edges with `add_edge`, then call `solve(source, sink)`.

    Standard paired forward/reverse residual-edge representation: `cap` IS
    the remaining residual capacity (not a separate `capacity - flow`
    computation) -- augmenting along edge `eid` by `f` does
    `edges[eid].cap -= f; edges[eid ^ 1].cap += f`. The flow actually
    carried by a forward edge `eid` at any point equals
    `edges[eid ^ 1].cap` (the reverse edge starts at 0 and only ever moves
    in lockstep with the forward edge's consumption), which is what
    `trace_paths` uses to recover the final routing without a separate
    bookkeeping field.

    `solve` does NOT take a target flow value: it augments one shortest
    path at a time and stops the instant the shortest remaining augmenting
    path has non-negative true cost, since total cost is convex in flow
    value for this class of graph (no negative cycles -- see module
    docstring). This is exactly what lets the optimal vehicle count `k`
    fall out of a single pass."""

    def __init__(self, n_nodes: int):
        self.n = n_nodes
        self.adj: list[list[int]] = [[] for _ in range(n_nodes)]
        self.edges: list[_Edge] = []

    def add_edge(self, u: int, v: int, capacity: float, cost: float) -> int:
        """Adds a forward edge u->v and its zero-capacity reverse v->u.
        Returns the forward edge's index (always even; `idx ^ 1` is its
        paired reverse edge)."""
        idx = len(self.edges)
        self.edges.append(_Edge(to=v, cap=capacity, cost=cost, original_cap=capacity))
        self.edges.append(_Edge(to=u, cap=0.0, cost=-cost, original_cap=0.0))
        self.adj[u].append(idx)
        self.adj[v].append(idx + 1)
        return idx

    def _topo_shortest(self, s: int, topo_order: list[int]) -> list[float]:
        """DAG shortest-path DP: O(V+E), replacing Bellman-Ford for the
        INITIAL potentials only. Valid exactly when `topo_order` lists every
        node in an order such that every edge with positive residual
        capacity goes from an earlier position to a later one -- true before
        any flow has been pushed (every reverse edge starts at cap=0, so
        only "forward" edges are traversable), which is exactly the
        situation `solve()` calls this in. Our event-association graphs are
        always built in this shape: source, then each event's (u_i, v_i)
        pair in TIME order (gating only ever links an earlier event to a
        later one), then sink -- see `default_event_topo_order`."""
        dist = [INF] * self.n
        dist[s] = 0.0
        for u in topo_order:
            du = dist[u]
            if du == INF:
                continue
            for eid in self.adj[u]:
                e = self.edges[eid]
                if e.cap > EPS and du + e.cost < dist[e.to]:
                    dist[e.to] = du + e.cost
        return dist

    def _bellman_ford(self, s: int) -> list[float]:
        """SPFA-style Bellman-Ford: initial potentials, needed because the
        original graph has negative-cost edges (link arcs, entry arcs can
        also be negative) that plain Dijkstra cannot handle."""
        dist = [INF] * self.n
        dist[s] = 0.0
        in_queue = [False] * self.n
        dq: deque[int] = deque([s])
        in_queue[s] = True
        while dq:
            u = dq.popleft()
            in_queue[u] = False
            du = dist[u]
            if du == INF:
                continue
            for eid in self.adj[u]:
                e = self.edges[eid]
                if self.edges[eid].cap > EPS and du + e.cost < dist[e.to] - EPS:
                    dist[e.to] = du + e.cost
                    if not in_queue[e.to]:
                        dq.append(e.to)
                        in_queue[e.to] = True
        return dist

    def _dijkstra(self, s: int, potential: list[float]) -> tuple[list[float], list[int]]:
        """Dijkstra over reduced costs `cost(u,v) + potential[u] -
        potential[v]`, which Johnson's theorem guarantees are >= 0 on every
        residual edge as long as `potential` is a valid shortest-distance
        labelling from the previous round."""
        dist = [INF] * self.n
        dist[s] = 0.0
        prev_edge = [-1] * self.n
        visited = [False] * self.n
        pq: list[tuple[float, int]] = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            for eid in self.adj[u]:
                if visited[self.edges[eid].to] or self.edges[eid].cap <= EPS:
                    continue
                e = self.edges[eid]
                if potential[u] == INF or potential[e.to] == INF:
                    continue  # target unreachable in the original graph -- see module docstring
                reduced = e.cost + potential[u] - potential[e.to]
                nd = d + max(reduced, 0.0)
                if nd < dist[e.to] - EPS:
                    dist[e.to] = nd
                    prev_edge[e.to] = eid
                    heapq.heappush(pq, (nd, e.to))
        return dist, prev_edge

    def solve(self, s: int, t: int, topo_order: list[int] | None = None) -> tuple[float, float]:
        """Returns (total_cost, total_flow). Stops the moment the shortest
        remaining augmenting path's true cost is >= 0 -- see class
        docstring. Call `trace_paths` afterwards to recover the actual
        source->sink paths (trajectories) from the resulting flow.

        `topo_order`, when given, computes the INITIAL Johnson potentials via
        an O(V+E) topological DP instead of Bellman-Ford's O(V*E) worst case
        -- see `_topo_shortest`. Every subsequent round already uses
        Dijkstra over non-negative reduced costs regardless, so this only
        speeds up the first round, but that round is Bellman-Ford's only
        appearance in the whole solve."""
        potential = self._topo_shortest(s, topo_order) if topo_order else self._bellman_ford(s)
        if potential[t] == INF:
            return 0.0, 0.0

        total_cost = 0.0
        total_flow = 0.0

        while True:
            dist, prev_edge = self._dijkstra(s, potential)
            if dist[t] == INF:
                break
            for v in range(self.n):
                if dist[v] < INF:
                    potential[v] += dist[v]
            true_path_cost = potential[t] - potential[s]
            if true_path_cost >= -EPS:
                break

            # Walk back from t to s to find the bottleneck capacity and the
            # edges to augment.
            path_edges: list[int] = []
            v = t
            bottleneck = INF
            while v != s:
                eid = prev_edge[v]
                bottleneck = min(bottleneck, self.edges[eid].cap)
                path_edges.append(eid)
                v = self.edges[eid ^ 1].to

            for eid in path_edges:
                self.edges[eid].cap -= bottleneck
                self.edges[eid ^ 1].cap += bottleneck

            total_flow += bottleneck
            total_cost += true_path_cost * bottleneck

        return total_cost, total_flow

    def _flow_on(self, eid: int) -> float:
        """Flow currently carried by forward edge `eid`: equals its paired
        reverse edge's residual capacity (see class docstring)."""
        return self.edges[eid ^ 1].cap

    def trace_paths(self, s: int, t: int) -> list[list[int]]:
        """Decompose the flow left on the graph after `solve` into
        source->sink node paths (one per unit of flow, since every edge in
        our usage has capacity 1). Consumes flow as it traces (safe to call
        once after `solve`)."""
        paths: list[list[int]] = []
        while True:
            path = [s]
            u = s
            while u != t:
                nxt = None
                for eid in self.adj[u]:
                    e = self.edges[eid]
                    # A forward edge (original_cap>0) still carrying flow
                    # means real flow passed along it.
                    if e.original_cap > 0 and self._flow_on(eid) > EPS:
                        nxt = (eid, e.to)
                        break
                if nxt is None:
                    break
                eid, to = nxt
                # Consume one unit of this edge's flow: pushing 1 unit
                # "backward" along the reverse edge is exactly the standard
                # residual-graph operation for retracting flow already sent.
                self.edges[eid].cap += 1.0
                self.edges[eid ^ 1].cap -= 1.0
                path.append(to)
                u = to
            if u != t:
                break  # no more complete s->t paths with flow
            paths.append(path)
        return paths


# ---------------------------------------------------------------------------
# Domain-specific graph construction for event association
# ---------------------------------------------------------------------------


@dataclass
class EntryExitCosts:
    """Learned from the fraction of TRUE trajectory starts/ends observed at
    border vs interior cameras (architecture.md §3's own phrasing:
    "learned from the fraction of events at border cameras"). A camera
    class where real journeys rarely start pays a higher entry cost for
    starting a new trajectory there (and likewise for exits)."""

    entry_cost_border: float
    entry_cost_interior: float
    exit_cost_border: float
    exit_cost_interior: float

    def entry_cost(self, is_border: bool) -> float:
        return self.entry_cost_border if is_border else self.entry_cost_interior

    def exit_cost(self, is_border: bool) -> float:
        return self.exit_cost_border if is_border else self.exit_cost_interior


def fit_entry_exit_costs(
    events: list[DetectionEvent], city: CityConfig, smoothing: float = ENTRY_EXIT_SMOOTHING
) -> EntryExitCosts:
    """Fit entry/exit costs from ground-truth trajectory starts/ends in
    `events` (a training split). `cost = -log(P(is a true start | camera
    class))`, Laplace-smoothed so a class with zero observed starts/ends
    never produces a literal infinite cost."""
    border_cameras = {c.camera_id for c in city.cameras if c.is_border}

    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in events:
        if e.gt_vehicle_id is not None:
            by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    n_border = n_interior = 0
    n_start_border = n_start_interior = 0
    n_end_border = n_end_interior = 0

    for evs in by_vehicle.values():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        for idx, e in enumerate(evs_sorted):
            is_border = e.camera_id in border_cameras
            if is_border:
                n_border += 1
            else:
                n_interior += 1
            if idx == 0:
                if is_border:
                    n_start_border += 1
                else:
                    n_start_interior += 1
            if idx == len(evs_sorted) - 1:
                if is_border:
                    n_end_border += 1
                else:
                    n_end_interior += 1

    def rate(numerator: int, denominator: int) -> float:
        return (numerator + smoothing) / (denominator + 2 * smoothing)

    p_start_border = rate(n_start_border, n_border)
    p_start_interior = rate(n_start_interior, n_interior)
    p_end_border = rate(n_end_border, n_border)
    p_end_interior = rate(n_end_interior, n_interior)

    return EntryExitCosts(
        entry_cost_border=-math.log(p_start_border),
        entry_cost_interior=-math.log(p_start_interior),
        exit_cost_border=-math.log(p_end_border),
        exit_cost_interior=-math.log(p_end_interior),
    )


def detection_cost(event: DetectionEvent, scale: float = DETECTION_COST_SCALE) -> float:
    """Cost of the u_i -> v_i node-split arc: `scale * -log(confidence)`.
    Always >= 0 -- routing flow through ANY event pays this baseline cost
    of "explaining" the detection, small relative to a strong link's
    (potentially large-magnitude negative) `-s(i,j)` cost so it never
    dominates the linking decision, but still penalises confidently-poor
    reads over confidently-good ones when a trajectory has to choose
    between otherwise-similar candidates."""
    conf = max(event.plate_confidence, MIN_PLATE_CONFIDENCE)
    return scale * -math.log(conf)


def default_event_topo_order(n_events: int, source: int = 0, sink: int = 1) -> list[int]:
    """The standard topological order for an event-association graph built
    with the `SOURCE=0, SINK=1, u_i=2+2i, v_i=2+2i+1` convention, where
    events are indexed in TIME order (0..n_events-1): source first, then
    every event's (u_i, v_i) pair in increasing i, then sink. Valid because
    gating only ever links an earlier event to a later one, so every v_i ->
    u_j link arc has i < j -- see `MinCostFlowSolver._topo_shortest`."""
    order = [source]
    for i in range(n_events):
        order.append(2 + 2 * i)
        order.append(2 + 2 * i + 1)
    order.append(sink)
    return order


# ---------------------------------------------------------------------------
# Exact dominance pruning + connected-component decomposition.
#
# At production candidate density (architecture.md §7's gate, ~200
# candidates/event), the overwhelming majority of `v_i -> u_j` link arcs are
# weak (a wide spatio-temporal gate necessarily admits many implausible
# pairs alongside the true one) -- expensive for the solver to carry, and
# PROVABLY never needed for the optimal solution. Dropping them, and then
# solving each surviving weakly-connected component of the link graph
# independently, are both exact (lossless) transformations, not heuristics.
# ---------------------------------------------------------------------------

# No epsilon slack: the dominance argument below is an exact "costs no
# more" inequality, so a boundary tie is correctly prunable too (rerouting
# through the bypass costs the SAME, not more).
def prune_dominated_arcs(
    arcs: list[tuple[str, str, float]],
    exit_cost: dict[str, float],
    entry_cost: dict[str, float],
) -> list[tuple[str, str, float]]:
    """Drop link arc `v_i -> u_j` (cost `c`) whenever
        c >= exit_cost[i] + entry_cost[j]
    i.e. whenever routing directly through `i`'s own exit arc and `j`'s own
    entry arc ("the bypass") is at least as cheap as using the link.

    Under the current arc-cost formula (module docstring's "THE ARC-COST
    DERIVATION": `c = -s(i,j) + exit_cost[i] + entry_cost[j]`), this
    condition is ALGEBRAICALLY EXACTLY `s(i,j) <= 0`:
        c >= exit_cost[i] + entry_cost[j]
        -s(i,j) + exit_cost[i] + entry_cost[j] >= exit_cost[i] + entry_cost[j]
        -s(i,j) >= 0  <=>  s(i,j) <= 0
    So this function, unchanged since before that fix, now keeps EXACTLY
    the arcs with positive fused log-odds (`s(i,j) > 0`) -- it needed no
    changes itself; the fix was entirely in how callers compute `c`.

    Proof this is lossless: take any feasible flow that sends a unit along
    `v_i -> u_j`. Every node-split arc has capacity 1, so that unit is the
    ONLY flow through `u_i -> v_i` and `u_j -> v_j` -- `i`'s exit arc and
    `j`'s entry arc are therefore both idle (each already has its one unit
    of through-flow committed to the link). Splice the link arc out: end
    `i`'s path with `v_i -> sink` instead, and start `j`'s path with
    `source -> u_j` instead. Every capacity constraint still holds (the
    previously-idle exit/entry arcs now carry the rerouted unit; the
    removed link arc carried exactly that unit before, no more). The
    result is still a feasible flow, and its cost changed by exactly
    `(exit_cost[i] + entry_cost[j]) - c`, which is <= 0 under the pruning
    condition -- so the reroute costs no more. Hence an optimal solution
    never strictly NEEDS a dominated arc: dropping it cannot raise the
    achievable minimum cost. `i`/`j` ending their own (possibly length-1)
    trajectories there is exactly the "uncovered event" fallback both
    callers already implement.
    """
    kept = []
    for pred_id, succ_id, cost in arcs:
        bypass = exit_cost[pred_id] + entry_cost[succ_id]
        if cost < bypass:
            kept.append((pred_id, succ_id, cost))
    return kept


def connected_components_by_arcs(
    event_ids: list[str], arcs: list[tuple[str, str, float]]
) -> list[list[str]]:
    """Weakly-connected components of the graph whose nodes are `event_ids`
    and whose (undirected) edges are `arcs`' (pred_id, succ_id) pairs. An
    event with no surviving arc at all comes back as its own singleton
    component. Union-find, O((n+m) alpha(n))."""
    parent: dict[str, str] = {e: e for e in event_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pred_id, succ_id, _ in arcs:
        union(pred_id, succ_id)

    groups: dict[str, list[str]] = {}
    for e in event_ids:
        groups.setdefault(find(e), []).append(e)
    return list(groups.values())


@dataclass
class AssociationGraph:
    """Maps event_ids to/from min-cost-flow node indices and holds the
    solver instance. `SOURCE`/`SINK` are fixed node ids 0/1; event `i`
    (0-indexed position in `event_ids`) occupies node-split `u_i = 2+2i`,
    `v_i = 2+2i+1`."""

    event_ids: list[str]
    solver: MinCostFlowSolver
    index_of: dict[str, int] = field(init=False)

    SOURCE = 0
    SINK = 1

    def __post_init__(self) -> None:
        self.index_of = {eid: i for i, eid in enumerate(self.event_ids)}

    def u(self, event_id: str) -> int:
        return 2 + 2 * self.index_of[event_id]

    def v(self, event_id: str) -> int:
        return 2 + 2 * self.index_of[event_id] + 1


def build_association_graph(
    events: list[DetectionEvent],
    candidates: dict[str, list[str]],
    link_scores: dict[tuple[str, str], float],
    entry_exit: EntryExitCosts,
    city: CityConfig,
) -> AssociationGraph:
    """Build the time-ordered DAG (module docstring). `candidates` maps
    event_id -> candidate PREDECESSOR event_ids (gating.gate_candidates /
    blocking.apply_blocking's output shape); `link_scores[(pred_id,
    succ_id)]` is the fused `total_log_odds` for that pair
    (engine.scoring.fusion.score_pair). Every event gets an entry and exit
    arc regardless of whether it has any candidate links -- an event with
    no plausible neighbours simply becomes its own length-1 trajectory.

    Link arc cost is `-score + exit_cost[pred_id] + entry_cost[succ_id]`
    (module docstring's "THE ARC-COST DERIVATION") -- the ACTUAL entry/exit
    costs each event would otherwise pay, read from the same
    `entry_cost_of`/`exit_cost_of` dicts used for that event's own
    node-split arcs, so the two are never allowed to drift apart."""
    border_cameras = {c.camera_id for c in city.cameras if c.is_border}
    event_ids = [e.event_id for e in events]
    n_nodes = 2 + 2 * len(event_ids)
    solver = MinCostFlowSolver(n_nodes)
    graph = AssociationGraph(event_ids=event_ids, solver=solver)

    entry_cost_of: dict[str, float] = {}
    exit_cost_of: dict[str, float] = {}
    for e in events:
        is_border = e.camera_id in border_cameras
        entry = entry_exit.entry_cost(is_border)
        exit_ = entry_exit.exit_cost(is_border)
        entry_cost_of[e.event_id] = entry
        exit_cost_of[e.event_id] = exit_
        u_i, v_i = graph.u(e.event_id), graph.v(e.event_id)
        solver.add_edge(graph.SOURCE, u_i, 1.0, entry)
        solver.add_edge(u_i, v_i, 1.0, detection_cost(e))
        solver.add_edge(v_i, graph.SINK, 1.0, exit_)

    known_events = set(event_ids)
    for succ_id, pred_ids in candidates.items():
        if succ_id not in known_events:
            continue
        for pred_id in pred_ids:
            if pred_id not in known_events:
                continue
            score = link_scores.get((pred_id, succ_id))
            if score is None:
                continue
            cost = -score + exit_cost_of[pred_id] + entry_cost_of[succ_id]
            solver.add_edge(graph.v(pred_id), graph.u(succ_id), 1.0, cost)

    return graph


def solve_trajectories(graph: AssociationGraph) -> tuple[list[list[str]], float, float]:
    """Runs the SSP solver to completion and decodes the resulting flow
    into trajectories (ordered lists of event_ids). Returns (trajectories,
    total_cost, total_flow)."""
    total_cost, total_flow = graph.solver.solve(graph.SOURCE, graph.SINK)
    node_paths = graph.solver.trace_paths(graph.SOURCE, graph.SINK)

    id_of_node: dict[int, str] = {}
    for eid in graph.event_ids:
        id_of_node[graph.u(eid)] = eid
        id_of_node[graph.v(eid)] = eid

    trajectories: list[list[str]] = []
    covered: set[str] = set()
    for path in node_paths:
        seq: list[str] = []
        for node in path:
            eid = id_of_node.get(node)
            if eid is not None and (not seq or seq[-1] != eid):
                seq.append(eid)
        if seq:
            trajectories.append(seq)
            covered.update(seq)

    # An event with no beneficial link in either direction has a strictly
    # positive standalone entry+detection+exit cost, so SSP -- which only
    # ever augments net-negative-cost paths -- correctly never spends flow
    # on it. That must not make the event vanish from the output: every
    # detection is accounted for somewhere, even as its own length-1
    # trajectory (see engine.association.window for the same fix, and
    # docs/decisions.md "Day 3a").
    for eid in graph.event_ids:
        if eid not in covered:
            trajectories.append([eid])

    return trajectories, total_cost, total_flow


def trajectories_from_events(
    events: list[DetectionEvent],
    candidates: dict[str, list[str]],
    link_scores: dict[tuple[str, str], float],
    entry_exit: EntryExitCosts,
    city: CityConfig,
    links_by_pair: dict[tuple[str, str], LinkEvidence] | None = None,
) -> list[Trajectory]:
    """End-to-end: build the graph, solve, and package results as
    `Trajectory` contract objects (decoded_plate/plate_confidence left for
    Day 3b's consensus decoder -- populated here with the first event's own
    argmax/confidence as a placeholder so the contract is always valid)."""
    graph = build_association_graph(events, candidates, link_scores, entry_exit, city)
    event_paths, _, _ = solve_trajectories(graph)

    events_by_id = {e.event_id: e for e in events}
    trajectories: list[Trajectory] = []
    for i, path in enumerate(event_paths):
        path_events = [events_by_id[eid] for eid in path]
        links: list[LinkEvidence] = []
        if links_by_pair:
            for a, b in zip(path[:-1], path[1:], strict=True):
                link = links_by_pair.get((a, b))
                if link is not None:
                    links.append(link)
        first = path_events[0]
        trajectories.append(
            Trajectory(
                trajectory_id=f"traj_{i:06d}",
                event_ids=path,
                decoded_plate=first.plate_argmax,
                plate_confidence=first.plate_confidence,
                start_time=path_events[0].timestamp,
                end_time=path_events[-1].timestamp,
                camera_sequence=[e.camera_id for e in path_events],
                links=links,
                gt_vehicle_id=first.gt_vehicle_id,
            )
        )
    return trajectories


__all__ = [
    "AssociationGraph",
    "EntryExitCosts",
    "MinCostFlowSolver",
    "build_association_graph",
    "connected_components_by_arcs",
    "default_event_topo_order",
    "detection_cost",
    "fit_entry_exit_costs",
    "prune_dominated_arcs",
    "solve_trajectories",
    "trajectories_from_events",
]
