"""Sliding-window association for online operation (architecture.md §3, §7).

~30 minute windows, ~50% overlap (a 15-minute step): each window is solved
independently by `mincostflow`, and an "open track" -- one whose last event
falls inside the OVERLAP zone at the tail of the window, meaning it might
continue -- carries its last event forward as a seed for the next window.
That seed event is included in the next window's own event set NATURALLY
(it already falls inside the next window's time range, since the overlap
zone is shared by construction); the only special treatment it gets is a
REDUCED entry cost, so the solver doesn't re-charge a fresh "vehicle
entered the city" penalty for a trajectory that really entered back in an
earlier window (architecture.md §3: "the last event of each open track
seeds the next window at reduced entry cost").

Trajectory identity is stitched across windows by event_id: if a window's
locally-solved path starts at an event that was carried over from the
previous window, that path is treated as a CONTINUATION of the global
trajectory that owned it, not a new one."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engine.association.gating import Gate, gate_candidates
from engine.association.mincostflow import (
    EntryExitCosts,
    MinCostFlowSolver,
    connected_components_by_arcs,
    default_event_topo_order,
    detection_cost,
    prune_dominated_arcs,
)
from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import LinkEvidence, Trajectory
from engine.scoring.fusion import FusionModel, score_pairs_batch, score_pairs_batch_totals

DEFAULT_WINDOW_SIZE_S = 1800.0  # 30 minutes
DEFAULT_WINDOW_STEP_S = 900.0  # 15 minutes -> 50% overlap
# A carried-over event already "paid" its entry cost in an earlier window;
# charging it again would unfairly bias the solver against continuing a
# real open track relative to (incorrectly) starting a fresh one. Not zero,
# so a track that genuinely should terminate (no good link exists in the
# new window) still can via its normal exit arc -- this only affects how
# cheap re-entering as a "start" is, not whether continuing is possible.
CARRIED_ENTRY_COST_FACTOR = 0.05


def _window_bounds(
    t_min: float, t_max: float, size: float, step: float
) -> list[tuple[float, float]]:
    bounds = []
    start = t_min
    while start <= t_max:
        bounds.append((start, start + size))
        start += step
    return bounds


@dataclass
class WindowStats:
    n_events: int
    n_arcs_before: int
    n_arcs_after: int
    n_components: int
    largest_component: int


def _solve_component(
    component_events: list[DetectionEvent],
    component_arcs: list[tuple[str, str, float]],
    entry_cost_of: dict[str, float],
    exit_cost_of: dict[str, float],
) -> list[list[str]]:
    """Solve ONE weakly-connected component's own min-cost flow, with its own
    local source/sink (engine.association.mincostflow.prune_dominated_arcs /
    connected_components_by_arcs's docstrings: components share no event
    nodes, so summing each component's optimum is exact)."""
    local_events = sorted(component_events, key=lambda e: e.timestamp)
    local_index = {e.event_id: i for i, e in enumerate(local_events)}
    n = len(local_events)
    solver = MinCostFlowSolver(2 + 2 * n)

    def u(eid: str) -> int:
        return 2 + 2 * local_index[eid]

    def v(eid: str) -> int:
        return 2 + 2 * local_index[eid] + 1

    for e in local_events:
        solver.add_edge(0, u(e.event_id), 1.0, entry_cost_of[e.event_id])
        solver.add_edge(u(e.event_id), v(e.event_id), 1.0, detection_cost(e))
        solver.add_edge(v(e.event_id), 1, 1.0, exit_cost_of[e.event_id])
    for pred_id, succ_id, cost in component_arcs:
        solver.add_edge(v(pred_id), u(succ_id), 1.0, cost)

    solver.solve(0, 1, topo_order=default_event_topo_order(n))
    node_paths = solver.trace_paths(0, 1)

    id_of_node: dict[int, str] = {}
    for e in local_events:
        id_of_node[u(e.event_id)] = e.event_id
        id_of_node[v(e.event_id)] = e.event_id

    event_paths: list[list[str]] = []
    covered: set[str] = set()
    for path in node_paths:
        seq: list[str] = []
        for node in path:
            eid = id_of_node.get(node)
            if eid is not None and (not seq or seq[-1] != eid):
                seq.append(eid)
        if seq:
            event_paths.append(seq)
            covered.update(seq)

    # SSP only ever augments a PROFITABLE (net negative-cost) path -- an
    # event with no beneficial link in either direction (e.g. a vehicle
    # seen at exactly one camera) has a strictly positive standalone
    # entry+detection+exit cost, so the solver correctly never "spends" flow
    # on it and it ends up with zero flow through its own u_i->v_i arc. That
    # is mathematically correct cost-minimisation, but it must not translate
    # into the event vanishing from every trajectory: every detection is
    # reported somewhere, even if only as its own length-1 trajectory (a
    # genuine "seen once" outcome, not a modelling failure). See
    # docs/decisions.md "Day 3a" -- this was caught by the reviewer's
    # full-day coverage test after being missed by the small-scale one.
    for e in local_events:
        if e.event_id not in covered:
            event_paths.append([e.event_id])

    return event_paths


def _solve_one_window(
    window_events: list[DetectionEvent],
    carried_event_ids: set[str],
    gate: Gate,
    fusion_model: FusionModel,
    entry_exit: EntryExitCosts,
    city: CityConfig,
    link_bias: float = 0.0,
) -> tuple[list[list[str]], dict[tuple[str, str], LinkEvidence], WindowStats]:
    """Gate + score + min-cost-flow solve a single window's events. Returns
    (event_id paths, link evidence by (from,to) pair actually USED, stats).

    Performance shape (docs/decisions.md): candidate pools are wide (a
    correctness requirement -- gating.py's whole point is not to drop the
    true predecessor), so most gated arcs are weak. Rather than handing all
    of them to one big solver, this (1) scores every candidate NUMERICALLY
    ONLY (`score_pairs_batch_totals`, no LinkEvidence objects), (2) drops
    every PROVABLY-unneeded arc (`prune_dominated_arcs`), (3) solves each
    surviving weakly-connected component on its own (smaller graphs, and
    isolated events skip the solver entirely), and only THEN (4) builds full
    `LinkEvidence` (`score_pairs_batch`) for the pairs the solution actually
    USES -- the WHY panel only ever needs evidence for a used link, not
    every candidate that was considered and rejected.

    Link arc cost is `-s(i,j) + exit_cost(i) + entry_cost(j)` (see
    engine.association.mincostflow module docstring, "THE ARC-COST
    DERIVATION") using the ACTUAL `exit_cost_of`/`entry_cost_of` dicts built
    below -- which already apply `CARRIED_ENTRY_COST_FACTOR` wherever an
    event is a carried-over seed -- so the link-arc cost and each event's
    own node-split arcs never disagree about what that event's entry/exit
    actually costs. Each pair's prior also uses that pair's SUCCESSOR's own
    gated-candidate count (`n_j`, from `gate_result.candidates`, counted
    BEFORE dominance pruning) rather than a fixed constant -- see
    engine.scoring.fusion module docstring's "THE PER-SUCCESSOR FIX".

    `link_bias` (beta, in nats; default 0.0, i.e. no change from the
    corrected formula) is SUBTRACTED from every link arc's cost -- link arc
    cost is `-s(i,j) + exit_cost(i) + entry_cost(j) - beta`, so the solver
    links iff `s(i,j) > -beta` instead of the plain Bayesian `s(i,j) > 0`
    (mincostflow.py's "THE ARC-COST DERIVATION" is the beta=0 case). A
    positive beta makes linking EASIER (recovers exactly the kind of
    weak-plate-evidence true link that a correct kinematic model no longer
    clears at rush-hour density -- see docs/decisions.md's beta sweep on a
    TRAIN-seed day, `eval/calibrate_link_bias.py`); a negative beta makes it
    stricter. `prune_dominated_arcs`'s bypass comparison is UNCHANGED by
    beta (the bypass literally routes through the exit/entry arcs, which
    beta never touches), so pruning stays the exact `s(i,j) <= -beta`
    condition at any beta -- see that function's docstring and
    tests/test_pruning_and_decomposition.py's beta-parametrized cases.
    Default 0.0 is a sensitivity-sweep/calibration knob only, never set away
    from 0.0 except via a calibrated default in eval/run_pipeline.py."""
    gate_result = gate_candidates(window_events, gate)
    events_by_id = {e.event_id: e for e in window_events}
    index_of = {e.event_id: i for i, e in enumerate(window_events)}
    n_candidates_by_succ_id: dict[str, int] = {
        succ_id: len(pred_ids) for succ_id, pred_ids in gate_result.candidates.items()
    }

    border_cameras = {c.camera_id for c in city.cameras if c.is_border}
    entry_cost_of: dict[str, float] = {}
    exit_cost_of: dict[str, float] = {}
    for e in window_events:
        is_border = e.camera_id in border_cameras
        entry = entry_exit.entry_cost(is_border)
        if e.event_id in carried_event_ids:
            entry *= CARRIED_ENTRY_COST_FACTOR
        entry_cost_of[e.event_id] = entry
        exit_cost_of[e.event_id] = entry_exit.exit_cost(is_border)

    # Gather every gated (pred, succ) pair as index arrays into
    # `window_events` and score them all NUMERICALLY in ONE batch (no
    # LinkEvidence yet -- see function docstring).
    batch_pred_idx: list[int] = []
    batch_succ_idx: list[int] = []
    batch_n_candidates: list[int] = []
    for succ_id, pred_ids in gate_result.candidates.items():
        if succ_id in carried_event_ids:
            # A carried-over seed's TRUE predecessor is its own previous
            # event from an earlier window, outside this window's node set
            # entirely -- it must never accept an in-window predecessor
            # here, or the solver can find it cheaper to attach some
            # genuinely-new event ahead of it, producing a local path that
            # doesn't start at the seed. That breaks the stitching
            # invariant in solve_windowed (which identifies a continuation
            # by `path[0] in carry_owner`): the seed would then be
            # re-emitted as part of a brand-new global trajectory while
            # still belonging to its old one, duplicating it in the output.
            continue
        n_j = n_candidates_by_succ_id[succ_id]
        for pred_id in pred_ids:
            batch_pred_idx.append(index_of[pred_id])
            batch_succ_idx.append(index_of[succ_id])
            batch_n_candidates.append(n_j)

    n_arcs_before = len(batch_pred_idx)
    kept_arcs: list[tuple[str, str, float]] = []
    if batch_pred_idx:
        totals = score_pairs_batch_totals(
            window_events,
            np.array(batch_pred_idx),
            np.array(batch_succ_idx),
            fusion_model,
            n_candidates_per_pair=np.array(batch_n_candidates),
        )
        all_arcs = [
            (
                window_events[batch_pred_idx[k]].event_id,
                window_events[batch_succ_idx[k]].event_id,
                -float(totals[k])
                + exit_cost_of[window_events[batch_pred_idx[k]].event_id]
                + entry_cost_of[window_events[batch_succ_idx[k]].event_id]
                - link_bias,
            )
            for k in range(n_arcs_before)
        ]
        kept_arcs = prune_dominated_arcs(all_arcs, exit_cost_of, entry_cost_of)

    all_event_ids = [e.event_id for e in window_events]
    components = connected_components_by_arcs(all_event_ids, kept_arcs)
    # Map each event to its component index to bucket kept_arcs per
    # component in one O(arcs) pass rather than re-scanning kept_arcs per
    # component.
    comp_of: dict[str, int] = {}
    for ci, comp in enumerate(components):
        for eid in comp:
            comp_of[eid] = ci
    arcs_by_component_idx: list[list[tuple[str, str, float]]] = [[] for _ in components]
    for pred_id, succ_id, cost in kept_arcs:
        arcs_by_component_idx[comp_of[pred_id]].append((pred_id, succ_id, cost))

    event_paths: list[list[str]] = []
    for ci, comp in enumerate(components):
        if len(comp) == 1:
            event_paths.append(list(comp))
            continue
        comp_events = [events_by_id[eid] for eid in comp]
        event_paths.extend(
            _solve_component(comp_events, arcs_by_component_idx[ci], entry_cost_of, exit_cost_of)
        )

    # Full LinkEvidence (with the per-channel breakdown the WHY panel needs)
    # is only ever built for pairs the FINAL solution actually uses --
    # consecutive event_ids within a solved path are exactly those.
    used_pred_idx: list[int] = []
    used_succ_idx: list[int] = []
    used_n_candidates: list[int] = []
    for seq in event_paths:
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            used_pred_idx.append(index_of[a])
            used_succ_idx.append(index_of[b])
            used_n_candidates.append(n_candidates_by_succ_id[b])

    links_by_pair: dict[tuple[str, str], LinkEvidence] = {}
    if used_pred_idx:
        links = score_pairs_batch(
            window_events,
            np.array(used_pred_idx),
            np.array(used_succ_idx),
            fusion_model,
            n_candidates_per_pair=np.array(used_n_candidates),
        )
        for link in links:
            links_by_pair[(link.from_event_id, link.to_event_id)] = link

    largest_component = max((len(c) for c in components), default=0)
    stats = WindowStats(
        n_events=len(window_events),
        n_arcs_before=n_arcs_before,
        n_arcs_after=len(kept_arcs),
        n_components=len(components),
        largest_component=largest_component,
    )
    return event_paths, links_by_pair, stats


@dataclass
class WindowedSolveResult:
    trajectories: list[Trajectory]
    n_windows: int


def solve_windowed(
    events: list[DetectionEvent],
    gate: Gate,
    fusion_model: FusionModel,
    entry_exit: EntryExitCosts,
    city: CityConfig,
    window_size_s: float = DEFAULT_WINDOW_SIZE_S,
    step_s: float = DEFAULT_WINDOW_STEP_S,
    on_window: Callable[[int, int, WindowStats, float], None] | None = None,
    link_bias: float = 0.0,
) -> WindowedSolveResult:
    """Solve the full event stream window-by-window, stitching open tracks
    across window boundaries by carried event_id (see module docstring).

    `on_window`, when given, is called once per NON-EMPTY window with
    `(window_index, n_windows, stats, elapsed_s)` -- purely for progress
    reporting (eval.run_pipeline uses it); it never affects the result.

    `link_bias` (beta, in nats): forwarded unchanged to `_solve_one_window`
    -- see its docstring. Default 0.0 leaves production behaviour untouched;
    a nonzero value is only ever set from a value calibrated by
    `eval/calibrate_link_bias.py` on a TRAINING day, never on the evaluated
    dataset (data/run1)."""
    events_sorted = sorted(events, key=lambda e: e.timestamp)
    if not events_sorted:
        return WindowedSolveResult(trajectories=[], n_windows=0)

    events_by_id = {e.event_id: e for e in events_sorted}
    t_min = events_sorted[0].timestamp.timestamp()
    t_max = events_sorted[-1].timestamp.timestamp()
    windows = _window_bounds(t_min, t_max, window_size_s, step_s)
    overlap_s = max(window_size_s - step_s, 0.0)

    # global_trajectory_id -> ordered event_ids accumulated so far
    open_tracks: dict[str, list[str]] = {}
    open_track_links: dict[str, list[LinkEvidence]] = {}
    finished: list[tuple[list[str], list[LinkEvidence]]] = []
    next_id = 0

    carry_owner: dict[str, str] = {}  # event_id -> global trajectory id
    # Every event_id already placed into some trajectory (open or finished)
    # by a PREVIOUS window. Windows overlap in TIME, so an event can fall
    # inside more than one window's raw time range -- but it must only ever
    # be handed to the solver once. The sole exception is a carried seed
    # (the last event of a still-open track), which is DELIBERATELY
    # reprocessed so the next window can link new events onto it.
    consumed_event_ids: set[str] = set()

    for window_i, (window_start, window_end) in enumerate(windows):
        window_events = [
            e
            for e in events_sorted
            if window_start <= e.timestamp.timestamp() < window_end
            and (e.event_id not in consumed_event_ids or e.event_id in carry_owner)
        ]
        if not window_events:
            continue

        t_window_start = time.perf_counter()
        event_paths, links_by_pair, stats = _solve_one_window(
            window_events,
            set(carry_owner),
            gate,
            fusion_model,
            entry_exit,
            city,
            link_bias=link_bias,
        )
        if on_window is not None:
            on_window(window_i, len(windows), stats, time.perf_counter() - t_window_start)
        for path in event_paths:
            consumed_event_ids.update(path)

        new_carry_owner: dict[str, str] = {}
        for path in event_paths:
            first = path[0]
            path_links = [
                links_by_pair[(a, b)]
                for a, b in zip(path[:-1], path[1:], strict=True)
                if (a, b) in links_by_pair
            ]
            if first in carry_owner:
                gid = carry_owner[first]
                open_tracks[gid].extend(path[1:])
                open_track_links[gid].extend(path_links)
            else:
                gid = f"traj_{next_id:06d}"
                next_id += 1
                open_tracks[gid] = list(path)
                open_track_links[gid] = path_links

            last_ts = events_by_id[path[-1]].timestamp.timestamp()
            if overlap_s > 0 and last_ts >= window_end - overlap_s:
                # Might continue into the next window -- keep it open.
                new_carry_owner[path[-1]] = gid
            else:
                finished.append((open_tracks.pop(gid), open_track_links.pop(gid)))

        carry_owner = new_carry_owner

    for gid, evids in open_tracks.items():
        finished.append((evids, open_track_links[gid]))

    trajectories: list[Trajectory] = []
    for i, (event_ids, links) in enumerate(finished):
        path_events = [events_by_id[eid] for eid in event_ids]
        first = path_events[0]
        trajectories.append(
            Trajectory(
                trajectory_id=f"traj_{i:06d}",
                event_ids=event_ids,
                decoded_plate=first.plate_argmax,
                plate_confidence=first.plate_confidence,
                start_time=path_events[0].timestamp,
                end_time=path_events[-1].timestamp,
                camera_sequence=[e.camera_id for e in path_events],
                links=links,
                gt_vehicle_id=first.gt_vehicle_id,
            )
        )

    return WindowedSolveResult(trajectories=trajectories, n_windows=len(windows))


__all__ = [
    "CARRIED_ENTRY_COST_FACTOR",
    "DEFAULT_WINDOW_SIZE_S",
    "DEFAULT_WINDOW_STEP_S",
    "WindowStats",
    "WindowedSolveResult",
    "solve_windowed",
]
