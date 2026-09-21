"""Shared evaluation metrics for the simulator's calibration guard, the
appearance-scaling report (docs/decisions.md, "Day 1b"), and the Day 2
scoring-channel evaluation (per-channel / fused AUC, ECE). Day 3-4 will add
IDF1 / ID-switch / trajectory metrics here per architecture.md section 8.
"""

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import Trajectory

QUERY_CAP_DEFAULT = 3000


def whole_plate_accuracy(
    events: list[DetectionEvent], true_plate_by_vehicle: dict[str, str]
) -> float:
    """Fraction of events whose decoded (argmax) plate exactly matches the
    vehicle's true 10-slot plate. This is "per-read OCR accuracy" as the
    published ANPR figures (85-95%) define it."""
    if not events:
        return float("nan")
    correct = sum(
        1 for e in events if e.plate_argmax == true_plate_by_vehicle[e.gt_vehicle_id]
    )
    return correct / len(events)


def appearance_rank1(
    events: list[DetectionEvent], query_cap: int = QUERY_CAP_DEFAULT, seed: int = 0
) -> tuple[float, int, int]:
    """Appearance-only rank-1 retrieval accuracy: for each query event whose
    vehicle has >=2 events in this gallery (a query needs a possible correct
    match to exist at all — standard Re-ID practice), find its nearest
    neighbour by cosine similarity among ALL OTHER events in the full
    gallery (excluding itself) and check whether it belongs to the same
    vehicle. The query side is capped at `query_cap` for compute; the
    gallery is always the full event set, so gallery-size crowding effects
    are measured honestly.

    Returns (rank1_accuracy, n_events, n_eligible_queries).
    """
    if not events:
        return float("nan"), 0, 0

    embs = np.array([e.embedding for e in events], dtype=np.float32)
    gt_ids = np.array([e.gt_vehicle_id for e in events])

    counts = Counter(gt_ids.tolist())
    eligible_idx = np.array([i for i, g in enumerate(gt_ids) if counts[g] >= 2])
    if len(eligible_idx) == 0:
        return float("nan"), len(events), 0

    rng = np.random.default_rng(seed)
    if len(eligible_idx) > query_cap:
        query_idx = np.sort(rng.choice(eligible_idx, size=query_cap, replace=False))
    else:
        query_idx = eligible_idx

    correct = 0
    batch = 1000
    for start in range(0, len(query_idx), batch):
        chunk = query_idx[start : start + batch]
        sims = embs[chunk] @ embs.T
        for row_i, global_i in enumerate(chunk):
            sims[row_i, global_i] = -np.inf
        nn = np.argmax(sims, axis=1)
        correct += int(np.sum(gt_ids[nn] == gt_ids[chunk]))

    rank1 = correct / len(query_idx)
    return rank1, len(events), len(query_idx)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-based Mann-Whitney U statistic
    (tie-aware, no sklearn dependency). `labels` are 0/1. -inf/+inf scores
    (e.g. a plate channel's alignment search, or a kinematic hard gate) are
    mapped to finite extremes before ranking so they still compare
    correctly relative to every other score."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    finite = scores[np.isfinite(scores)]
    lo, hi = (finite.min() - 1.0, finite.max() + 1.0) if finite.size else (-1.0, 1.0)
    scores = np.where(np.isneginf(scores), lo, scores)
    scores = np.where(np.isposinf(scores), hi, scores)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    rank = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        rank += j - i + 1
        i = j + 1

    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE: SUM_bin (n_bin/n) * |mean(predicted) - mean(actual)|."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return float(ece)


def sample_labeled_pairs(
    events: list[DetectionEvent], seed: int = 0, max_positive_pairs: int = 4000
) -> list[tuple[DetectionEvent, DetectionEvent, int]]:
    """Build labelled (event_a, event_b, label) candidate pairs for scoring-
    channel evaluation. Positives are real consecutive camera-to-camera
    passages of the same ground-truth vehicle. Negatives are *hard*: for
    each positive's destination camera, the closest-in-time event from a
    DIFFERENT vehicle at that same camera — i.e. a different vehicle that
    plausibly could have been mistaken for the true one, not an arbitrary
    random pair a trivial time-gap heuristic would separate for free. This
    stands in for what Day 3's spatio-temporal gating would hand the scorer
    (gating.py doesn't exist yet in Day 2)."""
    rng = random.Random(seed)

    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in events:
        if e.gt_vehicle_id is not None:
            by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    positives: list[tuple[DetectionEvent, DetectionEvent]] = []
    for evs in by_vehicle.values():
        evs_sorted = sorted(evs, key=lambda e: e.timestamp)
        for a, b in zip(evs_sorted[:-1], evs_sorted[1:], strict=True):
            positives.append((a, b))
    rng.shuffle(positives)
    positives = positives[:max_positive_pairs]

    events_by_camera: dict[str, list[DetectionEvent]] = {}
    for e in events:
        events_by_camera.setdefault(e.camera_id, []).append(e)

    labeled: list[tuple[DetectionEvent, DetectionEvent, int]] = []
    for a, b in positives:
        labeled.append((a, b, 1))
        candidates = [
            e for e in events_by_camera[b.camera_id] if e.gt_vehicle_id != a.gt_vehicle_id
        ]
        if candidates:
            neg_b = min(candidates, key=lambda e: abs((e.timestamp - b.timestamp).total_seconds()))
            labeled.append((a, neg_b, 0))

    rng.shuffle(labeled)
    return labeled


# ---------------------------------------------------------------------------
# Trajectory-level metrics (docs/decisions.md "Day 3a" Part 5).
#
# Pairwise channel AUC saturated in Day 2 (1.0000 on three of four strata)
# and is now a weak, misleading proxy for system quality -- it says nothing
# about whether the GLOBAL association (gating -> fusion -> min-cost flow)
# actually reconstructs correct end-to-end vehicle trajectories. IDF1 (and
# its components) is the real metric from here on: identity-level precision
# and recall over whole trajectories, computed via Hungarian (optimal
# bipartite) matching between predicted trajectories and ground-truth
# vehicles, exactly as architecture.md §8 names it.
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryMetrics:
    idf1: float
    id_precision: float
    id_recall: float
    id_switches: int
    fragmentation: int
    trajectory_completeness: float
    n_predicted_trajectories: int
    n_gt_vehicles: int
    n_gt_events: int
    n_predicted_events: int
    idtp: float
    idfp: float
    idfn: float


def compute_trajectory_metrics(
    predicted_trajectories: list[Trajectory], events: list[DetectionEvent]
) -> TrajectoryMetrics:
    """IDF1 / ID-precision / ID-recall / ID-switches / fragmentation /
    trajectory completeness, computed by OPTIMAL (Hungarian) matching
    between predicted trajectories and ground-truth vehicles.

    Ground truth is `events` grouped by `gt_vehicle_id`. `IDTP(G, P)` for a
    (ground-truth, predicted) pair is the number of events they share; the
    Hungarian algorithm finds the one-to-one matching that MAXIMISES total
    shared events (`scipy.optimize.linear_sum_assignment` on the negated
    overlap matrix -- rectangular inputs are supported natively, so
    predicted and ground-truth counts need not match).

        IDFN = total ground-truth events NOT captured by their matched
               predicted trajectory
        IDFP = total predicted events NOT actually belonging to their
               matched ground-truth vehicle
        IDPrecision = IDTP / (IDTP + IDFP)
        IDRecall    = IDTP / (IDTP + IDFN)
        IDF1        = 2*IDTP / (2*IDTP + IDFP + IDFN)

    ID-switches: summed, per ground-truth vehicle, over its own events in
    time order, the number of times the assigned predicted-trajectory id
    changes from one event to the next.

    Fragmentation: summed, per ground-truth vehicle, `max(distinct
    predicted trajectories its events landed in, 1) - 1` -- zero if a
    vehicle's whole journey lands in one predicted trajectory.

    Trajectory completeness: averaged, per ground-truth vehicle, the best
    single predicted trajectory's coverage fraction of that vehicle's
    events (a "mostly captured by one trajectory" score, distinct from
    IDF1's globally-optimal matching)."""
    gt_groups: dict[str, list[str]] = {}
    for e in events:
        if e.gt_vehicle_id is not None:
            gt_groups.setdefault(e.gt_vehicle_id, []).append(e.event_id)
    events_by_id = {e.event_id: e for e in events}
    for gid in gt_groups:
        gt_groups[gid].sort(key=lambda eid: events_by_id[eid].timestamp)

    pred_groups: dict[str, list[str]] = {
        t.trajectory_id: list(t.event_ids) for t in predicted_trajectories
    }

    gt_ids = list(gt_groups.keys())
    pred_ids = list(pred_groups.keys())
    gt_sets = {g: set(evs) for g, evs in gt_groups.items()}
    pred_sets = {p: set(evs) for p, evs in pred_groups.items()}

    n_gt, n_pred = len(gt_ids), len(pred_ids)
    overlap = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt_ids):
        g_set = gt_sets[g]
        for j, p in enumerate(pred_ids):
            overlap[i, j] = len(g_set & pred_sets[p])

    idtp = 0.0
    if n_gt and n_pred:
        row_ind, col_ind = linear_sum_assignment(-overlap)
        for r, c in zip(row_ind, col_ind, strict=True):
            idtp += overlap[r, c]

    n_gt_events = sum(len(s) for s in gt_sets.values())
    n_pred_events = sum(len(s) for s in pred_sets.values())
    idfn = n_gt_events - idtp
    idfp = n_pred_events - idtp

    id_precision = idtp / (idtp + idfp) if (idtp + idfp) > 0 else float("nan")
    id_recall = idtp / (idtp + idfn) if (idtp + idfn) > 0 else float("nan")
    denom = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / denom if denom > 0 else float("nan")

    event_to_pred: dict[str, str] = {}
    for p, evs in pred_groups.items():
        for eid in evs:
            event_to_pred[eid] = p

    id_switches = 0
    fragmentation = 0
    completeness_scores: list[float] = []
    for g in gt_ids:
        evs_sorted = gt_groups[g]
        pred_seq = [event_to_pred.get(eid) for eid in evs_sorted]
        id_switches += sum(1 for a, b in zip(pred_seq[:-1], pred_seq[1:], strict=True) if a != b)
        distinct_preds = {p for p in pred_seq if p is not None}
        fragmentation += max(len(distinct_preds) - 1, 0)
        if evs_sorted:
            counts = Counter(p for p in pred_seq if p is not None)
            best = max(counts.values()) if counts else 0
            completeness_scores.append(best / len(evs_sorted))

    trajectory_completeness = (
        sum(completeness_scores) / len(completeness_scores) if completeness_scores else float("nan")
    )

    return TrajectoryMetrics(
        idf1=idf1,
        id_precision=id_precision,
        id_recall=id_recall,
        id_switches=id_switches,
        fragmentation=fragmentation,
        trajectory_completeness=trajectory_completeness,
        n_predicted_trajectories=n_pred,
        n_gt_vehicles=n_gt,
        n_gt_events=n_gt_events,
        n_predicted_events=n_pred_events,
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
    )
