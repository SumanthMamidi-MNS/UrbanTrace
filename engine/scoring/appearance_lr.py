"""Appearance likelihood ratio (architecture.md §3, "Appearance term").

Two pieces, summed:

1. A density-ratio LR on cosine distance between the two events' Re-ID
   embeddings: `p1(d)` (same-vehicle distance distribution) vs `p0(d)`
   (different-vehicle distribution), both fit as smoothed histograms on
   labelled training pairs — not a hand-tuned threshold.
2. A discrete colour/type term using confusion matrices estimated on
   training data, marginalised over the unknown true colour/type exactly
   like the plate channel marginalises over the unknown true plate (same
   noisy-channel LR shape, just over a 8/5-way categorical instead of a
   26-way one).

The total is clamped to [-CLAMP, CLAMP] so this channel can never single-
handedly dominate the fusion (architecture.md is explicit that all three
channels stay independent evidence, not a veto).
"""

import math
from dataclasses import dataclass

import numpy as np

from engine.contracts.events import DetectionEvent

FLOOR = 1e-9
CLAMP = 20.0
N_BINS = 40
# Cosine distance of two unit vectors lies in [0, 2].
DIST_MIN, DIST_MAX = 0.0, 2.0
HIST_SMOOTHING = 1.0
CATEGORICAL_SMOOTHING = 1.0


def cosine_distance(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    cos_sim = float(np.dot(av, bv) / denom) if denom > 0 else 0.0
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


@dataclass
class AppearanceModel:
    bin_edges: np.ndarray  # (N_BINS + 1,)
    log_p1: np.ndarray  # (N_BINS,) log P(distance falls in bin | same vehicle)
    log_p0: np.ndarray  # (N_BINS,) log P(distance falls in bin | different vehicle)
    colors: list[str]
    types: list[str]
    color_prior: dict[str, float]
    color_confusion: dict[str, dict[str, float]]  # confusion[true][observed]
    type_prior: dict[str, float]
    type_confusion: dict[str, dict[str, float]]

    def _bin_index(self, d: float) -> int:
        idx = int(np.searchsorted(self.bin_edges, d, side="right") - 1)
        return min(max(idx, 0), len(self.log_p1) - 1)

    def distance_log_lr(self, d: float) -> float:
        idx = self._bin_index(d)
        return float(self.log_p1[idx] - self.log_p0[idx])

    def color_log_lr(self, color_a: str, color_b: str) -> float:
        return _categorical_log_lr(
            color_a, color_b, self.color_prior, self.color_confusion, self.colors
        )

    def type_log_lr(self, type_a: str, type_b: str) -> float:
        return _categorical_log_lr(
            type_a, type_b, self.type_prior, self.type_confusion, self.types
        )

    @classmethod
    def uniform_default(cls, colors: list[str], types: list[str]) -> "AppearanceModel":
        """A weak, unfit default: p1 mildly favours small distances over
        p0's flat prior. Useful for standalone testing; real use should
        always go through `fit_appearance_model` on a training split."""
        bin_edges = np.linspace(DIST_MIN, DIST_MAX, N_BINS + 1)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        p0 = np.full(N_BINS, 1.0 / N_BINS)
        p1_raw = np.exp(-3.0 * centers)
        p1 = p1_raw / p1_raw.sum()
        color_prior = {c: 1.0 / len(colors) for c in colors}
        color_confusion = {
            c: {c2: (0.8 if c2 == c else 0.2 / (len(colors) - 1)) for c2 in colors} for c in colors
        }
        type_prior = {t: 1.0 / len(types) for t in types}
        type_confusion = {
            t: {t2: (0.8 if t2 == t else 0.2 / (len(types) - 1)) for t2 in types} for t in types
        }
        return cls(
            bin_edges=bin_edges,
            log_p1=np.log(p1),
            log_p0=np.log(p0),
            colors=colors,
            types=types,
            color_prior=color_prior,
            color_confusion=color_confusion,
            type_prior=type_prior,
            type_confusion=type_confusion,
        )


def _categorical_log_lr(
    obs_a: str,
    obs_b: str,
    prior: dict[str, float],
    confusion: dict[str, dict[str, float]],
    values: list[str],
) -> float:
    """Marginalise over the unknown true category, exactly like the plate
    channel marginalises over the unknown true plate character:
    num = SUM_true pi(true) P(a|true) P(b|true); den = the two marginals'
    product."""
    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for true_v in values:
        p_true = max(prior.get(true_v, 0.0), FLOOR)
        p_a = max(confusion.get(true_v, {}).get(obs_a, 0.0), FLOOR)
        p_b = max(confusion.get(true_v, {}).get(obs_b, 0.0), FLOOR)
        num += p_true * p_a * p_b
        den_a += p_true * p_a
        den_b += p_true * p_b
    den = den_a * den_b
    return math.log(max(num, FLOOR)) - math.log(max(den, FLOOR))


@dataclass
class AppearanceLRResult:
    log_lr: float
    distance: float
    distance_lr: float
    color_lr: float
    type_lr: float


def score_appearance_pair(
    event_a: DetectionEvent, event_b: DetectionEvent, model: AppearanceModel
) -> AppearanceLRResult:
    d = cosine_distance(event_a.embedding, event_b.embedding)
    distance_lr = model.distance_log_lr(d)
    color_lr = model.color_log_lr(event_a.attributes.color, event_b.attributes.color)
    type_lr = model.type_log_lr(event_a.attributes.vehicle_type, event_b.attributes.vehicle_type)
    total = distance_lr + color_lr + type_lr
    total = max(-CLAMP, min(CLAMP, total))
    return AppearanceLRResult(
        log_lr=total, distance=d, distance_lr=distance_lr, color_lr=color_lr, type_lr=type_lr
    )


def _fit_categorical(
    events: list[DetectionEvent],
    true_by_vehicle: dict[str, str],
    getter,
    values: list[str],
    smoothing: float = CATEGORICAL_SMOOTHING,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    true_counts = dict.fromkeys(values, 0)
    confusion_counts = {v: dict.fromkeys(values, 0) for v in values}

    for e in events:
        true_v = true_by_vehicle.get(e.gt_vehicle_id) if e.gt_vehicle_id else None
        if true_v is None or true_v not in true_counts:
            continue
        obs_v = getter(e)
        true_counts[true_v] += 1
        if obs_v in confusion_counts[true_v]:
            confusion_counts[true_v][obs_v] += 1

    total = sum(true_counts.values())
    denom = total + smoothing * len(values)
    prior = {v: (true_counts[v] + smoothing) / denom for v in values}

    confusion: dict[str, dict[str, float]] = {}
    for v in values:
        row = confusion_counts[v]
        row_total = sum(row.values())
        row_denom = row_total + smoothing * len(values)
        confusion[v] = {w: (row[w] + smoothing) / row_denom for w in values}

    return prior, confusion


# ---------------------------------------------------------------------------
# Batched scoring (Day 4 performance pass).
#
# Per-pair work in `score_appearance_pair` is dominated by re-deriving norms
# from scratch and re-walking the categorical marginalisation sum for every
# pair. Both are actually per-EVENT (embeddings/attributes) or per-MODEL
# (the categorical LR only ever depends on the two OBSERVED labels, never on
# which events they came from) quantities:
#   - embeddings/norms: stack once, O(n_events); distances become a gather
#     + row-wise dot product per pair.
#   - colour/type LR: `_categorical_log_lr(obs_a, obs_b, ...)` is a pure
#     function of the two label strings -- there are only
#     len(colors)^2 / len(types)^2 distinct outcomes total, so the whole
#     small table is precomputed ONCE (cached on the model) and a pair's
#     contribution becomes a single 2-D lookup.
# ---------------------------------------------------------------------------


@dataclass
class AppearanceBatchArrays:
    embeddings: np.ndarray  # (N, D) float64
    norms: np.ndarray  # (N,)
    color_idx: np.ndarray  # (N,) int, index into model.colors
    type_idx: np.ndarray  # (N,) int, index into model.types


def _categorical_lr_matrix(
    prior: dict[str, float], confusion: dict[str, dict[str, float]], values: list[str]
) -> np.ndarray:
    n = len(values)
    matrix = np.empty((n, n))
    for i, va in enumerate(values):
        for j, vb in enumerate(values):
            matrix[i, j] = _categorical_log_lr(va, vb, prior, confusion, values)
    return matrix


def _model_lr_matrices(model: AppearanceModel) -> tuple[np.ndarray, np.ndarray]:
    """Cached (n_colors, n_colors) / (n_types, n_types) log-LR lookup tables,
    computed once per model instance and reused across every batch call."""
    cached = getattr(model, "_batch_lr_matrices", None)
    if cached is not None:
        return cached
    color_matrix = _categorical_lr_matrix(model.color_prior, model.color_confusion, model.colors)
    type_matrix = _categorical_lr_matrix(model.type_prior, model.type_confusion, model.types)
    model._batch_lr_matrices = (color_matrix, type_matrix)
    return color_matrix, type_matrix


def precompute_appearance_batch_arrays(
    events: list[DetectionEvent], model: AppearanceModel
) -> AppearanceBatchArrays:
    n = len(events)
    embeddings = np.array([e.embedding for e in events], dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1)
    color_pos = {c: i for i, c in enumerate(model.colors)}
    type_pos = {t: i for i, t in enumerate(model.types)}
    color_idx = np.fromiter(
        (color_pos.get(e.attributes.color, 0) for e in events), dtype=np.int64, count=n
    )
    type_idx = np.fromiter(
        (type_pos.get(e.attributes.vehicle_type, 0) for e in events), dtype=np.int64, count=n
    )
    return AppearanceBatchArrays(
        embeddings=embeddings, norms=norms, color_idx=color_idx, type_idx=type_idx
    )


# Chunk size for batched pair scoring, in PAIRS -- same rationale as
# `engine.scoring.plate_lr.PAIR_CHUNK_SIZE`. Here the per-pair gather is
# `arrays.embeddings[pred_idx]`/`[succ_idx]`, each (n_pairs, EMBEDDING_DIM)
# float64 (EMBEDDING_DIM=128): at full-city pair counts (millions) that is
# multiple GiB per gather, on top of whatever the plate channel is holding
# at the same time in `engine.scoring.fusion`. Scoring PAIR_CHUNK_SIZE pairs
# at a time bounds each chunk's gather to a few hundred MB regardless of
# total pair count.
PAIR_CHUNK_SIZE = 200_000


def _score_appearance_pairs_chunk(
    arrays: AppearanceBatchArrays,
    pred_idx: np.ndarray,
    succ_idx: np.ndarray,
    model: AppearanceModel,
) -> np.ndarray:
    """Core per-chunk math, unchanged from the original (pre-chunking)
    `score_appearance_pairs_batch` body."""
    ea = arrays.embeddings[pred_idx]
    eb = arrays.embeddings[succ_idx]
    denom = arrays.norms[pred_idx] * arrays.norms[succ_idx]
    dot = np.einsum("pd,pd->p", ea, eb)
    cos_sim = np.where(denom > 0, dot / np.where(denom > 0, denom, 1.0), 0.0)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    distance = 1.0 - cos_sim

    bin_idx = np.searchsorted(model.bin_edges, distance, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(model.log_p1) - 1)
    distance_lr = model.log_p1[bin_idx] - model.log_p0[bin_idx]

    color_matrix, type_matrix = _model_lr_matrices(model)
    color_lr = color_matrix[arrays.color_idx[pred_idx], arrays.color_idx[succ_idx]]
    type_lr = type_matrix[arrays.type_idx[pred_idx], arrays.type_idx[succ_idx]]

    total = distance_lr + color_lr + type_lr
    return np.clip(total, -CLAMP, CLAMP)


def score_appearance_pairs_batch(
    arrays: AppearanceBatchArrays,
    pred_idx: np.ndarray,
    succ_idx: np.ndarray,
    model: AppearanceModel,
) -> np.ndarray:
    """Batched appearance LR. Agrees with `score_appearance_pair` to float
    precision (tests/test_batch_scoring.py).

    Processes pairs in chunks of `PAIR_CHUNK_SIZE` (see its docstring) so
    peak memory stays bounded regardless of how many pairs are scored in one
    call; results are numerically identical to scoring everything in one
    shot since each pair's math is fully independent of every other pair's."""
    pred_idx = np.asarray(pred_idx, dtype=np.int64)
    succ_idx = np.asarray(succ_idx, dtype=np.int64)
    p = len(pred_idx)
    if p == 0:
        return np.zeros(0)

    total = np.empty(p, dtype=np.float64)
    for start in range(0, p, PAIR_CHUNK_SIZE):
        end = min(start + PAIR_CHUNK_SIZE, p)
        total[start:end] = _score_appearance_pairs_chunk(
            arrays, pred_idx[start:end], succ_idx[start:end], model
        )
    return total


def fit_appearance_model(
    train_events: list[DetectionEvent],
    true_color_by_vehicle: dict[str, str],
    true_type_by_vehicle: dict[str, str],
    colors: list[str],
    types: list[str],
    max_diff_pairs: int = 200_000,
    seed: int = 0,
) -> AppearanceModel:
    """Fit p1(d)/p0(d) histograms and colour/type confusion matrices on a
    training split. `true_*_by_vehicle` come from the simulator's ground
    truth, standing in for whatever labelled data a real deployment would
    use to calibrate this channel."""
    by_vehicle: dict[str, list[DetectionEvent]] = {}
    for e in train_events:
        if e.gt_vehicle_id is not None:
            by_vehicle.setdefault(e.gt_vehicle_id, []).append(e)

    same_d: list[float] = []
    for evs in by_vehicle.values():
        if len(evs) < 2:
            continue
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                same_d.append(cosine_distance(evs[i].embedding, evs[j].embedding))

    rng = np.random.default_rng(seed)
    n_events = len(train_events)
    ids = [e.gt_vehicle_id for e in train_events]
    embs = np.array([e.embedding for e in train_events], dtype=np.float64)
    n_target = min(max_diff_pairs, max(len(same_d) * 5, 5000))
    diff_d: list[float] = []
    if n_events >= 2:
        tries = 0
        max_tries = n_target * 5
        while len(diff_d) < n_target and tries < max_tries:
            i, j = rng.integers(0, n_events, size=2)
            tries += 1
            if i == j or ids[i] == ids[j]:
                continue
            diff_d.append(cosine_distance(embs[i], embs[j]))

    bin_edges = np.linspace(DIST_MIN, DIST_MAX, N_BINS + 1)
    same_hist, _ = np.histogram(same_d, bins=bin_edges)
    diff_hist, _ = np.histogram(diff_d, bins=bin_edges)
    same_p = (same_hist + HIST_SMOOTHING) / (same_hist.sum() + HIST_SMOOTHING * N_BINS)
    diff_p = (diff_hist + HIST_SMOOTHING) / (diff_hist.sum() + HIST_SMOOTHING * N_BINS)

    color_prior, color_confusion = _fit_categorical(
        train_events, true_color_by_vehicle, lambda e: e.attributes.color, colors
    )
    type_prior, type_confusion = _fit_categorical(
        train_events, true_type_by_vehicle, lambda e: e.attributes.vehicle_type, types
    )

    return AppearanceModel(
        bin_edges=bin_edges,
        log_p1=np.log(same_p),
        log_p0=np.log(diff_p),
        colors=colors,
        types=types,
        color_prior=color_prior,
        color_confusion=color_confusion,
        type_prior=type_prior,
        type_confusion=type_confusion,
    )
