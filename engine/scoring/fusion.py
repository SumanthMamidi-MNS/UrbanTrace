"""Fuse the three evidence channels into one LinkEvidence (architecture.md §3):

    total_log_odds = prior_log_odds + plate_lr + appearance_lr + kinematic_lr

`prior_log_odds` is derived from the number of spatio-temporally plausible
candidates the gate actually handed a given successor event: with N
equally-plausible candidates and (at most) one true link among them, the
prior odds that any *specific* candidate is the true one is roughly
1/(N-1) absent any other evidence, so `prior_log_odds = -log(max(N-1, 1))`.

THE PER-SUCCESSOR FIX (post-Day-3a): N must be THAT SUCCESSOR's own gated
candidate count, not a fixed constant. A constant `DEFAULT_EXPECTED_CANDIDATES
= 5.0` applied to every link regardless of how many candidates the gate
actually produced (architecture.md §7's gate hands ~218 candidates/event at
full-city scale) gives every link an unearned `-log(4) = -1.39` prior
instead of the correct `-log(217) ~= -5.4` -- roughly 4 nats of free
evidence on every single link, independent of how genuinely ambiguous that
link was. `eval/reports/error_analysis.json` traced 93.2% of full-city
UrbanTrace's wrong links to this plus the entry/exit arc-cost bug fixed in
`engine.association.mincostflow`/`engine.association.window`.

`engine.association.window._solve_one_window` now computes each successor's
own `n_j = len(gate_result.candidates[succ_id])` (the gate's own candidate
list length, counted BEFORE dominance pruning -- pruning happens
downstream of scoring and must not shrink the pool that defines the
prior) and threads it through `score_pairs_batch_totals`/
`score_pairs_batch` as `n_candidates_per_pair`. The scalar `score_pair`
(used by the API's WHY panel, which has no batch/gate context of its own)
takes an optional `n_candidates` argument for the same purpose, falling
back to `ctx.expected_candidates_per_window` -- now a fallback constant
for contexts with no real gate count available, not the production value.

Note on the `physically_impossible` flag Day 3's clone detector needs
(architecture.md §3, "Clone detection falls out"): `LinkEvidence` is a
frozen Day 1 contract with no such field, and this pairing scores every
candidate the same way regardless of what downstream does with it. The flag
is not lost, though — by construction `kinematic_lr == -inf` if and only if
the kinematic hard gate fired (see engine/scoring/kinematic_lr.py), so a
consumer recovers "physically impossible" losslessly from
`link.kinematic_lr == float("-inf")` without a schema change.
"""

import math
from dataclasses import dataclass

import numpy as np

from engine.contracts.events import DetectionEvent
from engine.contracts.trajectory import LinkEvidence
from engine.scoring.appearance_lr import (
    AppearanceModel,
    precompute_appearance_batch_arrays,
    score_appearance_pair,
    score_appearance_pairs_batch,
)
from engine.scoring.kinematic_lr import (
    KinematicModel,
    precompute_event_kinematic_arrays,
)
from engine.scoring.plate_lr import (
    PlatePriors,
    precompute_plate_batch_arrays,
    score_plate_pair,
    score_plate_pairs_batch,
)

DEFAULT_EXPECTED_CANDIDATES = 5.0

# Names accepted by the `channels` mask below (score_pairs_batch/
# score_pairs_batch_totals). Not enforced anywhere production-critical --
# only eval/ablation.py constructs a non-None mask.
CHANNEL_NAMES = frozenset({"plate", "appearance", "kinematic"})


@dataclass
class FusionModel:
    plate_priors: PlatePriors
    kinematic_model: KinematicModel
    appearance_model: AppearanceModel
    expected_candidates_per_window: float = DEFAULT_EXPECTED_CANDIDATES
    # Ablation-only channel mask (eval/ablation.py, architecture.md §8's
    # "plate-only -> +kinematics -> +appearance" ablation row). `None`
    # (the default, used everywhere in production) is current behaviour,
    # bit-for-bit: every channel contributes to `total_log_odds` exactly as
    # before this field existed. A non-None set of `CHANNEL_NAMES` restricts
    # `total_log_odds` (in score_pairs_batch/score_pairs_batch_totals only —
    # see their own `channels` parameter) to the SUM of just the named
    # channels' log-LRs plus the prior; a masked-out channel's own log-LR is
    # still computed and reported on `LinkEvidence` (so a WHY-panel-style
    # breakdown stays honest about what was actually measured), it just
    # contributes 0 to the fused total that ranking/min-cost-flow actually
    # sees. Because `engine.association.window.solve_windowed` forwards this
    # same `FusionModel` instance untouched into both batch-scoring calls,
    # setting this field is enough to ablate the ENTIRE production pipeline
    # (gate -> fuse -> windowed min-cost flow) without touching window.py.
    channels: frozenset[str] | None = None


def prior_log_odds(expected_candidates_per_window: float) -> float:
    n_competitors = max(expected_candidates_per_window - 1.0, 1.0)
    return -math.log(n_competitors)


def _prior_log_odds_array(n_candidates: np.ndarray) -> np.ndarray:
    """Vectorised sibling of `prior_log_odds`, applied elementwise to a
    per-pair array of gated-candidate counts (one count per pair's
    SUCCESSOR event, repeated across every predecessor that shares it)."""
    n_competitors = np.maximum(n_candidates - 1.0, 1.0)
    return -np.log(n_competitors)


def score_pair(
    event_a: DetectionEvent,
    event_b: DetectionEvent,
    ctx: FusionModel,
    n_candidates: float | None = None,
) -> LinkEvidence:
    """Score one candidate (event_a -> event_b) pair, populating every field
    of the LinkEvidence contract. Caller is responsible for only calling
    this on spatio-temporally plausible pairs (Day 3's gating) — this
    function itself applies no gate beyond the kinematic channel's own hard
    physical-impossibility check.

    `n_candidates`, when given, is `event_b`'s own gated-candidate pool size
    (module docstring's "THE PER-SUCCESSOR FIX") and drives the prior via
    `prior_log_odds(n_candidates)`. Omitted (the default -- e.g. the API's
    WHY panel, which re-scores one specific pair without re-running the
    gate), the prior falls back to the constant `ctx.expected_candidates_per_window`,
    exactly as before this fix."""
    plate_result = score_plate_pair(
        event_a.plate_posterior, event_b.plate_posterior, ctx.plate_priors
    )
    appearance_result = score_appearance_pair(event_a, event_b, ctx.appearance_model)
    kinematic_result = ctx.kinematic_model.score(
        event_a.camera_id, event_a.timestamp, event_b.camera_id, event_b.timestamp
    )

    prior = prior_log_odds(
        n_candidates if n_candidates is not None else ctx.expected_candidates_per_window
    )
    total = prior + plate_result.log_lr + appearance_result.log_lr + kinematic_result.log_lr

    return LinkEvidence(
        from_event_id=event_a.event_id,
        to_event_id=event_b.event_id,
        plate_lr=plate_result.log_lr,
        appearance_lr=appearance_result.log_lr,
        kinematic_lr=kinematic_result.log_lr,
        prior_log_odds=prior,
        total_log_odds=total,
        delta_t_s=kinematic_result.delta_t_s,
        expected_t_s=kinematic_result.expected_t_s,
        skipped_cameras=kinematic_result.skipped_cameras,
    )


@dataclass
class _BatchScoreArrays:
    plate_lr: np.ndarray
    appearance_lr: np.ndarray
    kin: object  # engine.scoring.kinematic_lr.KinematicBatchResult
    prior: np.ndarray  # one prior per pair -- see `_resolve_prior_array`
    total: np.ndarray


def _resolve_channels(
    ctx: FusionModel, channels: frozenset[str] | None
) -> frozenset[str] | None:
    """The explicit `channels` argument (when given) overrides `ctx.channels`
    for this one call; otherwise `ctx.channels` (default `None`, i.e. every
    channel) applies. Either way, `None` means "current behaviour,
    bit-for-bit" -- no masking."""
    return channels if channels is not None else ctx.channels


def _resolve_prior_array(
    n_pairs: int, ctx: FusionModel, n_candidates_per_pair: np.ndarray | None
) -> np.ndarray:
    """Per-pair prior array (module docstring's "THE PER-SUCCESSOR FIX").
    `n_candidates_per_pair`, when given, is one gated-candidate COUNT per
    pair -- that pair's SUCCESSOR's own `len(gate_result.candidates[...])`,
    counted before dominance pruning (`engine.association.window`) -- fed
    through `_prior_log_odds_array`. `None` (the default) reproduces the
    pre-fix constant prior exactly, broadcast to every pair: callers that
    don't have a real gate candidate count (ablation/legacy call sites,
    tests scoring synthetic pairs) keep getting `ctx.expected_candidates_per_window`."""
    if n_candidates_per_pair is None:
        return np.full(n_pairs, prior_log_odds(ctx.expected_candidates_per_window))
    n_arr = np.asarray(n_candidates_per_pair, dtype=np.float64)
    return _prior_log_odds_array(n_arr)


def _score_pairs_batch_arrays(
    events: list[DetectionEvent],
    pred_idx: np.ndarray,
    succ_idx: np.ndarray,
    ctx: FusionModel,
    channels: frozenset[str] | None = None,
    n_candidates_per_pair: np.ndarray | None = None,
) -> _BatchScoreArrays:
    """Shared numeric core of `score_pairs_batch`/`score_pairs_batch_totals`:
    precomputes each channel's per-EVENT arrays once (O(n_events)) and
    returns every channel's per-pair array plus the fused total, with no
    `LinkEvidence` construction at all.

    `channels` (falling back to `ctx.channels`, see `FusionModel`) is an
    ablation-only mask: when it is not `None`, a channel whose name is
    absent from it contributes 0 to `total` -- its own per-pair array
    (`plate_lr`/`appearance_lr`/`kin.log_lr`) is still computed and returned
    unchanged, so callers keep the real per-channel numbers even while the
    fused total ignores them. `channels=None` (the default) reproduces the
    pre-ablation total exactly: `prior + plate_lr + appearance_lr + kin.log_lr`.

    `n_candidates_per_pair`: see `_resolve_prior_array`."""
    resolved = _resolve_channels(ctx, channels)

    plate_arrays = precompute_plate_batch_arrays(
        [e.plate_posterior for e in events], ctx.plate_priors
    )
    plate_lr, _ = score_plate_pairs_batch(plate_arrays, pred_idx, succ_idx)

    appearance_arrays = precompute_appearance_batch_arrays(events, ctx.appearance_model)
    appearance_lr = score_appearance_pairs_batch(
        appearance_arrays, pred_idx, succ_idx, ctx.appearance_model
    )

    cam_idx, epoch_s, tod_idx = precompute_event_kinematic_arrays(events, ctx.kinematic_model)
    kin = ctx.kinematic_model.score_batch(cam_idx, epoch_s, tod_idx, pred_idx, succ_idx)

    prior = _resolve_prior_array(len(pred_idx), ctx, n_candidates_per_pair)
    if resolved is None:
        total = prior + plate_lr + appearance_lr + kin.log_lr
    else:
        plate_term = plate_lr if "plate" in resolved else np.zeros_like(plate_lr)
        appearance_term = (
            appearance_lr if "appearance" in resolved else np.zeros_like(appearance_lr)
        )
        kinematic_term = kin.log_lr if "kinematic" in resolved else np.zeros_like(kin.log_lr)
        total = prior + plate_term + appearance_term + kinematic_term
    return _BatchScoreArrays(
        plate_lr=plate_lr, appearance_lr=appearance_lr, kin=kin, prior=prior, total=total
    )


def score_pairs_batch_totals(
    events: list[DetectionEvent],
    pred_idx: np.ndarray,
    succ_idx: np.ndarray,
    ctx: FusionModel,
    channels: frozenset[str] | None = None,
    n_candidates_per_pair: np.ndarray | None = None,
) -> np.ndarray:
    """Numeric-only sibling of `score_pairs_batch`: just the fused
    `total_log_odds` per pair, with no `LinkEvidence` construction at all.
    Meant for graph-building at candidate-pool scale (hundreds of
    candidates/event), where most arcs get pruned as dominated
    (engine.association.mincostflow.prune_dominated_arcs) and never need
    their full per-channel breakdown -- only the SURVIVING, and eventually
    the actually-USED, pairs are worth the cost of a full `LinkEvidence`
    (engine.association.window builds those separately, only for pairs that
    end up on a solved path).

    `channels`: see `_score_pairs_batch_arrays`/`FusionModel.channels`.
    `None` (default) falls back to `ctx.channels` and reproduces the
    pre-ablation total exactly.

    `n_candidates_per_pair`: see `_resolve_prior_array`. `None` (default)
    reproduces the pre-fix constant-prior total exactly."""
    pred_idx = np.asarray(pred_idx, dtype=np.int64)
    succ_idx = np.asarray(succ_idx, dtype=np.int64)
    if len(pred_idx) == 0:
        return np.zeros(0)
    return _score_pairs_batch_arrays(
        events, pred_idx, succ_idx, ctx, channels, n_candidates_per_pair
    ).total


def score_pairs_batch(
    events: list[DetectionEvent],
    pred_idx: np.ndarray,
    succ_idx: np.ndarray,
    ctx: FusionModel,
    channels: frozenset[str] | None = None,
    n_candidates_per_pair: np.ndarray | None = None,
) -> list[LinkEvidence]:
    """Batched sibling of `score_pair`: scores every (pred, succ) pair in one
    shot by precomputing each channel's per-EVENT arrays once (O(n_events))
    instead of redoing per-event work for every pair (see
    engine/scoring/{plate,appearance,kinematic}_lr.py module docstrings for
    each channel's own precomputation). `pred_idx[k]`/`succ_idx[k]` are
    positions into `events`. Agrees with repeated `score_pair` calls to
    within 1e-6 per channel and in total (tests/test_batch_scoring.py) --
    including when a real per-successor `n_candidates_per_pair` is passed
    here and the matching `n_candidates` is passed to each `score_pair`
    call, per pair (module docstring's "THE PER-SUCCESSOR FIX").

    Uses `LinkEvidence.model_construct` rather than the normal constructor:
    every field below is already a plain, correctly-typed Python float/list
    computed by this module, so re-running pydantic's validation on every
    one of what can be millions of pairs in a full run buys nothing but
    wall-clock time. Callers scoring a large CANDIDATE pool (rather than a
    small, already-decided set of pairs) should prefer
    `score_pairs_batch_totals` and only call this on the pairs that survive
    pruning/solving -- see engine.association.window.

    `n_candidates_per_pair`: see `_resolve_prior_array`. Each returned
    `LinkEvidence.prior_log_odds` reports the ACTUAL per-link prior used
    (one value per pair when this is given, not one constant for every
    link)."""
    pred_idx = np.asarray(pred_idx, dtype=np.int64)
    succ_idx = np.asarray(succ_idx, dtype=np.int64)
    n_pairs = len(pred_idx)
    if n_pairs == 0:
        return []

    arrays = _score_pairs_batch_arrays(
        events, pred_idx, succ_idx, ctx, channels, n_candidates_per_pair
    )
    plate_lr, appearance_lr, kin, prior, total = (
        arrays.plate_lr,
        arrays.appearance_lr,
        arrays.kin,
        arrays.prior,
        arrays.total,
    )

    event_ids = [e.event_id for e in events]
    links: list[LinkEvidence] = []
    for k in range(n_pairs):
        links.append(
            LinkEvidence.model_construct(
                from_event_id=event_ids[pred_idx[k]],
                to_event_id=event_ids[succ_idx[k]],
                plate_lr=float(plate_lr[k]),
                appearance_lr=float(appearance_lr[k]),
                kinematic_lr=float(kin.log_lr[k]),
                prior_log_odds=float(prior[k]),
                total_log_odds=float(total[k]),
                delta_t_s=float(kin.delta_t_s[k]),
                expected_t_s=float(kin.expected_t_s[k]),
                skipped_cameras=kin.skipped_cameras[k],
            )
        )
    return links
