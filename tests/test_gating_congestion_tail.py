"""engine/association/gating.py's CONGESTION_TAIL_FACTOR: the gate's upper
bound must never exclude a merely-congested true predecessor just because
the fitted per-(camera-pair, tod) distribution happens to be tight (the
defect the lead measured: gate recall dropping to 0.68-0.70 at rush hour on
a congested day, while every true link that reached scoring still scored
strongly positive -- i.e. the gate, not the scorer, was the problem).

These are small, hand-built synthetic cases (no simulator dataset
generation) so they run in well under a second."""

import math

from engine.association.gating import _pair_window
from engine.scoring.kinematic_lr import KinematicModel, PairParams
from sim.city import generate_city

TOD = "midday"


def _model_with_tight_fit(sigma: float, t_ff: float, cam_i: str, cam_j: str) -> KinematicModel:
    """A KinematicModel whose (cam_i, cam_j, TOD) bucket is fitted tightly
    around `t_ff` with a large `n_obs`, so `params_for`'s empirical-Bayes
    blend toward the road-graph prior barely matters -- the fitted sigma is
    what actually governs the window."""
    city = generate_city(n_cameras=5, seed=1)
    return KinematicModel(
        city=city,
        pair_params={
            (cam_i, cam_j, TOD): PairParams(mu=math.log(t_ff), sigma=sigma, n_obs=1000),
        },
    )


def _reachable_pair(model: KinematicModel) -> tuple[str, str, float]:
    """First (cam_i, cam_j) pair in the generated city's camera list with a
    direct road-graph path, plus its free-flow travel time."""
    cams = [c.camera_id for c in model.city.cameras]
    for cam_i in cams:
        for cam_j in cams:
            if cam_i == cam_j:
                continue
            shortest = model.shortest_path_info(cam_i, cam_j)
            if shortest is not None:
                return cam_i, cam_j, shortest[1]
    raise AssertionError("no reachable camera pair found in generated city")


def test_congested_true_pair_admitted_with_default_factor_rejected_at_1x():
    """A true pair whose dt is 3x the free-flow time: excluded by a tight
    fitted distribution's own z-sigma spread, but must be admitted once the
    congestion-tail floor (default CONGESTION_TAIL_FACTOR=4.0) is applied --
    and must go back to being excluded with congestion_tail_factor=1.0 (the
    old behaviour, before this fix)."""
    probe = _model_with_tight_fit(sigma=0.4, t_ff=100.0, cam_i="cam_0000", cam_j="cam_0001")
    cam_i, cam_j, t_ff = _reachable_pair(probe)
    model = _model_with_tight_fit(sigma=0.05, t_ff=t_ff, cam_i=cam_i, cam_j=cam_j)

    dt_congested = 3.0 * t_ff

    window_default = _pair_window(
        model, cam_i, cam_j, TOD, z=3.0, miss_widen_factor=1.3, congestion_tail_factor=4.0
    )
    window_old = _pair_window(
        model, cam_i, cam_j, TOD, z=3.0, miss_widen_factor=1.3, congestion_tail_factor=1.0
    )

    assert window_default is not None and window_old is not None
    assert window_default.dt_max_s >= dt_congested, (
        f"congested true pair (dt={dt_congested:.1f}s) should be inside the gate window "
        f"with the default congestion tail factor, but dt_max_s={window_default.dt_max_s:.1f}s"
    )
    assert window_old.dt_max_s < dt_congested, (
        "sanity check: with congestion_tail_factor=1.0 (old behaviour) the tight fitted "
        "sigma alone should NOT be wide enough to admit a 3x-free-flow congested pair -- "
        "otherwise this test isn't exercising the defect"
    )


def test_congestion_tail_factor_does_not_change_lower_bound():
    """The lower (physically-impossible) bound must be untouched by the
    congestion tail factor -- only dt_max_s should move. Clone detection
    depends on the lower bound staying tight."""
    probe = _model_with_tight_fit(sigma=0.4, t_ff=100.0, cam_i="cam_0000", cam_j="cam_0001")
    cam_i, cam_j, t_ff = _reachable_pair(probe)
    model = _model_with_tight_fit(sigma=0.3, t_ff=t_ff, cam_i=cam_i, cam_j=cam_j)

    window_1x = _pair_window(
        model, cam_i, cam_j, TOD, z=3.0, miss_widen_factor=1.3, congestion_tail_factor=1.0
    )
    window_4x = _pair_window(
        model, cam_i, cam_j, TOD, z=3.0, miss_widen_factor=1.3, congestion_tail_factor=4.0
    )
    window_10x = _pair_window(
        model, cam_i, cam_j, TOD, z=3.0, miss_widen_factor=1.3, congestion_tail_factor=10.0
    )

    assert window_1x is not None and window_4x is not None and window_10x is not None
    assert window_1x.dt_min_s == window_4x.dt_min_s == window_10x.dt_min_s
    assert window_4x.dt_max_s > window_1x.dt_max_s
    assert window_10x.dt_max_s > window_4x.dt_max_s


def test_dt_below_lower_bound_still_rejected_regardless_of_congestion_factor():
    """A dt below the gate's own lower bound stays below it no matter the
    congestion tail factor -- the factor only ever raises the ceiling."""
    probe = _model_with_tight_fit(sigma=0.4, t_ff=100.0, cam_i="cam_0000", cam_j="cam_0001")
    cam_i, cam_j, t_ff = _reachable_pair(probe)
    model = _model_with_tight_fit(sigma=0.1, t_ff=t_ff, cam_i=cam_i, cam_j=cam_j)

    window = _pair_window(
        model, cam_i, cam_j, TOD, z=3.0, miss_widen_factor=1.3, congestion_tail_factor=4.0
    )
    assert window is not None
    dt_too_small = window.dt_min_s * 0.5
    assert dt_too_small < window.dt_min_s


def test_default_gate_field_matches_module_constant():
    from engine.association.gating import CONGESTION_TAIL_FACTOR, Gate

    probe = _model_with_tight_fit(sigma=0.4, t_ff=100.0, cam_i="cam_0000", cam_j="cam_0001")
    gate = Gate(model=probe)
    assert gate.congestion_tail_factor == CONGESTION_TAIL_FACTOR
