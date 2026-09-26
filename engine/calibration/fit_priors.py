"""Fit the plate-format prior, kinematic travel-time model, and appearance
density/confusion model on a TRAINING split (architecture.md §3, §10).

architecture.md §10 names "priors over-fitted to our own simulator" as a
risk, mitigated by fitting on a train split and reporting on a held-out
split generated with a different seed. Splitting one seed's events 80/20
would not actually satisfy that — it would still be one noise realisation.
`generate_train_holdout_datasets` below generates two independent data
realisations (different vehicle populations, different OCR/appearance
noise) that still share one camera network, because a kinematic model
fitted on train's (camera_i, camera_j) pairs is only meaningful evaluated
against held-out candidate pairs drawn from *that same* city.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from engine.association.gating import Gate, gate_candidates
from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent
from engine.contracts.plate import RTO_ALPHABET, STATE_ALPHABET
from engine.scoring.appearance_lr import AppearanceModel, fit_appearance_model
from engine.scoring.kinematic_lr import (
    KinematicModel,
    NegativeKinematicPriors,
    PairParams,
    fit_kinematic_model,
    fit_negative_kinematic_priors,
)
from engine.scoring.plate_lr import PlatePriors
from sim.city import generate_city
from sim.corruption import CorruptionConfig
from sim.generate import GeneratedDataset, generate_dataset_with_city
from sim.vehicles import COLORS, VEHICLE_TYPES

_STATE_IDX = {c: i for i, c in enumerate(STATE_ALPHABET)}
_RTO_IDX = {c: i for i, c in enumerate(RTO_ALPHABET)}


def generate_train_holdout_datasets(
    n_cameras: int,
    n_vehicles_train: int,
    n_vehicles_holdout: int,
    hours: int,
    city_seed: int,
    train_seed: int,
    holdout_seed: int,
    clone_fraction: float = 0.0,
    holdout_near_miss_fraction: float = 0.0,
) -> tuple[GeneratedDataset, GeneratedDataset]:
    """One shared city, two independent data realisations. `train_seed` and
    `holdout_seed` must differ; both must differ from nothing in particular
    with respect to `city_seed` (the city is shared on purpose).

    `holdout_near_miss_fraction` (docs/decisions.md, "Day 2b") is applied
    only to the held-out split: near-miss registrations are an evaluation
    stress condition (stratum S4), not something the plate-prior/appearance/
    kinematic fitting on train needs to see."""
    if train_seed == holdout_seed:
        raise ValueError("train_seed and holdout_seed must differ (architecture.md §10)")
    city = generate_city(n_cameras=n_cameras, seed=city_seed)
    train_cfg = CorruptionConfig(clone_fraction=clone_fraction)
    holdout_cfg = CorruptionConfig(clone_fraction=clone_fraction)
    train_ds = generate_dataset_with_city(
        city, n_vehicles_train, hours, train_seed, clone_fraction, train_cfg
    )
    holdout_ds = generate_dataset_with_city(
        city,
        n_vehicles_holdout,
        hours,
        holdout_seed,
        clone_fraction,
        holdout_cfg,
        near_miss_fraction=holdout_near_miss_fraction,
    )
    return train_ds, holdout_ds


def fit_plate_priors(
    train_events: list[DetectionEvent], smoothing: float = 1.0
) -> PlatePriors:
    """Empirical state-code / RTO-code priors from training events'
    plate_argmax reads (not ground truth directly — a real deployment would
    only ever have historical OCR reads to fit this from). Laplace-smoothed
    over the full syntactic vocabulary (26x26 state combos, 10x10 RTO
    combos) so no syntactically valid combination is ever given literally
    zero prior mass."""
    n_state, n_rto = len(STATE_ALPHABET), len(RTO_ALPHABET)
    state_counts = np.zeros((n_state, n_state))
    rto_counts = np.zeros((n_rto, n_rto))

    for e in train_events:
        plate = e.plate_argmax
        if len(plate) != 10:
            continue
        s0, s1 = plate[0], plate[1]
        if s0 in _STATE_IDX and s1 in _STATE_IDX:
            state_counts[_STATE_IDX[s0], _STATE_IDX[s1]] += 1
        r0, r1 = plate[2], plate[3]
        if r0 in _RTO_IDX and r1 in _RTO_IDX:
            rto_counts[_RTO_IDX[r0], _RTO_IDX[r1]] += 1

    state_matrix = state_counts + smoothing
    state_matrix /= state_matrix.sum()
    rto_matrix = rto_counts + smoothing
    rto_matrix /= rto_matrix.sum()
    return PlatePriors(state_matrix=state_matrix, rto_matrix=rto_matrix)


@dataclass
class FittedModels:
    plate_priors: PlatePriors
    kinematic_model: KinematicModel
    appearance_model: AppearanceModel


def fit_negative_kinematic_model(
    kinematic_model: KinematicModel, train_events: list[DetectionEvent]
) -> KinematicModel:
    """Stage 2 of the kinematic fit -- the kinematic-defect fix
    (docs/decisions.md): with p1 (`kinematic_model`) already fit, run the
    SAME spatio-temporal gate it defines over the training split and hand
    every gated candidate pair to `fit_negative_kinematic_priors`, which
    keeps only the ones the ground truth says are a different vehicle.
    That is p0's real sample space -- not a flat window (see
    engine/scoring/kinematic_lr.py's module docstring, "THE NULL MODEL").

    Lives here rather than inside `engine.scoring.kinematic_lr.
    fit_kinematic_model` itself because it needs `engine.association.
    gating`, which imports FROM kinematic_lr (`KinematicModel`,
    `time_of_day_bucket`) -- importing it back there would be circular.
    Also keeps `fit_kinematic_model`'s existing callers (eval/baselines.py,
    eval/blocking_report.py, eval/gating_report.py, and several tests that
    only need a plain p1 fit) from paying for an extra full gating pass they
    never asked for."""
    gate = Gate(model=kinematic_model)
    gate_result = gate_candidates(train_events, gate)
    neg_priors = fit_negative_kinematic_priors(
        train_events, gate_result.candidates, kinematic_model
    )
    return replace(kinematic_model, neg_priors=neg_priors)


def fit_all(train_dataset: GeneratedDataset) -> FittedModels:
    """Fit every scoring-channel model on one training dataset."""
    events = train_dataset.events
    true_color_by_vehicle = {v.gt_vehicle_id: v.color for v in train_dataset.vehicles}
    true_type_by_vehicle = {v.gt_vehicle_id: v.vehicle_type for v in train_dataset.vehicles}

    plate_priors = fit_plate_priors(events)
    kinematic_model = fit_kinematic_model(events, train_dataset.city)
    kinematic_model = fit_negative_kinematic_model(kinematic_model, events)
    appearance_model = fit_appearance_model(
        events, true_color_by_vehicle, true_type_by_vehicle, COLORS, VEHICLE_TYPES
    )
    return FittedModels(
        plate_priors=plate_priors,
        kinematic_model=kinematic_model,
        appearance_model=appearance_model,
    )


def save_fitted_models(models: FittedModels, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "plate_priors.npz",
        state_matrix=models.plate_priors.state_matrix,
        rto_matrix=models.plate_priors.rto_matrix,
    )
    np.savez(
        out_dir / "appearance_hist.npz",
        bin_edges=models.appearance_model.bin_edges,
        log_p1=models.appearance_model.log_p1,
        log_p0=models.appearance_model.log_p0,
    )
    am = models.appearance_model
    (out_dir / "appearance_categorical.json").write_text(
        json.dumps(
            {
                "colors": am.colors,
                "types": am.types,
                "color_prior": am.color_prior,
                "color_confusion": am.color_confusion,
                "type_prior": am.type_prior,
                "type_confusion": am.type_confusion,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    kin = models.kinematic_model
    neg = kin.neg_priors
    (out_dir / "kinematic_params.json").write_text(
        json.dumps(
            {
                "v_max_kmh": kin.v_max_kmh,
                "p_miss": kin.p_miss,
                "default_sigma": kin.default_sigma,
                "dt_min": kin.dt_min,
                "dt_max": kin.dt_max,
                "pair_params": [
                    {
                        "cam_i": k[0],
                        "cam_j": k[1],
                        "tod": k[2],
                        "mu": v.mu,
                        "sigma": v.sigma,
                        "n_obs": v.n_obs,
                    }
                    for k, v in kin.pair_params.items()
                ],
                "neg_priors": None
                if neg is None
                else {
                    "pair_params": [
                        {
                            "cam_i": k[0],
                            "cam_j": k[1],
                            "tod": k[2],
                            "mu": v.mu,
                            "sigma": v.sigma,
                            "n_obs": v.n_obs,
                        }
                        for k, v in neg.pair_params.items()
                    ],
                    "tod_pooled": {
                        tod: {"mu": v.mu, "sigma": v.sigma, "n_obs": v.n_obs}
                        for tod, v in neg.tod_pooled.items()
                    },
                    "global_mu_r": neg.global_mu_r,
                    "global_sigma_r": neg.global_sigma_r,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_fitted_models(in_dir: Path, city: CityConfig) -> FittedModels:
    in_dir = Path(in_dir)

    pp = np.load(in_dir / "plate_priors.npz")
    plate_priors = PlatePriors(state_matrix=pp["state_matrix"], rto_matrix=pp["rto_matrix"])

    ah = np.load(in_dir / "appearance_hist.npz")
    cat = json.loads((in_dir / "appearance_categorical.json").read_text(encoding="utf-8"))
    appearance_model = AppearanceModel(
        bin_edges=ah["bin_edges"],
        log_p1=ah["log_p1"],
        log_p0=ah["log_p0"],
        colors=cat["colors"],
        types=cat["types"],
        color_prior=cat["color_prior"],
        color_confusion=cat["color_confusion"],
        type_prior=cat["type_prior"],
        type_confusion=cat["type_confusion"],
    )

    kj = json.loads((in_dir / "kinematic_params.json").read_text(encoding="utf-8"))
    pair_params = {
        (p["cam_i"], p["cam_j"], p["tod"]): PairParams(
            mu=p["mu"], sigma=p["sigma"], n_obs=p["n_obs"]
        )
        for p in kj["pair_params"]
    }
    neg_json = kj.get("neg_priors")
    neg_priors = None
    if neg_json is not None:
        neg_priors = NegativeKinematicPriors(
            pair_params={
                (p["cam_i"], p["cam_j"], p["tod"]): PairParams(
                    mu=p["mu"], sigma=p["sigma"], n_obs=p["n_obs"]
                )
                for p in neg_json["pair_params"]
            },
            tod_pooled={
                tod: PairParams(mu=v["mu"], sigma=v["sigma"], n_obs=v["n_obs"])
                for tod, v in neg_json["tod_pooled"].items()
            },
            global_mu_r=neg_json["global_mu_r"],
            global_sigma_r=neg_json["global_sigma_r"],
        )
    kinematic_model = KinematicModel(
        city=city,
        pair_params=pair_params,
        v_max_kmh=kj["v_max_kmh"],
        p_miss=kj["p_miss"],
        default_sigma=kj["default_sigma"],
        dt_min=kj["dt_min"],
        dt_max=kj["dt_max"],
        neg_priors=neg_priors,
    )

    return FittedModels(
        plate_priors=plate_priors,
        kinematic_model=kinematic_model,
        appearance_model=appearance_model,
    )
