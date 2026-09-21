"""Generate eval/reports/stratified_auc.json: per-channel + fused AUC for
each of the four evaluation strata (docs/decisions.md, "Day 2b"), plus
stratum frequencies and one frequency-weighted composite figure.

This is the headline the project defends, replacing the single scalar
"fused AUC" from the first Day 2 pass: a plate is a near-unique identifier,
so uniform negative sampling made the plate channel look artificially
perfect and hid the one case (clones) the whole architecture exists for.

Usage:
    python -m eval.stratified_auc_report
"""

import json
from pathlib import Path

from engine.calibration.calibrate import score_labeled_pairs
from engine.calibration.fit_priors import fit_all
from engine.scoring.fusion import FusionModel
from eval.metrics import roc_auc
from eval.stratified import (
    LabeledPair,
    compute_stratum_frequencies,
    sample_clone_pairs,
    sample_degraded_pairs,
    sample_plate_similar_pairs,
    sample_routine_pairs,
)
from sim.city import generate_city
from sim.corruption import CorruptionConfig
from sim.generate import generate_dataset_with_city

OUT_PATH = Path("eval/reports/stratified_auc.json")

N_CAMERAS = 20
CITY_SEED = 1
TRAIN_SEED = 101
HOLDOUT_SEED = 202
STRESS_SEED = 303
N_VEHICLES_TRAIN = 4000
N_VEHICLES_HOLDOUT = 6000
N_VEHICLES_STRESS = 3000
HOURS = 24
CLONE_FRACTION = 0.02
NEAR_MISS_FRACTION = 0.02
MAX_PAIRS = 2000

# Deliberately higher-noise than the calibrated simulator defaults, purely
# to get a large-enough S3 sample: the default corruption rate makes
# ">=4 combined unread/misread slots across a pair" too rare to measure
# reliably (n=68 observed against the default p_occlude=0.03). This mirrors
# architecture.md §8's own stress-sweep methodology (OCR error 0-40%).
STRESS_CORRUPTION = CorruptionConfig(
    p_occlude=0.35,
    occlude_run_len_min=2,
    occlude_run_len_max=5,
    p_char_correct=0.90,
    posterior_peak_mass_min=0.55,
    posterior_peak_mass_max=0.75,
    clone_fraction=0.0,
)


def _auc_row(labeled_pairs: list[LabeledPair], fusion_model: FusionModel) -> dict:
    if not labeled_pairs:
        return {
            "n_pairs": 0,
            "n_positive": 0,
            "plate_auc": None,
            "appearance_auc": None,
            "kinematic_auc": None,
            "fused_auc": None,
        }
    plate, appearance, kinematic, fused, labels = score_labeled_pairs(labeled_pairs, fusion_model)
    return {
        "n_pairs": len(labels),
        "n_positive": int(labels.sum()),
        "plate_auc": float(roc_auc(plate, labels)),
        "appearance_auc": float(roc_auc(appearance, labels)),
        "kinematic_auc": float(roc_auc(kinematic, labels)),
        "fused_auc": float(roc_auc(fused, labels)),
    }


def build_report() -> dict:
    city = generate_city(n_cameras=N_CAMERAS, seed=CITY_SEED)

    train_ds = generate_dataset_with_city(
        city, N_VEHICLES_TRAIN, HOURS, TRAIN_SEED, clone_fraction=CLONE_FRACTION
    )
    holdout_ds = generate_dataset_with_city(
        city,
        N_VEHICLES_HOLDOUT,
        HOURS,
        HOLDOUT_SEED,
        clone_fraction=CLONE_FRACTION,
        near_miss_fraction=NEAR_MISS_FRACTION,
    )
    stress_ds = generate_dataset_with_city(
        city,
        N_VEHICLES_STRESS,
        HOURS,
        STRESS_SEED,
        clone_fraction=0.0,
        corruption_config=STRESS_CORRUPTION,
    )

    models = fit_all(train_ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )

    s1 = sample_routine_pairs(holdout_ds, seed=0, max_positive_pairs=MAX_PAIRS)
    s2 = sample_clone_pairs(holdout_ds, seed=0, max_pairs=MAX_PAIRS)
    s3 = sample_degraded_pairs(stress_ds, seed=0, max_pairs=MAX_PAIRS)
    s4 = sample_plate_similar_pairs(holdout_ds, seed=0, max_pairs=MAX_PAIRS)

    matrix = {
        "routine": _auc_row(s1, fusion_model),
        "clone": _auc_row(s2, fusion_model),
        "degraded": _auc_row(s3, fusion_model),
        "plate_similar": _auc_row(s4, fusion_model),
    }

    freq = compute_stratum_frequencies(holdout_ds)
    frequencies = {
        "routine": freq.freq_routine,
        "clone": freq.freq_clone,
        "degraded": freq.freq_degraded,
        "plate_similar": freq.freq_plate_similar,
        "n_total_pairs_in_holdout": freq.n_total_pairs,
        "note": (
            "Measured on the (non-stress) held-out split, at "
            f"clone_fraction={CLONE_FRACTION}, near_miss_fraction={NEAR_MISS_FRACTION}. "
            "'degraded' here is the NATURAL rate under default corruption, not the "
            "boosted stress dataset used to actually measure the degraded stratum's AUC "
            "(see the stress_dataset block below) -- weighting by this natural rate is "
            "what keeps the composite figure honest about real-world prevalence."
        ),
    }

    weighted_fused_auc = (
        frequencies["routine"] * matrix["routine"]["fused_auc"]
        + frequencies["clone"] * matrix["clone"]["fused_auc"]
        + frequencies["degraded"] * matrix["degraded"]["fused_auc"]
        + frequencies["plate_similar"] * matrix["plate_similar"]["fused_auc"]
    )
    weighted_plate_auc = (
        frequencies["routine"] * matrix["routine"]["plate_auc"]
        + frequencies["clone"] * matrix["clone"]["plate_auc"]
        + frequencies["degraded"] * matrix["degraded"]["plate_auc"]
        + frequencies["plate_similar"] * matrix["plate_similar"]["plate_auc"]
    )

    return {
        "description": (
            "Stratified pairwise AUC: plate-only is near-perfect on routine traffic "
            "and collapses to chance on clones; fusion holds across all four strata. "
            "A single scalar AUC hides exactly this -- see docs/decisions.md, 'Day 2b'."
        ),
        "config": {
            "n_cameras": N_CAMERAS,
            "city_seed": CITY_SEED,
            "train_seed": TRAIN_SEED,
            "holdout_seed": HOLDOUT_SEED,
            "stress_seed": STRESS_SEED,
            "n_vehicles_train": N_VEHICLES_TRAIN,
            "n_vehicles_holdout": N_VEHICLES_HOLDOUT,
            "n_vehicles_stress": N_VEHICLES_STRESS,
            "clone_fraction": CLONE_FRACTION,
            "near_miss_fraction": NEAR_MISS_FRACTION,
            "min_degraded_slots": 4,
            "max_plate_similar_edit_distance": 2,
            "stress_corruption": STRESS_CORRUPTION.__dict__,
        },
        "matrix": matrix,
        "stratum_frequencies": frequencies,
        "frequency_weighted": {
            "plate_auc": float(weighted_plate_auc),
            "fused_auc": float(weighted_fused_auc),
        },
    }


def main() -> None:
    report = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=lambda o: float(o)), encoding="utf-8")
    print(json.dumps(report["matrix"], indent=2))
    print(json.dumps(report["stratum_frequencies"], indent=2))
    print(json.dumps(report["frequency_weighted"], indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
