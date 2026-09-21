"""Generate eval/reports/clone_overlap_auc.json: clone-stratum AUC split by
route overlap (docs/decisions.md, "Day 3a", Part 0).

Pre-3a, EVERY clone's route was independently sampled, so clone/source
camera sequences were disjoint with overwhelming probability -- the
kinematic hard gate fired on essentially every clone/source pair for free,
making the clone stratum's kinematic AUC of 1.0000 a structural artifact of
the simulator, not a learned result. `clone_route_overlap` (default 0.5,
sim/vehicles.py) now draws half of all clones' routes from the SAME
origin/destination as their source, so their camera sequences genuinely
overlap and `dt` between them is often physically plausible -- the
realistic hard case a real clone detector actually has to handle.

Usage:
    python -m eval.clone_overlap_report
"""

import json
from pathlib import Path

from engine.calibration.calibrate import score_labeled_pairs
from engine.calibration.fit_priors import fit_all
from engine.scoring.fusion import FusionModel
from eval.metrics import roc_auc
from eval.stratified import LabeledPair, sample_clone_pairs_by_overlap
from sim.city import generate_city
from sim.generate import generate_dataset_with_city

OUT_PATH = Path("eval/reports/clone_overlap_auc.json")

N_CAMERAS = 20
CITY_SEED = 1
TRAIN_SEED = 101
HOLDOUT_SEED = 202
N_VEHICLES_TRAIN = 4000
N_VEHICLES_HOLDOUT = 6000
HOURS = 24
CLONE_FRACTION = 0.05  # boosted vs. the production default (0.02) purely to
# get large-enough disjoint/overlap sub-samples in one holdout run.
CLONE_ROUTE_OVERLAP = 0.5
MAX_PAIRS = 2000


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
        city,
        N_VEHICLES_TRAIN,
        HOURS,
        TRAIN_SEED,
        clone_fraction=CLONE_FRACTION,
        clone_route_overlap=CLONE_ROUTE_OVERLAP,
    )
    holdout_ds = generate_dataset_with_city(
        city,
        N_VEHICLES_HOLDOUT,
        HOURS,
        HOLDOUT_SEED,
        clone_fraction=CLONE_FRACTION,
        clone_route_overlap=CLONE_ROUTE_OVERLAP,
    )

    models = fit_all(train_ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )

    split = sample_clone_pairs_by_overlap(holdout_ds, seed=0, max_pairs=MAX_PAIRS)
    matrix = {
        "disjoint": _auc_row(split["disjoint"], fusion_model),
        "overlap": _auc_row(split["overlap"], fusion_model),
    }

    n_clone_vehicles = sum(1 for v in holdout_ds.vehicles if v.is_clone)
    n_overlap_vehicles = sum(1 for v in holdout_ds.vehicles if v.is_clone and v.route_overlap)

    return {
        "description": (
            "Clone stratum AUC split by whether the clone's route was drawn "
            "disjoint from (easy case) or overlapping with (realistic hard case) "
            "its source vehicle's route. See docs/decisions.md, 'Day 3a', Part 0."
        ),
        "config": {
            "n_cameras": N_CAMERAS,
            "city_seed": CITY_SEED,
            "train_seed": TRAIN_SEED,
            "holdout_seed": HOLDOUT_SEED,
            "n_vehicles_train": N_VEHICLES_TRAIN,
            "n_vehicles_holdout": N_VEHICLES_HOLDOUT,
            "clone_fraction": CLONE_FRACTION,
            "clone_route_overlap": CLONE_ROUTE_OVERLAP,
        },
        "n_clone_vehicles_holdout": n_clone_vehicles,
        "n_overlap_clone_vehicles_holdout": n_overlap_vehicles,
        "matrix": matrix,
    }


def main() -> None:
    report = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=lambda o: float(o)), encoding="utf-8")
    print(json.dumps(report["matrix"], indent=2))
    print(f"n_clone_vehicles_holdout={report['n_clone_vehicles_holdout']}")
    print(f"n_overlap_clone_vehicles_holdout={report['n_overlap_clone_vehicles_holdout']}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
