"""Generate eval/reports/appearance_scaling.json: appearance-only rank-1
retrieval accuracy as a function of gallery size (500/1500/5000/20000
identities). See docs/decisions.md, "Day 1b" — this curve is itself evidence
for the project's core thesis: appearance alone decays sharply at city
scale, which is exactly why plate + appearance + kinematics must be fused.

Usage:
    python -m eval.appearance_scaling_report
"""

import json
from pathlib import Path

from eval.metrics import appearance_rank1, whole_plate_accuracy
from sim.generate import generate_dataset

N_CAMERAS = 20
SEED = 42
GALLERY_SIZES = [500, 1500, 5000, 20000]
OUT_PATH = Path("eval/reports/appearance_scaling.json")


def main() -> None:
    points = []
    for n_vehicles in GALLERY_SIZES:
        ds = generate_dataset(
            n_cameras=N_CAMERAS, n_vehicles=n_vehicles, hours=24, seed=SEED, clone_fraction=0.0
        )
        true_plate_by_vehicle = {v.gt_vehicle_id: v.true_plate for v in ds.vehicles}
        plate_acc = whole_plate_accuracy(ds.events, true_plate_by_vehicle)
        rank1, n_events, n_eligible = appearance_rank1(ds.events)
        point = {
            "gallery_size": n_vehicles,
            "n_events": n_events,
            "n_eligible_queries": n_eligible,
            "appearance_rank1": round(rank1, 4),
            "whole_plate_accuracy": round(plate_acc, 4),
        }
        points.append(point)
        print(point, flush=True)

    report = {
        "description": (
            "Appearance-only rank-1 retrieval accuracy vs gallery size (number of "
            "distinct vehicle identities). Demonstrates that appearance alone decays "
            "at city scale -- the reason UrbanTrace fuses plate + appearance + kinematics "
            "rather than relying on any single channel."
        ),
        "n_cameras": N_CAMERAS,
        "seed": SEED,
        "points": points,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
