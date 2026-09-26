"""Stress sweep (architecture.md §8: "OCR error rate 0->40%, camera miss
rate 0->30%. The headline claim is that our gap over Baseline A widens as
conditions get worse"). Answers that claim PLAINLY, per axis.

Methodology: the FusionModel/gate/entry-exit costs are fit ONCE, on a
nominal-corruption train split generated from a SHARED city (same city
across every sweep point, per architecture.md §10). Every sweep point then
generates a FRESH in-memory held-out dataset -- same city, worsening
corruption -- and is evaluated against that SAME FIXED, already-calibrated
engine. This is deliberately the "conditions drift after you've deployed
and calibrated" scenario (you don't get to instantly retune your plate-
confusion prior when the weather gets worse), not a re-fit-per-point one:
Baseline A has no fitted model to go stale, so this is the fair, realistic
way to ask whether UrbanTrace's edge holds up (or grows) as the field gets
harder, not just whether a freshly-retuned model can always catch up.

OCR noise axis: `CorruptionConfig.p_char_correct` is swept (p_occlude held
at a mildly elevated 0.03 throughout) and the ACTUAL measured whole-plate
accuracy (`eval.metrics.whole_plate_accuracy`) is reported at each point,
not the knob value -- calibrated by a quick offline sweep to span roughly
88% down to 50%.

Camera-miss axis: `CorruptionConfig.p_miss` in {0, 0.1, 0.2, 0.3}.

Usage:
    python -m eval.stress
"""

import json
import time
from pathlib import Path

from engine.association.gating import Gate
from engine.association.mincostflow import fit_entry_exit_costs
from engine.association.window import solve_windowed
from engine.calibration.fit_priors import fit_all
from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent
from engine.scoring.fusion import FusionModel
from eval.metrics import compute_trajectory_metrics, whole_plate_accuracy
from eval.run_pipeline import _build_exact_match_baseline
from sim.city import generate_city
from sim.corruption import CorruptionConfig
from sim.generate import generate_dataset_with_city
from sim.vehicles import Vehicle

OUT_PATH = Path("eval/reports/stress_sweep.json")

N_CAMERAS = 20
N_VEHICLES = 2000
N_VEHICLES_TRAIN = 1200
HOURS = 3
CLONE_FRACTION = 0.02
CITY_SEED = 8301
TRAIN_SEED = 8401
EVAL_SEED_OCR_BASE = 8500
EVAL_SEED_MISS_BASE = 8600

# Calibrated offline (10 cameras/400 vehicles/1h, see decisions log below) so
# the MEASURED whole-plate accuracy spans roughly 88% down to 50%; p_occlude
# held at a mildly-elevated constant so degradation comes primarily from
# character-level misreads, not occlusion runs.
OCR_POINTS = [
    {"p_char_correct": 0.99, "p_occlude": 0.03},
    {"p_char_correct": 0.965, "p_occlude": 0.03},
    {"p_char_correct": 0.955, "p_occlude": 0.03},
    {"p_char_correct": 0.94, "p_occlude": 0.03},
    {"p_char_correct": 0.93, "p_occlude": 0.03},
]

MISS_POINTS = [0.0, 0.1, 0.2, 0.3]


def _true_plate_by_vehicle(vehicles: list[Vehicle]) -> dict[str, str]:
    return {v.gt_vehicle_id: v.true_plate for v in vehicles}


def _evaluate_point(
    events: list[DetectionEvent],
    gate: Gate,
    fusion_model: FusionModel,
    entry_exit,
    city: CityConfig,
) -> tuple[float, float, int]:
    """Returns (sutra_idf1, baseline_a_idf1, n_events)."""
    result = solve_windowed(events, gate, fusion_model, entry_exit, city)
    sutra_metrics = compute_trajectory_metrics(result.trajectories, events)
    baseline_trajectories = _build_exact_match_baseline(events, gate)
    baseline_metrics = compute_trajectory_metrics(baseline_trajectories, events)
    return sutra_metrics.idf1, baseline_metrics.idf1, len(events)


def build_report() -> dict:
    city = generate_city(n_cameras=N_CAMERAS, seed=CITY_SEED)
    train_ds = generate_dataset_with_city(
        city=city,
        n_vehicles=N_VEHICLES_TRAIN,
        hours=HOURS,
        seed=TRAIN_SEED,
        clone_fraction=CLONE_FRACTION,
        corruption_config=CorruptionConfig(clone_fraction=CLONE_FRACTION),
    )
    models = fit_all(train_ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(train_ds.events, city)
    gate = Gate(model=models.kinematic_model)

    ocr_rows = []
    for i, point in enumerate(OCR_POINTS):
        t0 = time.time()
        cfg = CorruptionConfig(
            p_char_correct=point["p_char_correct"],
            p_occlude=point["p_occlude"],
            clone_fraction=CLONE_FRACTION,
        )
        ds = generate_dataset_with_city(
            city=city,
            n_vehicles=N_VEHICLES,
            hours=HOURS,
            seed=EVAL_SEED_OCR_BASE + i,
            clone_fraction=CLONE_FRACTION,
            corruption_config=cfg,
        )
        measured_acc = whole_plate_accuracy(ds.events, _true_plate_by_vehicle(ds.vehicles))
        sutra_idf1, baseline_idf1, n_events = _evaluate_point(
            ds.events, gate, fusion_model, entry_exit, city
        )
        gap = sutra_idf1 - baseline_idf1
        row = {
            "knob_p_char_correct": point["p_char_correct"],
            "knob_p_occlude": point["p_occlude"],
            "measured_whole_plate_accuracy": measured_acc,
            "sutra_idf1": sutra_idf1,
            "baseline_a_idf1": baseline_idf1,
            "gap": gap,
            "n_events": n_events,
            "elapsed_s": time.time() - t0,
        }
        ocr_rows.append(row)
        print(
            f"OCR point p_char={point['p_char_correct']}: measured_acc={measured_acc:.3f} "
            f"sutra_idf1={sutra_idf1:.4f} baseline_idf1={baseline_idf1:.4f} gap={gap:.4f} "
            f"n_events={n_events} ({row['elapsed_s']:.1f}s)",
            flush=True,
        )

    miss_rows = []
    for j, p_miss in enumerate(MISS_POINTS):
        t0 = time.time()
        cfg = CorruptionConfig(p_miss=p_miss, clone_fraction=CLONE_FRACTION)
        ds = generate_dataset_with_city(
            city=city,
            n_vehicles=N_VEHICLES,
            hours=HOURS,
            seed=EVAL_SEED_MISS_BASE + j,
            clone_fraction=CLONE_FRACTION,
            corruption_config=cfg,
        )
        sutra_idf1, baseline_idf1, n_events = _evaluate_point(
            ds.events, gate, fusion_model, entry_exit, city
        )
        gap = sutra_idf1 - baseline_idf1
        row = {
            "p_miss": p_miss,
            "sutra_idf1": sutra_idf1,
            "baseline_a_idf1": baseline_idf1,
            "gap": gap,
            "n_events": n_events,
            "elapsed_s": time.time() - t0,
        }
        miss_rows.append(row)
        print(
            f"miss point p_miss={p_miss}: sutra_idf1={sutra_idf1:.4f} "
            f"baseline_idf1={baseline_idf1:.4f} gap={gap:.4f} n_events={n_events} "
            f"({row['elapsed_s']:.1f}s)",
            flush=True,
        )

    ocr_gap_best, ocr_gap_worst = ocr_rows[0]["gap"], ocr_rows[-1]["gap"]
    miss_gap_best, miss_gap_worst = miss_rows[0]["gap"], miss_rows[-1]["gap"]
    ocr_widens = ocr_gap_worst > ocr_gap_best
    miss_widens = miss_gap_worst > miss_gap_best

    widen_answer = (
        f"OCR-noise axis: gap {'WIDENS' if ocr_widens else 'does NOT widen'} as accuracy "
        f"drops -- {ocr_gap_best:.4f} at {ocr_rows[0]['measured_whole_plate_accuracy']:.1%} "
        f"measured accuracy vs {ocr_gap_worst:.4f} at "
        f"{ocr_rows[-1]['measured_whole_plate_accuracy']:.1%}. "
        f"Camera-miss axis: gap {'WIDENS' if miss_widens else 'does NOT widen'} as p_miss "
        f"rises -- {miss_gap_best:.4f} at p_miss=0 vs {miss_gap_worst:.4f} at p_miss=0.3. "
        + (
            "Both axes support the architecture.md §8 headline claim."
            if ocr_widens and miss_widens
            else "At least one axis does NOT support the headline claim as measured here."
        )
    )

    return {
        "description": (
            "Stress sweep (architecture.md §8) on 20-camera/2000-vehicle/3h in-memory "
            "datasets, one FRESH dataset per point, SAME city throughout. The engine "
            "(FusionModel/gate/entry-exit) is fit ONCE on a nominal-corruption train "
            f"split ({N_VEHICLES_TRAIN} vehicles, different seed, same city) and then "
            "evaluated UNCHANGED against every point's held-out data as corruption "
            "worsens -- see module docstring for why this (not a re-fit-per-point design) "
            "is the realistic test. Measured on the FIXED engine (per-successor prior + "
            "corrected arc cost). OCR-axis measured_whole_plate_accuracy is the ACTUAL "
            "measured value (eval.metrics.whole_plate_accuracy), not the corruption knob."
        ),
        "dataset": {
            "n_cameras": N_CAMERAS,
            "n_vehicles_per_point": N_VEHICLES,
            "n_vehicles_train": N_VEHICLES_TRAIN,
            "hours": HOURS,
            "clone_fraction": CLONE_FRACTION,
            "city_seed": CITY_SEED,
            "train_seed": TRAIN_SEED,
        },
        "ocr_noise_sweep": ocr_rows,
        "camera_miss_sweep": miss_rows,
        "widen_or_not_answer": widen_answer,
    }


def main() -> None:
    report = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
