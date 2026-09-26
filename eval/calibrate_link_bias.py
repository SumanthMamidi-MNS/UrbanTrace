"""CLI: calibrate `link_bias` (beta, in nats -- engine.association.window's
`solve_windowed(..., link_bias=...)`) on a TRAINING day.

THE PROBLEM THIS CALIBRATES (docs/decisions.md's beta sweep). After fixing
the kinematic null-hypothesis defect, a small ablation city's IDF1 improved
(0.981 -> 0.992) but data/run1 (50 cameras, 20k vehicles, 24h, seed 42, the
project's evaluation set) got WORSE (IDF1 0.9899 -> 0.9891, ID switches
754 -> 1893): at rush-hour density a CORRECT kinematic model says travel
time is barely discriminative, so true links with weak plate evidence no
longer clear the honest Bayesian `s(i,j) > 0` threshold that the old
(broken) model's inflated bonus used to carry for free. The fix to the
kinematic model was correct; the decision threshold was simply never
calibrated for IDF1 at realistic density. `link_bias` reopens exactly that
threshold as a single calibratable knob: link arc cost becomes
`-s(i,j) + exit_cost(i) + entry_cost(j) - beta`, so the solver links iff
`s(i,j) > -beta` (beta=0 is the plain, uncalibrated Bayesian threshold; a
positive beta makes linking easier, recovering weak-evidence true links).

WHY THIS SCRIPT EXISTS, RATHER THAN JUST SWEEPING ON data/run1: data/run1
is the project's EVALUATION set (architecture.md sec 10 / docs/decisions.md
"Day 2": priors -- and, by the same logic, any decision threshold -- must be
calibrated on a split independent of what gets reported). Choosing beta by
looking at IDF1 on run1 itself would be calibrating on the test set. This
script instead builds its own TRAINING day, on a seed that must differ from
run1's (42), sweeps beta on THAT day only, and reports the sweep table for
a human (or eval/run_pipeline.py's `--link-bias` default) to pick from.
data/run1 is never read or written by this script.

WHY A SMALLER CITY, NOT JUST FEWER VEHICLES (the task's own instruction:
"shorten the day or reduce cameras proportionally rather than just thinning
vehicles"). run1's rush-hour-density effect is a PER-CAMERA phenomenon: how
many candidate predecessors compete for one successor event during a rush
hour, at one camera. Thinning only the vehicle count on the full 50-camera
network would spread the same (smaller) population across the same many
cameras and make every camera's traffic artificially SPARSE -- hiding
exactly the density effect this calibration exists to reproduce. Instead
this scales city size (`sim.city.generate_city`'s `n_cameras`) AND vehicle
count down by the SAME factor, `SCALE`, which keeps vehicles-per-camera --
and hence rush-hour reads-per-camera -- equal to run1's (20,000/50 = 400).
Hours are left at run1's full 24 (not shortened): `sim.vehicles`'s
minute-of-day demand curve (`_build_minute_cdf`) is always built over a
fixed 24-hour profile and then simply CLIPPED to `hours*60` minutes when
`hours < 24`, which would pile samples up at the cutoff instead of
compressing the day's actual rush-hour timing -- shortening the day would
therefore distort exactly the rush-hour density this script needs to be
realistic, not just smaller.

Priors (plate/kinematic/appearance) and entry/exit costs are fit on a THIRD,
independent seed on the SAME (small) city -- `eval/run_pipeline.py`'s own
pattern (`DEFAULT_TRAIN_VEHICLES`, `DEFAULT_TRAIN_SEED_OFFSET`), reused here
via direct import so the two scripts' "how many independent seeds does this
need and why" story stays in exactly one place.

Usage:
    python -m eval.calibrate_link_bias
    python -m eval.calibrate_link_bias --scale 0.3 --train-day-seed 20260922
"""

import json
import time
from pathlib import Path

import typer
from rich.console import Console

from engine.association.gating import Gate
from engine.association.mincostflow import fit_entry_exit_costs
from engine.association.window import solve_windowed
from engine.calibration.fit_priors import fit_all
from engine.scoring.fusion import FusionModel
from eval.metrics import compute_trajectory_metrics
from eval.run_pipeline import DEFAULT_TRAIN_SEED_OFFSET, DEFAULT_TRAIN_VEHICLES
from sim.city import generate_city
from sim.congestion import CongestionConfig
from sim.corruption import CorruptionConfig
from sim.generate import generate_dataset_with_city

app = typer.Typer(add_completion=False)
console = Console()

# data/run1 itself (the dataset this script must NEVER touch, let alone
# tune on) is 50 cameras / 20,000 vehicles / 24h / seed 42 / clone_fraction
# 0.02, corruption = CorruptionConfig() defaults (verified equal to
# data/run1/config.json's "corruption" block). Every constant below is
# expressed relative to those numbers so the "same config, different scale
# and seed" claim in the module docstring is checkable by inspection.
RUN1_SEED = 42
RUN1_CAMERAS = 50
RUN1_VEHICLES = 20_000
RUN1_HOURS = 24
RUN1_CLONE_FRACTION = 0.02

DEFAULT_SCALE = 0.3  # -> 15 cameras, 6,000 vehicles (see module docstring)
DEFAULT_TRAIN_DAY_SEED = 20260922  # must differ from RUN1_SEED

DEFAULT_BETAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)

# "Beats beta=0 by a clear margin" (task step 3): a threshold small enough
# not to be noise (the sweep's own beta=0 IDF1 already tells us the floor)
# but large enough that adopting a nonzero default is worth the risk of
# creating more (even if fewer wrong-direction) links than the pure
# Bayesian threshold. Chosen relative to the magnitude of the regression
# this is meant to fix (run1's post-fix IDF1 drop was ~0.0008, 0.9899 ->
# 0.9891) -- a genuine fix should clear a materially larger bar than the
# regression itself, not just technically exceed 0.0.
CLEAR_MARGIN_IDF1 = 0.003

REPORT_PATH = Path("eval/reports/link_bias_calibration.json")


def _build_training_day(
    scale: float, train_day_seed: int, congestion: bool, congestion_config: CongestionConfig | None
):
    """The day beta is swept on -- see module docstring for why it scales
    city+vehicles together rather than thinning vehicles alone. `congestion`
    matches the CLI's own default (on, BPR) so this sweep reflects the same
    rush-hour density effect eval/run_pipeline.py's evaluated dataset sees,
    rather than sweeping beta against unrealistic free-flow travel times."""
    n_cameras = max(4, round(RUN1_CAMERAS * scale))
    n_vehicles = round(RUN1_VEHICLES * scale)
    city = generate_city(n_cameras=n_cameras, seed=train_day_seed)
    cfg = CorruptionConfig(clone_fraction=RUN1_CLONE_FRACTION)
    day_ds = generate_dataset_with_city(
        city=city,
        n_vehicles=n_vehicles,
        hours=RUN1_HOURS,
        seed=train_day_seed,
        clone_fraction=RUN1_CLONE_FRACTION,
        corruption_config=cfg,
        congestion=congestion,
        congestion_config=congestion_config,
    )
    return city, n_cameras, n_vehicles, day_ds


@app.command()
def main(
    scale: float = typer.Option(
        DEFAULT_SCALE,
        help="Fraction of run1's 50 cameras / 20,000 vehicles to use for the training day.",
    ),
    train_day_seed: int = typer.Option(
        DEFAULT_TRAIN_DAY_SEED,
        help="Seed for the training day's city+vehicles+corruption. Must differ from run1's 42.",
    ),
    train_seed_offset: int = typer.Option(
        DEFAULT_TRAIN_SEED_OFFSET,
        help="Added to train_day_seed for the (yet-another-seed) priors-fitting split.",
    ),
    priors_vehicles: int = typer.Option(
        DEFAULT_TRAIN_VEHICLES, help="Vehicle count for the priors-fitting split."
    ),
    congestion: bool = typer.Option(
        True,
        "--congestion/--no-congestion",
        help=(
            "Density-dependent (BPR) link travel times for BOTH the training day and the "
            "priors split, matching `python -m sim.generate`'s own default (on). Keeping "
            "this on by default means beta is swept under the same rush-hour density "
            "regime the evaluated dataset (if generated with congestion) actually has -- "
            "sweeping on free-flow data by accident would silently miscalibrate beta."
        ),
    ),
    betas: str = typer.Option(
        ",".join(str(b) for b in DEFAULT_BETAS),
        help="Comma-separated beta values to sweep (each is a full training-day solve).",
    ),
) -> None:
    beta_grid = tuple(float(b) for b in betas.split(",") if b.strip())
    if train_day_seed == RUN1_SEED:
        raise typer.BadParameter(
            f"train_day_seed must differ from run1's seed ({RUN1_SEED}) -- "
            "calibrating on run1 itself would be tuning on the evaluation set."
        )

    n_cameras = max(4, round(RUN1_CAMERAS * scale))
    n_vehicles = round(RUN1_VEHICLES * scale)
    congestion_config = CongestionConfig(enabled=True) if congestion else None
    console.print(
        f"[bold]Building TRAINING day[/bold]: {n_cameras} cameras, {n_vehicles} vehicles, "
        f"{RUN1_HOURS}h, seed={train_day_seed}, clone_fraction={RUN1_CLONE_FRACTION}, "
        f"congestion={'on (BPR)' if congestion else 'off (free flow)'} "
        f"(scale={scale} of run1's {RUN1_CAMERAS}cam/{RUN1_VEHICLES}veh -- NOT data/run1 itself, "
        f"vehicles-per-camera={n_vehicles / n_cameras:.0f} matches run1's "
        f"{RUN1_VEHICLES / RUN1_CAMERAS:.0f})"
    )
    t0 = time.time()
    city, n_cameras, n_vehicles, day_ds = _build_training_day(
        scale, train_day_seed, congestion, congestion_config
    )
    t1 = time.time()
    n_gt_vehicles_true = len({e.gt_vehicle_id for e in day_ds.events if e.gt_vehicle_id})
    console.print(
        f"[green]Training day built[/green]: {len(day_ds.events)} events, "
        f"{n_gt_vehicles_true} true trajectories (vehicles with >=1 event) in {t1 - t0:.1f}s"
    )

    priors_seed = train_day_seed + train_seed_offset
    console.print(
        f"[bold]Fitting priors[/bold] on a THIRD, independent seed={priors_seed}, "
        f"{priors_vehicles} vehicles, same (small) training-day city -- "
        f"eval/run_pipeline.py's own train-split pattern"
    )
    priors_ds = generate_dataset_with_city(
        city=city,
        n_vehicles=priors_vehicles,
        hours=RUN1_HOURS,
        seed=priors_seed,
        clone_fraction=RUN1_CLONE_FRACTION,
        congestion=congestion,
        congestion_config=congestion_config,
    )
    t2 = time.time()
    models = fit_all(priors_ds)
    fusion_model = FusionModel(
        plate_priors=models.plate_priors,
        kinematic_model=models.kinematic_model,
        appearance_model=models.appearance_model,
    )
    entry_exit = fit_entry_exit_costs(priors_ds.events, city)
    gate = Gate(model=models.kinematic_model)
    t3 = time.time()
    console.print(
        f"[green]Fitted priors[/green] in {t3 - t2:.1f}s (priors-split generation {t2 - t1:.1f}s)"
    )

    sweep: list[dict] = []
    for beta in beta_grid:
        console.print(f"[bold]beta={beta}[/bold]: solving the training day...")
        tb0 = time.time()
        result = solve_windowed(
            day_ds.events, gate, fusion_model, entry_exit, city, link_bias=beta
        )
        tb1 = time.time()
        metrics = compute_trajectory_metrics(result.trajectories, day_ds.events)
        row = {
            "beta": beta,
            "idf1": metrics.idf1,
            "id_precision": metrics.id_precision,
            "id_recall": metrics.id_recall,
            "id_switches": metrics.id_switches,
            "fragmentation": metrics.fragmentation,
            "trajectory_completeness": metrics.trajectory_completeness,
            "n_predicted_trajectories": metrics.n_predicted_trajectories,
            "n_gt_vehicles": metrics.n_gt_vehicles,
            "n_events": len(day_ds.events),
            "n_windows": result.n_windows,
            "solve_s": tb1 - tb0,
        }
        sweep.append(row)
        console.print(
            f"  idf1={metrics.idf1:.4f} id_switches={metrics.id_switches} "
            f"fragmentation={metrics.fragmentation} "
            f"n_pred_traj={metrics.n_predicted_trajectories} "
            f"(n_gt_vehicles={metrics.n_gt_vehicles}) solve_s={tb1 - tb0:.1f}"
        )

    baseline_row = next(r for r in sweep if r["beta"] == 0.0)
    best_row = max(sweep, key=lambda r: r["idf1"])
    margin = best_row["idf1"] - baseline_row["idf1"]
    chosen_clears_margin = best_row["beta"] != 0.0 and margin > CLEAR_MARGIN_IDF1
    chosen_beta = best_row["beta"] if chosen_clears_margin else 0.0

    if chosen_clears_margin:
        decision = (
            f"beta={chosen_beta} beats beta=0 by {margin:.4f} IDF1 "
            f"(> the {CLEAR_MARGIN_IDF1} clear-margin threshold) on the training day -- "
            f"adopting it as eval/run_pipeline.py's new default."
        )
    else:
        decision = (
            f"best beta swept was {best_row['beta']} at idf1={best_row['idf1']:.4f}, only "
            f"{margin:.4f} above beta=0's {baseline_row['idf1']:.4f} -- below the "
            f"{CLEAR_MARGIN_IDF1} clear-margin threshold, so keeping beta=0 as the default "
            f"(no change to eval/run_pipeline.py)."
        )
    console.print(f"[bold]Decision:[/bold] {decision}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "description": (
            "Beta (link_bias) calibration sweep on a TRAINING day -- NEVER data/run1 "
            "(the project's evaluation set, seed=42). This training day has its own "
            f"smaller city+vehicle population, scaled down together (not just thinned) "
            f"from run1's {RUN1_CAMERAS} cameras / {RUN1_VEHICLES} vehicles by factor "
            f"scale={scale}, so vehicles-per-camera (and hence rush-hour reads-per-camera) "
            f"matches run1's exactly: {n_vehicles} vehicles / {n_cameras} cameras = "
            f"{n_vehicles / n_cameras:.0f} veh/cam vs run1's "
            f"{RUN1_VEHICLES / RUN1_CAMERAS:.0f} veh/cam. hours={RUN1_HOURS} (full day, "
            f"unshortened -- see module docstring on why shortening would distort rush-hour "
            f"timing via sim.vehicles' minute-of-day demand curve clipping). "
            f"clone_fraction={RUN1_CLONE_FRACTION} and all other CorruptionConfig fields are "
            f"defaults, identical to run1's config.json. Training day seed={train_day_seed} "
            f"(run1's seed is {RUN1_SEED} -- must differ, checked at runtime). Priors "
            f"(plate/kinematic/appearance) and entry/exit costs are fit on a THIRD "
            f"independent seed={priors_seed} ({priors_vehicles} vehicles), same city -- "
            "eval/run_pipeline.py's own train-split pattern, reused via direct import."
        ),
        "run1_reference": {
            "seed": RUN1_SEED,
            "cameras": RUN1_CAMERAS,
            "vehicles": RUN1_VEHICLES,
            "hours": RUN1_HOURS,
            "clone_fraction": RUN1_CLONE_FRACTION,
            "note": "reference numbers only -- data/run1 itself was never read by this script",
        },
        "training_day": {
            "seed": train_day_seed,
            "scale": scale,
            "cameras": n_cameras,
            "vehicles": n_vehicles,
            "hours": RUN1_HOURS,
            "clone_fraction": RUN1_CLONE_FRACTION,
            "vehicles_per_camera": n_vehicles / n_cameras,
            "n_events": len(day_ds.events),
            "n_gt_vehicles_true": n_gt_vehicles_true,
            "generation_s": t1 - t0,
            "congestion": congestion,
        },
        "priors_split": {
            "seed": priors_seed,
            "seed_offset": train_seed_offset,
            "vehicles": priors_vehicles,
            "hours": RUN1_HOURS,
            "clone_fraction": RUN1_CLONE_FRACTION,
            "generation_and_fit_s": t3 - t1,
            "congestion": congestion,
        },
        "sweep": sweep,
        "clear_margin_idf1": CLEAR_MARGIN_IDF1,
        "baseline_beta0": baseline_row,
        "best_by_idf1": best_row,
        "chosen_beta": chosen_beta,
        "decision": decision,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {REPORT_PATH}")


if __name__ == "__main__":
    app()
