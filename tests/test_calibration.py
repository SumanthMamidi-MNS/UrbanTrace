"""Calibration guard (see docs/decisions.md): the simulator's default
corruption knobs must reproduce published real-world ANPR figures, not
whatever flatters our own method. If defaults miss these bands, the fix is
to retune sim/corruption.py + sim/vehicles.py defaults — never to widen the
assertion bands here.

- whole-plate per-read OCR accuracy: 85-95%, at any scale.
- appearance-only rank-1 retrieval accuracy: 65-75% at benchmark
  (VeRi-776-like, ~1500 identity) scale.
- appearance-only rank-1 must NOT collapse at full city scale (20k
  identities): >=35% floor. A flat (type, colour)-only appearance space
  crashes to ~15% here — see docs/decisions.md, "Day 1b" — so this floor is
  what actually guards the hierarchical (class -> archetype -> instance)
  embedding design, not just the benchmark-scale band above.
"""

from eval.metrics import appearance_rank1, whole_plate_accuracy
from sim.generate import generate_dataset

PLATE_ACC_LOW, PLATE_ACC_HIGH = 0.85, 0.95
RANK1_BENCHMARK_LOW, RANK1_BENCHMARK_HIGH = 0.65, 0.75
RANK1_CITY_SCALE_FLOOR = 0.35

# n_cameras=20 (not the production 50) keeps this test fast; rank-1 at a
# given identity count is governed by the embedding model and gallery size,
# not camera count, so this doesn't change what's being calibrated.
N_CAMERAS = 20


def _true_plate_by_vehicle(ds) -> dict[str, str]:
    return {v.gt_vehicle_id: v.true_plate for v in ds.vehicles}


def test_benchmark_scale_calibration_matches_published_anpr_figures():
    ds = generate_dataset(
        n_cameras=N_CAMERAS, n_vehicles=1500, hours=24, seed=42, clone_fraction=0.0
    )
    assert len(ds.events) > 1000

    plate_acc = whole_plate_accuracy(ds.events, _true_plate_by_vehicle(ds))
    rank1_acc, n_events, n_eligible = appearance_rank1(ds.events)

    assert PLATE_ACC_LOW <= plate_acc <= PLATE_ACC_HIGH, (
        f"whole-plate per-read accuracy {plate_acc:.4f} outside [{PLATE_ACC_LOW}, {PLATE_ACC_HIGH}]"
    )
    assert RANK1_BENCHMARK_LOW <= rank1_acc <= RANK1_BENCHMARK_HIGH, (
        f"appearance-only rank-1 accuracy {rank1_acc:.4f} outside "
        f"[{RANK1_BENCHMARK_LOW}, {RANK1_BENCHMARK_HIGH}] at benchmark (1500-identity) scale "
        f"(n_events={n_events}, n_eligible_queries={n_eligible})"
    )


def test_city_scale_rank1_does_not_collapse():
    ds = generate_dataset(
        n_cameras=N_CAMERAS, n_vehicles=20000, hours=24, seed=42, clone_fraction=0.0
    )
    assert len(ds.events) > 10000

    plate_acc = whole_plate_accuracy(ds.events, _true_plate_by_vehicle(ds))
    rank1_acc, n_events, n_eligible = appearance_rank1(ds.events)

    assert PLATE_ACC_LOW <= plate_acc <= PLATE_ACC_HIGH, (
        f"whole-plate per-read accuracy {plate_acc:.4f} outside "
        f"[{PLATE_ACC_LOW}, {PLATE_ACC_HIGH}] at city (20k-vehicle) scale"
    )
    assert rank1_acc >= RANK1_CITY_SCALE_FLOOR, (
        f"appearance-only rank-1 accuracy {rank1_acc:.4f} collapsed below the "
        f"{RANK1_CITY_SCALE_FLOOR} city-scale floor (n_events={n_events}, "
        f"n_eligible_queries={n_eligible}) — the appearance space is too coarse again"
    )
