"""Stratified pairwise evaluation (docs/decisions.md, "Day 2b").

This is the project's thesis, encoded as a test: a plate is a near-unique
identifier, so it dominates on routine traffic -- but on clones (same true
plate, different vehicle) the plate channel is structurally blind, and only
fusion with appearance + kinematics can still tell the vehicles apart.

If the clone-stratum fused AUC assertion below ever fails, DO NOT loosen it:
per the task spec, that would mean the clone case needs Day 3's kinematic
hard gate to be caught at the trajectory level rather than the pair level,
which is a load-bearing design finding, not a flaky test.
"""

import pytest

from eval.stratified_auc_report import build_report

CLONE_PLATE_AUC_CEILING = 0.60
CLONE_FUSED_AUC_FLOOR = 0.90


@pytest.fixture(scope="module")
def report():
    return build_report()


def test_routine_stratum_plate_channel_is_near_perfect(report):
    """The everyday case: two unrelated vehicles almost always have
    completely different plates, so plate-only is (correctly) excellent
    here -- this is what makes the clone/plate-similar strata meaningful by
    contrast, not a bug in either measurement."""
    row = report["matrix"]["routine"]
    assert row["plate_auc"] > 0.99
    assert row["fused_auc"] > 0.99


def test_clone_stratum_plate_only_collapses_to_chance_but_fusion_holds(report):
    """The thesis test. Same true plate, different vehicle: plate-only must
    be indistinguishable from guessing, and fusion must still work."""
    row = report["matrix"]["clone"]
    assert row["n_pairs"] > 20, f"clone stratum sample too small ({row['n_pairs']} pairs)"
    assert row["plate_auc"] < CLONE_PLATE_AUC_CEILING, (
        f"plate-only AUC on the clone stratum is {row['plate_auc']:.4f}, expected < "
        f"{CLONE_PLATE_AUC_CEILING} (same true plate should make plate evidence useless)"
    )
    assert row["fused_auc"] > CLONE_FUSED_AUC_FLOOR, (
        f"fused AUC on the clone stratum is {row['fused_auc']:.4f}, did not clear "
        f"{CLONE_FUSED_AUC_FLOOR}. Per the task spec: DO NOT weaken this assertion -- "
        f"this means the clone case needs Day 3's kinematic hard gate at the trajectory "
        f"level rather than the pair level, which must be reported as a design finding."
    )


def test_degraded_stratum_sample_is_large_enough(report):
    """The default corruption rate makes this stratum too rare to measure
    reliably (n=68 observed) -- the boosted stress dataset must actually
    fix that."""
    row = report["matrix"]["degraded"]
    assert row["n_pairs"] >= 500, f"degraded stratum sample too small ({row['n_pairs']} pairs)"


def test_plate_similar_stratum_plate_channel_is_measurably_weaker_than_routine(report):
    """Near-miss plates (edit distance <= 2) should still let the plate
    channel discriminate somewhat (unlike clones, they're not identical),
    but measurably worse than the routine stratum's near-total separation."""
    routine = report["matrix"]["routine"]
    similar = report["matrix"]["plate_similar"]
    assert similar["n_pairs"] > 20, f"plate-similar sample too small ({similar['n_pairs']} pairs)"
    assert similar["plate_auc"] < routine["plate_auc"]


def test_every_stratum_has_a_finite_fused_auc(report):
    for name, row in report["matrix"].items():
        assert row["fused_auc"] is not None, f"stratum {name} produced no fused AUC at all"
        assert 0.0 <= row["fused_auc"] <= 1.0


def test_stratum_frequencies_sum_to_one(report):
    freq = report["stratum_frequencies"]
    total = freq["routine"] + freq["clone"] + freq["degraded"] + freq["plate_similar"]
    assert abs(total - 1.0) < 1e-9


def test_frequency_weighted_headline_is_between_worst_and_best_stratum(report):
    fused_values = [row["fused_auc"] for row in report["matrix"].values()]
    weighted = report["frequency_weighted"]["fused_auc"]
    assert min(fused_values) - 1e-9 <= weighted <= max(fused_values) + 1e-9
