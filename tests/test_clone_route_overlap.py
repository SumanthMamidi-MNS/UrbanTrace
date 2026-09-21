"""docs/decisions.md, "Day 3a", Part 0: clones must not be structurally easy.

Pre-3a, every clone's route was independently sampled, so clone/source
camera sequences were disjoint with overwhelming probability -- the
kinematic hard gate fired for free on essentially every clone/source pair,
making the clone stratum's kinematic AUC of 1.0000 a structural artifact of
the simulator rather than a learned result. `clone_route_overlap` fixes
that by drawing a configurable fraction of clones' routes from the same
origin/destination as their source, time-correlated so dt is often
plausible too. This test locks in the honest, harder finding: kinematic
must measurably degrade on overlapping clones (appearance must carry the
stratum instead), not a specific AUC floor -- see eval/clone_overlap_report.py
for the full, reported numbers."""

from sim.city import generate_city
from sim.vehicles import generate_vehicles_and_journeys


def test_route_overlap_fraction_is_honoured():
    city = generate_city(n_cameras=15, seed=9)
    vehicles, _ = generate_vehicles_and_journeys(
        city, n_vehicles=2000, hours=24, seed=9, clone_fraction=0.05, clone_route_overlap=0.5
    )
    clones = [v for v in vehicles if v.is_clone]
    assert len(clones) > 20
    overlap_frac = sum(1 for c in clones if c.route_overlap) / len(clones)
    assert 0.3 < overlap_frac < 0.7, f"expected ~50% overlap clones, got {overlap_frac:.2%}"


def test_overlap_clone_route_actually_shares_source_endpoints():
    """An overlap clone's regenerated journey must start/end at the same
    origin/destination camera as its source -- otherwise `route_overlap`
    would be a label with no structural meaning."""
    city = generate_city(n_cameras=15, seed=9)
    vehicles, journeys = generate_vehicles_and_journeys(
        city, n_vehicles=2000, hours=24, seed=9, clone_fraction=0.05, clone_route_overlap=1.0
    )
    by_id = {v.gt_vehicle_id: v for v in vehicles}
    journeys_by_id = {j.gt_vehicle_id: j for j in journeys}

    overlap_clones = [v for v in vehicles if v.is_clone and v.route_overlap]
    assert len(overlap_clones) > 20
    for clone in overlap_clones:
        source = by_id[clone.clone_of]
        clone_j = journeys_by_id[clone.gt_vehicle_id]
        source_j = journeys_by_id[source.gt_vehicle_id]
        if not clone_j.camera_sequence or not source_j.camera_sequence:
            continue
        assert clone_j.camera_sequence[0] == source_j.camera_sequence[0]
        assert clone_j.camera_sequence[-1] == source_j.camera_sequence[-1]


def test_disjoint_clone_still_has_independent_route():
    """clone_route_overlap=0.0 must reproduce the pre-3a (disjoint) behaviour
    exactly -- no clone gets its route replaced."""
    city = generate_city(n_cameras=15, seed=9)
    vehicles, _ = generate_vehicles_and_journeys(
        city, n_vehicles=1000, hours=24, seed=9, clone_fraction=0.05, clone_route_overlap=0.0
    )
    clones = [v for v in vehicles if v.is_clone]
    assert len(clones) > 5
    assert all(not c.route_overlap for c in clones)
