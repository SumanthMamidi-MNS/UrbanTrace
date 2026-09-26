"""Vehicle population and journey (route + timing) generation.

Vehicles are generated with a true 10-slot canonical plate, a colour/type,
and a latent 128-d appearance embedding drawn from a THREE-LEVEL hierarchical
mixture: (type, colour) class (~40 buckets) -> model archetype (a vocabulary
of ~250 make/model-like centres) -> per-instance quirk noise. This mirrors
real Re-ID appearance structure (many vehicles share a body colour and
vehicle type; a smaller number also share a specific make/model silhouette;
individual vehicles still differ by damage, stickers, roof racks, etc.) and
is what keeps appearance-only retrieval from collapsing at city scale (see
docs/decisions.md, "Day 1b"). Journeys pick a random border-camera
origin/destination pair, route via a mildly randomised shortest path, and
traverse edges at speed_limit * lognormal noise, honouring a time-of-day
demand profile with rush-hour peaks.
"""

import bisect
import itertools
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import networkx as nx

from engine.contracts.city import CityConfig
from engine.contracts.plate import BLANK
from sim.congestion import CongestionConfig, VolumeCounts, make_congestion_fn, record_volume

SIM_EPOCH = datetime(2026, 1, 1, 0, 0, 0)

STATE_CODES = [
    "MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "WB",
    "AP", "TS", "KL", "MP", "HR", "PB", "BR",
]
_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

VEHICLE_TYPES = ["car", "bike", "truck", "bus", "auto"]
VEHICLE_TYPE_WEIGHTS = [0.45, 0.30, 0.08, 0.07, 0.10]

COLORS = ["white", "black", "silver", "grey", "red", "blue", "green", "yellow"]
COLOR_WEIGHTS = [0.22, 0.16, 0.18, 0.12, 0.10, 0.10, 0.07, 0.05]

EMBEDDING_DIM = 128

# Edge traversal speed noise: multiplicative lognormal factor on speed limit.
EDGE_SPEED_NOISE_SIGMA = 0.12
# Route randomisation: multiplicative noise on edge travel-time weight when
# computing the shortest path (keeps routes plausible but not identical).
ROUTE_WEIGHT_NOISE_LOW = 0.85
ROUTE_WEIGHT_NOISE_HIGH = 1.25

# Hierarchical latent embedding: class (type, colour) -> model archetype ->
# per-instance quirk. Weights below control how much of a vehicle's identity
# comes from each level; tuned (see tests/test_calibration.py and
# docs/decisions.md "Day 1b") so that appearance-only rank-1 retrieval lands
# at the published ~70% figure at benchmark (VeRi-776) scale *and* does not
# collapse at full city scale (20k vehicles).
N_ARCHETYPES = 1200
EMBEDDING_WEIGHT_CLASS = 0.15
EMBEDDING_WEIGHT_ARCHETYPE = 0.8
EMBEDDING_WEIGHT_INSTANCE = 1.4

# Route-overlap clone injection (docs/decisions.md, "Day 3a"): how far
# (seconds) an overlapping clone's start time is jittered from its source's
# actual start time. Small enough that the two journeys' passages genuinely
# interleave in time along the shared route (so some cross-vehicle dt's fall
# inside the kinematic model's plausible window), large enough that they are
# still two distinct trips, not a synchronised pair.
CLONE_OVERLAP_JITTER_S = 120.0


@dataclass
class Vehicle:
    gt_vehicle_id: str
    true_plate: str  # 10-char canonical string, may contain '_' blanks
    color: str
    vehicle_type: str
    embedding: list[float]  # latent, 128-d, L2-normalised
    is_clone: bool = False
    clone_of: str | None = None
    # A "near-miss": a genuinely different vehicle, registered close in
    # sequence at the same RTO, whose plate differs from another vehicle's
    # by only a couple of characters (docs/decisions.md, "Day 2b") --
    # e.g. MH12AB1234 and MH12AB1235, both real, both on the road. Distinct
    # from a clone: the plate is SIMILAR, not identical.
    is_near_miss: bool = False
    near_miss_of: str | None = None
    # True iff this clone's route was deliberately drawn from the same
    # origin/destination as its source vehicle (see `clone_route_overlap`
    # on `generate_vehicles_and_journeys` and docs/decisions.md, "Day 3a").
    # Meaningless (always False) for a non-clone vehicle.
    route_overlap: bool = False


@dataclass
class Passage:
    gt_vehicle_id: str
    camera_id: str
    timestamp: datetime


@dataclass
class Journey:
    gt_vehicle_id: str
    camera_sequence: list[str]
    passages: list[Passage] = field(default_factory=list)


def _sample_true_plate(rng: random.Random) -> str:
    state = rng.choice(STATE_CODES)
    rto = f"{rng.randint(1, 99):02d}"
    n_series_letters = rng.choices([1, 2], weights=[0.3, 0.7])[0]
    series = "".join(rng.choice(_LETTERS) for _ in range(n_series_letters))
    series_slots = series.ljust(2, "_")
    n_digits = rng.choices([1, 2, 3, 4], weights=[0.05, 0.1, 0.15, 0.7])[0]
    number = "".join(str(rng.randint(0, 9)) for _ in range(n_digits))
    number_slots = number.rjust(4, "_")
    return state + rto + series_slots + number_slots


def _unit_vector(rng: random.Random, dim: int) -> list[float]:
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _class_centers(rng: random.Random) -> dict[tuple[str, str], list[float]]:
    centers = {}
    for vtype in VEHICLE_TYPES:
        for color in COLORS:
            centers[(vtype, color)] = _unit_vector(rng, EMBEDDING_DIM)
    return centers


def _archetype_centers(rng: random.Random, n: int = N_ARCHETYPES) -> list[list[float]]:
    return [_unit_vector(rng, EMBEDDING_DIM) for _ in range(n)]


def _hierarchical_embedding(
    class_center: list[float],
    archetype_center: list[float],
    vrng: random.Random,
) -> list[float]:
    """Mix class, model-archetype, and per-instance quirk components into one
    latent appearance embedding (see module docstring).

    All three components are unit vectors before mixing. This matters: a raw
    (un-normalised) 128-d gaussian has norm ~sqrt(128)=11.3, which would swamp
    the unit-norm class/archetype terms regardless of its nominal weight —
    that was the original bug (see docs/decisions.md, "Day 1b") that made the
    mixing weights meaningless. With all three on the same norm-1 footing,
    the weights genuinely control how much of a vehicle's identity comes from
    each level."""
    instance_vec = _unit_vector(vrng, EMBEDDING_DIM)
    raw = [
        EMBEDDING_WEIGHT_CLASS * c + EMBEDDING_WEIGHT_ARCHETYPE * a + EMBEDDING_WEIGHT_INSTANCE * n
        for c, a, n in zip(class_center, archetype_center, instance_vec, strict=True)
    ]
    return _normalize(raw)


def _demand_weight(hour_frac: float) -> float:
    """Traffic demand weight for a given hour-of-day (0-24), with morning
    and evening rush peaks."""

    def bump(center: float, width: float) -> float:
        return math.exp(-0.5 * ((hour_frac - center) / width) ** 2)

    return 0.3 + bump(8.5, 1.2) + bump(18.0, 1.5)


def _build_minute_cdf() -> tuple[list[float], float]:
    minutes_per_day = 24 * 60
    profile = [_demand_weight((m / 60.0) % 24.0) for m in range(minutes_per_day)]
    cdf = list(itertools.accumulate(profile))
    return cdf, cdf[-1]


def _sample_minute_of_day(vrng: random.Random, cdf: list[float], total: float) -> int:
    x = vrng.random() * total
    idx = bisect.bisect_right(cdf, x)
    return min(idx, len(cdf) - 1)


def _sample_start_time(vrng: random.Random, hours: int, cdf: list[float], total: float) -> datetime:
    minute_of_day = _sample_minute_of_day(vrng, cdf, total)
    n_days = max(1, math.ceil(hours / 24))
    day = vrng.randrange(0, n_days)
    minute = day * 24 * 60 + minute_of_day
    max_minute = hours * 60 - 1
    minute = min(minute, max_minute)
    seconds_jitter = vrng.uniform(0, 60)
    return SIM_EPOCH + timedelta(minutes=minute, seconds=seconds_jitter)


def _route_with_noise(
    graph: nx.DiGraph, origin: str, dest: str, vrng: random.Random
) -> list[str] | None:
    perturb: dict[tuple[str, str], float] = {}

    def weight_fn(u: str, v: str, data: dict) -> float:
        key = (u, v)
        if key not in perturb:
            perturb[key] = vrng.uniform(ROUTE_WEIGHT_NOISE_LOW, ROUTE_WEIGHT_NOISE_HIGH)
        return data["weight"] * perturb[key]

    try:
        return nx.shortest_path(graph, origin, dest, weight=weight_fn)
    except nx.NetworkXNoPath:
        return None


def _walk_route(
    graph: nx.DiGraph,
    route: list[str],
    start_time: datetime,
    node_to_camera: dict[str, str],
    vrng: random.Random,
    vehicle_id: str,
    edge_time_fn: Callable[[str, str, datetime], float] | None = None,
    volume_sink: VolumeCounts | None = None,
    epoch: datetime = SIM_EPOCH,
    bucket_minutes: int = 5,
) -> tuple[list[str], list[Passage]]:
    """Walk `route` (a list of road-graph node ids) forward in time from
    `start_time`, emitting a Passage at every node that hosts a camera.
    Shared by the main per-vehicle journey generation and by clone
    route-overlap regeneration (`_inject_clones`) so both build passages the
    same way.

    `edge_time_fn` and `volume_sink` are the congestion knob's hooks (see
    `sim/congestion.py` and `generate_vehicles_and_journeys`'s two-pass
    scheme below); both default to None/off, in which case this function's
    arithmetic is byte-for-byte the original (pre-congestion) code path --
    that branch is deliberately left untouched so `congestion=False` keeps
    reproducing the pre-existing behaviour exactly."""
    t = start_time
    passages: list[Passage] = []
    camera_sequence: list[str] = []

    if route[0] in node_to_camera:
        cam_id = node_to_camera[route[0]]
        passages.append(Passage(gt_vehicle_id=vehicle_id, camera_id=cam_id, timestamp=t))
        camera_sequence.append(cam_id)

    for u, v in zip(route[:-1], route[1:], strict=True):
        edge_data = graph.get_edge_data(u, v)
        speed_kmh = edge_data["speed_limit_kmh"]
        length_m = edge_data["length_m"]

        if volume_sink is not None:
            record_volume(volume_sink, u, v, t, epoch, bucket_minutes)

        if edge_time_fn is None:
            noise_factor = vrng.lognormvariate(0, EDGE_SPEED_NOISE_SIGMA)
            effective_speed_ms = max(speed_kmh, 1.0) * 1000.0 / 3600.0 / max(noise_factor, 1e-3)
            travel_s = length_m / effective_speed_ms
        else:
            # Same per-edge lognormal noise draw as the free-flow branch
            # (same vrng call, same position in the call sequence across
            # pass 1/pass 2 -- see generate_vehicles_and_journeys), with the
            # BPR congestion multiplier applied on top of the free-flow time.
            free_flow_travel_s = length_m / (max(speed_kmh, 1.0) * 1000.0 / 3600.0)
            noise_factor = vrng.lognormvariate(0, EDGE_SPEED_NOISE_SIGMA)
            congestion_multiplier = edge_time_fn(u, v, t)
            travel_s = free_flow_travel_s * noise_factor * congestion_multiplier
        t = t + timedelta(seconds=travel_s)

        if v in node_to_camera:
            cam_id = node_to_camera[v]
            passages.append(Passage(gt_vehicle_id=vehicle_id, camera_id=cam_id, timestamp=t))
            camera_sequence.append(cam_id)

    return camera_sequence, passages


def generate_vehicles_and_journeys(
    city: CityConfig,
    n_vehicles: int,
    hours: int,
    seed: int,
    clone_fraction: float = 0.02,
    near_miss_fraction: float = 0.0,
    clone_route_overlap: float = 0.5,
    congestion: bool = False,
    congestion_config: CongestionConfig | None = None,
) -> tuple[list[Vehicle], list[Journey]]:
    """Generate a vehicle population and their journeys (route + ground-truth
    camera passages). Deterministic given `seed`.

    `congestion=False` (the default here -- callers that want it on opt in
    explicitly; `sim.generate`'s CLI defaults it on for new datasets) takes
    exactly the original single-pass free-flow-plus-lognormal-noise code
    path via `_generate_core`, unchanged, so it keeps reproducing
    pre-existing output byte-for-byte.

    `congestion=True` runs `_generate_core` twice (see `sim/congestion.py`):
    pass 1 at free flow to collect per-(edge, bucket) volumes; pass 2
    re-derives BPR travel-time multipliers from those volumes and re-runs
    the identical pipeline (same seeds, so identical routes/start times/
    per-edge noise draws -- see `_walk_route`'s docstring) with them
    applied. Two passes, not an iterative MSA loop: routes aren't
    re-optimized against congested times here, so a third pass would only
    recompute the same volumes."""
    if not congestion:
        return _generate_core(
            city, n_vehicles, hours, seed, clone_fraction, near_miss_fraction, clone_route_overlap
        )

    cfg = congestion_config or CongestionConfig()
    volume_counts: VolumeCounts = {}
    _generate_core(
        city,
        n_vehicles,
        hours,
        seed,
        clone_fraction,
        near_miss_fraction,
        clone_route_overlap,
        volume_sink=volume_counts,
    )

    graph = city.to_digraph()
    congestion_fn = make_congestion_fn(graph, volume_counts, SIM_EPOCH, cfg)
    return _generate_core(
        city,
        n_vehicles,
        hours,
        seed,
        clone_fraction,
        near_miss_fraction,
        clone_route_overlap,
        edge_time_fn=congestion_fn,
        bucket_minutes=cfg.bucket_minutes,
    )


def _generate_core(
    city: CityConfig,
    n_vehicles: int,
    hours: int,
    seed: int,
    clone_fraction: float = 0.02,
    near_miss_fraction: float = 0.0,
    clone_route_overlap: float = 0.5,
    edge_time_fn: Callable[[str, str, datetime], float] | None = None,
    volume_sink: VolumeCounts | None = None,
    bucket_minutes: int = 5,
) -> tuple[list[Vehicle], list[Journey]]:
    """The actual generation pipeline (vehicles -> routes -> walk -> clone
    injection -> near-miss injection), parameterised by the congestion
    hooks so `generate_vehicles_and_journeys` can run it once (off) or
    twice (on, pass 1 free-flow / pass 2 congested) -- see that function's
    docstring. `edge_time_fn=None, volume_sink=None` (the defaults) is the
    original pre-congestion code path, untouched."""
    rng = random.Random(seed)
    graph = city.to_digraph()
    class_centers = _class_centers(rng)
    archetype_centers = _archetype_centers(rng)

    node_to_camera = {cam.node_id: cam.camera_id for cam in city.cameras}
    border_nodes = [cam.node_id for cam in city.cameras if cam.is_border]
    if len(border_nodes) < 2:
        border_nodes = [cam.node_id for cam in city.cameras]

    cdf, total_weight = _build_minute_cdf()

    vehicles: list[Vehicle] = []
    journeys: list[Journey] = []

    for i in range(n_vehicles):
        vrng = random.Random(f"{seed}:vehicle:{i}")
        vehicle_id = f"veh_{i:06d}"
        vtype = vrng.choices(VEHICLE_TYPES, weights=VEHICLE_TYPE_WEIGHTS)[0]
        color = vrng.choices(COLORS, weights=COLOR_WEIGHTS)[0]
        plate = _sample_true_plate(vrng)

        class_center = class_centers[(vtype, color)]
        archetype_center = archetype_centers[vrng.randrange(N_ARCHETYPES)]
        embedding = _hierarchical_embedding(class_center, archetype_center, vrng)

        vehicle = Vehicle(
            gt_vehicle_id=vehicle_id,
            true_plate=plate,
            color=color,
            vehicle_type=vtype,
            embedding=embedding,
        )
        vehicles.append(vehicle)

        origin, dest = vrng.sample(border_nodes, k=2) if len(border_nodes) >= 2 else (
            border_nodes[0],
            border_nodes[0],
        )
        route = _route_with_noise(graph, origin, dest, vrng)
        if not route or len(route) < 2:
            journeys.append(Journey(gt_vehicle_id=vehicle_id, camera_sequence=[]))
            continue

        start_time = _sample_start_time(vrng, hours, cdf, total_weight)
        camera_sequence, passages = _walk_route(
            graph,
            route,
            start_time,
            node_to_camera,
            vrng,
            vehicle_id,
            edge_time_fn=edge_time_fn,
            volume_sink=volume_sink,
            bucket_minutes=bucket_minutes,
        )

        journeys.append(
            Journey(gt_vehicle_id=vehicle_id, camera_sequence=camera_sequence, passages=passages)
        )

    camera_to_node = {cam.camera_id: cam.node_id for cam in city.cameras}
    _inject_clones(
        vehicles,
        journeys,
        graph,
        camera_to_node,
        node_to_camera,
        hours,
        cdf,
        total_weight,
        seed,
        clone_fraction,
        clone_route_overlap,
        rng,
        edge_time_fn=edge_time_fn,
        volume_sink=volume_sink,
        bucket_minutes=bucket_minutes,
    )
    _inject_near_miss_plates(vehicles, near_miss_fraction, rng)

    return vehicles, journeys


def _inject_clones(
    vehicles: list[Vehicle],
    journeys: list[Journey],
    graph: nx.DiGraph,
    camera_to_node: dict[str, str],
    node_to_camera: dict[str, str],
    hours: int,
    cdf: list[float],
    total_weight: float,
    seed: int,
    clone_fraction: float,
    clone_route_overlap: float,
    rng: random.Random,
    edge_time_fn: Callable[[str, str, datetime], float] | None = None,
    volume_sink: VolumeCounts | None = None,
    bucket_minutes: int = 5,
) -> None:
    """Give a fraction of vehicles the same true plate as another vehicle.

    A `clone_route_overlap` fraction of clones have their route REPLACED
    with a fresh route sharing their source vehicle's origin/destination
    camera, so their camera sequences genuinely overlap in space and `dt`
    between source and clone passages is often physically plausible -- the
    realistic hard case (a cloned plate operating in the same neighbourhood
    as the original), not just the trivially-easy case of two vehicles that
    never go near each other. The remaining `1 - clone_route_overlap`
    fraction keep their independently-sampled route from the main loop
    above, which is disjoint from the source's with overwhelming
    probability (docs/decisions.md, "Day 3a" -- pre-3a, ALL clones were this
    disjoint case, which made the clone stratum's kinematic AUC of 1.0 a
    structural artifact rather than a learned result)."""
    if clone_fraction <= 0 or len(vehicles) < 2:
        return
    n_clones = round(len(vehicles) * clone_fraction)
    if n_clones == 0:
        return
    candidate_indices = list(range(len(vehicles)))
    rng.shuffle(candidate_indices)
    clone_indices = set(candidate_indices[:n_clones])
    # Sources are only ever drawn from non-clone vehicles, so a source's
    # plate can never itself be overwritten later in this loop.
    source_pool = [i for i in range(len(vehicles)) if i not in clone_indices]
    if not source_pool:
        return
    for idx in clone_indices:
        source_idx = rng.choice(source_pool)
        source = vehicles[source_idx]
        clone = vehicles[idx]
        clone.true_plate = source.true_plate
        clone.is_clone = True
        clone.clone_of = source.gt_vehicle_id

        if rng.random() >= clone_route_overlap:
            continue  # keep the independently-generated (disjoint) route

        source_journey = journeys[source_idx]
        if len(source_journey.camera_sequence) < 2:
            continue  # source has no usable origin/destination to share
        origin_node = camera_to_node.get(source_journey.camera_sequence[0])
        dest_node = camera_to_node.get(source_journey.camera_sequence[-1])
        if not origin_node or not dest_node or origin_node == dest_node:
            continue

        vrng = random.Random(f"{seed}:clone_overlap:{idx}")
        route = _route_with_noise(graph, origin_node, dest_node, vrng)
        if not route or len(route) < 2 or not source_journey.passages:
            continue

        clone.route_overlap = True
        # Time-correlate the clone with its source: a clone plate used in
        # the same corridor is only a genuinely hard case if it also passes
        # through at a broadly similar time (a plausible dt is what makes
        # the kinematic hard gate stop firing for free). A small jitter
        # around the source's own start time -- not an independent
        # full-day-demand-profile draw -- is what produces that; too large
        # a jitter and the two journeys never interleave in time even
        # though they share a route, reproducing the pre-3a artifact by a
        # different route (docs/decisions.md, "Day 3a").
        source_start = source_journey.passages[0].timestamp
        jitter = timedelta(seconds=vrng.uniform(-CLONE_OVERLAP_JITTER_S, CLONE_OVERLAP_JITTER_S))
        start_time = source_start + jitter
        epoch_end = SIM_EPOCH + timedelta(hours=hours)
        start_time = max(SIM_EPOCH, min(start_time, epoch_end - timedelta(seconds=1)))
        camera_sequence, passages = _walk_route(
            graph,
            route,
            start_time,
            node_to_camera,
            vrng,
            clone.gt_vehicle_id,
            edge_time_fn=edge_time_fn,
            volume_sink=volume_sink,
            bucket_minutes=bucket_minutes,
        )
        journeys[idx] = Journey(
            gt_vehicle_id=clone.gt_vehicle_id,
            camera_sequence=camera_sequence,
            passages=passages,
        )


def _perturb_number_slots(number_slots: str, rng: random.Random, max_delta: int = 2) -> str:
    """Nudge a plate's 4-character (blank-padded) number segment by a small
    delta, preserving its digit width (so a 3-digit number stays 3 digits,
    not becoming a 4-digit number with a leading zero -- those are different
    plates in different ways). Returns the input unchanged if there's no
    digit to perturb."""
    digits = number_slots.lstrip(BLANK)
    if not digits:
        return number_slots
    n_digits = len(digits)
    value = int(digits)
    max_value = 10**n_digits - 1
    delta = rng.choice([d for d in range(-max_delta, max_delta + 1) if d != 0])
    new_value = min(max(value + delta, 0), max_value)
    new_digits = str(new_value).zfill(n_digits)
    return new_digits.rjust(4, BLANK)


def _inject_near_miss_plates(
    vehicles: list[Vehicle], near_miss_fraction: float, rng: random.Random
) -> None:
    """Give a fraction of (non-clone) vehicles a plate that's a near-miss of
    another (non-clone) vehicle's plate: same state + RTO + series, number
    nudged by 1-2 -- e.g. MH12AB1234 and MH12AB1235. This isn't a contrived
    adversarial construction: sequential registrations within one RTO
    genuinely produce plates this close together, and both are real
    vehicles on the road (docs/decisions.md, "Day 2b"). Distinct from a
    clone: the plate is similar, never identical, and the target keeps its
    own independent appearance/route like any other vehicle."""
    if near_miss_fraction <= 0 or len(vehicles) < 2:
        return
    eligible = [i for i, v in enumerate(vehicles) if not v.is_clone]
    n_near_miss = round(len(vehicles) * near_miss_fraction)
    if n_near_miss == 0 or len(eligible) < 2:
        return
    rng.shuffle(eligible)
    target_indices = set(eligible[:n_near_miss])
    source_pool = [i for i in eligible if i not in target_indices]
    if not source_pool:
        return
    for idx in target_indices:
        source = vehicles[rng.choice(source_pool)]
        target = vehicles[idx]
        prefix = source.true_plate[:6]
        new_number_slots = _perturb_number_slots(source.true_plate[6:], rng)
        new_plate = prefix + new_number_slots
        if new_plate == source.true_plate:
            continue  # degenerate perturbation (e.g. no digits to nudge)
        target.true_plate = new_plate
        target.is_near_miss = True
        target.near_miss_of = source.gt_vehicle_id
