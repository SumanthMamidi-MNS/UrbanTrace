"""Corruption model: ground-truth passages -> noisy DetectionEvents.

All knobs live in CorruptionConfig. The single most important property here
is that OCR corruption emits a genuine per-slot POSTERIOR (not just a
sampled character): most mass on the sampled/"read" character, with the
remainder leaked to its visually-confusable group.
"""

import random
from dataclasses import dataclass

from engine.contracts.city import CityConfig
from engine.contracts.events import DetectionEvent, VehicleAttributes
from engine.contracts.plate import BLANK, PLATE_SLOTS, SlotPosterior
from sim.vehicles import COLORS, VEHICLE_TYPES, Passage, Vehicle

# Visually-confusable OCR character groups (within-group substitutions are
# far likelier than cross-group / random substitutions).
CONFUSABLE_GROUPS: list[set[str]] = [
    {"O", "0", "D", "Q"},
    {"I", "1", "L", "T"},
    {"B", "8", "6"},
    {"S", "5"},
    {"G", "6", "C"},
    {"Z", "2", "7"},
    {"A", "4"},
    {"M", "N", "H"},
    {"U", "V"},
    {"E", "F"},
    {"P", "R"},
    {"3", "9", "8"},
]

# Colour/type confusion clusters (visually similar under field conditions).
COLOR_CONFUSABLE_GROUPS: list[set[str]] = [
    {"white", "silver", "grey"},
    {"black", "grey"},
    {"blue", "green"},
]
TYPE_CONFUSABLE_GROUPS: list[set[str]] = [
    {"car", "auto"},
    {"truck", "bus"},
]


@dataclass
class CorruptionConfig:
    # --- OCR / plate ---
    # Tuned (see tests/test_calibration.py) so whole-plate per-read accuracy
    # lands at the published 85-95% figure for field ANPR.
    p_char_correct: float = 0.99
    p_char_confusable: float = 0.85
    p_blank_correct: float = 0.995
    posterior_peak_mass_min: float = 0.55
    posterior_peak_mass_max: float = 0.85
    # Fraction of the non-peak mass that leaks to the read character's
    # visually-confusable group; the rest spreads uniformly over the whole
    # alphabet. 0.85, not 0.999 (see docs/decisions.md, "Day 1c"): raising
    # this to 0.999 in Day 1b starved every out-of-group character of real
    # probability mass to fix an on-disk storage problem, which is the wrong
    # layer to fix it in — an out-of-group OCR error must stay unlikely, not
    # become effectively impossible. Storage compactness is now handled by
    # the codec's explicit residual term (engine/contracts/codec.py), not by
    # distorting this probability model.
    posterior_group_leak: float = 0.85

    # --- occlusion (contiguous unread slots -> uniform posterior) ---
    p_occlude: float = 0.03
    occlude_run_len_min: int = 2
    occlude_run_len_max: int = 4

    # --- missed detection (camera never fires for the passage) ---
    p_miss: float = 0.05

    # --- appearance ---
    # Tuned jointly with sim.vehicles' hierarchical embedding weights (see
    # docs/decisions.md, "Day 1b") so appearance-only rank-1 retrieval lands
    # at the published ~70% figure at benchmark (1500-identity) scale *and*
    # does not collapse at full city (20k-identity) scale.
    embedding_noise_sigma_min: float = 0.130
    embedding_noise_sigma_max: float = 0.142

    # --- colour / type attribute confusion ---
    color_p_correct: float = 0.8
    type_p_correct: float = 0.85
    attr_confidence_min: float = 0.55
    attr_confidence_max: float = 0.95

    # --- clone injection (consumed by sim.vehicles.generate_vehicles_and_journeys) ---
    clone_fraction: float = 0.02


def _char_groups_in_alphabet(char: str, alphabet: list[str]) -> set[str]:
    members: set[str] = set()
    alpha_set = set(alphabet)
    for group in CONFUSABLE_GROUPS:
        if char in group:
            members |= group & alpha_set
    members.discard(char)
    return members


def _sample_read_char(
    true_char: str, alphabet: list[str], rng: random.Random, cfg: CorruptionConfig
) -> str:
    if true_char == BLANK:
        if rng.random() < cfg.p_blank_correct:
            return BLANK
        others = [a for a in alphabet if a != BLANK]
        return rng.choice(others) if others else BLANK

    if rng.random() < cfg.p_char_correct:
        return true_char

    confusable = _char_groups_in_alphabet(true_char, alphabet)
    if confusable and rng.random() < cfg.p_char_confusable:
        return rng.choice(list(confusable))

    others = [a for a in alphabet if a != true_char]
    return rng.choice(others) if others else true_char


def _build_posterior(
    read_char: str, alphabet: list[str], rng: random.Random, cfg: CorruptionConfig
) -> SlotPosterior:
    """Build a peaked-but-honest posterior around `read_char`. Most of the
    non-peak mass goes to the read char's real confusable group (if any);
    the remainder spreads uniformly over the rest of the alphabet. Characters
    with no defined confusable group (e.g. J/K/W/X/Y) get no group leak at
    all — their non-peak mass spreads uniformly over the whole alphabet
    instead. (Day 1b synthesised a fake 2-character leak-set for this case to
    keep the on-disk encoding tiny; that conflated a storage concern with the
    probability model and was reverted in Day 1c — see docs/decisions.md.
    Storage compactness is now the codec's job via an explicit residual
    term.)"""
    peak_mass = rng.uniform(cfg.posterior_peak_mass_min, cfg.posterior_peak_mass_max)
    remaining_mass = 1.0 - peak_mass

    leak_targets = _char_groups_in_alphabet(read_char, alphabet)

    probs: dict[str, float] = {read_char: peak_mass}

    if leak_targets:
        leak_mass_total = remaining_mass * cfg.posterior_group_leak
        rest_mass_total = remaining_mass - leak_mass_total
        per_leak = leak_mass_total / len(leak_targets)
        for g in leak_targets:
            probs[g] = per_leak
    else:
        rest_mass_total = remaining_mass

    rest_symbols = [a for a in alphabet if a != read_char and a not in leak_targets]
    if rest_symbols:
        per_rest = rest_mass_total / len(rest_symbols)
        for s in rest_symbols:
            probs[s] = per_rest
    else:
        probs[read_char] += rest_mass_total

    # Renormalise to absorb float drift.
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return SlotPosterior(probs=probs)


def corrupt_plate(
    true_plate: str, rng: random.Random, cfg: CorruptionConfig
) -> list[SlotPosterior]:
    """Corrupt a 10-char canonical true plate into a list of 10 SlotPosteriors."""
    posteriors = []
    for slot_idx, true_char in enumerate(true_plate):
        alphabet = PLATE_SLOTS[slot_idx]
        read_char = _sample_read_char(true_char, alphabet, rng, cfg)
        posteriors.append(_build_posterior(read_char, alphabet, rng, cfg))
    return posteriors


def apply_occlusion(
    posteriors: list[SlotPosterior], rng: random.Random, cfg: CorruptionConfig
) -> list[SlotPosterior]:
    """With probability p_occlude, blank out a contiguous run of slots to a
    uniform (uninformative) posterior."""
    if rng.random() >= cfg.p_occlude:
        return posteriors
    n = len(posteriors)
    run_len = rng.randint(cfg.occlude_run_len_min, min(cfg.occlude_run_len_max, n))
    start = rng.randint(0, n - run_len)
    out = list(posteriors)
    for i in range(start, start + run_len):
        out[i] = SlotPosterior.uniform(PLATE_SLOTS[i])
    return out


def _confuse_categorical(
    true_value: str,
    groups: list[set[str]],
    all_values: list[str],
    p_correct: float,
    rng: random.Random,
    confidence_min: float,
    confidence_max: float,
) -> tuple[str, float]:
    if rng.random() < p_correct:
        observed = true_value
    else:
        confusable: set[str] = set()
        for g in groups:
            if true_value in g:
                confusable |= g
        confusable.discard(true_value)
        if confusable:
            observed = rng.choice(list(confusable))
        else:
            others = [v for v in all_values if v != true_value]
            observed = rng.choice(others) if others else true_value

    confidence = rng.uniform(confidence_min, confidence_max)
    return observed, confidence


def corrupt_embedding(latent: list[float], sigma: float, rng: random.Random) -> list[float]:
    noisy = [x + rng.gauss(0, sigma) for x in latent]
    norm = sum(x * x for x in noisy) ** 0.5 or 1.0
    return [x / norm for x in noisy]


class Corruptor:
    """Stateful corruption pipeline: ground-truth passages -> DetectionEvents.

    Deterministic given the seed passed at construction.
    """

    def __init__(self, city: CityConfig, seed: int, config: CorruptionConfig | None = None):
        self.config = config or CorruptionConfig()
        self._rng = random.Random(f"{seed}:corruption")
        self._event_counter = 0
        self._camera_sigmas: dict[str, float] = {}
        cam_rng = random.Random(f"{seed}:camera-sigma")
        for cam in city.cameras:
            self._camera_sigmas[cam.camera_id] = cam_rng.uniform(
                self.config.embedding_noise_sigma_min, self.config.embedding_noise_sigma_max
            )

    def _next_event_id(self) -> str:
        eid = f"evt_{self._event_counter:08d}"
        self._event_counter += 1
        return eid

    def corrupt_passage(self, passage: Passage, vehicle: Vehicle) -> DetectionEvent | None:
        cfg = self.config
        rng = self._rng

        if rng.random() < cfg.p_miss:
            return None

        posteriors = corrupt_plate(vehicle.true_plate, rng, cfg)
        posteriors = apply_occlusion(posteriors, rng, cfg)
        plate_argmax = "".join(sp.argmax() for sp in posteriors)
        plate_confidence = sum(sp.probs[sp.argmax()] for sp in posteriors) / len(posteriors)

        sigma = self._camera_sigmas.get(passage.camera_id, cfg.embedding_noise_sigma_max)
        embedding = corrupt_embedding(vehicle.embedding, sigma, rng)

        color, color_conf = _confuse_categorical(
            vehicle.color,
            COLOR_CONFUSABLE_GROUPS,
            COLORS,
            cfg.color_p_correct,
            rng,
            cfg.attr_confidence_min,
            cfg.attr_confidence_max,
        )
        vtype, type_conf = _confuse_categorical(
            vehicle.vehicle_type,
            TYPE_CONFUSABLE_GROUPS,
            VEHICLE_TYPES,
            cfg.type_p_correct,
            rng,
            cfg.attr_confidence_min,
            cfg.attr_confidence_max,
        )

        return DetectionEvent(
            event_id=self._next_event_id(),
            camera_id=passage.camera_id,
            timestamp=passage.timestamp,
            plate_posterior=posteriors,
            plate_argmax=plate_argmax,
            plate_confidence=plate_confidence,
            embedding=embedding,
            attributes=VehicleAttributes(
                color=color,
                vehicle_type=vtype,
                color_confidence=color_conf,
                type_confidence=type_conf,
            ),
            crop_uri=None,
            source="sim",
            gt_vehicle_id=vehicle.gt_vehicle_id,
        )
