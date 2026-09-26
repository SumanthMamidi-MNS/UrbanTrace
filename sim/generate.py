"""CLI entry point: generate a synthetic city + vehicle population + noisy
DetectionEvent stream.

Usage:
    python -m sim.generate --cameras 50 --vehicles 20000 --hours 24 --seed 42 --out data/run1/
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from engine.contracts.city import CityConfig
from engine.contracts.codec import encode_event_compact
from engine.contracts.events import DetectionEvent
from sim.city import generate_city
from sim.congestion import CongestionConfig, congestion_config_to_dict
from sim.corruption import CorruptionConfig, Corruptor
from sim.vehicles import EMBEDDING_DIM, Journey, Vehicle, generate_vehicles_and_journeys

app = typer.Typer(add_completion=False)
console = Console()

_DEFAULT_OUT = Path("data/run1/")


@dataclass
class GeneratedDataset:
    city: CityConfig
    vehicles: list[Vehicle]
    journeys: list[Journey]
    events: list[DetectionEvent]


def generate_dataset_with_city(
    city: CityConfig,
    n_vehicles: int,
    hours: int,
    seed: int,
    clone_fraction: float = 0.02,
    corruption_config: CorruptionConfig | None = None,
    near_miss_fraction: float = 0.0,
    clone_route_overlap: float = 0.5,
    congestion: bool = False,
    congestion_config: CongestionConfig | None = None,
) -> GeneratedDataset:
    """Like `generate_dataset`, but reuses an already-built city instead of
    deriving one from `seed`. This is what lets a train/held-out split share
    one camera network while drawing independent vehicle populations and
    OCR/appearance noise realisations — see
    `engine.calibration.fit_priors.generate_train_holdout_datasets` and
    docs/decisions.md, "Day 2": architecture.md §10 names seed-sharing as a
    risk, but a kinematic model fitted on one city's camera pairs is only
    meaningful evaluated on candidate pairs from *that same* camera network,
    so the two splits must share a city while still using different data
    seeds."""
    cfg = corruption_config or CorruptionConfig(clone_fraction=clone_fraction)
    vehicle_list, journeys = generate_vehicles_and_journeys(
        city=city,
        n_vehicles=n_vehicles,
        hours=hours,
        seed=seed,
        clone_fraction=cfg.clone_fraction,
        near_miss_fraction=near_miss_fraction,
        clone_route_overlap=clone_route_overlap,
        congestion=congestion,
        congestion_config=congestion_config,
    )
    vehicles_by_id = {v.gt_vehicle_id: v for v in vehicle_list}

    corruptor = Corruptor(city=city, seed=seed, config=cfg)
    events: list[DetectionEvent] = []
    for journey in journeys:
        vehicle = vehicles_by_id[journey.gt_vehicle_id]
        for passage in journey.passages:
            event = corruptor.corrupt_passage(passage, vehicle)
            if event is not None:
                events.append(event)

    events.sort(key=lambda e: (e.timestamp, e.camera_id, e.event_id))

    return GeneratedDataset(city=city, vehicles=vehicle_list, journeys=journeys, events=events)


def generate_dataset(
    n_cameras: int,
    n_vehicles: int,
    hours: int,
    seed: int,
    clone_fraction: float = 0.02,
    corruption_config: CorruptionConfig | None = None,
    near_miss_fraction: float = 0.0,
    clone_route_overlap: float = 0.5,
    congestion: bool = False,
    congestion_config: CongestionConfig | None = None,
) -> GeneratedDataset:
    """Run the full pipeline (city -> vehicles/journeys -> corrupted events)
    and return everything in memory. Deterministic given `seed`.

    `congestion` defaults to False here (and in
    `generate_vehicles_and_journeys`) so every existing caller/test keeps
    reproducing pre-existing output byte-for-byte; the CLI below defaults it
    to True for newly-generated datasets (see docs/decisions.md)."""
    city = generate_city(n_cameras=n_cameras, seed=seed)
    return generate_dataset_with_city(
        city=city,
        n_vehicles=n_vehicles,
        hours=hours,
        seed=seed,
        clone_fraction=clone_fraction,
        corruption_config=corruption_config,
        near_miss_fraction=near_miss_fraction,
        clone_route_overlap=clone_route_overlap,
        congestion=congestion,
        congestion_config=congestion_config,
    )


def write_dataset(dataset: GeneratedDataset, out: Path, run_config: dict) -> None:
    """Write the compact on-disk format: sparse-posterior events.jsonl plus a
    companion float16 embeddings.npy (see docs/decisions.md, "Day 1b"). Read
    it back with engine.contracts.store.EventStore — never parse events.jsonl
    directly, since `embedding` is not in it."""
    out.mkdir(parents=True, exist_ok=True)

    n = len(dataset.events)
    embeddings = np.empty((n, EMBEDDING_DIM), dtype=np.float16)

    events_path = out / "events.jsonl"
    with events_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, event in enumerate(dataset.events):
            embeddings[i] = np.asarray(event.embedding, dtype=np.float32)
            compact = encode_event_compact(event, embedding_row=i)
            f.write(json.dumps(compact, separators=(",", ":")))
            f.write("\n")

    np.save(out / "embeddings.npy", embeddings)

    ground_truth = {
        "vehicles": {
            v.gt_vehicle_id: {
                "true_plate": v.true_plate,
                "color": v.color,
                "vehicle_type": v.vehicle_type,
                "is_clone": v.is_clone,
                "clone_of": v.clone_of,
            }
            for v in dataset.vehicles
        },
        "journeys": {
            j.gt_vehicle_id: {
                "camera_sequence": j.camera_sequence,
                "passages": [
                    {"camera_id": p.camera_id, "timestamp": p.timestamp.isoformat()}
                    for p in j.passages
                ],
            }
            for j in dataset.journeys
        },
    }
    (out / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2, sort_keys=True), encoding="utf-8"
    )

    (out / "city.json").write_text(
        json.dumps(dataset.city.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )

    (out / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )


@app.command()
def main(
    cameras: int = typer.Option(50, help="Number of cameras to place in the city."),
    vehicles: int = typer.Option(20000, help="Number of vehicles to generate."),
    hours: int = typer.Option(24, help="Simulated time span, in hours."),
    seed: int = typer.Option(42, help="Random seed (determinism)."),
    out: Path = typer.Option(_DEFAULT_OUT, help="Output directory."),
    clone_fraction: float = typer.Option(
        0.02, help="Fraction of vehicles that are plate clones."
    ),
    congestion: bool = typer.Option(
        True,
        help=(
            "Density-dependent (BPR) link travel times, on by default for new "
            "datasets. --no-congestion reproduces the pre-congestion free-flow "
            "behaviour byte-for-byte."
        ),
    ),
) -> None:
    console.print(f"[bold]Generating city[/bold]: {cameras} cameras, seed={seed}")
    console.print(f"[bold]Generating vehicles + journeys[/bold]: {vehicles} vehicles, {hours}h")
    console.print(f"[bold]Congestion[/bold]: {'on (BPR)' if congestion else 'off (free flow)'}")
    console.print("[bold]Corrupting passages into DetectionEvents[/bold]")

    corruption_config = CorruptionConfig(clone_fraction=clone_fraction)
    congestion_config = CongestionConfig(enabled=congestion) if congestion else None
    dataset = generate_dataset(
        n_cameras=cameras,
        n_vehicles=vehicles,
        hours=hours,
        seed=seed,
        clone_fraction=clone_fraction,
        corruption_config=corruption_config,
        congestion=congestion,
        congestion_config=congestion_config,
    )

    run_config = {
        "cameras": cameras,
        "vehicles": vehicles,
        "hours": hours,
        "seed": seed,
        "clone_fraction": clone_fraction,
        "corruption": corruption_config.__dict__,
        "congestion": congestion,
        "congestion_config": congestion_config_to_dict(congestion_config)
        if congestion_config is not None
        else None,
    }
    write_dataset(dataset, out, run_config)

    events_path = out / "events.jsonl"
    console.print(f"[green]Done[/green]: {len(dataset.events)} events written to {events_path}")


if __name__ == "__main__":
    app()
