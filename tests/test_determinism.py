from pathlib import Path

from sim.congestion import CongestionConfig, congestion_config_to_dict
from sim.generate import generate_dataset, write_dataset


def test_same_seed_byte_identical_events_jsonl(tmp_path: Path):
    ds_a = generate_dataset(n_cameras=10, n_vehicles=200, hours=6, seed=42)
    ds_b = generate_dataset(n_cameras=10, n_vehicles=200, hours=6, seed=42)

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    write_dataset(ds_a, out_a, run_config={"seed": 42})
    write_dataset(ds_b, out_b, run_config={"seed": 42})

    bytes_a = (out_a / "events.jsonl").read_bytes()
    bytes_b = (out_b / "events.jsonl").read_bytes()
    assert bytes_a == bytes_b
    assert len(bytes_a) > 0


def test_different_seed_differs():
    ds_a = generate_dataset(n_cameras=10, n_vehicles=200, hours=6, seed=42)
    ds_b = generate_dataset(n_cameras=10, n_vehicles=200, hours=6, seed=43)

    lines_a = [e.model_dump_json() for e in ds_a.events]
    lines_b = [e.model_dump_json() for e in ds_b.events]
    assert lines_a != lines_b


def test_same_seed_byte_identical_events_jsonl_congestion_on(tmp_path: Path):
    """The BPR congestion knob (docs/decisions.md, "Sim congestion
    calibration") must be just as deterministic as the free-flow path: same
    seed, same two-pass (free-flow volumes -> congested re-traverse)
    pipeline, same bytes on disk."""
    ds_a = generate_dataset(n_cameras=10, n_vehicles=200, hours=6, seed=42, congestion=True)
    ds_b = generate_dataset(n_cameras=10, n_vehicles=200, hours=6, seed=42, congestion=True)

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    write_dataset(ds_a, out_a, run_config={"seed": 42, "congestion": True})
    write_dataset(ds_b, out_b, run_config={"seed": 42, "congestion": True})

    bytes_a = (out_a / "events.jsonl").read_bytes()
    bytes_b = (out_b / "events.jsonl").read_bytes()
    assert bytes_a == bytes_b
    assert len(bytes_a) > 0


def test_congestion_on_differs_from_congestion_off():
    """Turning congestion on must actually change the generated timestamps
    (otherwise the BPR pass would be a no-op) while leaving the OFF path's
    output untouched -- the point of the knob."""
    ds_off = generate_dataset(n_cameras=10, n_vehicles=200, hours=6, seed=42, congestion=False)
    ds_on = generate_dataset(n_cameras=10, n_vehicles=200, hours=6, seed=42, congestion=True)

    lines_off = [e.model_dump_json() for e in ds_off.events]
    lines_on = [e.model_dump_json() for e in ds_on.events]
    assert lines_off != lines_on


def test_run_pipeline_train_split_matches_eval_dataset_congestion(tmp_path: Path, monkeypatch):
    """Guard against the correctness trap: if the EVALUATION dataset was
    generated with congestion on, `eval/run_pipeline.py`'s train split (used
    to fit the kinematic/appearance/plate priors) must be generated with
    congestion on too, or the model is fit on free-flow travel times and
    evaluated on congested ones with nothing in the output revealing it. A
    dataset generated before the congestion knob existed (no `congestion`
    key in config.json) must still fall back to congestion=False, exactly
    reproducing pre-existing behaviour."""
    from eval import run_pipeline as rp

    ds = generate_dataset(n_cameras=6, n_vehicles=80, hours=2, seed=100, congestion=True)
    data_dir = tmp_path / "congested"
    write_dataset(
        ds,
        data_dir,
        run_config={
            "cameras": 6,
            "vehicles": 80,
            "hours": 2,
            "seed": 100,
            "clone_fraction": 0.02,
            "congestion": True,
            "congestion_config": congestion_config_to_dict(CongestionConfig(enabled=True)),
        },
    )

    captured: dict = {}
    original = rp.generate_dataset_with_city

    def spy(*args, **kwargs):
        captured["congestion"] = kwargs.get("congestion")
        captured["congestion_config"] = kwargs.get("congestion_config")
        return original(*args, **kwargs)

    monkeypatch.setattr(rp, "generate_dataset_with_city", spy)

    out_dir = tmp_path / "pipeline"
    rp.main(
        data=data_dir,
        out=out_dir,
        train_vehicles=50,
        train_seed_offset=9001,
        max_hours=None,
        link_bias=0.0,
    )

    assert captured["congestion"] is True
    assert captured["congestion_config"] is not None
    assert captured["congestion_config"].enabled is True

    # Fallback: a dataset whose config lacks the `congestion` key (e.g.
    # generated before the knob existed) must resolve to congestion=False.
    old_config = {
        "cameras": 6,
        "vehicles": 80,
        "hours": 2,
        "seed": 100,
        "clone_fraction": 0.02,
    }
    congestion, congestion_config = rp._congestion_from_config(old_config)
    assert congestion is False
    assert congestion_config is None
