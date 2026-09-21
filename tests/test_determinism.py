from pathlib import Path

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
