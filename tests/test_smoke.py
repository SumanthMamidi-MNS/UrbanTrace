import time

from sim.generate import generate_dataset


def test_small_scale_smoke_run_is_fast():
    start = time.perf_counter()
    ds = generate_dataset(n_cameras=5, n_vehicles=100, hours=24, seed=42)
    elapsed = time.perf_counter() - start

    assert ds.events
    assert elapsed < 10.0, f"smoke run took {elapsed:.2f}s, expected < 10s"
