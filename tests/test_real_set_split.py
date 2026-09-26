"""engine.perception.build_real_set: the split-by-plate guarantee.

Runs under the MAIN venv (build_real_set.py needs only Pillow + stdlib xml
+ the grammar/contracts modules -- no torch). Exercises the real `build()`
pipeline end to end against a small synthetic XML+image fixture constructed
in a temp dir, not just the bare hashing function -- the property that
matters is that every ROW belonging to a given plate lands in the same
split, which only a real pipeline run can actually confirm.
"""

from pathlib import Path

from PIL import Image

from engine.perception.build_real_set import SOURCES, assign_split, build

# A repeated plate (simulating e.g. multiple video frames of one car) plus
# several distinct plates, spread across all three source subfolders.
FIXTURE_PLATES = [
    ("MH12AB1234", 6),  # repeats heavily, like the real MH47Y1124 example
    ("DL05CD5678", 1),
    ("KA03EF0099", 1),
    ("TN09GH4321", 1),
    ("WB22IJ0007", 1),
    ("RJ14KL8899", 1),
    ("GJ01MN1111", 1),
    ("PB08OP2222", 1),
]


def _write_fixture_xml(xml_path: Path, image_name: str, plate_text: str) -> None:
    xml_path.write_text(
        f"""<annotation>
  <filename>{image_name}</filename>
  <object>
    <name>{plate_text}</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>10</ymin>
      <xmax>110</xmax>
      <ymax>40</ymax>
    </bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )


def _build_fixture(raw_root: Path) -> None:
    counter = 0
    for i, (plate, n_repeats) in enumerate(FIXTURE_PLATES):
        source = SOURCES[i % len(SOURCES)]
        src_dir = raw_root / source
        src_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(n_repeats):
            stem = f"img_{counter:04d}"
            counter += 1
            img_path = src_dir / f"{stem}.jpg"
            Image.new("RGB", (200, 80), (200, 200, 200)).save(img_path)
            _write_fixture_xml(src_dir / f"{stem}.xml", img_path.name, plate)


def _stub_score_reading(crop_rgb, label: str) -> dict:
    """A trivial stand-in for make_forced_alignment_scorer's real,
    torch-based scorer: always reads the whole crop as one line. This test
    is about the split-by-plate guarantee, not forced alignment itself, so
    it stays torch-free and runs under the main venv."""
    return {
        "layout": "one_line",
        "text_line1": label,
        "text_line2": "",
        "one_line_loss": 0.0,
        "two_row_candidates": {},
        "best_split_k": None,
        "best_two_row_loss": float("inf"),
    }


def test_build_produces_zero_plate_overlap_between_splits(tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_dir = tmp_path / "real"
    _build_fixture(raw_root)

    stats = build(out_dir, raw_root, _stub_score_reading)

    assert stats["excluded"]["total"] == 0, stats["excluded"]
    assert stats["kept_unique_plates"] == len(FIXTURE_PLATES)

    plates_by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        csv_path = out_dir / split / "labels.csv"
        assert csv_path.exists()
        import csv as csv_module

        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv_module.DictReader(f))
        plates_by_split[split] = {r["canonical"] for r in rows}

    train_plates = plates_by_split["train"]
    val_plates = plates_by_split["val"]
    test_plates = plates_by_split["test"]

    assert not (train_plates & val_plates), "train/val plate overlap"
    assert not (train_plates & test_plates), "train/test plate overlap"
    assert not (val_plates & test_plates), "val/test plate overlap"

    # Every image of the heavily-repeated plate must land in ONE split.
    repeated_plate = FIXTURE_PLATES[0][0]
    owning_splits = [s for s, plates in plates_by_split.items() if repeated_plate in plates]
    assert len(owning_splits) == 1, f"plate {repeated_plate} split across {owning_splits}"
    n_images_for_repeated = sum(
        1
        for split in ("train", "val", "test")
        for row in _read_csv_rows(out_dir / split / "labels.csv")
        if row["canonical"] == repeated_plate
    )
    assert n_images_for_repeated == FIXTURE_PLATES[0][1]


def _read_csv_rows(csv_path: Path) -> list[dict]:
    import csv as csv_module

    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv_module.DictReader(f))


def test_assign_split_is_deterministic_and_covers_all_three_buckets():
    # Same plate always assigns to the same split, regardless of call order.
    plates = [f"MH{n:02d}AB{n:04d}" for n in range(1, 60)]
    first_pass = {p: assign_split(p) for p in plates}
    second_pass = {p: assign_split(p) for p in reversed(plates)}
    assert first_pass == second_pass

    seen_splits = set(first_pass.values())
    assert seen_splits == {"train", "val", "test"}, seen_splits
