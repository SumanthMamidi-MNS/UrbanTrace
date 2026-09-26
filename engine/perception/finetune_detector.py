"""STEP D of the detector fix: fine-tune the approved YOLO plate detector
(Koushim/yolov8-license-plate-detection weights as init) on Indian SCENE
images. Reached only because measured recall on video_frames_boxes stayed
poor (0.17-0.31) across every RGB-fix / higher-resolution / tiled-inference
combination (steps A-C; see eval/reports/detector_eval.json) -- the
detector itself needs to see more Indian scene photos, not just be run
differently.

Training data (three sources, combined into one YOLO-format dataset under
C:/sutra-data/ocr/detector_finetune/):
  - plate_boxes_2k's 181 scene images (`0000xxxx.jpg`) -- YOLO labels
    already exist in the source data, used as-is. (The other 1,902
    already-cropped `License (N).png` images are never used here either --
    they would teach the detector "the whole image is the plate", which is
    wrong for a detector that has to find a plate INSIDE a scene.)
  - 2 of video_frames_boxes' 3 videos (`vid-1`, `vid-2`) -- SPLIT BY VIDEO,
    never by frame: frames within one video are near-duplicates (the same
    plates a few frames apart), so holding out individual frames would leak
    the same scenes into both train and test. `vid-3` (50 frames) is held
    out ENTIRELY as this fine-tune's own test set and never enters
    training. This is a SMALL held-out set (one video, 50 frames) -- stated
    plainly, not a claim of a robust test.
  - indian_vehicle_xml images belonging to the OCR TRAIN split ONLY.
    Determined by re-running engine.perception.build_real_set's own
    split-by-plate logic (`_collect_candidates` + `assign_split`) --
    NEVER reverse-engineered from the OCR crop filenames -- so this is
    guaranteed to select exactly the same images build_real_set.py itself
    calls "train", and never an OCR val/test image (those must stay unseen
    for the end-to-end OCR evaluation the lead runs separately). Pascal-VOC
    `<bndbox>` boxes are converted to YOLO format.

A small held-out slice of this combined pool (last ~10%, deterministic
shuffle) serves as ultralytics' own train-time validation split for
early-stopping/monitoring; it is drawn from the SAME sources as training
data and is NOT a substitute for the vid-3 held-out test set above.

Runs under the SEPARATE OCR venv (torch/ultralytics). Batch size kept
modest by default -- this shares the GPU with a concurrent OCR fine-tune.
"""

import argparse
import random
import shutil
from pathlib import Path

from engine.perception.build_real_set import (
    DEFAULT_RAW_ROOT as XML_ROOT_DEFAULT,
)
from engine.perception.build_real_set import (
    BuildStats,
    _collect_candidates,
    assign_split,
)

DEFAULT_PLATE_BOXES_2K = Path("C:/sutra-data/ocr/raw/plate_boxes_2k")
DEFAULT_VIDEO_FRAMES_BOXES = Path("C:/sutra-data/ocr/raw/video_frames_boxes")
DEFAULT_WEIGHTS = Path("C:/sutra-data/ocr/weights/best.pt")
DEFAULT_DATASET_DIR = Path("C:/sutra-data/ocr/detector_finetune")
DEFAULT_RUN_DIR = Path("C:/sutra-data/ocr/runs/detector_finetune")

TRAIN_VIDEOS = ["vid-1", "vid-2"]
HOLDOUT_VIDEO = "vid-3"
VAL_FRACTION = 0.10
SHUFFLE_SEED = 42


def voc_box_to_yolo_line(box: tuple[float, float, float, float], img_w: int, img_h: int) -> str:
    xmin, ymin, xmax, ymax = box
    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def collect_plate_boxes_2k_scene_items(root: Path) -> list[tuple[Path, Path | None]]:
    """(image_path, label_txt_path) for the 181 numeric-named scene images."""
    img_dir, lbl_dir = root / "images", root / "labels"
    items: list[tuple[Path, Path | None]] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png") or not img_path.stem.isdigit():
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        items.append((img_path, lbl_path if lbl_path.exists() else None))
    return items


def collect_video_items(root: Path, video_names: list[str]) -> list[tuple[Path, Path | None]]:
    items: list[tuple[Path, Path | None]] = []
    for vid in video_names:
        vid_dir = root / vid
        images = sorted(
            p for p in vid_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for img_path in images:
            lbl_path = img_path.with_suffix(".txt")
            items.append((img_path, lbl_path if lbl_path.exists() else None))
    return items


def collect_xml_train_items(xml_root: Path) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """(image_path, VOC box) for every indian_vehicle_xml candidate whose
    plate was assigned to the OCR TRAIN split by build_real_set's own
    split-by-plate logic -- re-run here, never reverse-engineered."""
    stats = BuildStats()
    by_plate = _collect_candidates(xml_root, stats)
    items = []
    for plate, candidates in by_plate.items():
        if assign_split(plate) != "train":
            continue
        for cand in candidates:
            items.append((cand.image_path, cand.box))
    return items


def _write_yolo_pair(
    img_path: Path, label_lines: list[str], out_img_dir: Path, out_lbl_dir: Path, dest_stem: str
) -> None:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    dest_img = out_img_dir / f"{dest_stem}{img_path.suffix.lower()}"
    shutil.copy2(img_path, dest_img)
    (out_lbl_dir / f"{dest_stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")


def build_dataset(
    dataset_dir: Path,
    plate_boxes_2k_root: Path = DEFAULT_PLATE_BOXES_2K,
    video_frames_boxes_root: Path = DEFAULT_VIDEO_FRAMES_BOXES,
    xml_root: Path = XML_ROOT_DEFAULT,
) -> dict:
    """Builds <dataset_dir>/{images,labels}/{train,val}/ and
    <dataset_dir>/holdout_vid3/{images,labels}/ (the held-out test set,
    kept separate, never referenced by data.yaml). Returns a small stats
    dict."""
    from PIL import Image

    pool: list[tuple[str, Path, list[str]]] = []  # (source_tag, image_path, yolo_label_lines)

    for img_path, lbl_path in collect_plate_boxes_2k_scene_items(plate_boxes_2k_root):
        lines = lbl_path.read_text(encoding="utf-8").splitlines() if lbl_path else []
        pool.append(("plate_boxes_2k_scene", img_path, lines))

    for img_path, lbl_path in collect_video_items(video_frames_boxes_root, TRAIN_VIDEOS):
        lines = lbl_path.read_text(encoding="utf-8").splitlines() if lbl_path else []
        pool.append(("video_train", img_path, lines))

    for img_path, box in collect_xml_train_items(xml_root):
        with Image.open(img_path) as im:
            w, h = im.size
        pool.append(("xml_ocr_train", img_path, [voc_box_to_yolo_line(box, w, h)]))

    rng = random.Random(SHUFFLE_SEED)
    order = list(range(len(pool)))
    rng.shuffle(order)
    n_val = max(int(len(order) * VAL_FRACTION), 1)
    val_indices = set(order[:n_val])

    counts = {"train": 0, "val": 0, "by_source": {}}
    for i, (source_tag, img_path, lines) in enumerate(pool):
        split = "val" if i in val_indices else "train"
        counts[split] += 1
        counts["by_source"].setdefault(source_tag, 0)
        counts["by_source"][source_tag] += 1
        _write_yolo_pair(
            img_path,
            lines,
            dataset_dir / "images" / split,
            dataset_dir / "labels" / split,
            dest_stem=f"{source_tag}_{i:05d}",
        )

    # Held-out test set (vid-3): copied separately, NEVER referenced by
    # data.yaml, so it structurally cannot leak into training.
    holdout_dir = dataset_dir / "holdout_vid3"
    n_holdout = 0
    for img_path, lbl_path in collect_video_items(video_frames_boxes_root, [HOLDOUT_VIDEO]):
        lines = lbl_path.read_text(encoding="utf-8").splitlines() if lbl_path else []
        _write_yolo_pair(
            img_path,
            lines,
            holdout_dir / "images",
            holdout_dir / "labels",
            dest_stem=f"holdout_{n_holdout:05d}",
        )
        n_holdout += 1
    counts["holdout_vid3"] = n_holdout

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: plate\n",
        encoding="utf-8",
    )
    counts["data_yaml"] = str(data_yaml)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the YOLO plate detector on Indian scenes."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8, help="Kept modest -- shares the GPU.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="640 matches PlateDetector's default tile size, so training resolution "
        "is consistent with what the fine-tuned model will actually see at inference.",
    )
    parser.add_argument(
        "--build-only", action="store_true", help="Only build the dataset, don't train."
    )
    args = parser.parse_args()

    print(f"building dataset at {args.dataset_dir} ...", flush=True)
    counts = build_dataset(args.dataset_dir)
    print(counts, flush=True)

    if args.build_only:
        return

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    model.train(
        data=str(args.dataset_dir / "data.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(args.run_dir.parent),
        name=args.run_dir.name,
        exist_ok=True,
        # amp=False: ultralytics' AMP sanity check auto-downloads a reference
        # model (yolo26n.pt) from GitHub on first use -- an unauthorized
        # download this project's rules forbid. Disabling AMP avoids the
        # check entirely (costs some training speed/memory headroom, not
        # correctness) rather than relying on a cache that may not persist.
        amp=False,
    )
    print(f"training complete. weights in {args.run_dir / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
