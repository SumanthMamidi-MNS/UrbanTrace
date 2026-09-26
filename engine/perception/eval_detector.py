"""Evaluate the approved YOLO plate detector (engine.perception.detect,
`<data_dir>/ocr/weights/best.pt`) on three real, boxed datasets: precision,
recall (at a fixed operating confidence) and mAP@0.5 (threshold-independent,
standard PASCAL VOC-style average precision), reported SEPARATELY per set
-- the PRD asks for plates "in multi-lane traffic streams", so detector
accuracy matters as its own claim, not folded into OCR numbers.

Runs under the SEPARATE OCR venv (torch/ultralytics, via
engine.perception.detect.PlateDetector). Inference only -- no training, no
gradient, a few thousand single-image forward passes.

Datasets (see `<data_dir>/ocr/raw/README.md`, where `<data_dir>` is
`URBANTRACE_DATA_DIR`, see `engine.paths`):
  - `plate_boxes_2k/`: ~2,083 real images, YOLO-format `.txt` labels
    (`images/`, `labels/`; an image with no label file has zero plates).
    IMPORTANT: 1,902 of these (`License (N).png`) are ALREADY-CROPPED plate
    images (e.g. 263x81px) whose GT box is essentially the whole image --
    a plate detector looks for a plate INSIDE a scene, so it structurally
    cannot "find" anything in an image that already IS the plate. Only the
    remaining ~181 (`0000xxxx.jpg`) are real scene photos. `is_precropped`
    below excludes the crops by a filename-independent, robust rule (GT box
    area > 60% of the image, OR image area < 40,000 px^2 -- verified
    against the filename-based ground truth in this exact dataset to
    exclude all 1,902 true crops with zero false negatives, at the cost of
    3 borderline scene photos: two tight close-ups whose own GT box is
    itself large (00000182.jpg, 00000306.jpg), and one small-but-genuine
    scene photo (00000286.jpg, 222x166). `00000234.jpg` -- a product photo
    of Odia-script plates, legitimate but a rare script/format edge case --
    is correctly KEPT by this rule (2.25 MP image, GT box only 24% of it)
    and is included in the reported scene-only numbers.
  - `video_frames_boxes/`: 160 frames across `vid-1/2/3`, 1920x1080,
    YOLO-format `.txt` beside each frame -- the PRD's actual scenario
    ("plates in multi-lane traffic streams"): small, distant plates
    (~30px wide at full resolution).
  - `indian_vehicle_xml/`: the same 1,697-photo set build_real_set.py uses
    for OCR, but evaluated here for its Pascal-VOC `<bndbox>` as plate
    DETECTION ground truth (one box per image; label text is irrelevant to
    this script). Reuses `build_real_set.find_image_file` for the same
    mixed-case/double-extension image lookup rather than reimplementing it.

Metric methodology: for each image, all detections (from a low confidence
floor, to trace the full precision-recall curve) are greedily matched to
ground-truth boxes by descending confidence, IoU >= 0.5, each GT box
claimed by at most one detection (standard single-class PASCAL VOC
protocol). mAP@0.5 is the area under the resulting precision-recall curve
(monotonic precision envelope, integrated over recall). Precision/recall
are ALSO reported at a fixed operating threshold (`--conf-threshold`,
default 0.25, matching engine.perception.detect's own default) since that
is the number a deployment actually experiences, not just the
threshold-independent ranking metric.

Steps (each run and reported separately, see `main()`): (A) the RGB/BGR fix
in engine.perception.detect (no tiling, ultralytics' default imgsz);
(B) higher inference resolution (imgsz=1280, and each image's own native
size); (C) tiled inference (the new PlateDetector default), including
measured per-image wall-clock cost. Step D (fine-tuning), if run, is a
separate training script, not this one.

Output: eval/reports/detector_eval.json, one entry per dataset per step.
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from engine.paths import get_data_dir
from engine.perception.build_real_set import SOURCES as XML_SOURCES
from engine.perception.build_real_set import find_image_file
from engine.perception.detect import DEFAULT_WEIGHTS_PATH, PlateDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = REPO_ROOT / "eval" / "reports" / "detector_eval.json"

DEFAULT_PLATE_BOXES_2K = get_data_dir() / "ocr/raw/plate_boxes_2k"
DEFAULT_VIDEO_FRAMES_BOXES = get_data_dir() / "ocr/raw/video_frames_boxes"
DEFAULT_XML_ROOT = get_data_dir() / "ocr/raw/indian_vehicle_xml"

Box = tuple[float, float, float, float]

LOW_CONF_FLOOR = 0.001  # to trace the full precision-recall curve for mAP
IOU_THRESHOLD = 0.5

# plate_boxes_2k crop-vs-scene rule -- see module docstring. Validated
# against this exact dataset's two filename families (License (N).png vs
# 0000xxxx.jpg, used ONLY for that validation, never as part of the rule
# itself): excludes all 1,902 true crops with zero false negatives, at the
# cost of 3 borderline scene photos.
PRECROPPED_AREA_FRACTION_THRESHOLD = 0.6
PRECROPPED_MIN_IMAGE_AREA = 40_000


def iou_xyxy(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    inter = iw * ih
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Box:
    x0 = (cx - w / 2) * img_w
    y0 = (cy - h / 2) * img_h
    x1 = (cx + w / 2) * img_w
    y1 = (cy + h / 2) * img_h
    return (x0, y0, x1, y1)


def load_yolo_boxes(txt_path: Path, img_w: int, img_h: int) -> list[Box]:
    if not txt_path.exists():
        return []
    boxes = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, w, h = parts[:5]
        boxes.append(yolo_to_xyxy(float(cx), float(cy), float(w), float(h), img_w, img_h))
    return boxes


def is_precropped_plate_image(gt_boxes: list[Box], img_w: int, img_h: int) -> bool:
    """True if this image is already a tight plate crop (so a plate
    DETECTOR cannot meaningfully be evaluated on it -- see module
    docstring). Filename-independent: either the GT box already fills most
    of the frame, or the frame itself is tiny."""
    if img_w * img_h < PRECROPPED_MIN_IMAGE_AREA:
        return True
    if not gt_boxes:
        return False
    max_frac = max(((b[2] - b[0]) * (b[3] - b[1])) / (img_w * img_h) for b in gt_boxes)
    return max_frac > PRECROPPED_AREA_FRACTION_THRESHOLD


def load_plate_boxes_2k(
    root: Path, exclude_precropped: bool = True
) -> tuple[list[tuple[Path, list[Box]]], int]:
    """Returns (pairs, n_excluded_as_precropped)."""
    img_dir, lbl_dir = root / "images", root / "labels"
    pairs = []
    n_excluded = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = load_yolo_boxes(lbl_dir / (img_path.stem + ".txt"), w, h)
        if exclude_precropped and is_precropped_plate_image(boxes, w, h):
            n_excluded += 1
            continue
        pairs.append((img_path, boxes))
    return pairs, n_excluded


def load_video_frames_boxes(root: Path) -> list[tuple[Path, list[Box]]]:
    pairs = []
    for vid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            p for p in vid_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for img_path in images:
            with Image.open(img_path) as im:
                w, h = im.size
            boxes = load_yolo_boxes(img_path.with_suffix(".txt"), w, h)
            pairs.append((img_path, boxes))
    return pairs


def load_indian_vehicle_xml(root: Path) -> list[tuple[Path, list[Box]]]:
    pairs = []
    for source in XML_SOURCES:
        for xf in sorted((root / source).rglob("*.xml")):
            img_path = find_image_file(xf)
            if img_path is None:
                continue
            tree = ET.parse(xf)
            bbox_el = tree.find("object/bndbox")
            boxes: list[Box] = []
            if bbox_el is not None:
                boxes.append(
                    (
                        float(bbox_el.find("xmin").text),
                        float(bbox_el.find("ymin").text),
                        float(bbox_el.find("xmax").text),
                        float(bbox_el.find("ymax").text),
                    )
                )
            pairs.append((img_path, boxes))
    return pairs


def load_holdout_vid3(holdout_dir: Path) -> list[tuple[Path, list[Box]]]:
    """Loads engine.perception.finetune_detector's held-out vid-3 set
    (flat images/ + labels/, YOLO format) -- the STEP D test set that never
    enters training."""
    img_dir, lbl_dir = holdout_dir / "images", holdout_dir / "labels"
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = load_yolo_boxes(lbl_dir / (img_path.stem + ".txt"), w, h)
        pairs.append((img_path, boxes))
    return pairs


def match_detections(
    detections: list[tuple[Box, float]], gt_boxes: list[Box], iou_threshold: float
) -> list[tuple[float, bool]]:
    """One image's detections (box, confidence), greedily matched to GT
    boxes by descending confidence -- standard single-class PASCAL VOC
    protocol. Returns (confidence, is_true_positive) per detection."""
    matched_gt = [False] * len(gt_boxes)
    results = []
    for box, conf in sorted(detections, key=lambda d: -d[1]):
        best_iou, best_idx = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if matched_gt[gi]:
                continue
            iou = iou_xyxy(box, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt[best_idx] = True
            results.append((conf, True))
        else:
            results.append((conf, False))
    return results


def compute_ap(detections: list[tuple[float, bool]], n_gt: int) -> float:
    """Standard PASCAL VOC average precision: area under the
    monotonic-envelope precision-recall curve, integrated over recall."""
    if n_gt == 0:
        return 0.0
    if not detections:
        return 0.0
    ordered = sorted(detections, key=lambda d: -d[0])
    tp = np.array([1.0 if is_tp else 0.0 for _, is_tp in ordered])
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = 0.0
    for i in range(1, len(recall)):
        if recall[i] != recall[i - 1]:
            ap += (recall[i] - recall[i - 1]) * precision[i]
    return float(ap)


def evaluate_detector_on_dataset(
    detector: PlateDetector,
    image_gt_pairs: list[tuple[Path, list[Box]]],
    conf_threshold: float,
    iou_threshold: float = IOU_THRESHOLD,
    detect_kwargs: dict | None = None,
    measure_timing: bool = False,
) -> dict:
    """`detect_kwargs` (e.g. {"imgsz": 1280} or {"tiled": True}) are passed
    straight through to `detector.detect()` for every image -- this is how
    steps A/B/C below are actually distinguished. `measure_timing=True`
    times each `detect()` call (wall clock, GPU included) and reports the
    mean per-image cost; off by default since it adds a small per-call
    overhead of its own and most steps don't need it."""
    detect_kwargs = dict(detect_kwargs or {})
    native_imgsz = detect_kwargs.get("imgsz") == "native"
    if native_imgsz:
        del detect_kwargs["imgsz"]

    all_detections: list[tuple[float, bool]] = []
    n_gt_total = 0
    tp_at_thresh = fp_at_thresh = n_pred_at_thresh = 0
    timings_s: list[float] = []

    for img_path, gt_boxes in image_gt_pairs:
        image = np.asarray(Image.open(img_path).convert("RGB"))
        call_kwargs = dict(detect_kwargs)
        if native_imgsz:
            h, w = image.shape[:2]
            call_kwargs["imgsz"] = ((max(w, h) + 31) // 32) * 32
        if measure_timing:
            t0 = time.perf_counter()
            dets = detector.detect(image, conf_threshold=LOW_CONF_FLOOR, **call_kwargs)
            timings_s.append(time.perf_counter() - t0)
        else:
            dets = detector.detect(image, conf_threshold=LOW_CONF_FLOOR, **call_kwargs)
        det_tuples = [(d.box_xyxy, d.confidence) for d in dets]
        matched = match_detections(det_tuples, gt_boxes, iou_threshold)
        all_detections.extend(matched)
        n_gt_total += len(gt_boxes)

        at_thresh = [(c, is_tp) for c, is_tp in matched if c >= conf_threshold]
        tp_at_thresh += sum(1 for _, is_tp in at_thresh if is_tp)
        fp_at_thresh += sum(1 for _, is_tp in at_thresh if not is_tp)
        n_pred_at_thresh += len(at_thresh)

    map50 = compute_ap(all_detections, n_gt_total)
    precision = tp_at_thresh / n_pred_at_thresh if n_pred_at_thresh else 0.0
    recall = tp_at_thresh / n_gt_total if n_gt_total else 0.0

    result = {
        "n_images": len(image_gt_pairs),
        "n_gt_boxes": n_gt_total,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "n_pred_boxes_at_conf_threshold": n_pred_at_thresh,
        "precision_at_conf_threshold": precision,
        "recall_at_conf_threshold": recall,
        "map50": map50,
    }
    if measure_timing and timings_s:
        # Drop the first call (model/CUDA warmup) from the mean -- it is not
        # representative of steady-state per-frame cost.
        steady = timings_s[1:] or timings_s
        result["mean_ms_per_image"] = float(np.mean(steady) * 1000)
        result["first_call_ms"] = float(timings_s[0] * 1000)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the YOLO plate detector.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--plate-boxes-2k", type=Path, default=DEFAULT_PLATE_BOXES_2K)
    parser.add_argument("--video-frames-boxes", type=Path, default=DEFAULT_VIDEO_FRAMES_BOXES)
    parser.add_argument("--xml-root", type=Path, default=DEFAULT_XML_ROOT)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--finetuned-weights",
        type=Path,
        default=None,
        help=(
            "STEP D: also evaluate this checkpoint (engine.perception.finetune_detector's "
            "output) on the held-out vid-3 test set ONLY "
            "(<data_dir>/ocr/detector_finetune/holdout_vid3/), tiled inference, and "
            "compare against the approved (pre-fine-tune) weights on the SAME held-out set."
        ),
    )
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=get_data_dir() / "ocr/detector_finetune/holdout_vid3",
    )
    parser.add_argument(
        "--only-step-d",
        action="store_true",
        help="Skip steps A/B/C (already in --out from a previous run) and only run/append "
        "the --finetuned-weights held-out comparison.",
    )
    args = parser.parse_args()

    detector = PlateDetector(weights_path=args.weights)

    scene_pairs, n_excluded_precropped = load_plate_boxes_2k(
        args.plate_boxes_2k, exclude_precropped=True
    )
    all_pairs_unfiltered, _ = load_plate_boxes_2k(args.plate_boxes_2k, exclude_precropped=False)
    video_pairs = load_video_frames_boxes(args.video_frames_boxes)
    xml_pairs = load_indian_vehicle_xml(args.xml_root)

    print(
        f"plate_boxes_2k: {len(all_pairs_unfiltered)} total, "
        f"{n_excluded_precropped} excluded as pre-cropped plate images, "
        f"{len(scene_pairs)} scene images kept for detector evaluation",
        flush=True,
    )

    if args.only_step_d and args.out.exists():
        report = json.loads(args.out.read_text(encoding="utf-8"))
        report["generated_at"] = datetime.now(UTC).isoformat()
        _maybe_run_step_d(report, args)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
        return

    report: dict = {
        "weights": str(args.weights),
        "conf_threshold": args.conf_threshold,
        "iou_threshold": IOU_THRESHOLD,
        "generated_at": datetime.now(UTC).isoformat(),
        "description": (
            "Detector evaluation, run in steps (see engine/perception/eval_detector.py's "
            "module docstring). plate_boxes_2k numbers below (all steps) are on the "
            f"{len(scene_pairs)} SCENE images only ({n_excluded_precropped} already-cropped "
            "plate images excluded -- see plate_boxes_2k_filter); video_frames_boxes numbers "
            f"are on all {len(video_pairs)} frames; indian_vehicle_xml on all "
            f"{len(xml_pairs)} images."
        ),
        "rgb_bug": {
            "verdict": "confirmed and fixed",
            "explanation": (
                "ultralytics' model.predict() assumes a raw numpy array is BGR (OpenCV "
                "convention) and flips it BGR->RGB internally before the model sees it "
                "(ultralytics/data/loaders.py: 'NumPy color inputs are assumed to use "
                "OpenCV-compatible BGR order'). engine.perception.detect.PlateDetector "
                "previously passed genuine RGB straight through, which ultralytics then "
                "wrongly 'corrected' -- feeding the model a channel-swapped image on every "
                "call. Verified: flipping PIL's RGB output to BGR (image[..., ::-1]) is "
                "byte-identical to cv2.imread's native BGR output for the same file, and "
                "produces different (and, per the source comment, correct) detections than "
                "passing RGB directly. Fixed inside PlateDetector.detect() itself (flips "
                "RGB->BGR before the ultralytics call), so callers (eval_detector.py, "
                "video_to_events.py) needed no changes -- both already pass RGB via PIL."
            ),
        },
        "plate_boxes_2k_filter": {
            "total_images": len(all_pairs_unfiltered),
            "excluded_precropped": n_excluded_precropped,
            "kept_scene_images": len(scene_pairs),
            "rule": (
                f"GT box area > {PRECROPPED_AREA_FRACTION_THRESHOLD * 100:.0f}% of image, OR "
                f"image area < {PRECROPPED_MIN_IMAGE_AREA} px^2. Validated against this "
                "dataset's two filename families (used only for validation, not the rule "
                "itself): excludes all 1,902 true crops, 0 false negatives; 3 borderline "
                "scene photos also excluded (2 tight close-ups, 1 small-but-genuine scene). "
                "00000234.jpg (Odia-script plate product photo, a legitimate rare edge case) "
                "is correctly kept."
            ),
        },
        "steps": {},
    }

    step_configs: list[tuple[str, dict, bool]] = [
        ("A_rgb_fix_no_tiling", {"tiled": False}, False),
        ("B_imgsz_1280", {"tiled": False, "imgsz": 1280}, False),
        ("B_imgsz_native", {"tiled": False, "imgsz": "native"}, False),
        ("C_tiled_default", {"tiled": True}, True),
    ]

    for step_name, detect_kwargs, time_it in step_configs:
        print(f"=== step {step_name} ({detect_kwargs}) ===", flush=True)
        step_result: dict = {}

        print(f"  plate_boxes_2k scenes ({len(scene_pairs)} images)...", flush=True)
        step_result["plate_boxes_2k_scenes"] = evaluate_detector_on_dataset(
            detector, scene_pairs, args.conf_threshold, detect_kwargs=detect_kwargs
        )

        print(f"  video_frames_boxes ({len(video_pairs)} images)...", flush=True)
        step_result["video_frames_boxes"] = evaluate_detector_on_dataset(
            detector,
            video_pairs,
            args.conf_threshold,
            detect_kwargs=detect_kwargs,
            measure_timing=time_it,
        )

        report["steps"][step_name] = step_result
        print(f"  {step_name}: {json.dumps(step_result, indent=2)}", flush=True)

    print(f"indian_vehicle_xml ({len(xml_pairs)} images, RGB-fixed, no tiling)...", flush=True)
    report["indian_vehicle_xml_rgb_fixed"] = evaluate_detector_on_dataset(
        detector, xml_pairs, args.conf_threshold, detect_kwargs={"tiled": False}
    )

    _maybe_run_step_d(report, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


def _maybe_run_step_d(report: dict, args: argparse.Namespace) -> None:
    """STEP D: if --finetuned-weights was given, evaluate it (tiled
    inference) on the held-out vid-3 set ONLY, alongside the approved
    (pre-fine-tune) weights on that SAME set, so the fine-tune's effect is
    isolated from everything steps A-C already changed."""
    if args.finetuned_weights is None:
        return
    if not args.holdout_dir.exists():
        print(f"WARNING: holdout dir {args.holdout_dir} not found -- skipping step D", flush=True)
        return

    holdout_pairs = load_holdout_vid3(args.holdout_dir)
    print(
        f"=== step D: fine-tuned detector on held-out vid-3 ({len(holdout_pairs)} frames) ===",
        flush=True,
    )

    pre_finetune_detector = PlateDetector(weights_path=args.weights)
    before = evaluate_detector_on_dataset(
        pre_finetune_detector, holdout_pairs, args.conf_threshold, detect_kwargs={"tiled": True}
    )
    print(f"  before (approved weights): {json.dumps(before, indent=2)}", flush=True)

    finetuned_detector = PlateDetector(weights_path=args.finetuned_weights)
    after = evaluate_detector_on_dataset(
        finetuned_detector, holdout_pairs, args.conf_threshold, detect_kwargs={"tiled": True}
    )
    print(f"  after (fine-tuned weights): {json.dumps(after, indent=2)}", flush=True)

    report["step_D_finetuned"] = {
        "finetuned_weights": str(args.finetuned_weights),
        "held_out_set": "video_frames_boxes/vid-3 (50 frames) -- NEVER used in fine-tune training",
        "note": (
            "Small held-out set (one video, 50 frames) -- stated plainly, not a claim of a "
            "robust test. Both readings use tiled inference (the PlateDetector default)."
        ),
        "before_approved_weights": before,
        "after_finetuned_weights": after,
    }


if __name__ == "__main__":
    main()
