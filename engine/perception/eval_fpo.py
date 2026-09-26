"""Evaluate a fast-plate-ocr ONNX checkpoint (zero-shot pretrained hub model,
or a fine-tuned export) on a real-plate labels CSV, using the SAME metrics
and definitions as engine/perception/eval_ocr.py (see that module's
docstring): whole-plate accuracy and character accuracy computed in
CANONICAL 10-slot space (via engine.perception.fpo_adapter, which reuses
engine.perception.ctc_to_slots's Indian plate grammar), per-condition
breakdown, PER-UNIQUE-PLATE accuracy, and confidence calibration (ECE) using
whole-plate confidence = the PRODUCT of each slot's peak probability (not
their mean -- see eval_ocr.py FINDING 2).

Runs under the SEPARATE OCR venv (needs `fast_plate_ocr` + `onnxruntime`).
`fpo_adapter.py` itself stays numpy-only and is tested from the MAIN venv;
this script is the thin onnxruntime-touching glue around it, exactly like
eval_ocr.py is the torch-touching glue around ctc_to_slots.py.

Crops are fed to the model AS-IS, whole, with no DESIGN A one-line/two-row
best-of-two search (unlike eval_ocr.py): fast-plate-ocr has a fixed number
of classification heads over one resized crop, with no notion of a physical
line break to split on, so there is no second "reading" to choose between.
Real two-line plates are simply resized (no aspect-ratio padding, per the
model's own `plate_config.yaml`: `keep_aspect_ratio: false`) into the
model's expected `(img_height, img_width)` input, exactly as the model's own
preprocessing does for any other image.

The `dataset` field in the written report is always "real" here -- this
script only ever runs against the held-out real Indian plate test split
(never touches TRAIN/VAL for evaluation purposes).
"""

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fast_plate_ocr.core.process import preprocess_image, read_and_resize_plate_image
from fast_plate_ocr.inference.config import PlateConfig

from engine.perception.fpo_adapter import fpo_output_to_canonical_slots

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "eval" / "reports"


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10
) -> float:
    """Identical implementation to eval_ocr.py's, kept in sync deliberately
    (small enough that a shared import would be more indirection than
    value) so ECE numbers from the two scripts are directly comparable."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _confidence_from_slots(slots) -> float:
    """PRODUCT of each slot's peak probability (eval_ocr.py FINDING 2),
    via exp(sum(log(.))) for numerical stability."""
    log_peaks = [math.log(s.probs[s.argmax()]) for s in slots]
    return float(math.exp(sum(log_peaks)))


def run_inference(
    session: ort.InferenceSession,
    rows: list[dict],
    root: Path,
    plate_cfg: PlateConfig,
    batch_size: int,
) -> np.ndarray:
    """Returns raw "plate" head output, shape (len(rows), max_plate_slots,
    vocabulary_size), already softmaxed (the model's own final Dense layer
    has activation="softmax" baked into the exported graph)."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    plate_output_name = "plate" if "plate" in output_names else output_names[0]

    all_outputs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        imgs = [
            read_and_resize_plate_image(
                root / row["image_path"],
                img_height=plate_cfg.img_height,
                img_width=plate_cfg.img_width,
                image_color_mode=plate_cfg.image_color_mode,
                keep_aspect_ratio=plate_cfg.keep_aspect_ratio,
                interpolation_method=plate_cfg.interpolation,
                padding_color=plate_cfg.padding_color,
            )
            for row in batch_rows
        ]
        x = preprocess_image(np.stack(imgs, axis=0))
        (out,) = session.run([plate_output_name], {input_name: x})
        all_outputs.append(out)
    return np.concatenate(all_outputs, axis=0)


def evaluate(
    session: ort.InferenceSession,
    rows: list[dict],
    root: Path,
    plate_cfg: PlateConfig,
    batch_size: int,
    num_bins: int,
    group_by_plate: bool = False,
) -> dict:
    plate_outputs = run_inference(session, rows, root, plate_cfg, batch_size)

    whole_plate_correct = np.zeros(len(rows), dtype=bool)
    char_accuracy = np.zeros(len(rows), dtype=np.float64)
    confidence = np.zeros(len(rows), dtype=np.float64)
    conditions: list[str | None] = []

    for i, row in enumerate(rows):
        slots = fpo_output_to_canonical_slots(
            plate_outputs[i], plate_cfg.alphabet, plate_cfg.pad_char
        )
        canonical = row["canonical"]
        argmax_str = "".join(s.argmax() for s in slots)
        whole_plate_correct[i] = argmax_str == canonical
        matches = sum(1 for a, b in zip(argmax_str, canonical, strict=True) if a == b)
        char_accuracy[i] = matches / len(canonical)
        confidence[i] = _confidence_from_slots(slots)
        conditions.append(row.get("condition") or None)

    correct_f = whole_plate_correct.astype(np.float64)
    ece = expected_calibration_error(confidence, correct_f, num_bins)

    report: dict = {
        "n_samples": len(rows),
        "whole_plate_accuracy": float(whole_plate_correct.mean()) if len(rows) else 0.0,
        "char_accuracy": float(char_accuracy.mean()) if len(rows) else 0.0,
        "confidence_ece": ece,
    }

    if any(c is not None for c in conditions):
        by_condition: dict[str, dict] = {}
        cond_arr = np.array([c if c is not None else "unknown" for c in conditions])
        for cond in sorted(set(cond_arr.tolist())):
            mask = cond_arr == cond
            by_condition[cond] = {
                "n_samples": int(mask.sum()),
                "whole_plate_accuracy": float(whole_plate_correct[mask].mean()),
                "char_accuracy": float(char_accuracy[mask].mean()),
            }
        report["per_condition"] = by_condition

    if group_by_plate:
        report["group_by_plate"] = _group_by_plate_report(rows, whole_plate_correct, char_accuracy)

    return report


def _group_by_plate_report(
    rows: list[dict], whole_plate_correct: np.ndarray, char_accuracy: np.ndarray
) -> dict:
    """PER-PLATE accuracy: for each unique canonical plate, average its own
    images' correctness, THEN average those per-plate means across plates
    (unweighted by image count) -- identical methodology to eval_ocr.py's,
    so one car with many frames can't dominate either report."""
    by_plate: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_plate.setdefault(row["canonical"], []).append(i)

    per_plate_whole = []
    per_plate_char = []
    image_counts = []
    for indices in by_plate.values():
        idx_arr = np.array(indices)
        per_plate_whole.append(float(whole_plate_correct[idx_arr].mean()))
        per_plate_char.append(float(char_accuracy[idx_arr].mean()))
        image_counts.append(len(indices))

    whole_mean = float(np.mean(per_plate_whole)) if per_plate_whole else 0.0
    char_mean = float(np.mean(per_plate_char)) if per_plate_char else 0.0
    return {
        "n_unique_plates": len(by_plate),
        "whole_plate_accuracy_per_plate_mean": whole_mean,
        "char_accuracy_per_plate_mean": char_mean,
        "images_per_plate_min": min(image_counts) if image_counts else 0,
        "images_per_plate_max": max(image_counts) if image_counts else 0,
        "images_per_plate_mean": float(np.mean(image_counts)) if image_counts else 0.0,
    }


def _error_sample(
    rows: list[dict], plate_outputs: np.ndarray, plate_cfg: PlateConfig, n: int
) -> list[dict]:
    """A small sample of whole-plate failures (label vs prediction) for
    qualitative review -- what kind of errors remain."""
    samples = []
    for i, row in enumerate(rows):
        slots = fpo_output_to_canonical_slots(
            plate_outputs[i], plate_cfg.alphabet, plate_cfg.pad_char
        )
        argmax_str = "".join(s.argmax() for s in slots)
        canonical = row["canonical"]
        if argmax_str != canonical:
            samples.append(
                {
                    "image_path": row["image_path"],
                    "condition": row.get("condition"),
                    "label": canonical,
                    "prediction": argmax_str,
                    "confidence": _confidence_from_slots(slots),
                }
            )
        if len(samples) >= n:
            break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot/fine-tuned evaluation of a fast-plate-ocr ONNX checkpoint."
    )
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--plate-config", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--group-by-plate", action="store_true")
    parser.add_argument("--error-sample-size", type=int, default=15)
    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Exact one-line description of what was evaluated, stamped into the report.",
    )
    args = parser.parse_args()

    root = args.root or args.csv.parent
    plate_cfg = PlateConfig.from_yaml(args.plate_config)

    with args.csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {args.csv}")

    session = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])

    report = evaluate(
        session, rows, root, plate_cfg, args.batch_size, args.num_bins, args.group_by_plate
    )

    # Error sample re-uses one more inference pass's worth of outputs rather
    # than threading them out of `evaluate` -- CPU inference on this small
    # model is fast enough that re-running is simpler than plumbing.
    plate_outputs = run_inference(session, rows, root, plate_cfg, args.batch_size)
    report["error_sample"] = _error_sample(rows, plate_outputs, plate_cfg, args.error_sample_size)

    report["dataset"] = "real"
    report["model_name"] = args.onnx_model.stem
    report["model_path"] = str(args.onnx_model)
    report["csv"] = str(args.csv)
    report["description"] = args.description
    report["generated_at"] = datetime.now(UTC).isoformat()

    out_path = args.out
    if out_path is None:
        DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_REPORT_DIR / f"ocr_real_fpo_{args.onnx_model.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "error_sample"}, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
