"""Evaluate a trained CRNN checkpoint: whole-plate accuracy, per-character
accuracy, per-condition breakdown, and confidence calibration (ECE).

Runs under the SEPARATE OCR venv. Consumes a labels CSV in the same schema
engine.perception.synth_plates.py writes (image_path, canonical, text_line1,
text_line2, layout, style, condition) -- a real-data CSV must carry the same
columns for this to work (condition may be omitted; per-condition breakdown
is then skipped).

The report's "dataset" field is ALWAYS one of "synthetic" or "real" --
required at the CLI, never inferred -- specifically so a synthetic
development number can never be mistaken for the PRD's real-data >90% claim
when someone reads eval/reports/ocr_*.json later.

Whole-plate / per-character accuracy are computed in CANONICAL 10-slot
space (via engine.perception.ctc_to_slots), not on the raw decoded string:
both `canonical` (ground truth) and the model's argmax reconstruction are
always exactly 10 characters, so per-character accuracy is a simple
position-wise comparison -- no edit-distance alignment needed, unlike
variable-length whole-plate strings.

Confidence calibration (ECE) uses whole-plate confidence = the PRODUCT of
each slot's peak probability (FINDING 2), not their mean: ten slots at 0.98
average to 0.98, but the probability all ten are simultaneously correct is
~0.98**10 =~ 0.82. The mean systematically overstates whole-plate confidence
and inflates measured ECE -- this is also the number the engine would need
for a whole-plate confidence, since the plate LR itself is a PRODUCT over
slots (docs/architecture.md sec 3), not a mean.

FINDING 3 (calibration): if the checkpoint carries a fitted "temperature"
(engine.perception.crnn.fit_temperature, run via train_ocr.py
--fit-temperature), inference divides logits by it before softmax, and the
report includes BOTH the uncalibrated (T=1) and calibrated ECE, so the
effect of calibration is visible rather than silently applied. Decode
(whole-plate/char accuracy) is temperature-invariant by construction --
dividing logits by a positive scalar never changes the argmax -- so only
confidence, not accuracy, differs between the two.

DESIGN A (stop depending on a layout label): every crop is decoded BOTH as
one line (the whole crop) and as two stacked rows (top/bottom halves,
concatenated -- see engine.perception.ctc_to_slots.decode_two_line_plate),
and `decode_plate_best_of_two` picks whichever reading has the higher
plate-level log-probability. The CSV's `layout` column is never read for
this -- it is written by dataset builders (synth_plates.py,
build_real_set.py) as informational metadata only, e.g. for the
per-condition breakdown, never as an input to decoding. This is what lets
the real test set (whose two-row plates were found NOT to reliably wrap at
a fixed grammar boundary -- see build_real_set.py) be scored correctly
without trusting a layout label at all. The report's `reading_win_counts`
records how often each reading actually won.
"""

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from engine.perception.crnn import CRNN, IMG_HEIGHT, IMG_WIDTH, split_two_line_crop
from engine.perception.ctc_to_slots import decode_plate_best_of_two

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "eval" / "reports"

CROP_ROLES = ("whole", "top", "bottom")


class _CropDataset(Dataset):
    """Flattens eval rows into crops for EVERY role in CROP_ROLES,
    regardless of the CSV's `layout` column (DESIGN A: layout is never
    trusted at inference time). Keeps the row index and role so results can
    be regrouped per plate."""

    def __init__(self, rows: list[dict], root: Path):
        self.rows = rows
        self.root = root
        self.crops: list[tuple[int, str]] = [
            (i, role) for i in range(len(rows)) for role in CROP_ROLES
        ]

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row_idx, role = self.crops[idx]
        row = self.rows[row_idx]
        img = Image.open(self.root / row["image_path"]).convert("L")
        if role != "whole":
            arr = np.array(img)
            top, bottom = split_two_line_crop(arr)
            img = Image.fromarray(top if role == "top" else bottom)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, row_idx, role


def _collate(batch):
    images = torch.stack([b[0] for b in batch])
    meta = [(b[1], b[2]) for b in batch]
    return images, meta


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10
) -> float:
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


@torch.no_grad()
def run_inference(
    model: CRNN, rows: list[dict], root: Path, device: str, batch_size: int
) -> dict[int, dict]:
    """Returns {row_idx: {"single"|"top"/"bottom": RAW LOGITS (T,C) np.ndarray}}.
    Raw logits, not softmax: FINDING 3 needs confidence recomputed at two
    different temperatures without a second forward pass, and decode/argmax
    is temperature-invariant so this loses nothing for accuracy either."""
    model.eval()
    ds = _CropDataset(rows, root)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0
    )
    results: dict[int, dict] = {}
    for images, meta in loader:
        images = images.to(device)
        logits_np = model(images).cpu().numpy()
        for i, (row_idx, role) in enumerate(meta):
            results.setdefault(row_idx, {})[role] = logits_np[i]
    return results


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _decode_best_of_two(crop_logits: dict, temperature: float) -> tuple[list, str] | None:
    """DESIGN A: decode a crop's "whole"/"top"/"bottom" logits both ways and
    return (slots, which_reading_won) from decode_plate_best_of_two, or
    None if any required crop is missing from `crop_logits`."""
    if not all(role in crop_logits for role in CROP_ROLES):
        return None
    probs_whole = _softmax(crop_logits["whole"], temperature)
    probs_top = _softmax(crop_logits["top"], temperature)
    probs_bottom = _softmax(crop_logits["bottom"], temperature)
    return decode_plate_best_of_two(probs_whole, probs_top, probs_bottom)


def _confidence_from_slots(slots) -> float:
    """FINDING 2: the PRODUCT of each slot's peak probability, computed as
    exp(sum(log(.))) for numerical stability -- see module docstring."""
    log_peaks = [math.log(s.probs[s.argmax()]) for s in slots]
    return float(math.exp(sum(log_peaks)))


def evaluate(
    model: CRNN,
    rows: list[dict],
    root: Path,
    device: str,
    batch_size: int,
    num_bins: int,
    temperature: float = 1.0,
    group_by_plate: bool = False,
) -> dict:
    per_row_logits = run_inference(model, rows, root, device, batch_size)

    whole_plate_correct = np.zeros(len(rows), dtype=bool)
    char_accuracy = np.zeros(len(rows), dtype=np.float64)
    confidence_before = np.zeros(len(rows), dtype=np.float64)
    confidence_after = np.zeros(len(rows), dtype=np.float64)
    conditions: list[str | None] = []
    reading_win_counts = {"one_line": 0, "two_row": 0}

    for i, row in enumerate(rows):
        crop_logits = per_row_logits.get(i, {})
        # Decode at T=1 for accuracy -- decode is temperature-invariant
        # (argmax is unaffected by dividing logits by a positive scalar),
        # so this is exactly as correct as decoding at the calibrated T.
        result_t1 = _decode_best_of_two(crop_logits, 1.0)
        canonical = row["canonical"]
        if result_t1 is None:
            whole_plate_correct[i] = False
            char_accuracy[i] = 0.0
            confidence_before[i] = 0.0
            confidence_after[i] = 0.0
        else:
            slots_t1, which = result_t1
            reading_win_counts[which] = reading_win_counts.get(which, 0) + 1
            argmax_str = "".join(s.argmax() for s in slots_t1)
            whole_plate_correct[i] = argmax_str == canonical
            matches = sum(1 for a, b in zip(argmax_str, canonical, strict=True) if a == b)
            char_accuracy[i] = matches / len(canonical)
            confidence_before[i] = _confidence_from_slots(slots_t1)
            if temperature != 1.0:
                result_cal = _decode_best_of_two(crop_logits, temperature)
                confidence_after[i] = _confidence_from_slots(result_cal[0]) if result_cal else 0.0
            else:
                confidence_after[i] = confidence_before[i]
        conditions.append(row.get("condition") or None)

    correct_f = whole_plate_correct.astype(np.float64)
    ece_before = expected_calibration_error(confidence_before, correct_f, num_bins)
    ece_after = (
        expected_calibration_error(confidence_after, correct_f, num_bins)
        if temperature != 1.0
        else ece_before
    )

    report: dict = {
        "n_samples": len(rows),
        "whole_plate_accuracy": float(whole_plate_correct.mean()) if len(rows) else 0.0,
        "char_accuracy": float(char_accuracy.mean()) if len(rows) else 0.0,
        "temperature": temperature,
        "confidence_ece_before_calibration": ece_before,
        "confidence_ece_after_calibration": ece_after,
        "reading_win_counts": reading_win_counts,
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
    (unweighted by image count). Without this, a plate with many images
    (e.g. 41 video frames of one car) would dominate a plain per-image
    average -- this is what the reviewer's "one car with 41 frames can't
    dominate" requirement means in practice."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CRNN plate OCR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root dir image_path is relative to (default: csv's own directory).",
    )
    parser.add_argument(
        "--dataset-label",
        choices=["synthetic", "real"],
        required=True,
        help=(
            "Whether --csv is synthetic or real plate crops -- stamped into "
            "the report; only 'real' results can support the PRD's >90% claim."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument(
        "--group-by-plate",
        action="store_true",
        help="Also report accuracy averaged PER UNIQUE PLATE, not per image (see docstring).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = args.root or args.csv.parent

    with args.csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {args.csv}")

    model = CRNN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    temperature = float(ckpt.get("temperature", 1.0))
    if temperature != 1.0:
        print(f"applying calibrated temperature T={temperature:.4f} from checkpoint", flush=True)

    report = evaluate(
        model, rows, root, device, args.batch_size, args.num_bins, temperature, args.group_by_plate
    )
    report["dataset"] = args.dataset_label
    report["description"] = (
        f"{'SYNTHETIC development' if args.dataset_label == 'synthetic' else 'REAL, held-out'} "
        f"plate OCR evaluation -- {len(rows)} samples from {args.csv}. "
        + (
            "This is a development number, NOT the PRD's real-data accuracy claim."
            if args.dataset_label == "synthetic"
            else "Supports the PRD's real-Indian-plate accuracy claim."
        )
    )
    report["checkpoint"] = str(args.checkpoint)
    report["csv"] = str(args.csv)
    report["generated_at"] = datetime.now(UTC).isoformat()

    out_path = args.out
    if out_path is None:
        DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_REPORT_DIR / f"ocr_{args.dataset_label}_{args.csv.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
