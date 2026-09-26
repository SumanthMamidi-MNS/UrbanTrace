"""Build the REAL OCR labels/crops set from
<data_dir>/ocr/raw/indian_vehicle_xml/ (1,697 real photos, Pascal-VOC XML
per image: `<object><bndbox>` is the plate box, `<object><name>` is the
plate text), where `<data_dir>` is `URBANTRACE_DATA_DIR` (see
`engine.paths`). See `<data_dir>/ocr/raw/README.md` for provenance and the
lead's own spot-checks.

Runs under either venv (only needs Pillow, stdlib xml, and the grammar/
contracts modules below -- no torch), but is meant to be run once, offline,
to produce `<data_dir>/ocr/real/{train,val,test}/` in EXACTLY the schema
engine.perception.synth_plates.py already writes and train_ocr.py/
eval_ocr.py already read: image_path, canonical, text_line1, text_line2,
layout, style, condition.

Label parsing reuses the engine's own grammar code -- it does not
reimplement plate parsing:
  - `api.plate_grammar.enumerate_canonical_forms` splits a literal plate
    string into the canonical 10-slot form(s) consistent with the Indian
    plate grammar (state/RTO/series/number).
  - `engine.perception.ctc_to_slots.canonical_string_to_slots` re-validates
    the resulting canonical string's per-slot alphabet membership as a
    defensive check (always true by construction, but reused rather than
    trusted blindly).

Three exclusion reasons, each counted and reported (see `build()`'s
returned stats and the CLI's printed summary):
  - **grammar_invalid**: the label does not parse under the grammar at all.
    Inspection of this run's actual failures (not just the lead's 18-sample
    spot check) shows two distinct causes, not one: genuine typos (a digit
    where a letter belongs, e.g. `ML05G3550` labelled `ML0563550`), AND a
    structural gap in the canonical form itself -- some real plates
    (disproportionately Delhi-registered) use a genuine 3-LETTER series
    (e.g. `DL9CAE1359`), but `engine.contracts.plate`'s canonical form has
    room for only 2 series-letter slots (frozen contract, not something
    this script can special-case). A handful of labels are not plate text
    at all (car model names like "TERRANO", "CRETA" mistakenly used as the
    annotation).
  - **ambiguous**: `enumerate_canonical_forms` returns more than one
    grammar-consistent split with no wildcard to disambiguate (observed
    exactly once in this dataset: "AR08983" could be RTO=00+number=8983 or
    RTO=08+number=983). Rather than add a guessing rule on top of the reused
    parser, these are excluded too -- the volume is negligible (1 of 1,697).
  - **missing_image**: the XML's sibling image file could not be found
    (observed once: `State-wise_OLX/MH/MH5.xml`).

Layout and row split (one_line vs two_line, and WHERE a two-row plate
wraps): decided by FORCED ALIGNMENT under the synthetic-pretrained CRNN
checkpoint, not by the crop's aspect ratio. Aspect ratio was tried first
and abandoned after verifying it by eye on ~15 of the ~30 crops it flagged
as two-line: only about a third were genuinely two-line (e.g. ratio 0.94
`MH14EY5972` was a one-line plate with a loose annotation box, not a
two-line one -- the box, not the plate, was tall). Worse, the genuinely
two-line plates found this way did not reliably wrap at the
state+RTO+series / number boundary a naive splitter would assume: real
plates like `WB07D5106` are printed as "WB07D51" / "06" (wrapped
mid-number) and `KL01AU585` as "KL01" / "AU585" (wrapped after RTO, before
the series). No box-geometry heuristic can recover that.

Forced alignment (`make_forced_alignment_scorer`) instead asks the model
itself: for each crop, with ground-truth text L (the label, blanks
stripped), score (a) the ONE-LINE reading -- CTC loss of the whole crop
against L -- and (b) for each of a handful of grammar-plausible split
points k (`SPLIT_CANDIDATES`), the TWO-ROW reading -- CTC loss of the top
half against L[:k] plus the bottom half against L[k:]. Whichever scores
lower (higher likelihood) wins, and its split point becomes
`text_line1`/`text_line2`/`layout` in the CSV. This is done ONCE per real
crop when building the set (not at train or eval time -- see
engine.perception.ctc_to_slots.decode_plate_best_of_two and this project's
DESIGN A for how inference-time decoding separately stops needing a layout
label at all), and every candidate's score is written to
`<out>/alignment_audit.json` so the choice can be checked by eye rather
than trusted blindly (the lead's request: eyeball ~10 two-row assignments).

`SPLIT_CANDIDATES = (4, 5, 6, 7)`: state+RTO are always printed together on
the same row on every real plate observed (k=4 is the minimum plausible
split), and a 2-letter series is the longest plausible top-row addition
(k=6); k=7 is included specifically because it is what the real `WB07D5106`
example above needed (a wrap one character into the number) -- a
"handful" of structurally-motivated candidates, not an exhaustive scan of
every possible k.

This function needs the trained CRNN (`engine.perception.crnn`, torch), so
it is the ONLY part of this module that imports torch, and it does so
LAZILY inside `make_forced_alignment_scorer` -- importing this module, and
running `build()` with a caller-supplied scorer, still needs no torch (see
`tests/test_real_set_split.py`, which passes a trivial stub scorer and runs
under the main venv).

Crop padding: 8% of the box's own width/height on each side, clamped to
image bounds -- a small, fixed padding meant to look like a plate-detector
crop (which rarely returns the mathematically exact plate rectangle).

Split: BY PLATE STRING, NEVER BY IMAGE (plates repeat -- e.g. `MH47Y1124`
appears 41 times, video frames of one car -- see the README). Every image
of a given plate goes to the same split. The assignment is a deterministic
hash of the canonical plate string (`hashlib.sha256`, not Python's
randomised built-in `hash()`), bucketed into ~60% train / 10% val / 30%
test of UNIQUE plates -- deterministic across runs and independent of
processing order. `tests/test_real_set_split.py` asserts zero canonical-
plate overlap between the produced splits.
"""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from api.plate_grammar import GrammarError, enumerate_canonical_forms
from engine.contracts.plate import BLANK
from engine.paths import get_data_dir
from engine.perception.ctc_to_slots import canonical_string_to_slots

DEFAULT_RAW_ROOT = get_data_dir() / "ocr/raw/indian_vehicle_xml"
DEFAULT_OUT = get_data_dir() / "ocr/real"
DEFAULT_CHECKPOINT = get_data_dir() / "ocr/runs/synth/best.pt"
SOURCES = ["State-wise_OLX", "google_images", "video_images"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".Jpg", ".Jpeg", ".Png"]

CROP_PADDING_FRAC = 0.08

TRAIN_FRAC = 0.60
VAL_FRAC = 0.10
# test gets the remainder (~0.30)

# Forced-alignment two-row split candidates -- see module docstring.
SPLIT_CANDIDATES = (4, 5, 6, 7)


class ScoreReadingFn(Protocol):
    def __call__(self, crop_rgb: np.ndarray, label: str) -> dict: ...

CSV_FIELDS = ["image_path", "canonical", "text_line1", "text_line2", "layout", "style", "condition"]


def find_image_file(xml_path: Path) -> Path | None:
    """The image sits beside its XML with a mixed-case .jpg/.jpeg/.png
    extension -- and occasionally the XML's own stem already ends in
    something like ".jpg" as literal filename text (e.g.
    "...071740.jpg.xml" whose image is "...071740.jpg.jpeg"), so this tries
    the stem plus each known extension directly before falling back to a
    case-insensitive directory scan."""
    stem = xml_path.name[: -len(".xml")]
    for ext in IMAGE_EXTS:
        candidate = xml_path.with_name(stem + ext)
        if candidate.exists():
            return candidate
    lower_target = stem.lower()
    for f in xml_path.parent.iterdir():
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if f.name[: -len(f.suffix)].lower() == lower_target:
            return f
    return None


def parse_plate_text(raw_text: str) -> str | None:
    """Returns the canonical 10-slot string, or None if `raw_text` fails to
    parse under the Indian plate grammar or parses ambiguously (both
    excluded -- see module docstring). Reuses api.plate_grammar's
    query-grammar parser (built for wildcarded search queries, but a
    label with no wildcards is just the degenerate case) rather than
    reimplementing plate parsing."""
    text = (raw_text or "").strip().upper().replace(" ", "").replace("-", "")
    try:
        forms = enumerate_canonical_forms(text)
    except GrammarError:
        return None
    if len(forms) != 1:
        return None
    canonical = forms[0]
    canonical_string_to_slots(canonical)  # defensive re-validation, not a second parser
    return canonical


def canonical_to_lines(canonical: str) -> tuple[str, str]:
    """(state+RTO+series with blanks stripped, number with blanks stripped)
    -- the GRAMMAR segments, same convention synth_plates.py uses for its
    `text_line1`/`text_line2` columns."""
    return canonical[:6].replace(BLANK, ""), canonical[6:].replace(BLANK, "")


def assign_split(plate: str) -> str:
    """Deterministic group split by PLATE STRING (never by image): a
    stable hash (sha256, not Python's process-randomised `hash()`) buckets
    each unique plate into train/val/test independent of processing order
    or how many images that plate has."""
    digest = hashlib.sha256(plate.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 1000
    if bucket < int(TRAIN_FRAC * 1000):
        return "train"
    if bucket < int((TRAIN_FRAC + VAL_FRAC) * 1000):
        return "val"
    return "test"


def crop_plate(
    image: Image.Image, box: tuple[float, float, float, float], padding_frac: float
) -> Image.Image:
    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin
    pad_x, pad_y = w * padding_frac, h * padding_frac
    cx0 = max(int(xmin - pad_x), 0)
    cy0 = max(int(ymin - pad_y), 0)
    cx1 = min(int(xmax + pad_x), image.width)
    cy1 = min(int(ymax + pad_y), image.height)
    return image.crop((cx0, cy0, cx1, cy1))


def make_forced_alignment_scorer(
    checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str | None = None
) -> ScoreReadingFn:
    """Returns a `score_reading(crop_rgb, label) -> dict` closure (DESIGN B,
    see module docstring) that scores the one-line reading and each
    SPLIT_CANDIDATES two-row split of a crop against `label` using the
    synthetic-pretrained CRNN checkpoint's own CTC loss, and returns the
    winning layout/text_line1/text_line2 plus every candidate's score for
    audit. Lazily imports torch so importing THIS MODULE, and calling
    `build()` with some other scorer, never needs it."""
    import torch

    from engine.perception.crnn import (
        CRNN,
        IMG_HEIGHT,
        IMG_WIDTH,
        build_ctc_loss,
        ctc_targets_from_labels,
        split_two_line_crop,
    )
    from engine.perception.ctc_to_slots import char_to_label

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN().to(resolved_device)
    ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ctc_loss_fn = build_ctc_loss()
    seq_len = IMG_WIDTH // 4

    def _tensor(arr_gray: np.ndarray):
        img = Image.fromarray(arr_gray).resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(resolved_device)

    @torch.no_grad()
    def _log_probs(tensor):
        logits = model(tensor)
        return torch.log_softmax(logits, dim=-1).permute(1, 0, 2).cpu()

    def _ctc_loss(log_probs, target: str) -> float:
        if not target:
            return float("inf")
        targets, target_lengths = ctc_targets_from_labels([target], char_to_label)
        input_lengths = torch.full((1,), seq_len, dtype=torch.long)
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        return float(loss.item())

    def score_reading(crop_rgb: np.ndarray, label: str) -> dict:
        gray = np.asarray(Image.fromarray(crop_rgb).convert("L"))
        top, bottom = split_two_line_crop(gray)

        whole_log_probs = _log_probs(_tensor(gray))
        top_log_probs = _log_probs(_tensor(top))
        bottom_log_probs = _log_probs(_tensor(bottom))

        one_line_loss = _ctc_loss(whole_log_probs, label)

        two_row_candidates: dict[int, float] = {}
        for k in SPLIT_CANDIDATES:
            if k < 1 or k >= len(label):
                continue
            loss = _ctc_loss(top_log_probs, label[:k]) + _ctc_loss(bottom_log_probs, label[k:])
            two_row_candidates[k] = loss

        if two_row_candidates:
            best_k = min(two_row_candidates, key=two_row_candidates.get)
            best_two_row_loss = two_row_candidates[best_k]
        else:
            best_k, best_two_row_loss = None, float("inf")

        if best_two_row_loss < one_line_loss:
            layout, text_line1, text_line2 = "two_line", label[:best_k], label[best_k:]
        else:
            layout, text_line1, text_line2 = "one_line", label, ""

        return {
            "layout": layout,
            "text_line1": text_line1,
            "text_line2": text_line2,
            "one_line_loss": one_line_loss,
            "two_row_candidates": two_row_candidates,
            "best_split_k": best_k,
            "best_two_row_loss": best_two_row_loss,
        }

    return score_reading


@dataclass
class Candidate:
    xml_path: Path
    image_path: Path
    box: tuple[float, float, float, float]
    source: str


@dataclass
class BuildStats:
    total_xml: int = 0
    excluded_grammar_invalid: int = 0
    excluded_ambiguous: int = 0
    excluded_missing_image: int = 0
    excluded_no_object_or_box: int = 0
    kept_images: int = 0
    kept_unique_plates: int = 0
    splits: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_xml": self.total_xml,
            "excluded": {
                "grammar_invalid": self.excluded_grammar_invalid,
                "ambiguous_parse": self.excluded_ambiguous,
                "missing_image": self.excluded_missing_image,
                "no_object_or_box": self.excluded_no_object_or_box,
                "total": (
                    self.excluded_grammar_invalid
                    + self.excluded_ambiguous
                    + self.excluded_missing_image
                    + self.excluded_no_object_or_box
                ),
            },
            "kept_images": self.kept_images,
            "kept_unique_plates": self.kept_unique_plates,
            "splits": self.splits,
        }


def _collect_candidates(raw_root: Path, stats: BuildStats) -> dict[str, list[Candidate]]:
    """Walk every XML under `raw_root`'s three source subfolders, parse and
    validate each label, and group surviving candidates by canonical plate
    string. Exclusions are tallied on `stats` in place."""
    by_plate: dict[str, list[Candidate]] = {}
    for source in SOURCES:
        src_dir = raw_root / source
        for xf in sorted(src_dir.rglob("*.xml")):
            stats.total_xml += 1
            tree = ET.parse(xf)
            name_el = tree.find("object/name")
            bbox_el = tree.find("object/bndbox")
            if name_el is None or bbox_el is None:
                stats.excluded_no_object_or_box += 1
                continue

            canonical = parse_plate_text(name_el.text or "")
            if canonical is None:
                # Distinguish "ambiguous" from "invalid" only for reporting;
                # parse_plate_text already made the exclusion decision.
                text = (name_el.text or "").strip().upper().replace(" ", "")
                try:
                    enumerate_canonical_forms(text)
                    stats.excluded_ambiguous += 1  # parsed, but len(forms) != 1
                except GrammarError:
                    stats.excluded_grammar_invalid += 1
                continue

            image_path = find_image_file(xf)
            if image_path is None:
                stats.excluded_missing_image += 1
                continue

            xmin = float(bbox_el.find("xmin").text)
            ymin = float(bbox_el.find("ymin").text)
            xmax = float(bbox_el.find("xmax").text)
            ymax = float(bbox_el.find("ymax").text)
            candidate = Candidate(
                xml_path=xf, image_path=image_path, box=(xmin, ymin, xmax, ymax), source=source
            )
            by_plate.setdefault(canonical, []).append(candidate)
    return by_plate


def build(out_dir: Path, raw_root: Path, score_reading_fn: ScoreReadingFn) -> dict:
    """`score_reading_fn` decides layout/text_line1/text_line2 per crop
    (DESIGN B forced alignment in production, via
    `make_forced_alignment_scorer`; a trivial stub in tests that don't need
    a real model -- see tests/test_real_set_split.py). Every call's result
    is also written to `<out_dir>/alignment_audit.json` for inspection."""
    stats = BuildStats()
    by_plate = _collect_candidates(raw_root, stats)

    plate_split = {plate: assign_split(plate) for plate in by_plate}
    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    audit_log: list[dict] = []
    reading_win_counts = {"one_line": 0, "two_line": 0}

    for plate, candidates in by_plate.items():
        split = plate_split[plate]
        line1_full, line2_full = canonical_to_lines(plate)
        full_label = line1_full + line2_full
        for cand in candidates:
            image = Image.open(cand.image_path).convert("RGB")
            crop = crop_plate(image, cand.box, CROP_PADDING_FRAC)
            crop_arr = np.asarray(crop)

            alignment = score_reading_fn(crop_arr, full_label)
            layout = alignment["layout"]
            text_line1 = alignment["text_line1"]
            text_line2 = alignment["text_line2"]
            reading_win_counts[layout] = reading_win_counts.get(layout, 0) + 1

            idx = len(rows_by_split[split])
            fname = f"{split}_{idx:06d}.jpg"
            img_dir = out_dir / split / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            crop.save(img_dir / fname, quality=92)
            rows_by_split[split].append(
                {
                    "image_path": f"images/{fname}",
                    "canonical": plate,
                    "text_line1": text_line1,
                    "text_line2": text_line2,
                    "layout": layout,
                    "style": "unknown",
                    "condition": cand.source,
                }
            )
            audit_log.append(
                {
                    "split": split,
                    "image_path": f"{split}/images/{fname}",
                    "canonical": plate,
                    "label": full_label,
                    **{k: v for k, v in alignment.items() if k not in ("text_line1", "text_line2")},
                }
            )

    for split, rows in rows_by_split.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        csv_path = split_dir / "labels.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "alignment_audit.json").write_text(
        json.dumps(audit_log, indent=2), encoding="utf-8"
    )

    stats.kept_images = sum(len(rows) for rows in rows_by_split.values())
    stats.kept_unique_plates = len(by_plate)
    stats.splits = {
        split: {
            "n_images": len(rows),
            "n_unique_plates": len({r["canonical"] for r in rows}),
        }
        for split, rows in rows_by_split.items()
    }
    stats_dict = stats.to_dict()
    stats_dict["reading_win_counts"] = reading_win_counts
    return stats_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the real OCR dataset from indian_vehicle_xml."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Synthetic-pretrained CRNN checkpoint used for forced alignment (DESIGN B).",
    )
    args = parser.parse_args()

    print(f"loading forced-alignment model from {args.checkpoint}", flush=True)
    score_reading_fn = make_forced_alignment_scorer(args.checkpoint)

    stats = build(args.out, args.raw_root, score_reading_fn)
    print(json.dumps(stats, indent=2))
    stats_path = args.out / "build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {stats_path}")
    print(f"wrote {args.out / 'alignment_audit.json'}")


if __name__ == "__main__":
    main()
