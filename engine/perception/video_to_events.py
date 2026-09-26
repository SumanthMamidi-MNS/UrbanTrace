"""Frames (a folder of images per camera) -> detector -> OCR -> DetectionEvent
JSONL, written in the engine's existing on-disk format (engine.contracts.codec
+ engine.contracts.store.EventStore) -- NOT a reinvented format. This is what
lets real footage run through the same linking engine the simulator feeds.

Runs under the SEPARATE OCR venv (torch/ultralytics). The on-disk output
(events.jsonl + embeddings.npy) is plain JSON + a float16 .npy, so it loads
back with engine.contracts.store.EventStore from the MAIN venv with zero
extra dependencies -- the L1/L3 contract this whole module exists to respect.

Input layout: `--input-dir` contains one subfolder per camera, each holding
that camera's frame images in chronological filename order:

    <input-dir>/<camera_id>/0001.jpg
    <input-dir>/<camera_id>/0002.jpg
    ...

Timestamps are NOT read from real capture metadata (a folder of images
carries none in general): each camera's frames are stamped
`--start-time + frame_index / --fps` seconds, independently per camera. This
is a documented simplification for the plumbing demo; a real deployment
would carry real per-frame timestamps through from the video container.

Two placeholders, both called out explicitly rather than silently shipped:

  - **Appearance embedding**: no approved Re-ID model exists for this task
    (the user explicitly ruled one out). `_placeholder_embedding` is a coarse
    colour-histogram descriptor of the plate crop, satisfying
    DetectionEvent's 128-d L2-normalised contract so the plumbing runs end to
    end, but it carries close to no real vehicle-identity signal. The
    appearance channel must not be trusted on video-sourced events until a
    real Re-ID model is added.
  - **Vehicle attributes** (colour/type): no attribute classifier was in
    scope for this task either. Every video-sourced event gets
    `color="unknown"`, `vehicle_type="unknown"` at low confidence.
"""

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from engine.contracts.codec import encode_event_compact
from engine.contracts.events import DetectionEvent, VehicleAttributes
from engine.contracts.plate import SlotPosterior
from engine.perception.crnn import CRNN, IMG_HEIGHT, IMG_WIDTH, split_two_line_crop
from engine.perception.ctc_to_slots import decode_plate_best_of_two
from engine.perception.detect import DEFAULT_WEIGHTS_PATH, PlateDetector
from engine.perception.fpo_adapter import fpo_output_to_canonical_slots

EMBEDDING_DIM = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Default fast-plate-ocr run: the fine-tuned checkpoint behind the measured
# 81.0% whole-plate / 94.3% char accuracy on the held-out real test set (vs.
# the CRNN's 44.3% whole-plate) -- see eval/reports and docs/decisions.md.
# Made the default reader for this reason.
DEFAULT_FPO_ONNX_PATH = Path(
    "C:/sutra-data/ocr/fpo/runs/2026-09-25_17-56-29/best.onnx"
)
DEFAULT_FPO_PLATE_CONFIG_PATH = Path(
    "C:/sutra-data/ocr/fpo/runs/2026-09-25_17-56-29/plate_config.yaml"
)

# One plate crop in -> (10 canonical SlotPosteriors, which_reading_won or
# None). `which` is only meaningful for the CRNN's one-line/two-row
# best-of-two search (see `_ocr_crop`); the fpo reader has no such choice
# and always returns None.
OcrFn = Callable[[np.ndarray], tuple[list[SlotPosterior], str | None]]


def _placeholder_embedding(crop_rgb: np.ndarray, dim: int = EMBEDDING_DIM) -> list[float]:
    """NOT a real Re-ID embedding -- see module docstring. A coarse,
    deterministic colour-histogram descriptor (16x16 RGB thumbnail, 768
    values pooled down to `dim` by averaging contiguous chunks), L2-
    normalised so it satisfies DetectionEvent's embedding contract."""
    if crop_rgb.size == 0:
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 1.0
        return vec.tolist()
    small = np.asarray(Image.fromarray(crop_rgb).resize((16, 16)), dtype=np.float32) / 255.0
    flat = small.flatten()  # 16*16*3 = 768 = dim * 6
    chunk = len(flat) / dim
    vec = np.array(
        [flat[int(i * chunk) : int((i + 1) * chunk)].mean() for i in range(dim)], dtype=np.float32
    )
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    else:
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 1.0
    return vec.tolist()


def _crop_to_tensor(crop_gray: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(crop_gray).resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


@torch.no_grad()
def _ocr_crop(
    model: CRNN, crop_rgb: np.ndarray, device: str, temperature: float = 1.0
) -> tuple[list[SlotPosterior], str]:
    """Runs OCR on one plate crop (RGB uint8) and returns (10 canonical
    SlotPosteriors, which_reading_won).

    DESIGN A: decodes the crop BOTH as one line (the whole crop) and as two
    stacked rows (top/bottom halves), and picks whichever reading has the
    higher plate-level log-probability (decode_plate_best_of_two) -- never
    a stored/assumed layout, since real plates don't reliably wrap where a
    naive splitter would expect (see build_real_set.py). Three forward
    passes per crop (whole/top/bottom), batched together.

    `temperature`: FINDING 3's calibration, read from the checkpoint by
    `main()` below and threaded through here so the real inference path is
    calibrated exactly like eval_ocr.py's evaluation, not just the eval
    script -- an uncalibrated confidence handed to the engine's plate LR is
    the same overconfidence problem either way."""
    gray = np.asarray(Image.fromarray(crop_rgb).convert("L"))
    top, bottom = split_two_line_crop(gray)

    batch = torch.cat(
        [_crop_to_tensor(gray), _crop_to_tensor(top), _crop_to_tensor(bottom)], dim=0
    ).to(device)
    probs = torch.softmax(model(batch) / temperature, dim=-1).cpu().numpy()
    return decode_plate_best_of_two(probs[0], probs[1], probs[2])


def make_crnn_reader(model: CRNN, device: str, temperature: float = 1.0) -> OcrFn:
    """Wraps `_ocr_crop` (unchanged CRNN path) as an `OcrFn`."""

    def reader(crop_rgb: np.ndarray) -> tuple[list[SlotPosterior], str | None]:
        return _ocr_crop(model, crop_rgb, device, temperature)

    return reader


def make_fpo_reader(session, plate_cfg) -> OcrFn:
    """Wraps a fast-plate-ocr ONNX session as an `OcrFn`: detector crop ->
    the SAME preprocessing eval_fpo.py uses (`resize_image` +
    `preprocess_image`, imported from `fast_plate_ocr.core.process` rather
    than duplicated) -> ONNX inference -> `fpo_output_to_canonical_slots`.

    `session`/`plate_cfg` are typed loosely (onnxruntime.InferenceSession /
    fast_plate_ocr.inference.config.PlateConfig) so this module's top-level
    imports stay onnxruntime/fast_plate_ocr-free; only `main()` (OCR venv
    only, same as the rest of this file) imports and constructs them.
    """
    from fast_plate_ocr.core.process import preprocess_image, resize_image

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    plate_output_name = "plate" if "plate" in output_names else output_names[0]

    def reader(crop_rgb: np.ndarray) -> tuple[list[SlotPosterior], str | None]:
        img = crop_rgb.astype(np.uint8)
        if plate_cfg.image_color_mode == "grayscale":
            import cv2

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized = resize_image(
            img,
            img_height=plate_cfg.img_height,
            img_width=plate_cfg.img_width,
            image_color_mode=plate_cfg.image_color_mode,
            keep_aspect_ratio=plate_cfg.keep_aspect_ratio,
            interpolation_method=plate_cfg.interpolation,
            padding_color=plate_cfg.padding_color,
        )
        x = preprocess_image(resized)
        (out,) = session.run([plate_output_name], {input_name: x})
        slots = fpo_output_to_canonical_slots(out[0], plate_cfg.alphabet, plate_cfg.pad_char)
        return slots, None

    return reader


def process_camera_folder(
    camera_id: str,
    frame_paths: list[Path],
    detector: PlateDetector,
    ocr_fn: OcrFn,
    start_time: datetime,
    fps: float,
    conf_threshold: float,
    event_counter_start: int,
    reading_win_counts: dict[str, int] | None = None,
) -> tuple[list[DetectionEvent], int]:
    events: list[DetectionEvent] = []
    counter = event_counter_start
    for frame_idx, frame_path in enumerate(frame_paths):
        image = np.asarray(Image.open(frame_path).convert("RGB"))
        timestamp = start_time + timedelta(seconds=frame_idx / fps)

        for det in detector.detect(image, conf_threshold=conf_threshold):
            slots, which = ocr_fn(det.crop)
            if reading_win_counts is not None and which is not None:
                reading_win_counts[which] = reading_win_counts.get(which, 0) + 1
            plate_argmax = "".join(s.argmax() for s in slots)
            plate_confidence = float(np.mean([s.probs[s.argmax()] for s in slots]))

            event = DetectionEvent(
                event_id=f"evt_video_{counter:08d}",
                camera_id=camera_id,
                timestamp=timestamp,
                plate_posterior=slots,
                plate_argmax=plate_argmax,
                plate_confidence=plate_confidence,
                embedding=_placeholder_embedding(det.crop),
                attributes=VehicleAttributes(
                    color="unknown",
                    vehicle_type="unknown",
                    color_confidence=0.5,
                    type_confidence=0.5,
                ),
                crop_uri=None,
                source="video",
                gt_vehicle_id=None,
            )
            events.append(event)
            counter += 1
    return events, counter


def write_events(events: list[DetectionEvent], out_dir: Path) -> None:
    """Mirrors sim/generate.py's write_dataset exactly (same events.jsonl +
    embeddings.npy pair), so EventStore reads either interchangeably."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(events)
    embeddings = np.empty((n, EMBEDDING_DIM), dtype=np.float16)

    events_path = out_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, event in enumerate(events):
            embeddings[i] = np.asarray(event.embedding, dtype=np.float32)
            compact = encode_event_compact(event, embedding_row=i)
            f.write(json.dumps(compact, separators=(",", ":")))
            f.write("\n")

    np.save(out_dir / "embeddings.npy", embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frames -> plate detector -> OCR -> DetectionEvent JSONL."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Contains one subfolder per camera."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reader",
        choices=["fpo", "crnn"],
        default="fpo",
        help="OCR reader: fast-plate-ocr (default, measured 81.0%% whole-plate on the held-out "
        "real test set) or the CRNN (44.3%%).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="CRNN OCR checkpoint (.pt). Required when --reader crnn.",
    )
    parser.add_argument(
        "--fpo-onnx",
        type=Path,
        default=DEFAULT_FPO_ONNX_PATH,
        help="fast-plate-ocr ONNX checkpoint. Used when --reader fpo.",
    )
    parser.add_argument(
        "--fpo-plate-config",
        type=Path,
        default=DEFAULT_FPO_PLATE_CONFIG_PATH,
        help="fast-plate-ocr plate_config.yaml matching --fpo-onnx. Used when --reader fpo.",
    )
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS_PATH, help="YOLO plate detector weights."
    )
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Assumed frame rate for synthetic timestamps."
    )
    parser.add_argument(
        "--start-time", type=str, default="2026-01-01T00:00:00", help="ISO datetime for frame 0."
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    args = parser.parse_args()

    if args.reader == "crnn" and args.checkpoint is None:
        parser.error("--checkpoint is required when --reader crnn")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = datetime.fromisoformat(args.start_time)

    detector = PlateDetector(weights_path=args.weights, device=device)

    reading_win_counts: dict[str, int] | None
    if args.reader == "crnn":
        ocr_model = CRNN().to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ocr_model.load_state_dict(ckpt["model_state"])
        ocr_model.eval()
        temperature = float(ckpt.get("temperature", 1.0))
        if temperature != 1.0:
            print(
                f"applying calibrated temperature T={temperature:.4f} from checkpoint", flush=True
            )
        ocr_fn = make_crnn_reader(ocr_model, device, temperature)
        reading_win_counts = {"one_line": 0, "two_row": 0}
    else:
        import onnxruntime as ort
        from fast_plate_ocr.inference.config import PlateConfig

        if not args.fpo_onnx.exists():
            raise FileNotFoundError(f"fast-plate-ocr ONNX model not found at {args.fpo_onnx}")
        if not args.fpo_plate_config.exists():
            raise FileNotFoundError(
                f"fast-plate-ocr plate_config.yaml not found at {args.fpo_plate_config}"
            )
        plate_cfg = PlateConfig.from_yaml(args.fpo_plate_config)
        session = ort.InferenceSession(str(args.fpo_onnx), providers=["CPUExecutionProvider"])
        ocr_fn = make_fpo_reader(session, plate_cfg)
        reading_win_counts = None
        print(f"using fast-plate-ocr reader: {args.fpo_onnx}", flush=True)

    camera_dirs = sorted(p for p in args.input_dir.iterdir() if p.is_dir())
    if not camera_dirs:
        raise FileNotFoundError(f"no per-camera subfolders found under {args.input_dir}")

    all_events: list[DetectionEvent] = []
    counter = 0
    for cam_dir in camera_dirs:
        frame_paths = sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not frame_paths:
            continue
        events, counter = process_camera_folder(
            camera_id=cam_dir.name,
            frame_paths=frame_paths,
            detector=detector,
            ocr_fn=ocr_fn,
            start_time=start_time,
            fps=args.fps,
            conf_threshold=args.conf_threshold,
            event_counter_start=counter,
            reading_win_counts=reading_win_counts,
        )
        all_events.extend(events)
        print(
            f"[{cam_dir.name}] {len(frame_paths)} frames -> {len(events)} plate events",
            flush=True,
        )

    all_events.sort(key=lambda e: (e.timestamp, e.camera_id, e.event_id))
    write_events(all_events, args.out)
    print(f"wrote {len(all_events)} events -> {args.out / 'events.jsonl'}", flush=True)
    if reading_win_counts is not None:
        print(f"reading_win_counts: {reading_win_counts}", flush=True)


if __name__ == "__main__":
    main()
