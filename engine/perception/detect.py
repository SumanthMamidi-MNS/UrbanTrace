"""Thin wrapper around the approved pretrained YOLOv8 plate detector
(Koushim/yolov8-license-plate-detection, MIT-licensed weights, Hugging Face)
-- image in, plate crops + boxes + confidences out.

Runs under the SEPARATE OCR venv (ultralytics + torch). The default weights
are now the FINE-TUNED detector,
C:/sutra-data/ocr/runs/detector_finetune/weights/best.pt (trained on top of
the pretrained baseline below -- see `engine/perception/finetune_detector.py`
and `eval/reports/detector_eval.json` / `detector_holdout_video.json` for
its measured accuracy). The original pretrained baseline,
C:/sutra-data/ocr/weights/best.pt (downloaded once via `hf download`, not
committed to the repo -- see docs/decisions.md, "OCR module"), remains
available via `PRETRAINED_WEIGHTS_PATH` for callers that explicitly want it
(e.g. an eval script comparing before/after fine-tuning).

Licence note (docs/decisions.md): the Ultralytics library itself is
AGPL-3.0; the detector weights are MIT. Acceptable for this prototype with
the user's approval -- a production deployment needs a commercial Ultralytics
licence or an Apache-licensed detector.

RGB/BGR (a real bug, found and fixed): this module's public contract is
that `detect()` takes an HxWx3 RGB array, matching every other image path
in this codebase (PIL-based, throughout engine/perception). But
ultralytics' `model.predict()` on a raw numpy array ASSUMES it is already
BGR (OpenCV convention) and flips it BGR->RGB internally before feeding the
model (confirmed in ultralytics/data/loaders.py: "NumPy color inputs are
assumed to use OpenCV-compatible BGR order"). Passing genuine RGB straight
through, as this module previously did, made ultralytics wrongly "correct"
already-correct pixels -- feeding the model a channel-swapped image on
every single call. Fixed by flipping RGB->BGR ourselves immediately before
the ultralytics call (`image[..., ::-1]`, verified byte-identical to
cv2.imread's native BGR output for the same file), so the public contract
(callers pass RGB) is unchanged but the model now actually receives the RGB
pixels it was trained on.

Tiled (sliced) inference is the DEFAULT inference path (`tiled=True`):
real CCTV/multi-lane frames are large (e.g. 1920x1080) with small, distant
plates (~30px wide); resized down to a single 640 (or even 1280) input,
a 30px plate shrinks below what the detector can resolve. Tiling instead
runs the detector on overlapping ~640px crops of the full-resolution frame
(so a plate is never seen smaller than it truly is) PLUS one full-frame
pass (catches large, near plates that straddle tile boundaries or are
already big enough not to need tiling), maps every tile's boxes back to
frame coordinates, and merges everything with class-agnostic NMS. Costs
roughly (n_tiles + 1) forward passes per frame instead of 1 -- see
eval_detector.py's measured per-frame timing for the actual number on this
hardware -- so `tiled=False` remains available for callers on a latency
budget (e.g. a live low-resolution feed where tiling would be wasted work).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# Pretrained baseline (Koushim/yolov8-license-plate-detection, see module
# docstring) -- kept available for callers that explicitly want the
# pre-fine-tuning weights.
PRETRAINED_WEIGHTS_PATH = Path("C:/sutra-data/ocr/weights/best.pt")

# Fine-tuned detector (module docstring) -- now the default.
DEFAULT_WEIGHTS_PATH = Path("C:/sutra-data/ocr/runs/detector_finetune/weights/best.pt")

# Tiling defaults -- see module docstring.
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 0.2
DEFAULT_TILE_NMS_IOU = 0.5


@dataclass
class PlateDetection:
    box_xyxy: tuple[float, float, float, float]  # pixel coords in the input image
    confidence: float
    crop: np.ndarray  # HxWx3 uint8 RGB crop, exactly box_xyxy cut out of the input


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def _class_agnostic_nms(
    boxes: list[tuple[float, float, float, float]], confidences: list[float], iou_threshold: float
) -> list[int]:
    """Returns indices to KEEP, highest confidence first, standard greedy
    NMS -- there is only one class (plate), so no per-class grouping."""
    order = sorted(range(len(boxes)), key=lambda i: -confidences[i])
    kept: list[int] = []
    for i in order:
        if all(_iou_xyxy(boxes[i], boxes[j]) < iou_threshold for j in kept):
            kept.append(i)
    return kept


def _generate_tiles(
    w: int, h: int, tile_size: int, overlap_frac: float
) -> list[tuple[int, int, int, int]]:
    """Sliding-window tile boxes (x0, y0, x1, y1) covering the whole
    (w, h) frame with `overlap_frac` overlap between neighbours. A frame
    dimension smaller than `tile_size` gets a single tile spanning it
    (no padding, no upsampling)."""

    def _starts(dim: int) -> list[int]:
        if dim <= tile_size:
            return [0]
        stride = max(int(tile_size * (1 - overlap_frac)), 1)
        starts = list(range(0, dim - tile_size, stride))
        starts.append(dim - tile_size)  # ensure the far edge is always covered
        return starts

    x_starts, y_starts = _starts(w), _starts(h)
    tiles = set()
    for y0 in y_starts:
        for x0 in x_starts:
            x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
            tiles.add((x0, y0, x1, y1))
    return sorted(tiles)


class PlateDetector:
    """Loads the YOLO model once; call `.detect(image)` per frame."""

    def __init__(self, weights_path: Path | str = DEFAULT_WEIGHTS_PATH, device: str | None = None):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"plate detector weights not found at {weights_path} -- "
                "see docs/decisions.md, 'OCR module' for the approved download."
            )
        self.model = YOLO(str(weights_path))
        self.device = device

    def _raw_boxes(
        self, image: np.ndarray, conf_threshold: float, imgsz: int | None
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """One ultralytics forward pass over the WHOLE `image` (no tiling,
        no cropping) -- returns (box_xyxy, confidence) pairs in `image`'s
        own pixel coordinates. `image` is RGB; flipped to BGR here (see
        module docstring) before the ultralytics call."""
        image_bgr = np.ascontiguousarray(image[..., ::-1])
        kwargs: dict = {"conf": conf_threshold, "device": self.device, "verbose": False}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        results = self.model.predict(source=image_bgr, **kwargs)
        if not results or results[0].boxes is None:
            return []
        out = []
        for box in results[0].boxes:
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            out.append((xyxy, conf))
        return out

    def _crop(self, image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        xi1, yi1 = max(int(round(x1)), 0), max(int(round(y1)), 0)
        xi2, yi2 = min(int(round(x2)), w), min(int(round(y2)), h)
        if xi2 <= xi1 or yi2 <= yi1:
            return None
        return image[yi1:yi2, xi1:xi2].copy()

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        imgsz: int | None = None,
        tiled: bool = True,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: float = DEFAULT_TILE_OVERLAP,
        tile_nms_iou: float = DEFAULT_TILE_NMS_IOU,
    ) -> list[PlateDetection]:
        """`image` is an HxWx3 RGB uint8 array (as returned by e.g.
        PIL.Image -> np.array). Returns one PlateDetection per detected
        plate, highest confidence first.

        `tiled=True` (default, see module docstring): runs one full-frame
        pass plus one pass per overlapping `tile_size`-px tile, maps tile
        boxes back to frame coordinates, and merges everything with
        class-agnostic NMS at `tile_nms_iou`. `tiled=False` runs a single
        full-frame pass only (cheaper, appropriate when the input is
        already small or latency-critical). `imgsz` (if given) is passed
        to ultralytics for every pass (full-frame and each tile); left
        `None` to use ultralytics' own default (640) at whatever
        resolution the pass's input already is.
        """
        h, w = image.shape[:2]
        boxes: list[tuple[float, float, float, float]] = []
        confidences: list[float] = []

        for box, conf in self._raw_boxes(image, conf_threshold, imgsz):
            boxes.append(box)
            confidences.append(conf)

        if tiled:
            for x0, y0, x1, y1 in _generate_tiles(w, h, tile_size, tile_overlap):
                tile = image[y0:y1, x0:x1]
                for (bx0, by0, bx1, by1), conf in self._raw_boxes(tile, conf_threshold, imgsz):
                    boxes.append((bx0 + x0, by0 + y0, bx1 + x0, by1 + y0))
                    confidences.append(conf)

        keep = _class_agnostic_nms(boxes, confidences, tile_nms_iou)

        detections: list[PlateDetection] = []
        for i in keep:
            crop = self._crop(image, boxes[i])
            if crop is None:
                continue
            detections.append(
                PlateDetection(box_xyxy=boxes[i], confidence=confidences[i], crop=crop)
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
