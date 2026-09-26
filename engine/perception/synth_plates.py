"""Synthetic Indian plate renderer for OCR pre-training.

Runs under the SEPARATE OCR venv (needs Pillow, numpy, opencv). Deterministic
given a seed: every sample's plate text, style, layout and augmentation draw
from `random.Random(f"{seed}:{split}:{i}")`, independent of iteration order
or batch size, matching the determinism convention already used by
sim/vehicles.py.

Plate grammar mirrors sim/vehicles.py's `_sample_true_plate` exactly (same
state-code list, same series-length and number-length weights), EXCEPT that
series letters exclude I and O: real Indian RTOs do not issue them as series
letters either, for the same reason this renderer must not -- in a bold
condensed font "O" is visually indistinguishable from "0". sim/vehicles.py
is out of scope for this change (a different track owns it), so this is a
deliberate, documented divergence, not a bug. This module is NOT imported
from sim/vehicles.py directly: that module pulls in networkx (for the road
graph), which the OCR venv does not install and should not need to -- see
docs/decisions.md, "OCR module".

Output layout, per split (train/val/test):
    <out>/<split>/images/<split>_NNNNNNN.jpg
    <out>/<split>/labels.csv   -- columns: image_path (relative to the split
        dir), canonical (10-char slot string incl. "_" padding), text_line1,
        text_line2 (grammar-based text, NOT the row split -- see below),
        layout, style, condition

Two-line labelling: a two-line plate's `text_line1` is the top row's actual
printed text (state+RTO+series) and `text_line2` is the bottom row's
(number) -- this is a GRAMMAR split, not a pixel-row split. The renderer
always saves ONE composite image per sample (both rows together, exactly
like a real plate-detector crop would hand over), so the pixel-level
top/bottom split is applied later, identically for synthetic and real data,
by `engine.perception.crnn.split_two_line_crop`.

Realism pipeline (every sample, regardless of `condition`): the plate is
rendered on its own canvas, optionally decorated with HSRP features (a
vertical blue "IND" strip and a chakra hologram -- real HSRP plates use
them, older plates don't, so most-but-not-all samples get them and the model
must learn to ignore them either way), then composited onto a random
vehicle-body-coloured background patch, given a few degrees of rotation and
a mild perspective warp, and re-cropped with a random margin around the
plate -- including an occasional slightly-too-tight crop that cuts into the
plate edge. This mimics what a real plate-detector crop actually looks like
(bumper/background context, off-centre text, imperfect box), which a tight
frontal centred render does not. Only AFTER this does the per-sample
`condition` augmenter (perspective/motion-blur/etc., including "clean" as a
no-op) apply its own, typically stronger, degradation on top.

HARD GUARANTEE (every character named in the label is fully visible): text
is laid out with a font sized to leave a safe margin from the plate edge on
every layout (`_fit_font_size`; the HSRP strip's reserved width is already
subtracted from the space text is fit against), and every character's
bounding box is tracked through rotation/paste/perspective/crop via a
parallel id-map (`_composite_and_recrop_safe`); if any character would end
up clipped, the composite's random parameters are re-sampled -- from the
sample's own seed, so still fully deterministic -- and retried. A crop may
cut into the plate's BORDER (that is realistic) but never into a character
(that is a labelling error). Without this, the renderer taught the model
that characters can be inferred past what the image shows -- see
docs/decisions.md's account of the same defect in the "occlusion" condition,
fixed the same way in spirit (never claim more than the image shows).

Conditions cover the PRD's named real-world degradations: perspective/angle,
motion blur, low light/glare, noise, JPEG artifacts, dirt/damage,
partial occlusion, rain, and low resolution. Each rendered sample gets
exactly ONE primary condition (including "clean"), so `eval_ocr.py` can
report a clean per-condition accuracy breakdown; real degradations often
compound, but a single dominant condition per sample keeps the breakdown
legible, which is the tradeoff made here.
"""

import argparse
import csv
import io
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from engine.contracts.plate import BLANK
from engine.paths import get_data_dir

# Mirrors sim/vehicles.py's STATE_CODES exactly -- kept as an independent
# constant (see module docstring: sim/vehicles.py cannot be imported here).
STATE_CODES = [
    "MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "WB",
    "AP", "TS", "KL", "MP", "HR", "PB", "BR",
]
_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
# Series letters exclude I and O -- see module docstring.
SERIES_LETTERS = [c for c in _LETTERS if c not in ("I", "O")]

STYLES = ["white_private", "yellow_commercial", "green_ev"]
STYLE_WEIGHTS = [0.70, 0.20, 0.10]
LAYOUTS = ["one_line", "two_line"]
LAYOUT_WEIGHTS = [0.75, 0.25]

CONDITIONS = [
    "clean", "perspective", "motion_blur", "gaussian_blur", "low_light",
    "glare", "noise", "jpeg", "dirt_damage", "occlusion", "rain", "low_res",
]
CONDITION_WEIGHTS = [0.15, 0.09, 0.08, 0.07, 0.08, 0.07, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07]

STYLE_COLORS = {
    "white_private": {"bg": (250, 250, 248), "fg": (10, 10, 10)},
    "yellow_commercial": {"bg": (255, 199, 44), "fg": (10, 10, 10)},
    "green_ev": {"bg": (30, 140, 70), "fg": (250, 250, 248)},
}
# Mild per-sample plate-colour variation (off-white / faded yellow / etc.):
# jitter each RGB channel independently within +-COLOR_JITTER.
COLOR_JITTER = 16

_VENDORED_FONT_PATH = Path(__file__).resolve().parent / "assets/fonts/SairaCondensed-Bold.ttf"
_DATA_DIR_FONT_PATH = get_data_dir() / "ocr/fonts/SairaCondensed-Bold.ttf"
# Prefer a font dropped in the data dir (e.g. a different weight/variant for
# local experimentation) if present, otherwise fall back to the font vendored
# into the repo (SIL OFL-licensed, see engine/perception/assets/fonts/OFL.txt)
# so this module -- and tests/test_synth_plate_safety.py, which imports it
# under the main venv -- works on a fresh clone with no URBANTRACE_DATA_DIR
# set and no machine-local font directory.
FONT_PATH = _DATA_DIR_FONT_PATH if _DATA_DIR_FONT_PATH.exists() else _VENDORED_FONT_PATH

ONE_LINE_SIZE = (400, 100)
TWO_LINE_SIZE = (240, 180)

# --- HSRP decoration (vertical "IND" strip + chakra hologram) ---
HSRP_PROBABILITY = 0.75
HSRP_STRIP_COLOR = (30, 55, 140)
HSRP_CHAKRA_FILL = (120, 145, 195)
HSRP_CHAKRA_OUTLINE = (50, 70, 130)
HSRP_SPOKE_COLOR = (235, 240, 250)

# --- realism compositing (background, rotation, perspective, recrop) ---
BACKGROUND_MARGIN_MIN = 0.08
BACKGROUND_MARGIN_MAX = 0.25
ROTATION_DEGREES_MAX = 5.0
COMPOSITE_PERSPECTIVE_SHIFT_MAX = 0.04
CROP_MARGIN_FRAC_MIN = 0.04
CROP_MARGIN_FRAC_MAX = 0.16
TIGHT_CROP_PROBABILITY = 0.15
# A "too-tight" crop can cut slightly INTO the plate edge (negative margin),
# not just omit the surrounding background -- this is the realistic failure
# mode a detector box sometimes makes, not merely "less padding".
TIGHT_CROP_MARGIN_MIN = -0.04
TIGHT_CROP_MARGIN_MAX = 0.02

# --- faux-bold/thin stroke-weight jitter (PIL's native stroke_width) ---
# Weighted toward 0 (the font's own weight): occasional extra-bold strokes
# approximate a stroke-weight jitter. PIL has no native erosion/thinning
# primitive, and the font is already a single bold weight, so "thin" is not
# separately simulated -- documented simplification.
STROKE_WIDTH_CHOICES = [0, 0, 0, 1, 1, 2]

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    size = max(size, 6)
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(str(FONT_PATH), size)
    return _FONT_CACHE[size]


def sample_plate(rng: random.Random) -> tuple[str, str, str]:
    """Returns (canonical_10char, grammar_line1, grammar_line2).
    grammar_line1 = state+RTO+series (no blanks), grammar_line2 = number (no
    blanks) -- these are the GRAMMAR segments, used verbatim as the one-line
    plate's full text (line1+line2) or the two-line plate's two rows."""
    state = rng.choice(STATE_CODES)
    rto = f"{rng.randint(1, 99):02d}"
    n_series_letters = rng.choices([1, 2], weights=[0.3, 0.7])[0]
    series = "".join(rng.choice(SERIES_LETTERS) for _ in range(n_series_letters))
    series_slots = series.ljust(2, BLANK)
    n_digits = rng.choices([1, 2, 3, 4], weights=[0.05, 0.1, 0.15, 0.7])[0]
    number = "".join(str(rng.randint(0, 9)) for _ in range(n_digits))
    number_slots = number.rjust(4, BLANK)
    canonical = state + rto + series_slots + number_slots
    return canonical, state + rto + series, number


def _jitter_color(c: tuple[int, int, int], rng: random.Random, delta: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, v + rng.randint(-delta, delta))) for v in c)


def _style_colors(style: str, rng: random.Random) -> dict[str, tuple[int, int, int]]:
    base = STYLE_COLORS[style]
    return {
        "bg": _jitter_color(base["bg"], rng, COLOR_JITTER),
        "fg": _jitter_color(base["fg"], rng, COLOR_JITTER // 2),
    }


# Fraction of the usable width text is allowed to actually fill -- the
# remainder is guaranteed empty margin on both sides. Font size is shrunk
# (see `_fit_font_size`) until text fits within this, on EVERY layout, so
# text never starts life within a hair of the plate edge before rotation,
# perspective and cropping even get a chance to clip it (docs: FINDING 1,
# "renderer clips characters the label still names").
TEXT_FIT_FRACTION = 0.92
MIN_FONT_SIZE = 10


def _measure_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    tracking: int,
    stroke_width: int = 0,
) -> int:
    total = 0
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_width)
        total += (bbox[2] - bbox[0]) + tracking
    return max(total - tracking, 0)


def _fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    base_size: int,
    usable_w: int,
    tracking: int,
    stroke_width: int,
) -> tuple[ImageFont.FreeTypeFont, int]:
    """Shrink `base_size` until `text` measures within
    `usable_w * TEXT_FIT_FRACTION`, so a safe margin from the plate edge is
    guaranteed by construction rather than by luck. Never shrinks below
    MIN_FONT_SIZE (a genuinely too-narrow usable_w is a config problem, not
    something to paper over by rendering illegible text)."""
    size = base_size
    target_w = usable_w * TEXT_FIT_FRACTION
    while size > MIN_FONT_SIZE:
        font = _load_font(size)
        if _measure_line(draw, text, font, tracking, stroke_width) <= target_w:
            return font, size
        size -= 1
    return _load_font(MIN_FONT_SIZE), MIN_FONT_SIZE


def _draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    tracking: int,
    stroke_width: int = 0,
) -> list[tuple[float, float, float, float]]:
    """Draws `text` and returns each character's absolute (x0, y0, x1, y1)
    ink bounding box, in the same order as `text` -- used to track every
    character through the compositing transforms (see
    `_composite_and_recrop_safe`)."""
    x, y = xy
    boxes: list[tuple[float, float, float, float]] = []
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=fill)
        abs_bbox = draw.textbbox((x, y), ch, font=font, stroke_width=stroke_width)
        boxes.append(abs_bbox)
        adv_bbox = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_width)
        x += (adv_bbox[2] - adv_bbox[0]) + tracking
    return boxes


def _hsrp_reserve_width(h: int, rng: random.Random) -> int:
    return max(int(h * rng.uniform(0.28, 0.36)), 14)


def _draw_hsrp_features(img: Image.Image, left_reserve: int, rng: random.Random) -> None:
    """Draw a chakra hologram + vertical 'IND' strip in the left `left_reserve`
    px of `img`, mutating it in place. See module docstring for why this is
    applied to only a majority of samples."""
    draw = ImageDraw.Draw(img)
    _, h = img.size
    margin = max(int(h * 0.06), 3)
    chakra_r = max(int(left_reserve * 0.42), 4)
    chakra_cx = margin + chakra_r
    chakra_cy = margin + chakra_r
    draw.ellipse(
        [chakra_cx - chakra_r, chakra_cy - chakra_r, chakra_cx + chakra_r, chakra_cy + chakra_r],
        fill=HSRP_CHAKRA_FILL,
        outline=HSRP_CHAKRA_OUTLINE,
    )
    for k in range(8):
        angle = k * math.pi / 4
        x2 = chakra_cx + chakra_r * 0.82 * math.cos(angle)
        y2 = chakra_cy + chakra_r * 0.82 * math.sin(angle)
        draw.line([chakra_cx, chakra_cy, x2, y2], fill=HSRP_SPOKE_COLOR, width=1)

    strip_w = max(int(left_reserve * 0.7), 5)
    strip_x0 = margin
    strip_y0 = chakra_cy + chakra_r + margin
    strip_y1 = h - margin
    if strip_y1 <= strip_y0:
        return
    draw.rectangle([strip_x0, strip_y0, strip_x0 + strip_w, strip_y1], fill=HSRP_STRIP_COLOR)

    font = _load_font(max(int(strip_w * 1.3), 8))
    label = Image.new("RGBA", (80, 24), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((0, 0), "IND", font=font, fill=(255, 255, 255, 255))
    bbox = label.getbbox()
    if bbox:
        label = label.crop(bbox)
    rotated = label.rotate(90, expand=True)
    px = strip_x0 + max((strip_w - rotated.width) // 2, 0)
    py = strip_y0 + max((strip_y1 - strip_y0 - rotated.height) // 2, 0)
    img.paste(rotated, (px, py), rotated)


def render_one_line(
    text: str, style: str, rng: random.Random
) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    w, h = ONE_LINE_SIZE
    colors = _style_colors(style, rng)
    img = Image.new("RGB", (w, h), colors["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, w - 3, h - 3], outline=(20, 20, 20), width=4)

    has_hsrp = rng.random() < HSRP_PROBABILITY
    left_reserve = _hsrp_reserve_width(h, rng) if has_hsrp else 0
    if has_hsrp:
        _draw_hsrp_features(img, left_reserve, rng)

    base_size = int(h * rng.uniform(0.52, 0.62))
    tracking = rng.randint(2, 9)
    stroke_width = rng.choice(STROKE_WIDTH_CHOICES)

    usable_x0 = left_reserve + 10 if has_hsrp else 8
    usable_w = max(w - usable_x0 - 8, 10)
    font, _ = _fit_font_size(draw, text, base_size, usable_w, tracking, stroke_width)
    text_w = _measure_line(draw, text, font, tracking, stroke_width)
    x = usable_x0 + max((usable_w - text_w) // 2, 0)
    y = (h - font.size) // 2 - int(font.size * 0.08)
    boxes = _draw_tracked_text(draw, (x, y), text, font, colors["fg"], tracking, stroke_width)
    return img, boxes


def render_two_line(
    line1: str, line2: str, style: str, rng: random.Random
) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    w, h = TWO_LINE_SIZE
    colors = _style_colors(style, rng)
    img = Image.new("RGB", (w, h), colors["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, w - 3, h - 3], outline=(20, 20, 20), width=4)

    has_hsrp = rng.random() < HSRP_PROBABILITY
    left_reserve = _hsrp_reserve_width(h, rng) if has_hsrp else 0
    if has_hsrp:
        _draw_hsrp_features(img, left_reserve, rng)

    half = h // 2
    base_size = int(half * rng.uniform(0.54, 0.64))
    tracking = rng.randint(2, 9)
    stroke_width = rng.choice(STROKE_WIDTH_CHOICES)
    usable_x0 = left_reserve + 8 if has_hsrp else 6
    usable_w = max(w - usable_x0 - 6, 10)

    # Both rows share ONE font size -- fit against whichever row is wider,
    # so the two rows look consistent rather than each finding its own size.
    fit_size = min(
        _fit_font_size(draw, line1, base_size, usable_w, tracking, stroke_width)[1],
        _fit_font_size(draw, line2, base_size, usable_w, tracking, stroke_width)[1],
    )
    font = _load_font(fit_size)

    boxes: list[tuple[float, float, float, float]] = []
    for line, y0 in [(line1, 0), (line2, half)]:
        text_w = _measure_line(draw, line, font, tracking, stroke_width)
        x = usable_x0 + max((usable_w - text_w) // 2, 0)
        y = y0 + (half - font.size) // 2 - int(font.size * 0.08)
        line_boxes = _draw_tracked_text(
            draw, (x, y), line, font, colors["fg"], tracking, stroke_width
        )
        boxes.extend(line_boxes)
    return img, boxes


def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _random_background(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """A flat-to-gradient patch in a random vehicle-body-ish colour -- stands
    in for the bumper/body/road context around a real plate crop."""
    w, h = size
    c1 = np.array([rng.randint(20, 220) for _ in range(3)], dtype=np.float32)
    c2 = np.clip(c1 + np.array([rng.randint(-40, 40) for _ in range(3)]), 0, 255).astype(np.float32)
    t = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
    arr = c1 * (1 - t) + c2 * t
    arr = np.broadcast_to(arr, (h, w, 3))
    return _to_pil(arr)


# FINDING 1 (renderer clips characters the label still names): every
# character's bounding box is tracked through rotation/paste/perspective/crop
# via a parallel nearest-neighbour "id map" (see `_char_id_map` and
# `_composite_and_recrop_safe`), and an attempt is rejected -- retried with
# freshly, deterministically re-sampled parameters -- unless every character
# keeps at least this fraction of its original pixel area. NEAREST resampling
# of a solid id-map rectangle loses at most a few boundary pixels to
# aliasing when the shape is genuinely fully inside the frame, so this is a
# generous margin for that noise while still catching real clipping (which
# typically removes 30-100% of a character's area, not a sliver of it).
CHAR_SURVIVAL_THRESHOLD = 0.90
MAX_COMPOSITE_ATTEMPTS = 40


def _char_id_map(
    size: tuple[int, int], char_boxes: list[tuple[float, float, float, float]]
) -> np.ndarray:
    """uint8 label map, same size as `size`: pixel value k+1 inside
    character k's ink bbox (1-indexed; 0 = background/no character), so it
    composes safely with PIL's rotate(fillcolor=0) and cv2's
    BORDER_CONSTANT/borderValue=0 for anything rotation or warp brings in
    from outside the original canvas -- that content is background too."""
    w, h = size
    id_map = np.zeros((h, w), dtype=np.uint8)
    for idx, (x0, y0, x1, y1) in enumerate(char_boxes, start=1):
        xi0, yi0 = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
        xi1, yi1 = min(int(math.ceil(x1)), w), min(int(math.ceil(y1)), h)
        if xi1 > xi0 and yi1 > yi0:
            id_map[yi0:yi1, xi0:xi1] = idx
    return id_map


def _composite_attempt(
    plate_img: Image.Image,
    char_boxes: list[tuple[float, float, float, float]],
    rng: random.Random,
) -> tuple[Image.Image, np.ndarray]:
    """One randomised background/rotate/perspective/crop attempt. Returns
    (final_rgb_image, final_id_map): the id map is carried through the
    IDENTICAL geometric ops as the RGB image (nearest-neighbour, zero-filled
    borders throughout), so comparing its per-character pixel counts
    before/after tells `_composite_and_recrop_safe` whether any character
    was clipped by this particular draw of parameters."""
    pw, ph = plate_img.size
    id_map = _char_id_map((pw, ph), char_boxes)

    margin_frac = rng.uniform(BACKGROUND_MARGIN_MIN, BACKGROUND_MARGIN_MAX)
    pad_w, pad_h = int(pw * margin_frac), int(ph * margin_frac)
    canvas_w, canvas_h = pw + 2 * pad_w, ph + 2 * pad_h
    canvas = _random_background((canvas_w, canvas_h), rng)

    angle = rng.uniform(-ROTATION_DEGREES_MAX, ROTATION_DEGREES_MAX)
    plate_rgba = plate_img.convert("RGBA")
    rotated = plate_rgba.rotate(angle, expand=True, resample=Image.BICUBIC)
    rw, rh = rotated.size
    id_rotated = Image.fromarray(id_map, mode="L").rotate(
        angle, expand=True, resample=Image.NEAREST, fillcolor=0
    )
    if id_rotated.size != (rw, rh):
        # Belt-and-suspenders: PIL's expand-size math is a pure function of
        # (size, angle), independent of pixel content, so this should never
        # actually trigger -- but a resize fallback is cheap insurance
        # against relying on that undocumented-but-observed behaviour.
        id_rotated = id_rotated.resize((rw, rh), Image.NEAREST)

    max_x, max_y = max(canvas_w - rw, 0), max(canvas_h - rh, 0)
    px = rng.randint(0, max_x) if max_x > 0 else 0
    py = rng.randint(0, max_y) if max_y > 0 else 0
    canvas.paste(rotated, (px, py), rotated)
    id_canvas_img = Image.new("L", (canvas_w, canvas_h), 0)
    id_canvas_img.paste(id_rotated, (px, py))
    id_canvas = np.array(id_canvas_img)

    arr = _to_np(canvas)
    ch, cw = arr.shape[:2]
    max_shift = rng.uniform(0.01, COMPOSITE_PERSPECTIVE_SHIFT_MAX)
    corners = [(0, 0), (cw, 0), (cw, ch), (0, ch)]
    src = np.float32(corners)
    dst = np.float32(
        [
            (
                x + rng.uniform(-max_shift, max_shift) * cw,
                y + rng.uniform(-max_shift, max_shift) * ch,
            )
            for x, y in corners
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, matrix, (cw, ch), borderMode=cv2.BORDER_REPLICATE)
    id_warped = cv2.warpPerspective(
        id_canvas,
        matrix,
        (cw, ch),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    composite = _to_pil(warped)

    x0, y0, x1, y1 = px, py, px + rw, py + rh
    tight = rng.random() < TIGHT_CROP_PROBABILITY
    if tight:
        # (b) a too-tight crop may cut into the plate BORDER, never into a
        # character -- enforced by the survival check in the caller, not by
        # restricting this margin range itself.
        margin_x = rng.uniform(TIGHT_CROP_MARGIN_MIN, TIGHT_CROP_MARGIN_MAX) * rw
        margin_y = rng.uniform(TIGHT_CROP_MARGIN_MIN, TIGHT_CROP_MARGIN_MAX) * rh
    else:
        margin_x = rng.uniform(CROP_MARGIN_FRAC_MIN, CROP_MARGIN_FRAC_MAX) * rw
        margin_y = rng.uniform(CROP_MARGIN_FRAC_MIN, CROP_MARGIN_FRAC_MAX) * rh

    cx0 = max(int(x0 - margin_x), 0)
    cy0 = max(int(y0 - margin_y), 0)
    cx1 = min(int(x1 + margin_x), cw)
    cy1 = min(int(y1 + margin_y), ch)
    if cx1 <= cx0 or cy1 <= cy0:
        cx0, cy0, cx1, cy1 = 0, 0, cw, ch

    final_img = composite.crop((cx0, cy0, cx1, cy1)).convert("RGB")
    final_id_map = id_warped[cy0:cy1, cx0:cx1]
    return final_img, final_id_map


def _chars_fully_present(id_map_before: np.ndarray, id_map_after: np.ndarray, n_chars: int) -> bool:
    for idx in range(1, n_chars + 1):
        before = int(np.sum(id_map_before == idx))
        after = int(np.sum(id_map_after == idx))
        if before == 0:
            continue  # degenerate zero-area box: never observed, never block on it
        if after < CHAR_SURVIVAL_THRESHOLD * before:
            return False
    return True


def _final_char_boxes(id_map: np.ndarray, n_chars: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for idx in range(1, n_chars + 1):
        ys, xs = np.where(id_map == idx)
        if len(xs) == 0:
            boxes.append((0, 0, 0, 0))
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def _composite_and_recrop_safe(
    plate_img: Image.Image,
    char_boxes: list[tuple[float, float, float, float]],
    seed_key: str,
) -> tuple[Image.Image, list[tuple[int, int, int, int]], int]:
    """Paste `plate_img` onto a larger random background, rotate it a few
    degrees, apply a mild perspective warp to the whole composite, then crop
    back down with a random margin around the plate's own bounding box --
    occasionally a "too-tight" crop. See module docstring for why this runs
    on every sample, before the per-condition augmenter.

    FINDING 1 fix: this is a hard guarantee, not a hope. Every character's
    bbox is tracked through the whole pipeline (`_composite_attempt`'s id
    map); if any character would end up clipped, the composite parameters
    are re-sampled -- deterministically, from `seed_key` + attempt number,
    independent of whatever entropy the caller's own rng already consumed --
    and retried, up to MAX_COMPOSITE_ATTEMPTS times. Returns
    (final_image, final_char_boxes, attempts_used) so callers can log the
    retry rate; `attempts_used == 0` means the first draw already worked.
    """
    id_map_before = _char_id_map(plate_img.size, char_boxes)
    n_chars = len(char_boxes)

    img, id_map_after, attempt = None, None, 0
    for attempt in range(MAX_COMPOSITE_ATTEMPTS):
        attempt_rng = random.Random(f"{seed_key}:composite:{attempt}")
        img, id_map_after = _composite_attempt(plate_img, char_boxes, attempt_rng)
        if _chars_fully_present(id_map_before, id_map_after, n_chars):
            return img, _final_char_boxes(id_map_after, n_chars), attempt

    print(
        f"[synth_plates] WARNING: composite retry budget exhausted for {seed_key} "
        f"({MAX_COMPOSITE_ATTEMPTS} attempts) -- using the last attempt anyway",
        flush=True,
    )
    return img, _final_char_boxes(id_map_after, n_chars), attempt


# --- Per-condition augmentations: each takes (PIL Image, random.Random) ->
# PIL Image. Applied AFTER _composite_and_recrop -- see module docstring.


def apply_clean(img: Image.Image, rng: random.Random) -> Image.Image:
    return img


def apply_perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = _to_np(img)
    h, w = arr.shape[:2]
    max_shift = rng.uniform(0.04, 0.14)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    src = np.float32(corners)
    dst = np.float32(
        [
            (
                cx + rng.uniform(-max_shift, max_shift) * w,
                cy + rng.uniform(-max_shift, max_shift) * h,
            )
            for cx, cy in corners
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return _to_pil(warped)


def apply_motion_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = _to_np(img)
    k = rng.choice([5, 7, 9, 11])
    angle = rng.uniform(0, 180)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    total = kernel.sum()
    kernel /= total if total != 0 else 1.0
    blurred = cv2.filter2D(arr, -1, kernel)
    return _to_pil(blurred)


def apply_gaussian_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    radius = rng.uniform(0.8, 2.4)
    return img.filter(ImageFilter.GaussianBlur(radius))


def apply_low_light(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = _to_np(img).astype(np.float32) / 255.0
    gamma = rng.uniform(1.6, 3.0)
    brightness = rng.uniform(0.35, 0.65)
    contrast = rng.uniform(0.6, 0.9)
    arr = np.power(np.clip(arr, 0, 1), gamma)
    mean = arr.mean()
    arr = (arr - mean) * contrast + mean
    arr = arr * brightness
    return _to_pil(arr * 255.0)


def apply_glare(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = _to_np(img).astype(np.float32)
    h, w = arr.shape[:2]
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    radius = rng.uniform(0.3, 0.6) * max(w, h)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1 - dist / radius, 0, 1) ** 2
    intensity = rng.uniform(120, 220)
    arr = arr + mask[..., None] * intensity
    return _to_pil(arr)


def apply_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = _to_np(img).astype(np.float32)
    sigma = rng.uniform(8, 30)
    noise_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    arr = arr + noise_rng.normal(0, sigma, arr.shape)
    return _to_pil(arr)


def apply_jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(8, 35)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_dirt_damage(img: Image.Image, rng: random.Random) -> Image.Image:
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = overlay.size
    for _ in range(rng.randint(2, 6)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.03, 0.12) * max(w, h)
        color = rng.choice([(40, 30, 20), (10, 10, 10), (90, 80, 60)])
        alpha = rng.randint(90, 200)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, alpha))
    return overlay.convert("RGB")


def apply_occlusion(img: Image.Image, rng: random.Random) -> Image.Image:
    """Partial occlusion (mud, tow bar, sticker) over part of the characters.
    Applied to the composited crop, where the text sits roughly in the middle
    ~40% of the height, so the band is placed inside the central region and is
    at most ~half the text height: characters stay legible but degraded.
    A full-height band would erase characters that the label still names,
    teaching the model to confidently hallucinate unseen characters -- the
    opposite of what the engine needs (unread slots must stay uncertain)."""
    arr = _to_np(img).copy()
    h, w = arr.shape[:2]
    band_w = rng.uniform(0.12, 0.40) * w
    band_h = rng.uniform(0.10, 0.20) * h
    x0 = rng.uniform(0, w - band_w)
    y0 = rng.uniform(0.28 * h, 0.72 * h - band_h)
    color = rng.choice([(0, 0, 0), (60, 60, 60), (120, 110, 90)])
    arr[int(y0) : int(y0 + band_h), int(x0) : int(x0 + band_w)] = color
    return _to_pil(arr)


def apply_rain(img: Image.Image, rng: random.Random) -> Image.Image:
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = overlay.size
    for _ in range(rng.randint(40, 90)):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        length = rng.uniform(6, 16)
        angle = rng.uniform(65, 80)
        dx = length * math.cos(math.radians(angle))
        dy = length * math.sin(math.radians(angle))
        draw.line([x, y, x + dx, y + dy], fill=(200, 210, 220, 90), width=1)
    return overlay.filter(ImageFilter.GaussianBlur(0.4)).convert("RGB")


def apply_low_res(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    scale = rng.uniform(0.2, 0.4)
    small = img.resize((max(int(w * scale), 8), max(int(h * scale), 8)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


AUGMENTERS = {
    "clean": apply_clean,
    "perspective": apply_perspective,
    "motion_blur": apply_motion_blur,
    "gaussian_blur": apply_gaussian_blur,
    "low_light": apply_low_light,
    "glare": apply_glare,
    "noise": apply_noise,
    "jpeg": apply_jpeg,
    "dirt_damage": apply_dirt_damage,
    "occlusion": apply_occlusion,
    "rain": apply_rain,
    "low_res": apply_low_res,
}

CSV_FIELDS = ["image_path", "canonical", "text_line1", "text_line2", "layout", "style", "condition"]


def render_dataset(n: int, seed: int, out_dir: Path, split: str, log_every: int = 2000) -> Path:
    """Render `n` samples for `split` (train/val/test) into
    <out_dir>/<split>/images/ + <out_dir>/<split>/labels.csv. Returns the
    labels.csv path. Deterministic: sample i always uses
    random.Random(f"{seed}:{split}:{i}"), independent of `n` or order."""
    split_dir = out_dir / split
    img_dir = split_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    retry_counts: list[int] = []
    for i in range(n):
        seed_key = f"{seed}:{split}:{i}"
        rng = random.Random(seed_key)
        canonical, grammar_line1, grammar_line2 = sample_plate(rng)
        style = rng.choices(STYLES, weights=STYLE_WEIGHTS)[0]
        layout = rng.choices(LAYOUTS, weights=LAYOUT_WEIGHTS)[0]
        condition = rng.choices(CONDITIONS, weights=CONDITION_WEIGHTS)[0]

        if layout == "one_line":
            full_text = grammar_line1 + grammar_line2
            img, char_boxes = render_one_line(full_text, style, rng)
            text_line1, text_line2 = full_text, ""
        else:
            img, char_boxes = render_two_line(grammar_line1, grammar_line2, style, rng)
            text_line1, text_line2 = grammar_line1, grammar_line2

        img, _final_boxes, attempts = _composite_and_recrop_safe(img, char_boxes, seed_key)
        retry_counts.append(attempts)
        img = AUGMENTERS[condition](img, rng)

        fname = f"{split}_{i:07d}.jpg"
        img.save(img_dir / fname, quality=92)
        rows.append(
            {
                "image_path": f"images/{fname}",
                "canonical": canonical,
                "text_line1": text_line1,
                "text_line2": text_line2,
                "layout": layout,
                "style": style,
                "condition": condition,
            }
        )
        if (i + 1) % log_every == 0:
            print(f"[{split}] rendered {i + 1}/{n}", flush=True)

    csv_path = split_dir / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if retry_counts:
        arr = np.array(retry_counts)
        n_retried = int(np.sum(arr > 0))
        n_hard_failed = int(np.sum(arr == MAX_COMPOSITE_ATTEMPTS - 1))
        print(
            f"[{split}] composite retries: mean={arr.mean():.3f} max={arr.max()} "
            f"needed_retry={n_retried}/{len(arr)} ({100 * n_retried / len(arr):.2f}%) "
            f"hard_failures={n_hard_failed}",
            flush=True,
        )
    print(f"[{split}] wrote {len(rows)} rows -> {csv_path}", flush=True)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render synthetic Indian plates for OCR pre-training."
    )
    parser.add_argument("--out", type=Path, default=get_data_dir() / "ocr/synth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=60000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=5000)
    args = parser.parse_args()

    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"font not found at {FONT_PATH} -- see docs/decisions.md, 'OCR module'"
        )

    render_dataset(args.n_train, args.seed, args.out, "train")
    render_dataset(args.n_val, args.seed + 1, args.out, "val")
    render_dataset(args.n_test, args.seed + 2, args.out, "test")


if __name__ == "__main__":
    main()
