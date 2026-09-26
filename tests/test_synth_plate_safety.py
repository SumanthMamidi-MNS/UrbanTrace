"""Render-time character-containment guarantee (reviewer's FINDING 1):
every character named in a synthetic plate's label must be fully inside the
final composited image -- a crop may cut into the plate's border, never into
a character. Runs under the MAIN venv: Pillow + opencv-python-headless were
added to pyproject.toml's dev extras specifically so this test does not need
the separate, much heavier OCR (torch) venv -- see that file's comment.
"""

import random

from engine.perception.synth_plates import (
    LAYOUT_WEIGHTS,
    LAYOUTS,
    MAX_COMPOSITE_ATTEMPTS,
    STYLE_WEIGHTS,
    STYLES,
    _composite_and_recrop_safe,
    render_one_line,
    render_two_line,
    sample_plate,
)

N_SAMPLES = 300


def _assert_all_boxes_inside(
    img_size: tuple[int, int], boxes: list[tuple[int, int, int, int]]
) -> None:
    w, h = img_size
    for box in boxes:
        x0, y0, x1, y1 = box
        assert x1 > x0 and y1 > y0, f"box {box} has zero/negative area -- character vanished"
        assert x0 >= 0 and y0 >= 0, f"box {box} starts outside final frame {img_size}"
        assert x1 <= w and y1 <= h, f"box {box} extends outside final frame {img_size}"


def test_no_labelled_character_is_clipped_across_many_samples():
    """Renders N_SAMPLES one-line and two-line plates (the real weighted
    layout/style distribution) and asserts every character's FINAL bbox
    (after background compositing, rotation, perspective warp and crop --
    the pipeline reviewer's FINDING 1 diagnosed as clipping the last
    character of two-line top rows, and long one-line plates) lies strictly
    inside the final image. Also asserts the retry safety net never
    exhausts its budget over this sample size, which would indicate the
    proactive font-fit (part a) has regressed."""
    hard_failures = 0
    retried = 0
    two_line_seen = False
    one_line_seen = False

    for i in range(N_SAMPLES):
        seed_key = f"safety_test:{i}"
        rng = random.Random(seed_key)
        canonical, line1, line2 = sample_plate(rng)
        style = rng.choices(STYLES, weights=STYLE_WEIGHTS)[0]
        layout = rng.choices(LAYOUTS, weights=LAYOUT_WEIGHTS)[0]

        if layout == "one_line":
            one_line_seen = True
            img, boxes = render_one_line(line1 + line2, style, rng)
        else:
            two_line_seen = True
            img, boxes = render_two_line(line1, line2, style, rng)

        assert len(boxes) == len(canonical.replace("_", ""))

        final_img, final_boxes, attempts = _composite_and_recrop_safe(img, boxes, seed_key)
        if attempts > 0:
            retried += 1
        if attempts == MAX_COMPOSITE_ATTEMPTS - 1:
            hard_failures += 1

        _assert_all_boxes_inside(final_img.size, final_boxes)

    assert one_line_seen, "no one-line sample drawn in this run -- widen N_SAMPLES or check weights"
    assert two_line_seen, "no two-line sample drawn in this run -- widen N_SAMPLES or check weights"
    assert hard_failures == 0, (
        f"{hard_failures}/{N_SAMPLES} samples exhausted the composite retry budget "
        f"({MAX_COMPOSITE_ATTEMPTS} attempts) -- the font-fit safety margin is too thin"
    )
    print(f"[test_synth_plate_safety] {retried}/{N_SAMPLES} samples needed at least one retry")
