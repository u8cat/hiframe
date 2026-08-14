#!/usr/bin/env python3
"""Report, and with --apply remove, the transparent margin around each logo.

The margin is measured by rendering the drawing and looking for the ink, then
translated back into user units so that the viewBox can be narrowed onto it. The
drawing itself is never touched, so nothing is rescaled or redrawn.

Narrowing happens once per run. Repeating it would eat into the drawing: each
crop lands between two pixels, and the sliver of ink outside the new viewBox is
then lost, so a second measurement finds a margin that is not really there. The
ink box is also grown by a pixel before being applied, and the result is checked
for ink pushed outside the viewBox, the file being restored if there is any.

    python3 tighten.py <logo_dir> [--apply]
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

RENDER = 1000   # px along the longer side, enough to place an edge precisely
SLACK = 1       # px of ink kept beyond the measured box, against rounding
TIGHT = SLACK + 1  # px of margin below which a logo counts as tight already


def root_tag(svg):
    return re.search(r"<svg\b[^>]*>", svg, re.S).group()


def attr(tag, name):
    m = re.search(rf'\b{name}\s*=\s*"([^"]*)"', tag)
    return m.group(1) if m else None


def viewbox_of(svg):
    """The user space rectangle the drawing occupies, or None."""
    tag = root_tag(svg)
    if vb := attr(tag, "viewBox"):
        parts = [float(x) for x in re.split(r"[\s,]+", vb.strip())]
        if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
            return parts
    # Without a viewBox the user unit is the pixel, so width and height give it.
    try:
        w = float(re.sub(r"[a-z%]+$", "", attr(tag, "width") or ""))
        h = float(re.sub(r"[a-z%]+$", "", attr(tag, "height") or ""))
        return [0.0, 0.0, w, h] if w > 0 and h > 0 else None
    except (TypeError, ValueError):
        return None


def set_box(tag, box):
    """The root tag with its viewBox, width and height set to `box`."""
    x, y, w, h = box
    for name, value in (("viewBox", f"{x:.8g} {y:.8g} {w:.8g} {h:.8g}"),
                        ("width", f"{w:.8g}"), ("height", f"{h:.8g}")):
        if re.search(rf'\b{name}\s*=\s*"[^"]*"', tag):
            tag = re.sub(rf'\b{name}\s*=\s*"[^"]*"', f'{name}="{value}"', tag, count=1)
        else:
            tag = tag[:-1].rstrip() + f' {name}="{value}">'
    return tag


def render(svg, box):
    """Rasterize `svg` over exactly `box`, so that pixels map to user units."""
    x, y, w, h = box
    width = RENDER if w >= h else max(1, round(RENDER * w / h))
    height = RENDER if h > w else max(1, round(RENDER * h / w))
    with tempfile.NamedTemporaryFile("w", suffix=".svg") as source, \
         tempfile.NamedTemporaryFile(suffix=".png") as out:
        source.write(svg.replace(root_tag(svg), set_box(root_tag(svg), box), 1))
        source.flush()
        subprocess.run(["rsvg-convert", "-w", str(width), "-h", str(height),
                        source.name, "-o", out.name], check=True, capture_output=True)
        return cv2.imread(out.name, cv2.IMREAD_UNCHANGED)


def ink_box(image):
    """Bounding box of anything not fully transparent, as (x0, y0, x1, y1)."""
    if image is None:
        return None
    mask = image[:, :, 3] > 0 if image.shape[2] == 4 else image.min(axis=2) < 250
    ys, xs = np.nonzero(mask)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def spills(svg, box):
    """Whether any ink of the drawing falls outside `box`."""
    x, y, w, h = box
    wide = [x - w / 4, y - h / 4, w * 1.5, h * 1.5]
    image = render(svg, wide)
    inked = ink_box(image)
    if not inked:
        return False
    ih, iw = image.shape[:2]
    x0, y0, x1, y1 = inked
    # The original box occupies the middle two thirds of the widened render.
    return (x0 < iw / 6 - 1 or x1 > iw * 5 / 6 + 1
            or y0 < ih / 6 - 1 or y1 > ih * 5 / 6 + 1)


def main(logo_dir, apply):
    print(f"{'logo':<38} {'margin L/R/T/B (%)':<24} action")
    changed = 0
    for path in sorted(Path(logo_dir).glob("*.svg")):
        svg = path.read_text(encoding="utf-8", errors="replace")
        box = viewbox_of(svg)
        if not box:
            print(f"{path.name:<38} {'?':<24} skipped, no viewBox or size")
            continue
        if (par := attr(root_tag(svg), "preserveAspectRatio")) and "none" in par:
            print(f"{path.name:<38} {'?':<24} skipped, preserveAspectRatio={par}")
            continue

        image = render(svg, box)
        inked = ink_box(image)
        if not inked:
            print(f"{path.name:<38} {'?':<24} skipped, renders empty")
            continue
        x0, y0, x1, y1 = inked
        h, w = image.shape[:2]
        left, right, top, bottom = x0 / w, (w - 1 - x1) / w, y0 / h, (h - 1 - y1) / h
        margins = f"{left:5.1%} {right:5.1%} {top:5.1%} {bottom:5.1%}".replace("%", "")

        # Measured in pixels, so that the slack left by a previous run, which is
        # a share of a short side and so looks large as a fraction, still counts
        # as tight and the run stays a no-op.
        if max(x0, w - 1 - x1, y0, h - 1 - y1) <= TIGHT:
            print(f"{path.name:<38} {margins:<24} already tight")
            continue
        if not apply:
            print(f"{path.name:<38} {margins:<24} would tighten")
            changed += 1
            continue

        # Keep a pixel of slack, so that rounding cannot bite into the drawing.
        mx, my, vw, vh = box
        sx, sy = vw / w, vh / h
        tight = [mx + (x0 - SLACK) * sx, my + (y0 - SLACK) * sy,
                 (x1 - x0 + 1 + 2 * SLACK) * sx, (y1 - y0 + 1 + 2 * SLACK) * sy]
        tightened = svg.replace(root_tag(svg), set_box(root_tag(svg), tight), 1)

        if spills(tightened, tight):
            print(f"{path.name:<38} {margins:<24} LEFT ALONE, would cut the drawing")
            continue
        path.write_text(tightened, encoding="utf-8")
        after = ink_box(render(tightened, tight))
        ah, aw = render(tightened, tight).shape[:2]
        worst = max(after[0] / aw, (aw - 1 - after[2]) / aw,
                    after[1] / ah, (ah - 1 - after[3]) / ah)
        print(f"{path.name:<38} {margins:<24} tightened, {worst:.2%} left")
        changed += 1

    print(f"\n{changed} logo(s) {'tightened' if apply else 'to tighten'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".", "--apply" in sys.argv)
