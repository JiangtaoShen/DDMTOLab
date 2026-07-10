"""Generate ui/assets/logo.png from the DDMTOLab logo mark geometry.

The mark is the two gradient-filled polygons from logo.svg (Visio export /
claude.ai/design "DDMTOLab GUI (Refined)"). Rasterized here with Pillow so the
GUI needs no SVG renderer (cairosvg) at runtime: main._load_logo falls back to
logo.png automatically.

Run:  python ui/assets/generate_logo.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# viewBox of the mark (matches the refined design header SVG)
VB_X, VB_Y, VB_W, VB_H = 0.0, 466.0, 128.0, 130.0
SS = 8            # supersampling factor for antialiasing
TARGET_H = 240    # output height in px (display scales it down)

# Left polygon: path "M37.4 545.13 ... Z" with translate(0, -34.5273)
_DY = -34.5273
POLY_A = [(x, y + _DY) for x, y in [
    (37.4, 545.13), (37.4, 584.53), (48.91, 595.28), (60.42, 595.28),
    (60.42, 531.98), (60.42, 530.06), (60.42, 501.28), (0.0, 535.81),
    (0.0, 566.5),
]]
# Right polygon: path "M0 595.28 ... Z" with translate(67.1363, 0)
_DX = 67.1363
POLY_B = [(x + _DX, y) for x, y in [
    (0.0, 595.28), (23.02, 580.89), (23.02, 510.6), (60.42, 531.98),
    (60.42, 501.28), (0.0, 466.76), (0.0, 495.53), (0.0, 497.45),
]]

# Horizontal gradients across each polygon's own bounding box
GRAD_A = ((0x00, 0xB0, 0xF0), (0x46, 0x72, 0xC4))   # left #00b0f0 -> right #4672c4
GRAD_B = ((0x46, 0x72, 0xC4), (0x00, 0xFE, 0xFE))   # left #4672c4 -> right #00fefe


def _poly_layer(points, c_left, c_right, size):
    w, h = size
    pts = [((x - VB_X) * SS, (y - VB_Y) * SS) for x, y in points]
    xs = [p[0] for p in pts]
    x0, x1 = min(xs), max(xs)

    grad = np.zeros((h, w, 4), np.uint8)
    t = np.clip((np.arange(w) - x0) / max(x1 - x0, 1e-9), 0.0, 1.0)
    for ch, (a, b) in enumerate(zip(c_left, c_right)):
        grad[:, :, ch] = (a + (b - a) * t).astype(np.uint8)[None, :]
    grad[:, :, 3] = 255

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    layer.paste(Image.fromarray(grad, "RGBA"), (0, 0), mask)
    return layer


def main():
    size = (int(VB_W * SS), int(VB_H * SS))
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    img.alpha_composite(_poly_layer(POLY_A, *GRAD_A, size=size))
    img.alpha_composite(_poly_layer(POLY_B, *GRAD_B, size=size))

    img = img.crop(img.getbbox())
    out_w = round(img.width * TARGET_H / img.height)
    img = img.resize((out_w, TARGET_H), Image.LANCZOS)

    out = Path(__file__).parent / "logo.png"
    img.save(out)
    print(f"wrote {out} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
