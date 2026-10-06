#!/usr/bin/env python3
"""Re-render plate-l02-feature-scaling.webp with a clean footer band.

Fix: the old PIL render let figure strokes (valley ellipse + zigzag)
cross the footer caption. This render confines ALL figure content to
y <= 1000, paints a clean footer band at y = 1060..1280, then stamps
the caption centered inside the band.

Numbers verified against l02-linear-regression.md, "feature scaling,
the hidden dial" subchapter: living area 1,000-3,000 sq ft, bedrooms
1-5; "taking hundreds of steps to drift down the flat floor".
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs229/assets"
W, H = 1920, 1280
BG = (247, 242, 234)
INK = (16, 26, 45)
PILL = (25, 35, 57)
WHITE = (255, 255, 255)
FB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

CONTENT_BOTTOM = 995  # nothing drawn below this except the footer band
BAND_TOP = 1060        # clean footer band: 1060..1280


def font(path, size):
    return ImageFont.truetype(path, size)


def text_w(d, text, f):
    b = d.textbbox((0, 0), text, font=f)
    return b[2] - b[0]


def fit_w(d, path, text, max_w, start, bold=True):
    s = start
    while s > 10 and text_w(d, text, font(path, s)) > max_w:
        s -= 2
    return font(path, s)


def centered(d, cx, y, text, f, fill=INK):
    d.text((cx - text_w(d, text, f) / 2, y), text, font=f, fill=fill)


def wrap(d, text, f, max_w):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if text_w(d, t, f) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)

# ---- clean footer band first: solid paper, nothing crosses it ----
d.rectangle([0, BAND_TOP, W, H], fill=BG)

# ---- title ----
tf = fit_w(d, FB, "Scale features before gradient descent", 1860, 140)
centered(d, W / 2, 64, "Scale features before gradient descent", tf)

# ---- panels ----
left_box = (60, 280, 920, CONTENT_BOTTOM)
right_box = (1000, 280, 1860, CONTENT_BOTTOM)
for (x0, y0, x1, y1) in (left_box, right_box):
    d.rounded_rectangle([x0, y0, x1, y1], radius=45, outline=INK, width=3)

hf = font(FB, 76)
centered(d, (left_box[0] + left_box[2]) / 2, left_box[1] + 55, "Before: unscaled", hf)
centered(d, (right_box[0] + right_box[2]) / 2, right_box[1] + 55, "After: scaled to 0 to 1", hf)

left_items = ["\u2022 Living area 1000 to 3000, bedrooms 1 to 5",
              "\u2022 Long narrow valley",
              "\u2022 Descent zigzags for hundreds of steps"]
right_items = ["\u2022 Round bowl",
               "\u2022 One alpha fits every knob",
               "\u2022 Descent walks straight to the bottom"]

DIAG_TOP = 760  # bullets must end above this; diagrams live 760..990


def bullets(box, items):
    x0, y0, x1, y1 = box
    size = 46
    while size > 28:
        bf = font(FR, size)
        step = int(size * 1.32)
        by = y0 + 175
        lines = []
        for item in items:
            for ln in wrap(d, item, bf, (x1 - x0) - 140):
                lines.append(ln)
            lines.append(None)  # item gap marker
        end = by + sum(step + 14 if ln is None else step for ln in lines) - 14
        if end <= DIAG_TOP:
            break
        size -= 2
    by = y0 + 175
    for ln in lines:
        if ln is None:
            by += 14
        else:
            centered(d, (x0 + x1) / 2, by, ln, bf)
            by += step
    return by


bullets(left_box, left_items)
bullets(right_box, right_items)

# ---- valley ellipse + zigzag (LEFT), 760..990 ----
ex0, ex1, ey0, ey1 = 420, 560, 765, 985
d.ellipse([ex0, ey0, ex1, ey1], outline=INK, width=10)
cx = (ex0 + ex1) / 2
pts = []
n = 8
for i in range(n + 1):
    y = ey0 + 18 + i * ((ey1 - ey0 - 36) / n)
    x = cx + (34 if i % 2 else -34)
    pts.append((x, y))
d.line(pts, fill=INK, width=8, joint="curve")

# ---- circle + straight arrow (RIGHT), 760..990 ----
ccx, ccy, r = 1430, 872, 115
d.ellipse([ccx - r, ccy - r, ccx + r, ccy + r], outline=INK, width=10)
ax0, ax1 = ccy - r + 20, ccy + r - 20
d.line([(ccx, ax0), (ccx, ax1)], fill=INK, width=12)
d.polygon([(ccx - 26, ax1 - 40), (ccx + 26, ax1 - 40), (ccx, ax1)], fill=INK)

# ---- center pill on top (may overlap panel edges, as in the original) ----
px0, py0, px1, py1 = 800, 560, 1120, 850
d.rounded_rectangle([px0, py0, px1, py1], radius=(py1 - py0) // 2, fill=PILL)
pf = fit_w(d, FB, "for every knob", (px1 - px0) - 90, 62)
lh = int(62 * 1.3)
centered(d, (px0 + px1) / 2, py0 + 52, "one alpha", pf, WHITE)
centered(d, (px0 + px1) / 2, py0 + 52 + lh, "for every knob", pf, WHITE)

# ---- footer caption, stamped INSIDE the clean band only ----
footer = ("Shell 4. Scaling turns the narrow valley into a round bowl. "
          "Feature scaling. Unscaled features make a narrow valley and "
          "zigzag descent. Scaled features make a round bowl and straight "
          "descent. Source: original plate for Stanford Frontier AI.")
ff = font(FR, 34)
lines = wrap(d, footer, ff, 1720)
lh = 44
y = BAND_TOP + (H - BAND_TOP - lh * len(lines)) / 2
for ln in lines:
    centered(d, W / 2, y, ln, ff)
    y += lh

# ---- self-check: the gutter between panels and the footer band stays clean ----
import numpy as np
px = np.asarray(im)
zone = px[1000:BAND_TOP]
dark = (zone[:, :, 0] < 100).sum()
assert dark == 0, f"{dark} dark pixels between content and footer band"

path = os.path.join(OUT, "plate-l02-feature-scaling.webp")
im.save(path, "WEBP", quality=88, method=6)
print("wrote", path, im.size)
