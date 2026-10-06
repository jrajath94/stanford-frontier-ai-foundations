#!/usr/bin/env python3
"""Regenerate three stale CS229 lesson plates with lesson-grounded numbers.

Deterministic PIL rendering (no AI image model, no API keys). Numbers are
copied from the lesson markdown and verified by arithmetic in the docstring.

1. plate-l03-sigmoid-vs-line.webp
   Lesson l03 (caption): "The line predicts 0.27 and 0.55 for two tumors.
   The sigmoid squeezes them to 0.57 and 0.63."
   sigmoid(0.27) = 0.5671 -> 0.57; sigmoid(0.55) = 0.6341 -> 0.63.
2. plate-l02-alpha-traces.webp
   Lesson l02 L297-306: alpha=0.01 -> theta_100 = 4*0.98^100 = 0.53;
   alpha=1.5 -> 4, -8, 16, -32, 64, -128.
3. plate-l05-gda-vs-qda.webp
   Lesson l05 L211-213: d=100 -> GDA about 5,251 knobs, QDA about 10,301.
   (Old plate footer read 10,200 vs 5,150, contradicting the lesson.)
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

def font(path, size):
    return ImageFont.truetype(path, size)

def text_w(draw, text, f):
    b = draw.textbbox((0, 0), text, font=f)
    return b[2] - b[0]

def fit(path, text, max_w, start):
    s = start
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    while s > 10 and text_w(probe, text, font(path, s)) > max_w:
        s -= 2
    return font(path, s)

def centered(draw, cx, y, text, f, fill=INK):
    w = text_w(draw, text, f)
    draw.text((cx - w / 2, y), text, font=f, fill=fill)

def panel(draw, x0, y0, x1, y1, r=45, sw=3):
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=INK, width=sw)

def pill(draw, x0, y0, x1, y1, lines, size=68):
    # auto-fit: shrink text until the widest line fits inside the pill
    f = font(FB, size)
    max_w = (x1 - x0) - 100
    while size > 20 and text_w(draw, max(lines, key=len), f) > max_w:
        size -= 2
        f = font(FB, size)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=(y1 - y0) // 2, fill=PILL)
    lh = int(size * 1.25)
    cy = (y0 + y1) / 2
    y = cy - lh * len(lines) / 2
    for ln in lines:
        centered(draw, (x0 + x1) / 2, y, ln, f, WHITE)
        y += lh

def wrap(draw, text, f, max_w):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if text_w(draw, t, f) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines

def make_plate(path, title_lines, left, right, pill_lines, footer,
               left_box, right_box, pill_box, body_size=58, pill_size=68):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    # title
    tf = fit(FB, max(title_lines, key=len), 1800, 140)
    ty = 96
    for ln in title_lines:
        centered(d, W / 2, ty, ln, tf)
        ty += 162
    # panels (drawn first; pill goes on top)
    boxes = []
    for (box, head, body) in [(left_box, left[0], left[1]),
                              (right_box, right[0], right[1])]:
        x0, y0, x1, y1 = box
        panel(d, x0, y0, x1, y1)
        cx = (x0 + x1) / 2
        hf = fit(FB, head, (x1 - x0) - 70, 76)
        centered(d, cx, y0 + 96, head, hf)
        boxes.append((box, head, body, cx))
    for (box, head, body, cx) in boxes:
        x0, y0, x1, y1 = box
        bf = font(FR, body_size)
        # shrink body if it would overflow the panel
        need = 0
        for item in body:
            need += len(wrap(d, item, bf, (x1 - x0) - 110)) + 0.4
        avail = (y1 - 50) - (y0 + 250)
        while need * body_size * 1.28 > avail and body_size > 30:
            body_size -= 2
            bf = font(FR, body_size)
            need = sum(len(wrap(d, item, bf, (x1 - x0) - 110)) + 0.4
                       for item in body)
        by = y0 + 246
        step = int(body_size * 1.3)
        for item in body:
            for ln in wrap(d, item, bf, (x1 - x0) - 110):
                centered(d, cx, by, ln, bf)
                by += step
            by += int(step * 0.45)
    # pill on top (may overlap panel edges, as in the originals)
    pill(d, *pill_box, pill_lines, size=pill_size)
    # footer
    ff = fit(FR, footer, 1800, 34)
    centered(d, W / 2, H - 88, footer, ff)
    im.save(os.path.join(OUT, path), "WEBP", quality=88, method=6)
    print("wrote", path, im.size)

# ---- 1. sigmoid vs line ----
make_plate(
    "plate-l03-sigmoid-vs-line.webp",
    ["The squeeze that fixes", "the line"],
    ("Before: the line", ["predicts 0.27", "predicts 0.55"]),
    ("After: the sigmoid", ["squeezes to 0.57", "squeezes to 0.63"]),
    ["smooth and", "monotone"],
    "The sigmoid turns any score into a probability. Source: original toy for the tumor job. Project: Stanford Frontier AI.",
    (60, 470, 590, 1110), (1330, 470, 1860, 1110), (615, 670, 1090, 910),
)

# ---- 2. alpha traces ----
make_plate(
    "plate-l02-alpha-traces.webp",
    ["Three learning rates,", "one bowl"],
    ("The setup", ["\u2022 J = theta squared", "\u2022 start at 4, bottom at 0"]),
    ("Three traces",
     ["\u2022 alpha 0.01 crawls: 100 steps reach 0.53",
      "\u2022 alpha 0.1 converges smooth",
      "\u2022 alpha 1.5 explodes: 4, minus 8, 16, minus 32, 64"]),
    ["watch the", "loss curve"],
    "Small alpha crawls, large alpha diverges. Source: original toy for the learning-rate rule. Project: Stanford Frontier AI.",
    (67, 470, 772, 1110), (1011, 470, 1794, 1110), (741, 680, 1042, 865),
)

# ---- 3. gda vs qda ----
make_plate(
    "plate-l05-gda-vs-qda.webp",
    ["One Sigma or two"],
    ("GDA", ["\u2022 one shared spread", "\u2022 boundary is a straight line"]),
    ("QDA", ["\u2022 each class its own spread",
             "\u2022 boundary curves around the tighter class"]),
    ["shared spread,", "straight line"],
    "Two spreads cost twice the knobs: 10,301 vs 5,251 at d = 100. Source: original toy for the covariance choice. Project: Stanford Frontier AI.",
    (89, 470, 733, 1110), (1211, 470, 1856, 1110), (672, 680, 1272, 890),
    body_size=56, pill_size=60,
)
