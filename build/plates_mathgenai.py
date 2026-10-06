#!/usr/bin/env python3
"""math-genai plate renderer. Warm-paper lesson plates via PIL, saved as webp.

Visual system: BG #F7F4EE, ink #1B2838, muted #5C6B7A, line #D9D3C7,
panel #FFFDF8, count #E7F1F8, new #E7F4EF, active #F4E6D4, chip #E6E2DA,
teal #1F7A72, orange #C46B2C, focus #1E4D8C, pink #F3D4D8, yellow #F6E7A8,
green #D9E8D3. 8px grid. Flat fills only.
"""
import os, math
from PIL import Image, ImageDraw, ImageFont

BG = "#F7F4EE"; INK = "#1B2838"; MUTED = "#5C6B7A"; LINE = "#D9D3C7"
PANEL = "#FFFDF8"; COUNT = "#E7F1F8"; NEW = "#E7F4EF"; ACTIVE = "#F4E6D4"
CHIP = "#E6E2DA"; TEAL = "#1F7A72"; ORANGE = "#C46B2C"; FOCUS = "#1E4D8C"
PINK = "#F3D4D8"; YELLOW = "#F6E7A8"; GREEN = "#D9E8D3"

ASSETS = "/home/hatch/workspace/stanford-frontier-ai/content/v2/math-genai/assets"

FR = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
FB = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
FMI = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # has math glyphs

def _font(path, size, bold=False):
    return ImageFont.truetype(path, size)

class Plate:
    def __init__(self, title, claim, footer, source="original toy",
                 width=960, inner_h=360):
        self.w = width
        self.top = 104
        self.h = self.top + inner_h + 72
        self.img = Image.new("RGB", (self.w, self.h), BG)
        self.d = ImageDraw.Draw(self.img)
        self.d.text((32, 36), title, font=_font(FB, 30), fill=INK)
        self.d.text((32, 72), claim, font=_font(FR, 16), fill=MUTED)
        self._title, self._claim, self._footer, self._source = title, claim, footer, source

    # -- primitives -----------------------------------------------------
    def panel(self, x, y, w, h, fill=PANEL, label=None):
        self.d.rounded_rectangle([x, y, x + w, y + h], radius=12,
                                 fill=fill, outline=LINE, width=2)
        if label:
            self.d.text((x + 16, y + 16), label, font=_font(FB, 14), fill=MUTED)

    def rect(self, x, y, w, h, fill, label=None, size=15, stroke=INK,
             rx=8, color=INK, bold=True):
        self.d.rounded_rectangle([x, y, x + w, y + h], radius=rx,
                                 fill=fill, outline=stroke, width=2)
        if label is not None:
            f = _font(FB, size) if bold else _font(FR, size)
            tw = self.d.textlength(label, font=f)
            self.d.text((x + (w - tw) / 2, y + (h - size) / 2 - 2), label,
                        font=f, fill=color)

    def chip(self, x, y, text, fill=CHIP, size=14):
        f = _font(FR, size)
        tw = max(48, self.d.textlength(text, font=f) + 28)
        self.d.rounded_rectangle([x, y, x + tw, y + 32], radius=16,
                                 fill=fill, outline=INK, width=2)
        self.d.text((x + (tw - self.d.textlength(text, font=f)) / 2, y + 6),
                    text, font=f, fill=INK)
        return tw

    def text(self, x, y, s, size=15, color=INK, bold=False, anchor="start",
             mono=False):
        f = _font(FB if bold else FR, size)
        self.d.text((x, y), s, font=f, fill=color,
                    anchor={"start": "la", "middle": "ma", "end": "ra"}.get(anchor, anchor))

    def arrow(self, x1, y1, x2, y2, label=None, color=INK, width=2):
        self.d.line([x1, y1, x2, y2], fill=color, width=width)
        ang = math.atan2(y2 - y1, x2 - x1); L = 11
        self.d.polygon([(x2, y2),
                        (x2 - L * math.cos(ang - 0.42), y2 - L * math.sin(ang - 0.42)),
                        (x2 - L * math.cos(ang + 0.42), y2 - L * math.sin(ang + 0.42))],
                       fill=color)
        if label:
            f = _font(FR, 13)
            tw = self.d.textlength(label, font=f)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if abs(y2 - y1) < 2:
                self.d.rectangle([mx - tw / 2 - 8, my - 26, mx + tw / 2 + 8, my - 2], fill=BG)
                self.d.text((mx - tw / 2, my - 24), label, font=f, fill=color)
            else:
                self.d.text((mx + 10, my - 8), label, font=f, fill=color)

    def line(self, x1, y1, x2, y2, color=LINE, width=2):
        self.d.line([x1, y1, x2, y2], fill=color, width=width)

    def circle(self, x, y, r, fill, outline=None):
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill,
                       outline=outline or fill, width=2)

    def curve(self, pts, color=INK, width=3):
        self.d.line(pts, fill=color, width=width, joint="curve")

    def bars(self, x, y_base, items, maxv, bar_w=56, gap=24, height=200,
             size=13, color=INK, neg_down=True):
        """items: list of (label, value, fill). Negative values hang below baseline."""
        xx = x
        for label, val, fill in items:
            bh = height * abs(val) / maxv
            top, bot = (y_base - bh, y_base) if val >= 0 else (y_base, y_base + bh)
            self.d.rectangle([xx, top, xx + bar_w, bot],
                             fill=fill, outline=INK, width=2)
            f = _font(FR, size)
            tw = self.d.textlength(label, font=f)
            self.d.text((xx + (bar_w - tw) / 2, y_base + 8), label, font=f, fill=MUTED)
            vw = self.d.textlength(f"{val}", font=_font(FB, size))
            vy = top - 24 if val >= 0 else bot + 8
            self.d.text((xx + (bar_w - vw) / 2, vy),
                        f"{val}", font=_font(FB, size), fill=color)
            xx += bar_w + gap
        return xx

    def axes(self, x0, y0, x1, y1):
        self.line(x0, y1, x0, y0, color=INK)
        self.line(x0, y0, x1, y0, color=INK)

    def save(self, name):
        f = _font(FR, 15)
        self.d.text((32, self.h - 40), self._footer, font=f, fill=INK)
        sf = _font(FR, 12)
        s = f"source: {self._source}"
        self.d.text((self.w - 32 - self.d.textlength(s, font=sf), self.h - 38),
                    s, font=sf, fill=MUTED)
        path = os.path.join(ASSETS, name)
        self.img.save(path, "WEBP", quality=88)
        print("saved", path)
        return path
