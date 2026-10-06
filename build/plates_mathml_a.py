#!/usr/bin/env python3
"""Plate generator for math-ml deepening. Chunk A: helpers + generic plates.
All numbers are computed in code. Palette follows build/VISUAL_SYSTEM.md."""
import math, os, re

OUT = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/math-ml/assets")
os.makedirs(OUT, exist_ok=True)

BG = "#F7F4EE"; INK = "#1B2838"; MUTED = "#5C6B7A"; LINE = "#D9D3C7"
PANEL = "#FFFDF8"; COUNT = "#E7F1F8"; NEWOBJ = "#E7F4EF"; ACTIVE = "#F4E6D4"
CHIP = "#E6E2DA"; TEAL = "#1F7A72"; ORANGE = "#C46B2C"; FOCUS = "#1E4D8C"
PINK = "#F3D4D8"; YELLOW = "#F6E7A8"; GREEN = "#D9E8D3"
SANS = "Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

class Plate:
    def __init__(self, w, h, title, claim, footer, source="original"):
        self.w, self.h = w, h
        self.p = []
        self.p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">')
        self.p.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{BG}"/>')
        self.text(24, 40, title, 30, 600, INK)
        self.text(24, 66, claim, 16, 450, MUTED)
        self.p.append(f'<line x1="24" y1="80" x2="{w-24}" y2="80" stroke="{LINE}" stroke-width="1.5"/>')
        short_footer = re.sub(r"^Shell \d+\.\s*", "", footer).split(" Shell")[0].rstrip()
        self.text(24, h - 16, short_footer, 14, 450, MUTED)
        self.text(w - 24, h - 16, f"Source: {source}", 13, 500, MUTED, anchor="end")

    def text(self, x, y, s, size=16, weight=450, color=INK, anchor="start", family=SANS):
        self.p.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')

    def mono(self, x, y, s, size=15, color=INK, anchor="start"):
        self.text(x, y, s, size, 500, color, anchor, MONO)

    def rect(self, x, y, w, h, fill=PANEL, stroke=LINE, r=8, sw=1.5):
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, color=LINE, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')

    def arrow(self, x1, y1, x2, y2, label=None, color=INK):
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"/>')
        ang = math.atan2(y2 - y1, x2 - x1)
        for da in (0.42, -0.42):
            a = ang + math.pi + da
            ex, ey = x2 + 10 * math.cos(a), y2 + 10 * math.sin(a)
            self.p.append(f'<line x1="{x2}" y1="{y2}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" stroke-width="2"/>')
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 8
            self.text(mx, my, label, 13, 500, color, anchor="middle")

    def chip(self, x, y, w, h, label, fill=CHIP):
        self.rect(x, y, w, h, fill, LINE, 999)
        self.text(x + w / 2, y + h / 2 + 5, label, 14, 500, INK, anchor="middle")

    def matrix(self, x, y, m, cell=46, fmt=None, color=INK):
        """Draw matrix m (list of rows) with brackets at (x,y) top-left."""
        n, d = len(m), len(m[0])
        mw, mh = d * cell, n * cell
        self.p.append(f'<path d="M {x+10} {y} L {x} {y} L {x} {y+mh} L {x+10} {y+mh}" stroke="{INK}" stroke-width="2" fill="none"/>')
        self.p.append(f'<path d="M {x+mw-10} {y} L {x+mw} {y} L {x+mw} {y+mh} L {x+mw-10} {y+mh}" stroke="{INK}" stroke-width="2" fill="none"/>')
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                s = fmt(v) if fmt else str(v)
                self.text(x + j * cell + cell / 2, y + i * cell + cell / 2 + 6, s, 16, 500, color, anchor="middle", family=MONO)

    def vec(self, x, y, comps, cell=46, vertical=False, fmt=None):
        if vertical:
            self.matrix(x, y, [[c] for c in comps], cell, fmt)
        else:
            self.matrix(x, y, [list(comps)], cell, fmt)

    def save(self, name):
        self.p.append("</svg>")
        path = os.path.join(OUT, name)
        open(path, "w").write("\n".join(self.p))
        return path

def _wrap(s, n=38):
    if len(s) <= n:
        return [s]
    words = s.split(" ")
    lines, cur = [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 <= n:
            cur = (cur + " " + w_).strip()
        else:
            lines.append(cur); cur = w_
    if cur:
        lines.append(cur)
    return lines

def bab(name, title, claim, left_title, left_lines, rule, right_title, right_lines,
        footer, source="original", left_fill=PANEL, right_fill=NEWOBJ):
    """Generic before/after plate. left_lines/right_lines: list of (text, color or None)."""
    P = Plate(760, 430, title, claim, footer, source)
    P.rect(24, 96, 336, 280, left_fill)
    P.rect(400, 96, 336, 280, right_fill)
    P.text(40, 126, left_title, 16, 600, INK)
    P.text(416, 126, right_title, 16, 600, INK)
    y = 156
    for s, c in left_lines:
        for wl in _wrap(s):
            P.mono(40, y, wl, 14.5, c or INK)
            y += 28
    y = 156
    for s, c in right_lines:
        for wl in _wrap(s):
            P.mono(416, y, wl, 14.5, c or INK)
            y += 28
    P.arrow(360, 236, 400, 236, None, TEAL)
    P.text(380, 222, rule, 13, 500, TEAL, anchor="middle")
    return P.save(name)

def vbar(name, title, claim, footer, items, source="original", ylabel=""):
    """items: list of (label, value, color). Vertical bars with exact value labels."""
    P = Plate(760, 440, title, claim, footer, source)
    n = len(items)
    maxv = max(v for _, v, _ in items)
    ax, ay, aw, ah = 64, 110, 640, 260
    P.line(ax, ay, ax, ay + ah, INK, 1.5)
    P.line(ax, ay + ah, ax + aw, ay + ah, INK, 1.5)
    if ylabel:
        P.text(24, ay + ah / 2, ylabel, 13, 500, MUTED)
    bw = aw / n
    for i, (lab, v, c) in enumerate(items):
        bh = (v / maxv) * ah if maxv else 0
        x = ax + i * bw + bw * 0.2
        w = bw * 0.6
        P.rect(x, ay + ah - bh, w, bh, c, LINE, 8)
        P.mono(ax + i * bw + bw / 2, ay + ah + 22, lab, 13, INK, anchor="middle")
        P.mono(ax + i * bw + bw / 2, ay + ah - bh - 8, f"{v:g}", 14, INK, anchor="middle")
    return P.save(name)

def curve(name, title, claim, footer, fn, x0, x1, marks, source="original",
          xlabel="", ylabel=""):
    """Plot fn over [x0,x1] with marked points: marks = [(x, label, color)]."""
    P = Plate(760, 440, title, claim, footer, source)
    ax, ay, aw, ah = 64, 110, 620, 250
    xs = [x0 + (x1 - x0) * i / 200 for i in range(201)]
    ys = [fn(x) for x in xs]
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.1 or 1
    def px(x): return ax + (x - x0) / (x1 - x0) * aw
    def py(v): return ay + ah - (v - ymin + pad) / (ymax - ymin + 2 * pad) * ah
    pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, ys))
    P.p.append(f'<polyline points="{pts}" fill="none" stroke="{FOCUS}" stroke-width="2.5"/>')
    P.line(ax, py(0) if ymin <= 0 <= ymax else ay + ah, ax + aw, py(0) if ymin <= 0 <= ymax else ay + ah, LINE, 1.5)
    P.line(ax, ay, ax, ay + ah, INK, 1.5)
    P.line(ax, ay + ah, ax + aw, ay + ah, INK, 1.5)
    if xlabel: P.text(ax + aw / 2, ay + ah + 28, xlabel, 13, 500, MUTED, anchor="middle")
    if ylabel: P.text(24, ay + ah / 2, ylabel, 13, 500, MUTED)
    for x, lab, c in marks:
        v = fn(x)
        P.line(px(x), py(v), px(x), ay + ah, c, 1.5, dash="5,4")
        P.p.append(f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="6" fill="{c}" stroke="{INK}" stroke-width="1.5"/>')
        P.mono(px(x), py(v) - 12, lab, 13, INK, anchor="middle")
    return P.save(name)

def scatter(name, title, claim, footer, points, line=None, source="original",
            xlabel="x", ylabel="y", xrange=None, yrange=None):
    """points: [(x,y,label,color)]. line: (slope, intercept) or None."""
    P = Plate(760, 440, title, claim, footer, source)
    ax, ay, aw, ah = 64, 110, 620, 250
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    x0, x1 = xrange or (min(xs) - 0.5, max(xs) + 0.5)
    y0, y1 = yrange or (min(ys) - 0.5, max(ys) + 0.5)
    def px(x): return ax + (x - x0) / (x1 - x0) * aw
    def py(y): return ay + ah - (y - y0) / (y1 - y0) * ah
    P.line(ax, ay, ax, ay + ah, INK, 1.5)
    P.line(ax, ay + ah, ax + aw, ay + ah, INK, 1.5)
    P.text(ax + aw / 2, ay + ah + 28, xlabel, 13, 500, MUTED, anchor="middle")
    P.text(24, ay + ah / 2, ylabel, 13, 500, MUTED)
    if line:
        s, b = line
        P.line(px(x0), py(s * x0 + b), px(x1), py(s * x1 + b), TEAL, 2.5)
    for x, y, lab, c in points:
        P.p.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="7" fill="{c}" stroke="{INK}" stroke-width="1.5"/>')
        if lab: P.text(px(x) + 10, py(y) - 8, lab, 13, 500, INK)
    return P.save(name)

def trace(name, title, claim, footer, series, source="original", xlabel="step", ylabel="value"):
    """series: list of (label, values, color, dashed?). Line traces of sequences."""
    P = Plate(760, 440, title, claim, footer, source)
    ax, ay, aw, ah = 72, 110, 600, 250
    allv = [v for _, vs, _, _ in series for v in vs]
    v0, v1 = min(allv), max(allv)
    pad = (v1 - v0) * 0.12 or 1
    n = max(len(vs) for _, vs, _, _ in series)
    def px(i): return ax + i / (n - 1) * aw if n > 1 else ax
    def py(v): return ay + ah - (v - v0 + pad) / (v1 - v0 + 2 * pad) * ah
    P.line(ax, ay, ax, ay + ah, INK, 1.5)
    P.line(ax, ay + ah, ax + aw, ay + ah, INK, 1.5)
    P.text(ax + aw / 2, ay + ah + 28, xlabel, 13, 500, MUTED, anchor="middle")
    P.text(28, ay + ah / 2, ylabel, 13, 500, MUTED)
    lx = ax + aw + 8
    for j, (lab, vs, c, dash) in enumerate(series):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vs))
        d = ' stroke-dasharray="6,4"' if dash else ""
        P.p.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.5"{d}/>')
        for i, v in enumerate(vs):
            P.p.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4.5" fill="{c}" stroke="{INK}" stroke-width="1"/>')
        P.text(lx, ay + 20 + j * 24, lab, 13, 500, INK)
        P.rect(lx - 22, ay + 10 + j * 24, 16, 10, c, c, 2)
    return P.save(name)
