#!/usr/bin/env python3
"""Warm-paper SVG plate factory for CS329A per build/VISUAL_SYSTEM.md.

Primitives on the 8px grid. One claim per plate. Every number computed
by the calling script, never hand-waved. Output goes to
content/v2/cs329a/assets/.
"""
import os, html, math

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/cs329a/assets")

BG = "#F7F4EE"; INK = "#1B2838"; MUTED = "#5C6B7A"; LINE = "#D9D3C7"
PANEL = "#FFFDF8"; COUNT = "#E7F1F8"; NEW = "#E7F4EF"; ACTIVE = "#F4E6D4"
CHIP = "#E6E2DA"; TEAL = "#1F7A72"; ORANGE = "#C46B2C"; FOCUS = "#1E4D8C"
PINK = "#F3D4D8"; YELLOW = "#F6E7A8"; GREEN = "#D9E8D3"

FONT = "Inter, 'Source Sans 3', 'IBM Plex Sans', system-ui, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

def esc(s):
    return html.escape(str(s), quote=True)

def avg_w(text, size):
    # rough Inter advance: 0.52 * size for mixed text, 0.6 for mono
    return len(str(text)) * size * 0.52

class Plate:
    def __init__(self, title, claim, footer, source="original toy",
                 width=960, inner_h=320):
        self.title = title; self.claim = claim; self.footer = footer
        self.source = source
        self.w = width
        self.top = 96            # below title + claim
        self.h = self.top + inner_h + 64
        self.e = []              # svg elements
        self.head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
            f'<rect width="{self.w}" height="{self.h}" fill="{BG}" rx="0"/>'
        )
        self.head += (
            f'<text x="32" y="44" font-size="30" font-weight="600" fill="{INK}">{esc(title)}</text>'
            f'<text x="32" y="72" font-size="16" font-weight="450" fill="{MUTED}">{esc(claim)}</text>'
        )

    def panel(self, x, y, w, h, fill=PANEL, label=None):
        self.e.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{LINE}" stroke-width="1.5" rx="12"/>')
        if label:
            self.e.append(f'<text x="{x+16}" y="{y+26}" font-size="14" font-weight="600" fill="{MUTED}">{esc(label)}</text>')

    def rect(self, x, y, w, h, fill, label=None, label_size=15, stroke=INK, rx=8,
             label_color=INK, bold=True):
        self.e.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" rx="{rx}"/>')
        if label is not None:
            wgt = 600 if bold else 500
            self.e.append(f'<text x="{x+w/2}" y="{y+h/2+label_size*0.35}" text-anchor="middle" '
                          f'font-size="{label_size}" font-weight="{wgt}" fill="{label_color}">{esc(label)}</text>')

    def chip(self, x, y, text, fill=CHIP, size=14):
        tw = max(48, avg_w(text, size) + 24)
        self.e.append(f'<rect x="{x}" y="{y}" width="{tw}" height="32" fill="{fill}" stroke="{INK}" stroke-width="1.5" rx="999"/>')
        self.e.append(f'<text x="{x+tw/2}" y="{y+21}" text-anchor="middle" font-size="{size}" font-weight="500" fill="{INK}">{esc(text)}</text>')
        return tw

    def arrow(self, x1, y1, x2, y2, label=None, color=INK):
        self.e.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"/>')
        ang = math.atan2(y2-y1, x2-x1)
        L = 10
        self.e.append(f'<polygon points="{x2},{y2} {x2-L*math.cos(ang-0.4)},{y2-L*math.sin(ang-0.4)} {x2-L*math.cos(ang+0.4)},{y2-L*math.sin(ang+0.4)}" fill="{color}"/>')
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            if abs(y2-y1) < 1:  # horizontal
                self.e.append(f'<rect x="{mx-avg_w(label,13)/2-8}" y="{my-24}" width="{avg_w(label,13)+16}" height="24" fill="{BG}"/>')
                self.e.append(f'<text x="{mx}" y="{my-6}" text-anchor="middle" font-size="13" font-weight="500" fill="{color}">{esc(label)}</text>')
            else:
                self.e.append(f'<text x="{mx+8}" y="{my}" font-size="13" font-weight="500" fill="{color}">{esc(label)}</text>')

    def text(self, x, y, s, size=15, color=INK, bold=False, anchor="start", mono=False):
        wgt = 600 if bold else (450 if not mono else 400)
        ff = f' font-family="{MONO}"' if mono else ''
        self.e.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{wgt}" fill="{color}"{ff}>{esc(s)}</text>')

    def poly(self, points, fill, stroke=INK):
        p = " ".join(f"{x},{y}" for x, y in points)
        self.e.append(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    def line(self, x1, y1, x2, y2, color=LINE, width=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.e.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>')

    def circle(self, cx, cy, r, fill, label=None, size=14):
        self.e.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{INK}" stroke-width="1.5"/>')
        if label is not None:
            self.e.append(f'<text x="{cx}" y="{cy+size*0.35}" text-anchor="middle" font-size="{size}" font-weight="600" fill="{INK}">{esc(label)}</text>')

    def save(self, name):
        foot = (f'<text x="32" y="{self.h-24}" font-size="15" font-weight="500" fill="{INK}">{esc(self.footer)}</text>'
                f'<text x="{self.w-32}" y="{self.h-24}" text-anchor="end" font-size="12" font-weight="450" fill="{MUTED}">source: {esc(self.source)}</text>')
        body = self.head + "".join(self.e) + foot + "</svg>"
        path = os.path.join(ASSETS, name)
        with open(path, "w") as f:
            f.write(body)
        return path

def hbar_bars(plate, x, y, w, items, maxv, bar_h=40, gap=16):
    """items: list of (label, value, fill). Returns end y."""
    for label, val, fill in items:
        bw = w * val / maxv
        plate.rect(x, y, bw, bar_h, fill, label=f"{label}: {val}", label_size=14,
                   stroke=INK, rx=8)
        y += bar_h + gap
    return y

def vbar_bars(plate, x, y_base, items, maxv, bar_w=56, gap=24, scale=1.0):
    """items: list of (label, value, fill). Bars rise from y_base."""
    xx = x
    for label, val, fill in items:
        bh = val / maxv * 220 * scale
        plate.rect(xx, y_base - bh, bar_w, bh, fill, label=None, stroke=INK, rx=8)
        plate.text(xx + bar_w/2, y_base - bh - 12, str(val), size=14, bold=True, anchor="middle")
        plate.text(xx + bar_w/2, y_base + 24, label, size=13, color=MUTED, anchor="middle")
        xx += bar_w + gap
    return xx
