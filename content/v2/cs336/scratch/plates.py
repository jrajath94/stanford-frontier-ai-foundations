"""Shared SVG lesson-plate generator for the CS336 v2 build.
Style: warm paper #F7F4EE, ink #1B2838, flat fills, no gradient/glow/shadow.
Every plate: title, one claim line, before/rule/after panels, footer with source+shell.
"""
import os, html

OUT = "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/assets"

BG = "#F7F4EE"; INK = "#1B2838"; MUT = "#5C6B7A"; LINE = "#D9D3C7"
PANEL = "#FFFDF8"; COUNT = "#E7F1F8"; NEWTOK = "#E7F4EF"; ACTIVE = "#F4E6D4"
CHIP = "#E6E2DA"; TEAL = "#1F7A72"; ORANGE = "#C46B2C"; FOCUS = "#1E4D8C"
PINK = "#F3D9D9"; BLUE = "#D9E6F5"; GREEN = "#DDEBD9"; YELLOW = "#F5EBCF"

def E(s):
    return html.escape(str(s), quote=True)

class Plate:
    def __init__(self, w=960, h=520):
        self.w, self.h = w, h
        self.parts = []
        self.parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="system-ui,-apple-system,\'Segoe UI\',sans-serif">'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>')

    def title(self, t, claim):
        self.parts.append(f'<text x="36" y="48" font-size="30" font-weight="700" fill="{INK}">{E(t)}</text>')
        self.parts.append(f'<text x="36" y="78" font-size="17" fill="{MUT}">{E(claim)}</text>')
        self.parts.append(f'<line x1="36" y1="96" x2="{self.w-36}" y2="96" stroke="{LINE}" stroke-width="1"/>')
        return 110

    def panel(self, x, y, w, h, label=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{PANEL}" stroke="{LINE}" stroke-width="1.5"/>')
        if label:
            self.parts.append(f'<text x="{x+14}" y="{y+24}" font-size="14" font-weight="700" fill="{MUT}">{E(label)}</text>')
        return (x, y, w, h)

    def chip(self, x, y, w, h, label, sub=None, fill=CHIP, hatch=False, tcolor=INK):
        if hatch:
            pid = f"h{x}{y}".replace(".", "")
            self.parts.append(
                f'<defs><pattern id="{pid}" width="8" height="8" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
                f'<rect width="8" height="8" fill="{fill}"/><line x1="0" y1="0" x2="0" y2="8" stroke="{MUT}" stroke-width="2" opacity="0.5"/></pattern></defs>')
            fillv = f"url(#{pid})"
        else:
            fillv = fill
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fillv}" stroke="{INK}" stroke-width="1.5"/>')
        self.parts.append(f'<text x="{x+w/2}" y="{y+h/2-2}" font-size="15" font-weight="600" fill="{tcolor}" text-anchor="middle">{E(label)}</text>')
        if sub:
            self.parts.append(f'<text x="{x+w/2}" y="{y+h/2+18}" font-size="12" fill="{MUT}" text-anchor="middle">{E(sub)}</text>')

    def arrow(self, x1, x2, y, label, color=FOCUS):
        self.parts.append(
            f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="3" marker-end="url(#ah)"/>')
        self.parts.append(
            f'<text x="{(x1+x2)/2}" y="{y-10}" font-size="14" font-weight="700" fill="{color}" text-anchor="middle">{E(label)}</text>')

    def arrow_down(self, x, y1, y2, label, color=FOCUS):
        self.parts.append(
            f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="3" marker-end="url(#ah)"/>')
        self.parts.append(
            f'<text x="{x+10}" y="{(y1+y2)/2}" font-size="14" font-weight="700" fill="{color}">{E(label)}</text>')

    def text(self, x, y, s, size=16, fill=INK, weight=400, anchor="start", width=None):
        w_attr = f' textLength="{width}" lengthAdjust="spacingAndGlyphs"' if width else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{w_attr}>{E(s)}</text>')

    def footer(self, s):
        self.parts.append(
            f'<text x="36" y="{self.h-24}" font-size="14" fill="{MUT}">{E(s)}</text>')

    def defs_arrow(self):
        self.parts.append(
            '<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">'
            '<path d="M0,0 L8,3 L0,6 Z" fill="#1E4D8C"/></marker></defs>')

    def save(self, name):
        self.parts.append('</svg>')
        path = os.path.join(OUT, name)
        open(path, "w").write("".join(self.parts))
        print("wrote", path)
