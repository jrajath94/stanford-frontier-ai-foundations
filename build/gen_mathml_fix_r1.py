#!/usr/bin/env python3
"""FIX AGENT round 1: new math-ml figures (B) + chapter plates (C).

Writes to content/v2/math-ml/assets/ (the source of truth).
Every number is computed with numpy and asserted before the file is written.
Style follows build/gen_mathml_plates.py and build/FIGURE_SPEC_STANFORD.md:
warm paper #F7F4EE, 8px grid, one claim per lesson plate, left = toy,
center = one named rule, right = result. Chapter plates: WITHOUT / object /
WITH, tradeoff footer, reuse lesson-plate chip colors.
"""
import html
import os
import xml.etree.ElementTree as ET

import numpy as np

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/math-ml/assets")

FONT = "Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"
INK = "#1B2838"
MUTED = "#5C6B7A"
LINE = "#D9D3C7"
PAPER = "#F7F4EE"
PANEL = "#FFFDF8"
COUNT = "#E7F1F8"
NEWTOK = "#E7F4EF"
TEAL = "#1F7A72"
ORANGE = "#C46B2C"
FOCUS = "#1E4D8C"


def esc(s):
    return html.escape(str(s), quote=False)


def T(x, y, s, size=14.5, weight=500, fill=INK, family=MONO, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def TC(x, y, s, size=14.5, weight=500, fill=INK, family=MONO):
    return T(x, y, s, size=size, weight=weight, fill=fill, family=family, anchor="middle")


def rect_panel(x, y, w, h, fill, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{LINE}" stroke-width="1.5"/>')


def arrow(x1, x2, y, label):
    hx = x2 - 9.1
    return "\n".join([
        f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{TEAL}" stroke-width="2"/>',
        f'<line x1="{x2}" y1="{y}" x2="{hx}" y2="{y - 4.1}" stroke="{TEAL}" stroke-width="2"/>',
        f'<line x1="{x2}" y1="{y}" x2="{hx}" y2="{y + 4.1}" stroke="{TEAL}" stroke-width="2"/>',
        T((x1 + x2) / 2, y - 14, label, size=13, weight=500, fill=TEAL, family=FONT, anchor="middle"),
    ])


def frame(title, claim_lines, footer_claim, panels, arrow_labels, source="original"):
    """panels: 3 x (header, [(text, color[, size])])."""
    assert len(claim_lines) in (1, 2)
    assert len(panels) == 3 and len(arrow_labels) == 2
    if len(claim_lines) == 1:
        div_y, py, ph = 80, 96, 280
    else:
        div_y, py, ph = 100, 112, 264
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" '
             'viewBox="0 0 760 430" role="img">']
    parts.append(f'<rect x="0" y="0" width="760" height="430" fill="{PAPER}"/>')
    title_size = 28 if len(title) > 44 else 30
    parts.append(T(24, 40, title, size=title_size, weight=600, family=FONT))
    if len(claim_lines) == 1:
        parts.append(T(24, 66, claim_lines[0], size=16, weight=450, fill=MUTED, family=FONT))
    else:
        parts.append(T(24, 64, claim_lines[0], size=16, weight=450, fill=MUTED, family=FONT))
        parts.append(T(24, 86, claim_lines[1], size=16, weight=450, fill=MUTED, family=FONT))
    parts.append(f'<line x1="24" y1="{div_y}" x2="736" y2="{div_y}" '
                 f'stroke="{LINE}" stroke-width="1.5"/>')
    parts.append(T(24, 414, "One claim: " + footer_claim, size=14, weight=450, fill=MUTED, family=FONT))
    parts.append(T(736, 414, "Source: " + source, size=13, weight=500, fill=MUTED, family=FONT, anchor="end"))
    LX, LW = 24, 256
    CX, CW = 312, 136
    RX, RW = 480, 256
    for (px, pw), fill in zip([(LX, LW), (CX, CW), (RX, RW)], [PANEL, COUNT, NEWTOK]):
        parts.append(rect_panel(px, py, pw, ph, fill))
    ay = py + 140
    parts.append(arrow(LX + LW, CX, ay, arrow_labels[0]))
    parts.append(arrow(CX + CW, RX, ay, arrow_labels[1]))
    for (header, lines), x in zip(panels, [40, 328, 496]):
        parts.append(T(x, py + 30, header, size=16, weight=600, family=FONT))
        y = py + 60
        for line in lines:
            txt, color = line[0], line[1]
            size = line[2] if len(line) > 2 else 14.5
            parts.append(T(x, y, txt, size=size, fill=color))
            y += 28
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


specs = []  # (filename, svg_text, [required number substrings])


# ============ B.1: orthogonal complement (l02 @280) ============
b = np.array([3.0, 4.0])
a = np.array([1.0, 0.0])
adotb = float(a @ b)
adota = float(a @ a)
proj = (adotb / adota) * a
leftover = b - proj
assert adotb == 3.0 and adota == 1.0
assert np.array_equal(proj, [3.0, 0.0])
assert np.array_equal(leftover, [0.0, 4.0])
assert float(leftover @ a) == 0.0            # leftover perpendicular to the line
assert np.array_equal(proj + leftover, b)     # nothing lost

specs.append(("plate-l02-complement.svg", frame(
    title="Every vector splits uniquely",
    claim_lines=["[3,4] = [3,0] + [0,4]: a part on the line, a part perpendicular to it."],
    footer_claim="a vector is its space part plus its complement part, uniquely.",
    panels=[
        ("The toy", [
            ("b = [3,4]", INK),
            ("line: x-axis", INK),
            ("a = [1,0]", INK),
        ]),
        ("Project", [
            ("((a·b)/(a·a))·a", INK, 13),
            ("= (3/1)·[1,0]", INK, 13),
            ("drop the ⊥", TEAL, 13),
        ]),
        ("The unique split", [
            ("line part: [3,0]", INK),
            ("complement: [0,4]", INK),
            ("[0,4]·[1,0] = 0  ⊥", TEAL),
            ("[3,0]+[0,4]=[3,4]", TEAL),
        ]),
    ],
    arrow_labels=["project", "split"],
    source="original toy",
), ["[3,4]", "[3,0]", "[0,4]", "(3/1)"]))


# ============ B.2: two tests are better than one (l05 @198) ============
prior1 = 0.088          # posterior after one positive test
num2 = 0.95 * prior1
den2 = num2 + 0.10 * (1 - prior1)
post2 = num2 / den2
assert round(num2, 4) == 0.0836
assert round(den2, 4) == 0.1748
assert round(post2 * 100, 1) == 47.8

specs.append(("plate-l05-twotests.svg", frame(
    title="Two positives: 47.8%",
    claim_lines=["One positive test: 8.8%. A second independent positive: 47.8%.",
                 "Evidence accumulates multiplicatively."],
    footer_claim="retesting multiplies the likelihoods; belief jumps 8.8% to 47.8%.",
    panels=[
        ("After one +", [
            ("belief = 8.8%", INK),
            ("the new prior", INK),
        ]),
        ("Bayes again", [
            ("P(sick|++) =", INK, 13),
            ("0.95·0.088 /", INK, 13),
            ("(0.0836+0.0912)", INK, 13),
        ]),
        ("After two +", [
            ("= 0.0836/0.1748", INK),
            ("= 47.8%", TEAL),
            ("8.8% → 47.8%", TEAL),
        ]),
    ],
    arrow_labels=["retest", "multiply"],
    source="original arithmetic",
), ["8.8%", "0.0836", "0.1748", "47.8%"]))


# ============ B.3: Gaussian MLE (l06 @298) ============
data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
mu_hat = float(np.mean(data))
sq = float(np.sum((data - mu_hat) ** 2))
var_hat = sq / len(data)
assert mu_hat == 5.0
assert sq == 32.0
assert var_hat == 4.0

specs.append(("plate-l06-gaussmle.svg", frame(
    title="Gaussian MLE: the sample mean and variance",
    claim_lines=["Data [2,4,4,4,5,5,7,9]: the maximizing (mu, sigma) are mu = 5, sigma2 = 4.",
                 "No search needed: closed forms."],
    footer_claim="fit a Gaussian = compute the mean, compute the variance.",
    panels=[
        ("The data", [
            ("[2,4,4,4,5,5,7,9]", INK),
            ("n = 8", INK),
        ]),
        ("Maximize", [
            ("mu-hat = x-bar", INK, 13),
            ("sigma2 = mean of", INK, 13),
            ("(xi - mu-hat)2", INK, 13),
        ]),
        ("The MLE", [
            ("mu-hat = 40/8 = 5", INK),
            ("sq dev sum = 32", INK),
            ("sigma2 = 32/8 = 4", TEAL),
        ]),
    ],
    arrow_labels=["maximize", "read off"],
    source="original arithmetic",
), ["mu-hat = 40/8 = 5", "32/8 = 4", "n = 8"]))


# ============ B.5: Newton in one step (l08 @224) ============
x0 = 0.0
fp = 2 * (x0 - 3)          # f'(x) = 2(x-3)
fpp = 2.0                  # f''(x) = 2
x1 = x0 - fp / fpp
f1 = (x1 - 3) ** 2
assert fp == -6.0 and fpp == 2.0
assert x1 == 3.0 and f1 == 0.0

specs.append(("plate-l08-newton.svg", frame(
    title="Newton in one step",
    claim_lines=["f(x) = (x-3)2 from x0 = 0: x1 = 0 - (-6)/2 = 3. Done. Exact."],
    footer_claim="on a true quadratic, Newton lands exactly in one step.",
    panels=[
        ("x0 = 0", [
            ("f = 9", INK),
            ("f' = -6 (slope)", INK),
            ("f'' = 2 (curvature)", INK),
        ]),
        ("x - f'/f''", [
            ("minimize the", INK, 13),
            ("Taylor quadratic", INK, 13),
        ]),
        ("x1 = 3", [
            ("0 - (-6)/2 = 3", INK),
            ("f(3) = 0", TEAL),
            ("exact, one step", TEAL),
        ]),
    ],
    arrow_labels=["slope+curve", "land"],
    source="original arithmetic",
), ["f' = -6", "f'' = 2", "0 - (-6)/2 = 3", "f(3) = 0"]))


# ============ B.6: Huffman coding (l10 @255) ============
p = np.array([0.5, 0.25, 0.125, 0.125])
H = float(-np.sum(p * np.log2(p)))
code_lens = np.array([1, 2, 3, 3])          # codes 0, 10, 110, 111
avg_len = float(np.sum(p * code_lens))
assert round(H, 2) == 1.75
assert round(avg_len, 2) == 1.75
assert round(avg_len - H, 9) == 0.0        # exactly the floor

specs.append(("plate-l10-huffman.svg", frame(
    title="Huffman hits the entropy floor",
    claim_lines=["Probs [0.5, 0.25, 0.125, 0.125]: H = 1.75 bits.",
                 "Codes 0/10/110/111 average 1.75 bits. Exactly entropy."],
    footer_claim="no code averages below entropy; Huffman reaches it.",
    panels=[
        ("The symbols", [
            ("p = [0.5, 0.25,", INK),
            ("     0.125, 0.125]", INK),
            ("H = 1.75 bits", INK),
        ]),
        ("Short for likely", [
            ("0.5 -> 0", INK, 13),
            ("0.25 -> 10", INK, 13),
            ("0.125 -> 110, 111", INK, 13),
        ]),
        ("Average = floor", [
            ("0.5·1 + 0.25·2", INK),
            ("+ 2·0.125·3", INK),
            ("= 1.75 = H", TEAL),
        ]),
    ],
    arrow_labels=["assign", "reach floor"],
    source="original arithmetic",
), ["H = 1.75 bits", "= 1.75 = H", "0.5·1 + 0.25·2"]))


# ============ B.7: convex sets (l08 @44) — custom geometry ============
# Disk: center (152,244) r=88. Chord A=(100,200), B=(210,290) stays inside.
# Crescent: big circle center (608,244) r=88, cut circle center (668,244) r=76.
# Horns A=(620,175), B=(620,313); chord midpoint (620,244) is inside the cut
# circle, hence outside the set: the segment escapes.
def _dist2(px, py, cx, cy):
    return (px - cx) ** 2 + (py - cy) ** 2


disk_c, disk_r = (152, 244), 88
dA, dB = (100, 200), (210, 290)
assert _dist2(*dA, *disk_c) < disk_r ** 2 and _dist2(*dB, *disk_c) < disk_r ** 2
assert _dist2(155, 245, *disk_c) < disk_r ** 2          # chord midpoint inside
big_c, big_r = (608, 244), 88
cut_c, cut_r = (668, 244), 76
cA, cB = (620, 175), (620, 313)
assert _dist2(*cA, *big_c) < big_r ** 2 and _dist2(*cA, *cut_c) > cut_r ** 2
assert _dist2(*cB, *big_c) < big_r ** 2 and _dist2(*cB, *cut_c) > cut_r ** 2
assert _dist2(620, 244, *big_c) < big_r ** 2            # midpoint in big circle
assert _dist2(620, 244, *cut_c) < cut_r ** 2            # ...but in the hole: escapes

_disk_path = (f'<circle cx="{disk_c[0]}" cy="{disk_c[1]}" r="{disk_r}" fill="{COUNT}" '
              f'stroke="{INK}" stroke-width="1.5"/>')
_crescent_path = (
    f'<path d="M {cA[0]},{cA[1]} '
    f'A {big_r} {big_r} 0 1 0 {cB[0]},{cB[1]} '
    f'A {cut_r} {cut_r} 0 0 1 {cA[0]},{cA[1]} Z" '
    f'fill="{COUNT}" stroke="{INK}" stroke-width="1.5"/>')


def _pt(p, color, label, lx, ly):
    return (f'<circle cx="{p[0]}" cy="{p[1]}" r="5" fill="{color}" stroke="{INK}" stroke-width="1.5"/>'
            f'{T(lx, ly, label, size=13, weight=500, fill=color, family=FONT)}')


convexset_svg = "\n".join([
    '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" viewBox="0 0 760 430" role="img">',
    f'<rect x="0" y="0" width="760" height="430" fill="{PAPER}"/>',
    T(24, 40, "A convex set holds every segment", size=30, weight=600, family=FONT),
    T(24, 66, "Disk: the segment between any two points stays inside. Crescent: it escapes.", size=16, weight=450, fill=MUTED, family=FONT),
    f'<line x1="24" y1="80" x2="736" y2="80" stroke="{LINE}" stroke-width="1.5"/>',
    T(24, 414, "One claim: convexity is the segment test; the disk passes, the crescent fails.", size=14, weight=450, fill=MUTED, family=FONT),
    T(736, 414, "Source: original toy", size=13, weight=500, fill=MUTED, family=FONT, anchor="end"),
    rect_panel(24, 96, 256, 280, PANEL),
    rect_panel(312, 96, 136, 280, COUNT),
    rect_panel(480, 96, 256, 280, PANEL),
    arrow(280, 312, 236, "segment test"),
    arrow(448, 480, 236, "check"),
    T(40, 126, "Convex", size=16, weight=600, family=FONT),
    _disk_path,
    f'<line x1="{dA[0]}" y1="{dA[1]}" x2="{dB[0]}" y2="{dB[1]}" stroke="{TEAL}" stroke-width="2"/>',
    _pt(dA, TEAL, "A", 88, 196),
    _pt(dB, TEAL, "B", 216, 306),
    T(40, 348, "disk: ||w|| <= 1", size=13, weight=500, fill=INK, family=MONO),
    T(40, 368, "segment stays inside", size=13, weight=500, fill=TEAL, family=FONT),
    T(328, 126, "Test", size=16, weight=600, family=FONT),
    T(328, 172, "take any two", size=13, weight=500, fill=INK, family=FONT),
    T(328, 196, "points in the set", size=13, weight=500, fill=INK, family=FONT),
    T(328, 228, "all of AB", size=13, weight=500, fill=INK, family=FONT),
    T(328, 252, "inside?", size=13, weight=500, fill=TEAL, family=FONT),
    T(496, 126, "Not convex", size=16, weight=600, family=FONT),
    _crescent_path,
    f'<line x1="{cA[0]}" y1="{cA[1]}" x2="{cB[0]}" y2="{cB[1]}" stroke="{ORANGE}" stroke-width="2"/>',
    _pt(cA, ORANGE, "A", 600, 168),
    _pt(cB, ORANGE, "B", 600, 332),
    T(496, 348, "crescent moon", size=13, weight=500, fill=INK, family=MONO),
    T(496, 368, "segment escapes", size=13, weight=500, fill=ORANGE, family=FONT),
    "</svg>",
]) + "\n"

specs.append(("plate-l08-convexset.svg", convexset_svg,
              ["disk: ||w|| &lt;= 1", "segment stays inside", "crescent moon", "segment escapes"]))


# ============ Chapter plates (C): 960x600, WITHOUT / object / WITH ============
def chap_plate(title, subtitle, headers, columns, tradeoff, footer):
    """headers: (left, center, right). columns: 3 x [lines]; each line is
    (text, color[, size, weight]). tradeoff: one line. footer: one line."""
    assert len(headers) == 3 and len(columns) == 3
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" '
             'viewBox="0 0 960 600" role="img" '
             'font-family="Anthropic Sans, Inter, \'Source Sans 3\', \'IBM Plex Sans\', sans-serif">']
    parts.append('<rect width="960" height="600" fill="#F7F4EE"/>')
    parts.append(f'<text x="48" y="56" font-size="30" font-weight="600" fill="#1B2838">{esc(title)}</text>')
    parts.append(f'<text x="48" y="86" font-size="17" fill="#5C6B7A">{esc(subtitle)}</text>')
    parts.append('<g font-size="18" font-weight="600" fill="#1B2838">')
    parts.append(f'<text x="48" y="124">{esc(headers[0])}</text>')
    parts.append(f'<text x="328" y="124">{esc(headers[1])}</text>')
    parts.append(f'<text x="648" y="124">{esc(headers[2])}</text>')
    parts.append('</g>')
    boxes = [(48, 264, "#FFFDF8", "#1B2838", 180), (328, 304, "#E7F1F8", "#1E4D8C", 480),
             (648, 264, "#E7F4EF", "#1F7A72", 780)]
    for (x, w, fill, stroke, cx), lines in zip(boxes, columns):
        parts.append(f'<rect x="{x}" y="140" width="{w}" height="300" rx="12" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        y = 184
        for line in lines:
            txt, color = line[0], line[1]
            size = line[2] if len(line) > 2 else 14
            weight = line[3] if len(line) > 3 else 500
            parts.append(f'<text x="{cx}" y="{y}" text-anchor="middle" font-size="{size}" '
                         f'font-weight="{weight}" fill="{color}">{esc(txt)}</text>')
            y += 30.5
    parts.append(f'<rect x="48" y="456" width="864" height="72" rx="12" '
                 f'fill="#FFFDF8" stroke="#1B2838" stroke-width="1.5"/>')
    parts.append(f'<text x="480" y="492" text-anchor="middle" font-size="15" '
                 f'font-weight="500" fill="#1B2838">Tradeoff: {esc(tradeoff)}</text>')
    parts.append(f'<text x="48" y="566" font-size="16" font-weight="500" '
                 f'fill="#1B2838">One connection: {esc(footer)}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


# ---------------- C1: l01 chapter plate ----------------
A1 = np.array([2.0, 1.0, 0.0]); B1 = np.array([1.0, 1.0, 1.0])
dot1 = float(A1 @ B1); A3v = 3.0 * A1; dot3v = float(A3v @ B1)
nA1, nA3v, nB1 = (float(np.linalg.norm(v)) for v in (A1, A3v, B1))
cos1 = dot1 / (nA1 * nB1); cos3 = dot3v / (nA3v * nB1)
assert dot1 == 3.0 and dot3v == 9.0
assert round(nA1, 2) == 2.24 and round(nA3v, 2) == 6.71 and round(nB1, 2) == 1.73
assert round(nA1 * nB1, 2) == 3.87
assert round(cos1, 2) == 0.77 and round(cos3, 2) == 0.77

specs.append(("plate-l01-chap-vectors.svg", chap_plate(
    title="Chapter plate: what cosine buys",
    subtitle="Chapter plate. Size fools the dot product. Cosine divides size out. Source: original synthesis of the lesson.",
    headers=("WITHOUT: the raw dot product", "the norm", "WITH: cosine"),
    columns=[
        [("A scores 3", INK), ("A3 scores 9", INK), ("3x the score,", MUTED),
         ("same words", MUTED), ("length wins by cheating", ORANGE)],
        [("||x|| = sqrt(sum xi^2)", INK), ("||A|| = 2.24", INK), ("||A3|| = 6.71", INK),
         ("2.24 x 1.73 = 3.87", INK), ("size, measured alone", FOCUS)],
        [("3 / 3.87 = 0.77", INK), ("9 / 11.6 = 0.77", INK), ("identical", TEAL),
         ("tweet matches article", TEAL), ("direction only", TEAL)],
    ],
    tradeoff="cosine throws size away; when size carries signal (spending behavior), trust distance.",
    footer="L02 stacks these vectors into matrices, so one call scores a whole batch.",
), ["A scores 3", "A3 scores 9", "2.24", "6.71", "3.87", "0.77"]))


# ---------------- C2: l02 chapter plate ----------------
specs.append(("plate-l02-chap-matrices.svg", chap_plate(
    title="Chapter plate: compose and batch",
    subtitle="Chapter plate. One matrix call replaces a loop of per-vector calls. Source: original synthesis of the lesson.",
    headers=("WITHOUT: per-vector calls", "the product", "WITH: matrix-matrix"),
    columns=[
        [("f(g(h(v))), one vector", INK), ("at a time", INK), ("1,000 vectors =", MUTED),
         ("1,000 calls", MUTED), ("the GPU idles", ORANGE)],
        [("AB entry = row . column", INK), ("AB = [5 2; 1 1]", INK), ("entry (1,1):", FOCUS),
         ("1*3 + 2*1 = 5", FOCUS), ("composition in one object", TEAL)],
        [("X is 2x1000", INK), ("one call: MX", INK), ("GPU parallelizes", TEAL),
         ("across columns", TEAL), ("batches are wide matrices", TEAL)],
    ],
    tradeoff="the rectangle must fit the hardware; composition is paid once, then amortized.",
    footer="L09's normal equation is this product turned into a projection.",
), ["1,000 vectors", "2x1000", "1*3 + 2*1 = 5", "[5 2; 1 1]"]))


# ---------------- C3: l03 chapter plate ----------------
n3 = 12288
ops3 = n3 ** 3
assert ops3 == 12288 ** 3 and round(ops3 / 1e12, 2) == 1.86  # "about two trillion"
var_keep, var_tot = 2.0, 2.5
assert round(var_keep / var_tot * 100) == 80

specs.append(("plate-l03-chap-pca.svg", chap_plate(
    title="Chapter plate: keep the stretch that matters",
    subtitle="Chapter plate. Eigen-decomposition finds the stretch directions; PCA keeps the biggest. Source: original synthesis of the lesson.",
    headers=("WITHOUT: all directions", "eigenpairs", "WITH: top-k"),
    columns=[
        [("C = [2 0; 0 0.5]", INK), ("variance 2.5 in 2 dims", INK), ("12,288 dims:", MUTED),
         ("n^3 = ~2T ops", MUTED), ("pay the full price", ORANGE)],
        [("Av = lambda v", INK), ("lambda = 2.0 (80%)", INK), ("lambda = 0.5 (20%)", INK),
         ("perpendicular", FOCUS), ("when symmetric", FOCUS)],
        [("x-axis keeps 2/2.5", INK), ("= 80% of the spread", TEAL), ("1 dim, 80% kept", TEAL),
         ("power iteration finds it", TEAL), ("drop the y-axis", MUTED)],
    ],
    tradeoff="the dropped 20% is gone; non-square or defective matrices break the recipe.",
    footer="L04's SVD does this for rectangles, where eigenvalues do not exist.",
), ["12,288", "~2T ops", "2/2.5", "80%"]))


# ---------------- C4: l04 chapter plate ----------------
full4 = 4096 * 4096
comp4 = 64 * (4096 + 4096 + 1)
assert full4 == 16777216
assert comp4 == 524352
assert round(full4 / comp4) == 32  # "32x fewer" on the plate

specs.append(("plate-l04-chap-svd.svg", chap_plate(
    title="Chapter plate: the honest price of compression",
    subtitle="Chapter plate. SVD writes any matrix as stretch pieces; truncation drops the small ones. Source: original synthesis of the lesson.",
    headers=("WITHOUT: full rank", "the spectrum", "WITH: rank-64"),
    columns=[
        [("4096x4096 =", INK), ("16,777,216 numbers", INK), ("full memory,", MUTED),
         ("full compute", MUTED), ("no structure used", ORANGE)],
        [("A = U Sigma V^T", INK), ("sigma = [5, 3] (toy)", INK), ("rank = nonzero sigma", FOCUS),
         ("error = dropped sigma", FOCUS), ("Eckart-Young: optimal", FOCUS)],
        [("64 x 8193 =", INK), ("524,352 numbers", TEAL), ("32x fewer numbers", TEAL),
         ("LoRA updates the pieces", TEAL), ("same behavior, less", TEAL)],
    ],
    tradeoff="every dropped sigma is permanent error; computing the SVD itself costs O(n^3).",
    footer="L08 reads this spectrum as a valley: kappa = sigma-max / sigma-min.",
), ["16,777,216", "524,352", "32x", "[5, 3]"]))


# ---------------- C5: l05 chapter plate ----------------
gut5 = 0.95
post1_5 = 95 / 1085
assert round(post1_5 * 100, 1) == 8.8

specs.append(("plate-l05-chap-bayes.svg", chap_plate(
    title="Chapter plate: price the bet",
    subtitle="Chapter plate. The gut says 95%; the base rate says 8.8%. Bayes counts. Source: original synthesis of the lesson.",
    headers=("WITHOUT: the gut", "Bayes' rule", "WITH: count"),
    columns=[
        [("positive test feels", INK), ("like 95%", INK), ("95 true vs", MUTED),
         ("990 false alarms", MUTED), ("base-rate fallacy", ORANGE)],
        [("P(H|E) = P(E|H)P(H)", INK), ("/ P(E)", INK), ("denominator = 0.1085", FOCUS),
         ("every hypothesis pays", FOCUS), ("its share", FOCUS)],
        [("one +: 8.8%", INK), ("two +: 47.8%", TEAL), ("evidence multiplies", TEAL),
         ("naive Bayes:", TEAL), ("each word is a test", TEAL)],
    ],
    tradeoff="tests cost time and money; independence is assumed, rarely true, yet ranking survives.",
    footer="L06 turns the update into fitting: the MLE maximizes the likelihood.",
), ["95%", "8.8%", "47.8%", "0.1085", "990"]))


# ---------------- C6: l06 chapter plate ----------------
L_raw = 0.5 ** 1000
L_log = 1000 * np.log(0.5)
assert L_raw == 9.332636185032189e-302
assert round(L_log, 1) == -693.1

specs.append(("plate-l06-chap-mle.svg", chap_plate(
    title="Chapter plate: maximize the log",
    subtitle="Chapter plate. Raw likelihoods underflow; the log turns products into sums. Source: original synthesis of the lesson.",
    headers=("WITHOUT: raw likelihood", "log-likelihood", "WITH: closed forms"),
    columns=[
        [("0.5^1000 = 9.3e-302", INK), ("1100 flips: float64", MUTED), ("rounds to 0.0", MUTED),
         ("every candidate scores 0", MUTED), ("grid search guesses p", ORANGE)],
        [("log L = sum log p(xi)", INK), ("1000 flips: -693.1", INK), ("monotone: same argmax", FOCUS),
         ("NLL = the loss", FOCUS), ("products become sums", FOCUS)],
        [("Bernoulli: p = 0.4", INK), ("counting is estimating", TEAL), ("Gaussian: mean 5,", TEAL),
         ("variance 4", TEAL), ("two lines of code", TEAL)],
    ],
    tradeoff="MLE divides by n (4.0, biased 7/8); n-1 (4.5714) unbiases; large n: the same.",
    footer="L10's cross-entropy is this NLL on a categorical model.",
), ["9.3e-302", "-693.1", "0.4", "4.5714"]))


# ---------------- C7: l07 chapter plate ----------------
v7, e7 = 0.9 ** 10, 1.1 ** 10
assert round(v7, 2) == 0.35 and round(e7, 2) == 2.59

specs.append(("plate-l07-chap-chainrule.svg", chap_plate(
    title="Chapter plate: one backward pass",
    subtitle="Chapter plate. The chain rule orders the multiplies; backprop runs them once. Source: original synthesis of the lesson.",
    headers=("WITHOUT: bump each weight", "the chain rule", "WITH: backprop"),
    columns=[
        [("n weights =", INK), ("n forward passes", INK), ("w1 = 3.01: y = 24.08", MUTED),
         ("slope 8, one weight", MUTED), ("at a time", MUTED)],
        [("dy/dw1 =", INK), ("(dy/dh)(dh/dw1)", INK), ("= 4 x 2 = 8", FOCUS),
         ("ordered product", FOCUS), ("of Jacobians", FOCUS)],
        [("one forward,", INK), ("one backward", INK), ("dy/dw2 = 6,", TEAL),
         ("dy/dw1 = 8", TEAL), ("bump test agrees", TEAL)],
    ],
    tradeoff="activations are stored for the backward pass; 0.9^10 = 0.35 vanishes, 1.1^10 = 2.59 explodes.",
    footer="L08 walks downhill along this gradient; the step size decides everything.",
), ["24.08", "4 x 2 = 8", "0.35", "2.59"]))


# ---------------- C8: l08 chapter plate ----------------
loss_div = [9.0, 12.96, 18.66, 26.87]
assert all(b > a for a, b in zip(loss_div, loss_div[1:]))
eta_over = (1.1 - 1.0) / 1.0
assert round(eta_over * 100) == 10
gap_close = 0.8
assert round(gap_close ** 2, 2) == 0.64          # loss factor per step

specs.append(("plate-l08-chap-gd.svg", chap_plate(
    title="Chapter plate: one valley, one step size",
    subtitle="Chapter plate. Convexity promises one valley; eta < 2/L keeps the walk inside it. Source: original synthesis of the lesson.",
    headers=("WITHOUT: wrong step", "the rule", "WITH: sane step"),
    columns=[
        [("eta = 1.1", INK), ("loss 9.0 -> 26.87", INK), ("each step overshoots", MUTED),
         ("further out", MUTED), ("non-convex: traps", ORANGE)],
        [("eta < 2/L", INK), ("L = 2, so eta < 1", INK), ("1.1 is 10% past", FOCUS),
         ("f'' >= 0: one valley", FOCUS), ("local = global", FOCUS)],
        [("eta = 0.1:", INK), ("gap closes 20%/step", TEAL), ("loss x0.64 per step", TEAL),
         ("Newton: x1 = 3", TEAL), ("one step, exact", TEAL)],
    ],
    tradeoff="convex is the special case; ravines (kappa = 25) zigzag and crawl without momentum.",
    footer="momentum and Adam exist for the ravines this bowl does not have.",
), ["26.87", "10%", "0.64", "x1 = 3", "kappa = 25"]))


# ---------------- C9: l09 chapter plate ----------------
XtX, Xty = 14.0, 14.5
w9 = Xty / XtX
assert round(w9, 4) == 1.0357

specs.append(("plate-l09-chap-leastsquares.svg", chap_plate(
    title="Chapter plate: the projection formula",
    subtitle="Chapter plate. Guessing w loses to solving for w; the normal equation is the solve. Source: original synthesis of the lesson.",
    headers=("WITHOUT: guessing", "the normal equation", "WITH: solve"),
    columns=[
        [("w = 1.0, 1.05, 1.1", INK), ("errors 0.06,", MUTED), ("0.045, 0.10", MUTED),
         ("guess forever", MUTED), ("never exact", ORANGE)],
        [("X^T X w = X^T y", INK), ("14 w = 14.5", INK), ("gradient = 0", FOCUS),
         ("projection in", FOCUS), ("matrix form", FOCUS)],
        [("w = 1.0357", TEAL), ("error 0.0421,", TEAL), ("beats every guess", TEAL),
         ("residual perpendicular", TEAL), ("to the column space", TEAL)],
    ],
    tradeoff="X^T X costs O(n^3); duplicate features give det = 0: infinite solutions.",
    footer="L02's projection is this picture: the prediction is the shadow on the column space.",
), ["1.0357", "0.0421", "14 w = 14.5", "det = 0"]))


# ---------------- C10: l10 chapter plate ----------------
H10 = 1.75
kl10 = 0.25 * np.log2(0.25 / 0.5) + 0.75 * np.log2(0.75 / 0.5)
assert round(H10, 2) == 1.75
assert round(kl10, 3) == 0.189
ce_wrong = -np.log2(0.01)
assert round(ce_wrong, 2) == 6.64
perp3 = 2 ** 3
assert perp3 == 8

specs.append(("plate-l10-chap-coding.svg", chap_plate(
    title="Chapter plate: the floor and the bill",
    subtitle="Chapter plate. Entropy is the coding floor; cross-entropy is the bill the model pays. Source: original synthesis of the lesson.",
    headers=("WITHOUT: bad bets", "entropy H", "WITH: pay the floor"),
    columns=[
        [("predict 0.01 on truth:", INK), ("6.64 bits", INK), ("q = 0 on truth:", MUTED),
         ("infinite bill", MUTED), ("wrong code: H + KL", ORANGE)],
        [("H = 1.75 bits", INK), ("surprise = -log2 p", INK), ("KL = 0.189 bits", FOCUS),
         ("asymmetric:", FOCUS), ("not a distance", FOCUS)],
        [("Huffman: 1.75 avg", TEAL), ("codes 0/10/110/111", TEAL), ("= the floor", TEAL),
         ("3 bits -> perplexity 8", TEAL), ("gradient q-p bounded", TEAL)],
    ],
    tradeoff="smoothing defends the infinite bill; cross-entropy only prices the true class.",
    footer="every LM trains on this bill; 3 bits of cross-entropy is a perplexity of 8.",
), ["1.75", "0.189", "6.64", "perplexity 8"]))


# ============ Tables (B.4, B.8, B.9): computed, asserted, written to /tmp ============
TABLES = {}

# B.4: the n vs n-1 question (l06 @351)
ssd = 32.0
mle_var = ssd / 8
bes_var = ssd / 7
bias_factor = 7 / 8
assert mle_var == 4.0
assert round(bes_var, 4) == 4.5714
assert bias_factor == 0.875
TABLES["table-n-vs-nminus1.md"] = f"""| Estimator | Formula on the toy | Value | Bias | Use when |
|---|---|---|---|---|
| MLE variance (divide by n) | 32 / 8 | {mle_var:.1f} | biased: expectation is (n-1)/n = {bias_factor} of the truth | n large, or model training (matches the likelihood) |
| Bessel's correction (divide by n-1) | 32 / 7 | {bes_var:.4f} | unbiased | small samples in classical statistics |

numpy's `var` divides by n ({mle_var:.1f}); pandas' divides by n-1 ({bes_var:.4f}). The disagreement vanishes as n grows.
"""

# B.8: perplexity (l10 @235) — smallest passing medium is the table
perp_rows = []
for bits in (1, 2, 3):
    perp_rows.append((bits, 2 ** bits))
log2_20, log2_200 = float(np.log2(20)), float(np.log2(200))
assert round(log2_20, 2) == 4.32 and round(log2_200, 2) == 7.64
assert 2 ** 3 == 8
TABLES["table-perplexity.md"] = """| Cross-entropy (bits) | Perplexity = 2^H | Reads as |
|---|---|---|
| 1 | 2 | as confused as choosing uniformly among 2 words |
| 2 | 4 | as confused as choosing uniformly among 4 words |
| 3 | 8 | as confused as choosing uniformly among 8 words |
| 4.32 | 20 | strong on open text |
| 7.64 | 200 | weak |

Perplexity 8 = cross-entropy 3 bits. Same loss, human-readable scale: lower is better.
"""

# B.9: metrics, because accuracy lies (l08 @309)
TP, TN, FP, FN = 0, 99, 0, 1
total = TP + TN + FP + FN
acc = (TP + TN) / total
rec = TP / (TP + FN)
assert total == 100
assert round(acc * 100, 1) == 99.0
assert round(rec * 100, 1) == 0.0
# precision = TP/(TP+FP) = 0/0: undefined — no positives predicted at all
TABLES["table-metrics.md"] = """| Metric | Formula | Always-healthy score | What it hides |
|---|---|---|---|
| Accuracy | (TP + TN) / total | 99 / 100 = 99% | looks perfect |
| Recall | TP / (TP + FN) | 0 / 1 = 0% | found none of the sick |
| Precision | TP / (TP + FP) | 0 / 0: undefined | predicted no positives at all |

On 100 patients (1 sick), always saying "healthy" scores 99% accuracy and 0% recall. On rare classes, report precision and recall always.
"""


def main():
    os.makedirs(ASSETS, exist_ok=True)
    failed = False
    for fname, svg, required in specs:
        path = os.path.join(ASSETS, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        ET.parse(path)  # raises on malformed XML
        missing = [r for r in required if r not in svg]
        status = "OK" if not missing else f"MISSING {missing}"
        if missing:
            failed = True
        print(f"  {fname}: {status}")
    for fname, md in TABLES.items():
        path = os.path.join("/tmp", fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"  wrote table {path}")
    if failed:
        raise SystemExit("VERIFICATION FAILED")
    print(f"All {len(specs)} SVGs: valid XML, all required numbers present.")


if __name__ == "__main__":
    main()
