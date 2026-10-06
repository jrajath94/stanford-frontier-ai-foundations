#!/usr/bin/env python3
"""Figure Enforcer: 12 deterministic SVG lesson plates for math-ml.

Every number on every plate is computed in this script (numpy) and asserted
against the lesson caption BEFORE the file is written. No number the code
did not compute. Style follows site/v2/math-ml/assets/plate-l01-cosine.svg
and build/FIGURE_SPEC_STANFORD.md (warm paper #F7F4EE, 8px grid, one claim
per plate, left = toy, center = one named rule, right = result).
"""
import html
import os
import xml.etree.ElementTree as ET

import numpy as np

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/site/v2/math-ml/assets")

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


def esc(s):
    return html.escape(str(s), quote=False)


def T(x, y, s, size=14.5, weight=500, fill=INK, family=MONO, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def rect_panel(x, y, w, h, fill):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            f'fill="{fill}" stroke="{LINE}" stroke-width="1.5"/>')


def arrow(x1, x2, y, label):
    hx = x2 - 9.1
    return "\n".join([
        f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{TEAL}" stroke-width="2"/>',
        f'<line x1="{x2}" y1="{y}" x2="{hx}" y2="{y - 4.1}" stroke="{TEAL}" stroke-width="2"/>',
        f'<line x1="{x2}" y1="{y}" x2="{hx}" y2="{y + 4.1}" stroke="{TEAL}" stroke-width="2"/>',
        T((x1 + x2) / 2, y - 14, label, size=13, weight=500, fill=TEAL, family=FONT, anchor="middle"),
    ])


def frame(title, claim_lines, footer_claim, panels, arrow_labels):
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
    title_size = 28 if len(title) > 44 else 30  # spec range 28-36px; keep long titles inside 736
    parts.append(T(24, 40, title, size=title_size, weight=600, family=FONT))
    if len(claim_lines) == 1:
        parts.append(T(24, 66, claim_lines[0], size=16, weight=450, fill=MUTED, family=FONT))
    else:
        parts.append(T(24, 64, claim_lines[0], size=16, weight=450, fill=MUTED, family=FONT))
        parts.append(T(24, 86, claim_lines[1], size=16, weight=450, fill=MUTED, family=FONT))
    parts.append(f'<line x1="24" y1="{div_y}" x2="736" y2="{div_y}" '
                 f'stroke="{LINE}" stroke-width="1.5"/>')
    parts.append(T(24, 414, "One claim: " + footer_claim, size=14, weight=450, fill=MUTED, family=FONT))
    parts.append(T(736, 414, "Source: original", size=13, weight=500, fill=MUTED, family=FONT, anchor="end"))
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


# ---------------- Plate 1: Cauchy-Schwarz ----------------
A = np.array([2.0, 1.0, 0.0])
B = np.array([1.0, 1.0, 1.0])
dot = float(A @ B)
nA, nB = float(np.linalg.norm(A)), float(np.linalg.norm(B))
prod = nA * nB
cosAB = dot / prod
A3 = 3.0 * A
dot3 = float(A @ A3)
nA3 = float(np.linalg.norm(A3))
assert dot == 3.0
assert round(nA, 2) == 2.24 and round(nB, 2) == 1.73
assert round(prod, 2) == 3.87
assert round(cosAB, 2) == 0.77
assert dot3 == 15.0 and round(nA3, 2) == 6.71
assert round(2.24 * 6.71) == 15.0  # lesson rounds 15.03 to 15

specs.append(("plate-l01-cauchyschwarz.svg", frame(
    title="Cauchy-Schwarz bounds the dot product",
    claim_lines=["cos(A,B) = 0.77 is trapped in [-1,1] by |x.y| <= ||x|| ||y||.",
                 "Equality only for scaled copies."],
    footer_claim="the dot product can never outrun the product of the norms.",
    panels=[
        ("The toy", [
            ("x·y = 3", INK),
            ("‖x‖ = 2.24, ‖y‖ = 1.73", INK),
            ("2.24 × 1.73 = 3.87", INK),
            ("|3| ≤ 3.87, holds", TEAL),
        ]),
        ("Normalize", [
            ("|x·y|", INK, 13),
            ("/(‖x‖ × ‖y‖)", INK, 13),
            ("≤ 1", INK, 13),
        ]),
        ("Trapped in [-1,1]", [
            ("cos(A,B) = 0.77", INK),
            ("-1 ≤ 0.77 ≤ 1", TEAL),
            ("equality: cos(A,A3)", MUTED),
            ("= 15/(2.24×6.71)", INK),
            ("= 15/15 = 1", TEAL),
        ]),
    ],
    arrow_labels=["divide", "bound"],
), ["2.24", "1.73", "3.87", "0.77", "15/15"]))


# ---------------- Plate 2: distance vs cosine ----------------
diff = A - A3
dist = float(np.linalg.norm(diff))
cosAA3 = float(A @ A3) / (nA * nA3)
assert round(dist, 2) == 4.47
assert round(cosAA3, 2) == 1.00

specs.append(("plate-l01-distance.svg", frame(
    title="Distance cares about size, cosine does not",
    claim_lines=["A vs A3: cosine 1.00, distance 4.47. Same direction, different measures."],
    footer_claim="distance measures separation, cosine measures direction.",
    panels=[
        ("The toy", [
            ("A  = [2,1,0]", INK),
            ("A3 = [6,3,0]  (3×A)", INK),
            ("A − A3 = [−4,−2,0]", INK),
        ]),
        ("Euclidean gap", [
            ("‖x − y‖", INK, 13),
            ("= √(16+4+0)", INK, 13),
        ]),
        ("Two answers", [
            ("cos(A,A3) = 1.00", INK),
            ("‖A−A3‖ = √20", INK),
            ("= 4.47", TEAL),
            ("same direction,", MUTED),
            ("different measures", MUTED),
        ]),
    ],
    arrow_labels=["subtract", "norm"],
), ["1.00", "4.47", "√20"]))


# ---------------- Plate 3: matrix-matrix product ----------------
Am = np.array([[1, 2], [0, 1]])
Bm = np.array([[3, 0], [1, 1]])
AB = Am @ Bm
assert np.array_equal(AB, [[5, 2], [1, 1]])
assert AB[0, 0] == 1 * 3 + 2 * 1 == 5
assert AB[1, 1] == 0 * 0 + 1 * 1 == 1

specs.append(("plate-l02-matmat.svg", frame(
    title="Matrix-matrix: every entry is a row-dot-column",
    claim_lines=["AB = [5 2. 1 1], entry by entry. The pair the transpose QA uses."],
    footer_claim="every entry of AB is one row-dot-column.",
    panels=[
        ("The inputs", [
            ("A = [1 2]", INK),
            ("    [0 1]", INK),
            ("B = [3 0]", INK),
            ("    [1 1]", INK),
        ]),
        ("Row·column", [
            ("entry (i,j)", INK, 13),
            ("= row i · col j", INK, 13),
        ]),
        ("AB, entry by entry", [
            ("AB = [5 2]", INK),
            ("     [1 1]", INK),
            ("(1,1): 1·3+2·1 = 5", INK),
            ("(2,2): 0·0+1·1 = 1", INK),
        ]),
    ],
    arrow_labels=["dot", "place"],
), ["[5 2]", "[1 1]", "1·3+2·1 = 5", "0·0+1·1 = 1"]))


# ---------------- Plate 4: outer vs dot product ----------------
u = np.array([2, 1])
v = np.array([3, 1])
outer = np.outer(u, v)
dotuv = int(u @ v)
assert np.array_equal(outer, [[6, 2], [3, 1]])
assert dotuv == 7
assert int(np.linalg.matrix_rank(outer)) == 1

specs.append(("plate-l02-outer.svg", frame(
    title="Outer product expands, dot product collapses",
    claim_lines=["u v^T = [6 2. 3 1], rank 1. u^T v = 7, one number."],
    footer_claim="shapes decide: the outer product expands, the dot product collapses.",
    panels=[
        ("The inputs", [
            ("u  = [2,1]  (column)", INK),
            ("v^T = [3,1]  (row)", INK),
        ]),
        ("Shape rule", [
            ("(n×1)(1×n)", INK, 13),
            ("= n×n", INK, 13),
            ("(1×n)(n×1)", INK, 13),
            ("= 1×1", INK, 13),
        ]),
        ("Expands, collapses", [
            ("u v^T = [6 2]", INK),
            ("        [3 1]", INK),
            ("rank 1: expands", TEAL),
            ("u^T v = 7, one number", INK),
        ]),
    ],
    arrow_labels=["multiply", "compare"],
), ["[6 2]", "[3 1]", "= 7", "rank 1"]))


# ---------------- Plate 5: Hadamard ----------------
H = Am * Bm  # elementwise
assert np.array_equal(H, [[3, 0], [0, 1]])
assert not np.array_equal(H, AB)

specs.append(("plate-l02-hadamard.svg", frame(
    title="Hadamard: same inputs, different meaning",
    claim_lines=["A circ B = [3 0. 0 1] vs AB = [5 2. 1 1]. No mixing."],
    footer_claim="Hadamard scales in place, matrix multiplication mixes.",
    panels=[
        ("Same inputs", [
            ("A = [1 2]", INK),
            ("    [0 1]", INK),
            ("B = [3 0]", INK),
            ("    [1 1]", INK),
        ]),
        ("Entrywise", [
            ("(A∘B)ij =", INK, 13),
            ("Aij · Bij", INK, 13),
            ("no mixing", TEAL, 13),
        ]),
        ("Two meanings", [
            ("A∘B = [3 0]", INK),
            ("      [0 1]", INK),
            ("AB  = [5 2]", INK),
            ("      [1 1]  (mixes)", MUTED),
        ]),
    ],
    arrow_labels=["multiply", "compare"],
), ["[3 0]", "[0 1]", "[5 2]", "[1 1]"]))


# ---------------- Plate 6: full-matrix eigenvalues ----------------
C = np.array([[4, 2], [1, 3]])
w = np.linalg.eigvals(C)
tr = float(np.trace(C))
det = float(np.linalg.det(C))
v1 = np.array([2, 1])
v2 = np.array([1, -1])
assert sorted(np.round(w).astype(int).tolist()) == [2, 5]
assert tr == 7.0 and round(det) == 10
assert np.array_equal(C @ v1, 5 * v1)  # [10, 5]
assert np.array_equal(C @ v2, 2 * v2)  # [2, -2]

specs.append(("plate-l03-fullmatrix.svg", frame(
    title="Eigenvalues of a full matrix, verified",
    claim_lines=["C = [4 2. 1 3]: eigenvalues 5 and 2, trace 7, determinant 10.",
                 "Each eigenvector verified by multiplication."],
    footer_claim="eigenvalues 5 and 2, each verified by multiplication.",
    panels=[
        ("C = [4 2; 1 3]", [
            ("C = [4 2]", INK),
            ("    [1 3]", INK),
            ("trace = 4+3 = 7", INK),
            ("det = 12−2 = 10", INK),
        ]),
        ("Solve det=0", [
            ("det(C−λI)=0", INK, 13),
            ("λ²−7λ+10=0", INK, 13),
            ("λ = 5, 2", TEAL, 13),
        ]),
        ("Verified", [
            ("C[2,1] = [10,5]", INK),
            ("= 5·[2,1]  ✓ λ=5", TEAL),
            ("C[1,−1] = [2,−2]", INK),
            ("= 2·[1,−1]  ✓ λ=2", TEAL),
        ]),
    ],
    arrow_labels=["solve", "verify"],
), ["[10,5]", "[2,−2]", "= 7", "= 10", "λ = 5, 2"]))


# ---------------- Plate 7: power iteration ----------------
xn = np.array([1.0, 0.0])
steps = []
for _ in range(3):
    x = C @ xn          # multiply the NORMALIZED vector each step
    xn = x / np.linalg.norm(x)
    steps.append((x.copy(), xn.copy()))
(x1, xn1), (x2, xn2), (x3, xn3) = steps
xn0 = np.array([1.0, 0.0])
target = np.array([2.0, 1.0]) / np.sqrt(5.0)
rayleigh = float(xn3 @ C @ xn3)
assert np.array_equal(np.round(xn1, 3), [0.970, 0.243])
assert np.array_equal(np.round(x2, 2), [4.37, 1.70])
assert np.array_equal(np.round(xn2, 3), [0.932, 0.362])
assert np.array_equal(np.round(x3, 2), [4.45, 2.02])
assert round(float(np.linalg.norm(x3)), 2) == 4.89
assert np.array_equal(np.round(xn3, 3), [0.911, 0.413])
assert np.array_equal(np.round(target, 3), [0.894, 0.447])
assert round(rayleigh, 2) == 4.96

specs.append(("plate-l03-power.svg", frame(
    title="Power iteration converges to the top eigenvector",
    claim_lines=["Three steps: [1,0] to [0.911,0.413], near [0.894,0.447].",
                 "Rayleigh quotient 4.96 vs true 5."],
    footer_claim="repeating Cx converges to the top eigenvector.",
    panels=[
        ("Three steps", [
            ("x0 = [1,0]", INK),
            ("x1 = [0.970,0.243]", INK),
            ("x2 = [0.932,0.362]", INK),
            ("x3 = [0.911,0.413]", TEAL),
        ]),
        ("Iterate", [
            ("x ← Cx/‖Cx‖", INK, 13),
        ]),
        ("Converged", [
            ("x3 = [0.911,0.413]", TEAL),
            ("target [0.894,0.447]", INK),
            ("Rayleigh = 4.96", INK),
            ("true λ1 = 5", MUTED),
        ]),
    ],
    arrow_labels=["iterate", "estimate"],
), ["0.911", "0.413", "0.894", "0.447", "4.96"]))


# ---------------- Plate 8: rectangular SVD ----------------
Ar = np.array([[3, 2], [2, 3], [-2, 2]])
AtA = Ar.T @ Ar
lam = np.linalg.eigvalsh(AtA)
sig = np.sqrt(lam)[::-1]
s2 = 1.0 / np.sqrt(2.0)
rv1 = np.array([s2, s2])
rv2 = np.array([s2, -s2])
u1 = Ar @ rv1 / sig[0]
u2 = Ar @ rv2 / sig[1]
assert np.array_equal(np.round(lam), [9.0, 25.0])
assert np.array_equal(np.round(sig), [5.0, 3.0])
assert np.array_equal(np.round(rv1, 4), [0.7071, 0.7071])
assert np.array_equal(np.round(rv2, 4), [0.7071, -0.7071])
assert np.array_equal(np.round(u1, 4), [0.7071, 0.7071, 0.0])
assert np.array_equal(np.round(u2, 4), [0.2357, -0.2357, -0.9428])
# entry (0,0) and (2,1) of A = U Σ Vᵀ, using lesson-rounded values
assert round(5 * 0.7071 * 0.7071, 2) == 2.5
assert round(3 * 0.2357 * 0.7071, 2) == 0.5
assert round(5 * 0.0 * 0.7071, 2) == 0.0
assert round(3 * (-0.9428) * (-0.7071), 2) == 2.0

specs.append(("plate-l04-rect.svg", frame(
    title="A full-rank rectangular SVD, verified",
    claim_lines=["3x2 matrix: singular values 5 and 3. Both entries of A rebuilt by hand."],
    footer_claim="UΣVᵀ rebuilds every entry of A.",
    panels=[
        ("A is 3×2", [
            ("A = [3 2]", INK),
            ("    [2 3]", INK),
            ("    [−2 2]", INK),
            ("AᵀA eig: 25, 9", INK),
            ("σ = [5, 3], rank 2", TEAL),
        ]),
        ("Build U", [
            ("u_i = Av_i/σ_i", INK, 13),
        ]),
        ("Rebuilt by hand", [
            ("u1 = [0.7071, 0.7071,", INK, 13),
            ("       0]", INK, 13),
            ("u2 = [0.2357, −0.2357,", INK, 13),
            ("       −0.9428]", INK, 13),
            ("(0,0): 2.5+0.5 = 3.0", INK),
            ("(2,1): 0+2.0 = 2.0", INK),
            ("both correct", TEAL),
        ]),
    ],
    arrow_labels=["build", "rebuild"],
), ["0.7071", "0.2357", "−0.9428", "3.0", "2.0", "σ = [5, 3]"]))


# ---------------- Plate 9: condition number ----------------
kappa = 5.0 / 3.0
assert round(kappa, 2) == 1.67

specs.append(("plate-l04-cond.svg", frame(
    title="The condition number is stretch imbalance",
    claim_lines=["kappa = 5/3 = 1.67: mild.",
                 "Large kappa means narrow valleys for gradient descent."],
    footer_claim="κ = σmax/σmin measures stretch imbalance.",
    panels=[
        ("The spectrum", [
            ("σ1 = 5", INK),
            ("σ2 = 3", INK),
        ]),
        ("κ ratio", [
            ("κ = σmax/σmin", INK, 13),
            ("= 5/3", INK, 13),
        ]),
        ("Mild", [
            ("κ = 1.67", TEAL),
            ("near 1: balanced", INK),
            ("huge κ: narrow valleys", INK),
            ("zigzag gradient steps", MUTED),
        ]),
    ],
    arrow_labels=["divide", "read"],
), ["5/3", "1.67"]))


# ---------------- Plate 10: independence ----------------
from fractions import Fraction
assert Fraction(4, 52) == Fraction(1, 13)
assert Fraction(2, 26) == Fraction(1, 13)
assert Fraction(4, 12) == Fraction(1, 3)
p_king = 4 / 52
p_face = 12 / 52
p_king_and_face = 4 / 52
p_product = p_king * p_face
assert round(p_king_and_face, 4) == 0.0769
assert round(p_product, 4) == 0.0178
ratio = (4 / 12) / (4 / 52)   # P(king|face) / P(king) = 13/3
assert ratio == 13 / 3
assert round(ratio, 2) == 4.33

specs.append(("plate-l05-independence.svg", frame(
    title="Independence: conditioning changes nothing",
    claim_lines=["P(ace|red) = P(ace): independent. P(king|face) ≈ 4.33× P(king): dependent."],
    footer_claim="independence means conditioning changes nothing.",
    panels=[
        ("Color vs rank", [
            ("P(ace) = 4/52 = 1/13", INK),
            ("P(ace|red) = 2/26", INK),
            ("= 1/13, independent", TEAL),
        ]),
        ("Test equality", [
            ("P(A|B)=P(A)?", INK, 13),
            ("or P(A∩B)=", INK, 13),
            ("P(A)·P(B)?", INK, 13),
        ]),
        ("Rank vs face", [
            ("P(king) = 4/52 = 1/13", INK),
            ("P(king|face) = 4/12", INK),
            ("= 1/3 ≈ 4.33× P(king)", INK),
            ("0.0769 vs 0.0178:", MUTED),
            ("product rule fails", MUTED),
        ]),
    ],
    arrow_labels=["compare", "contrast"],
), ["1/13", "1/3", "4.33× P(king)", "0.0769", "0.0178"]))


# ---------------- Plate 11: law of total probability ----------------
t1 = 0.95 * 0.01
t2 = 0.10 * 0.99
ptotal = t1 + t2
assert round(t1, 4) == 0.0095
assert round(t2, 4) == 0.099
assert round(ptotal, 4) == 0.1085

specs.append(("plate-l05-totalprob.svg", frame(
    title="The denominator, derived",
    claim_lines=["P(positive) = 0.0095 + 0.0990 = 0.1085.",
                 "Every hypothesis contributes its share."],
    footer_claim="the denominator sums every hypothesis share.",
    panels=[
        ("Two hypotheses", [
            ("sick: 1% of people", INK),
            ("P(+|sick) = 0.95", INK),
            ("healthy: 99%", INK),
            ("P(+|healthy) = 0.10", INK),
        ]),
        ("Add shares", [
            ("P(E) = Σ", INK, 13),
            ("P(E|H)·P(H)", INK, 13),
        ]),
        ("The denominator", [
            ("0.95 × 0.01 = 0.0095", INK),
            ("0.10 × 0.99 = 0.0990", INK),
            ("sum = 0.1085", TEAL),
        ]),
    ],
    arrow_labels=["weight", "add"],
), ["0.0095", "0.0990", "0.1085"]))


# ---------------- Plate 12: spam filter ----------------
p_spam, p_free_given_spam, p_free_given_legit = 0.20, 0.40, 0.05
p_free = p_free_given_spam * p_spam + p_free_given_legit * (1 - p_spam)
num = p_free_given_spam * p_spam
post = num / p_free
assert round(num, 2) == 0.08
assert round(p_free, 2) == 0.12
assert round(post, 3) == 0.667

specs.append(("plate-l05-spam.svg", frame(
    title="One word moves the belief",
    claim_lines=["P(spam|free) = 0.08/0.12 = 0.667. One word: 20% to 66.7%."],
    footer_claim="one word moved the belief from 20% to 66.7%.",
    panels=[
        ("The toy", [
            ("P(spam) = 0.20", INK),
            ("P(free|spam) = 0.40", INK),
            ("P(free|legit) = 0.05", INK),
        ]),
        ("Bayes flip", [
            ("P(free) =", INK, 13),
            ("0.08+0.04", INK, 13),
            ("= 0.12", INK, 13),
        ]),
        ("Belief moves", [
            ("P(spam|free)", INK),
            ("= 0.08/0.12", INK),
            ("= 0.667", TEAL),
            ("20% → 66.7%", TEAL),
        ]),
    ],
    arrow_labels=["total", "divide"],
), ["0.08/0.12", "0.667", "20%", "66.7%"]))


def main():
    assert len(specs) == 12, f"expected 12 plates, got {len(specs)}"
    os.makedirs(ASSETS, exist_ok=True)
    results = []
    for fname, svg, required in specs:
        path = os.path.join(ASSETS, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        # verify: valid XML/SVG + required numbers present as text
        ET.parse(path)  # raises on malformed XML
        raw = svg
        missing = [r for r in required if r not in raw]
        results.append((fname, path, required, missing))
    print(f"Wrote {len(results)} plates to {ASSETS}")
    failed = False
    for fname, path, required, missing in results:
        status = "OK" if not missing else f"MISSING {missing}"
        if missing:
            failed = True
        print(f"  {fname}: {status}  (numbers: {', '.join(required)})")
    if failed:
        raise SystemExit("VERIFICATION FAILED: some required numbers absent")
    print("All 12 plates: valid XML, all required numbers present.")


if __name__ == "__main__":
    main()
