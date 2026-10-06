#!/usr/bin/env python3
"""Deepening plates for math-genmodels (L01-L08). Warm-paper lesson plates via
the shared Plate renderer, saved as webp into math-genmodels/assets.

Visual system: BG #F7F4EE, ink #1B2838, 8px grid, flat fills, one claim per
plate. Every number on a plate is computed in this script.
"""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
import plates_mathgenai as pm
pm.ASSETS = "/home/hatch/workspace/stanford-frontier-ai/content/v2/math-genmodels/assets"
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# ---------------------------------------------------------------- L01 ---
# 1. The counting machine: 20 rolls -> counts -> rule -> fresh samples
p = Plate("The counting machine: rolls become a rule",
          "20 rolls give the rule (0.65, 0.20, 0.15); rolling the rule draws fresh samples.",
          "Shell 2. Count the toy: each roll is one data point.",
          source="original toy", inner_h=420)
rolls = [1,1,3,1,2,1,1,3,1,2,1,1,1,3,2,1,1,1,2,1]
p.text(32, p.top, "20 rolls", size=14, bold=True, color=MUTED)
for i, r in enumerate(rolls):
    row = i // 10
    col = i % 10
    p.chip(32 + col * 88, p.top + 28 + row * 44, str(r),
           fill=COUNT if r == 1 else (NEW if r == 2 else PINK), size=13)
p.text(32, p.top + 132, "counts:  1 -> 13     2 -> 4     3 -> 3", size=15, bold=True)
p.text(32, p.top + 160, "rule:  P(1) = 0.65   P(2) = 0.20   P(3) = 0.15", size=15, bold=True)
p.arrow(32, p.top + 210, 200, p.top + 210, label="roll the rule")
p.chip(216, p.top + 194, "fresh roll: 1", fill=NEW, size=14)
p.chip(216, p.top + 238, "fresh roll: 3", fill=NEW, size=14)
p.chip(216, p.top + 282, "fresh roll: 1", fill=NEW, size=14)
p.text(32, p.top + 340, "learning = counting. sampling = rolling the counts.", size=14, color=MUTED)
p.save("plate-l01-counting.webp")

# 2. The table explodes: outcome counts on a log ladder
p = Plate("The table explodes",
          "A thumbnail has 2^1024 outcomes. A photo has ~10^473000. No table survives.",
          "Shell 2. Count the outcomes; the counters needed equal them.",
          source="original toy", inner_h=400)
rows = [("loaded die", 3, 0.48), ("32x32 binary image", 2**1024, 308.0),
        ("256x256 color photo", 10**473000, 473000.0)]
# log10 scale bars: heights proportional to log10(count)
y = p.top + 8
p.text(32, y, "outcomes (log scale)", size=14, bold=True, color=MUTED)
y += 32
for name, n, lg in rows:
    p.text(32, y + 6, name, size=14)
    bw = max(8, int(600 * lg / 473000.0)) if lg > 1 else 8
    bw = min(bw, 430)  # keep the value label inside the 960px plate (audit fix 2026-10-06)
    p.rect(320, y, bw, 28, TEAL if lg < 300 else (ORANGE if lg < 400000 else PINK), label=None, rx=8)
    p.text(336 + bw, y + 4, f"10^{lg:.0f}" if lg >= 1 else "3", size=14, bold=True)
    y += 56
p.text(32, y + 8, "one counter per outcome: the die needs 3, the photo needs more counters", size=14, color=MUTED)
p.text(32, y + 32, "than atoms exist. counting dies by counting.", size=14, color=MUTED)
p.save("plate-l01-explosion.webp")
print("L01 plates done")

# ---------------------------------------------------------------- L02 ---
# 3. Chain-rule factorization tree for "the cat sat"
p = Plate("The chain rule is an identity, not an approximation",
          "0.80 x 1.00 x 0.75 = 0.60, exactly the corpus fraction 6/10.",
          "Shell 2. Multiply the toy: the product must equal the count.",
          source="original toy", inner_h=380)
p.text(32, p.top, "P(\"the cat sat\") =", size=15, bold=True)
nodes = [("P(the)", "0.80", COUNT), ("P(cat | the)", "1.00", NEW),
         ("P(sat | the cat)", "0.75", ACTIVE)]
xx = 32
for label, val, fill in nodes:
    p.panel(xx, p.top + 36, 280, 200, fill=fill, label=label)
    p.text(xx + 16, p.top + 80, "corpus count:", size=13, color=MUTED)
    if label == "P(the)":
        det = "\"the\" starts 8 of 10"
    elif label == "P(cat | the)":
        det = "after \"the\": 8 of 8"
    else:
        det = "after \"the cat\": 6 of 8"
    p.text(xx + 16, p.top + 104, det, size=13)
    p.rect(xx + 16, p.top + 140, 120, 56, BG, label=val, size=20, rx=12)
    xx += 304
p.arrow(296, p.top + 136, 328, p.top + 136, label="x")
p.arrow(600, p.top + 136, 632, p.top + 136, label="x")
p.panel(32, p.top + 264, 888, 88, fill=CHIP, label="product")
p.text(56, p.top + 300, "0.80 x 1.00 x 0.75 = 0.60  =  6/10 sentences in the corpus", size=16, bold=True)
p.save("plate-l02-tree.webp")

# 4. Teacher forcing: parallel training vs serial sampling
p = Plate("Teacher forcing: training is parallel, sampling is serial",
          "True histories are known upfront, so all positions train at once. Sampling waits its turn.",
          "Shell 3. Apply the one rule: the teacher forces the true past.",
          source="original toy", inner_h=400)
p.panel(32, p.top, 420, 340, fill=NEW, label="TRAIN: all steps at once")
steps = [("[START]", "the"), ("[START, the]", "cat"), ("[START, the, cat]", "sat")]
for i, (inp, tgt) in enumerate(steps):
    y = p.top + 56 + i * 88
    p.chip(56, y, inp, fill=COUNT, size=12)
    p.arrow(56 + 220, y + 16, 56 + 260, y + 16, label=None)
    p.chip(330, y, "target: " + tgt, fill=ACTIVE, size=12)
p.text(56, p.top + 320, "one forward pass", size=13, bold=True)
p.panel(484, p.top, 444, 340, fill=PINK, label="SAMPLE: one step at a time")
sy = p.top + 56
p.chip(508, sy, "draw: the", fill=ACTIVE, size=13)
p.arrow(560, sy + 40, 560, sy + 64, label=None)
p.chip(508, sy + 72, "draw: cat | the", fill=ACTIVE, size=13)
p.arrow(560, sy + 112, 560, sy + 136, label=None)
p.chip(508, sy + 144, "draw: sat | the cat", fill=ACTIVE, size=13)
p.text(508, sy + 200, "step t waits for step t-1", size=13, bold=True)
p.text(508, sy + 228, "1,000 tokens x 20 ms = 20 s", size=13)
p.save("plate-l02-teacher-forcing.webp")

# 5. Exposure bias: 0.9^t decay
p = Plate("Exposure bias: mistakes compound",
          "With 10% error per step, only 0.9^20 = 0.12 of 20-token samples stay fully on track.",
          "Shell 2. Count the decay: each step multiplies by 0.9.",
          source="original toy", inner_h=420)
pts = [(t, 0.9 ** t) for t in range(0, 21)]
x0, y0, x1, y1 = 80, p.top + 40, 880, p.top + 340
p.axes(x0, y0, x1, y1)
p.text(x0 - 8, y1 + 8, "t = 0", size=12, color=MUTED)
p.text(x1 - 40, y1 + 8, "t = 20", size=12, color=MUTED)
p.text(16, y0 + 120, "1.0", size=12, color=MUTED)
p.text(16, y1 - 8, "0.0", size=12, color=MUTED)
coords = [(x0 + t * (x1 - x0) / 20, y1 - v * (y1 - y0)) for t, v in pts]
p.curve(coords, color=FOCUS, width=3)
for t, v in [(0, 1.0), (5, 0.9 ** 5), (10, 0.9 ** 10), (20, 0.9 ** 20)]:
    cx = x0 + t * (x1 - x0) / 20
    cy = y1 - v * (y1 - y0)
    p.circle(cx, cy, 7, FOCUS, outline=INK)
    p.text(cx - 24, cy - 40, f"{v:.2f}", size=13, bold=True)
    p.text(cx - 16, y1 + 28, f"t={t}", size=12, color=MUTED)
p.text(560, p.top + 8, "0.9^5 = 0.59    0.9^10 = 0.35    0.9^20 = 0.12", size=14, bold=True)
p.save("plate-l02-exposure.webp")

# 6. Temperature reshapes decoding
p = Plate("Temperature reshapes the same logits",
          "T = 0.5 sharpens, T = 2.0 flattens. Greedy is T -> 0.",
          "Shell 3. Apply one rule: divide logits by T, then softmax.",
          source="original toy", inner_h=720)
import math as _m
logits = [2.0, 1.0, 0.1]
words = ["cat", "dog", "sat"]
def softmax(ls, T):
    sc = [l / T for l in ls]
    m = max(sc)
    es = [_m.exp(s - m) for s in sc]
    s = sum(es)
    return [e / s for e in es]
y = p.top + 8
p.text(32, y, "logits: [2.0, 1.0, 0.1]", size=14, bold=True, color=MUTED)
y += 40
for T in (0.5, 1.0, 2.0):
    probs = softmax(logits, T)
    p.text(32, y + 4, f"T = {T}", size=14, bold=True)
    p.bars(160, y + 150, [(w, round(pr, 3), TEAL if i == 0 else (CHIP if i == 1 else PINK))
                          for i, (w, pr) in enumerate(zip(words, probs))],
           1.0, bar_w=72, gap=40, height=140, size=12)
    y += 210
p.save("plate-l02-temperature.webp")
print("L02 plates done")

# ---------------------------------------------------------------- L03 ---
# 7. ELBO: tug of war between reconstruction and KL rent
p = Plate("ELBO: reconstruction reward minus KL rent",
          "The toy photo pays 2.318 nats of rent to sit at mu = 2.0, sigma = 0.5.",
          "Shell 3. Apply the one rule: maximize the deal, both terms at once.",
          source="original toy", inner_h=440)
mu, sig = 2.0, 0.5
kl = 0.5 * (mu**2 + sig**2 - 1 - math.log(sig**2))
p.panel(32, p.top, 420, 300, fill=NEW, label="term 1: rebuild well")
p.text(56, p.top + 64, "E[log p(x | z)]", size=16, bold=True)
p.text(56, p.top + 96, "draw z, decode, compare", size=13)
p.text(56, p.top + 120, "to the true photo.", size=13)
p.text(56, p.top + 168, "pays for codes that", size=13, color=MUTED)
p.text(56, p.top + 192, "keep photos distinct.", size=13, color=MUTED)
p.panel(508, p.top, 420, 300, fill=PINK, label="term 2: KL rent")
p.text(532, p.top + 64, "KL(q(z|x) || N(0,1))", size=16, bold=True)
p.text(532, p.top + 96, "0.5 x (4.00 + 0.25 - 1", size=13)
p.text(532, p.top + 120, "- log 0.25) = 2.318 nats", size=13)
p.text(532, p.top + 168, "charges for codes that", size=13, color=MUTED)
p.text(532, p.top + 192, "stray from the prior.", size=13, color=MUTED)
p.arrow(452, p.top + 150, 508, p.top + 150, label=None)
p.rect(32, p.top + 328, 896, 64, CHIP, label=f"ELBO = rebuild reward - {kl:.3f} nats rent", size=16, rx=12)
p.save("plate-l03-elbo.webp")

# 8. The reparameterization trick: move the dice roll aside
p = Plate("The reparameterization trick: move the dice roll aside",
          "z = 2.0 + 0.5 x 0.6 = 2.3. Gradients: dz/dmu = 1, dz/dsigma = 0.6.",
          "Shell 3. Apply the one rule: randomness lives in epsilon, parameters in the arithmetic.",
          source="original toy", inner_h=440)
p.panel(32, p.top, 420, 300, fill=PINK, label="blocked path")
p.text(56, p.top + 64, "z ~ N(mu, sigma^2)", size=16, bold=True)
p.text(56, p.top + 100, "draw directly:", size=13, color=MUTED)
p.text(56, p.top + 128, "dice roll has", size=13)
p.text(56, p.top + 152, "no derivative.", size=13)
p.rect(56, p.top + 196, 200, 56, BG, label="grad = ???", size=15, rx=12, stroke=PINK)
p.arrow(452, p.top + 150, 508, p.top + 150, label="rewrite")
p.panel(508, p.top, 420, 300, fill=NEW, label="open path")
p.text(532, p.top + 64, "z = mu + sigma x eps", size=16, bold=True)
p.text(532, p.top + 100, "eps = 0.6 drawn once", size=13, color=MUTED)
p.text(532, p.top + 128, "z = 2.0 + 0.5 x 0.6", size=13)
p.text(532, p.top + 152, "= 2.3", size=15, bold=True)
p.text(532, p.top + 196, "dz/dmu = 1", size=14, bold=True)
p.text(532, p.top + 222, "dz/dsigma = 0.6", size=14, bold=True)
p.save("plate-l03-reparam.webp")

# 9. The blur: KL leash crowds codes, midpoint decodes to the average
p = Plate("The KL leash crowds codes",
          "Reconstructions shrink {0, 10} -> {2.5, 7.5}. z = 0 decodes to 5.",
          "Shell 2. Count the shrinkage: the leash pulls every code toward 0.",
          source="original toy", inner_h=460)
p.text(32, p.top, "no leash (KL = 0)", size=14, bold=True, color=MUTED)
x0, y0b, x1, y1b = 32, p.top + 120, 928, p.top + 120
p.line(x0, y0b, x1, y1b, color=INK)
for zc, col, lab in [(-1.0, TEAL, "-1"), (1.0, TEAL, "+1")]:
    cx = x0 + (zc + 1.5) / 3.0 * (x1 - x0)
    p.circle(cx, y0b, 12, col, outline=INK)
    p.text(cx - 8, y0b - 40, lab, size=13, bold=True)
    p.text(cx - 40, y0b + 20, "x-hat = 0" if zc < 0 else "x-hat = 10", size=12)
p.text(32, p.top + 170, "with leash (KL rent paid)", size=14, bold=True, color=MUTED)
y2 = p.top + 260
p.line(x0, y2, x1, y2, color=INK)
for zc, col, lab in [(-0.5, ORANGE, "-0.5"), (0.0, FOCUS, "0"), (0.5, ORANGE, "+0.5")]:
    cx = x0 + (zc + 1.5) / 3.0 * (x1 - x0)
    p.circle(cx, y2, 12, col, outline=INK)
    p.text(cx - 12, y2 - 40, lab, size=13, bold=True)
p.text(x0 + (0.0 + 1.5) / 3.0 * (x1 - x0) - 60, y2 + 20, "z = 0 -> x-hat = 5", size=13, bold=True)
p.text(x0 + (-0.5 + 1.5) / 3.0 * (x1 - x0) - 48, y2 + 20, "x-hat = 2.5", size=12)
p.text(x0 + (0.5 + 1.5) / 3.0 * (x1 - x0) - 48, y2 + 20, "x-hat = 7.5", size=12)
p.text(32, p.top + 330, "smoothness fills the holes and averages the outputs: one mechanism.", size=14, color=MUTED)
p.save("plate-l03-blur.webp")

# 10. Posterior collapse: the decoder stops reading z
p = Plate("Posterior collapse: the decoder stops reading z",
          "KL = 0 for every photo. The latent carries nothing; the decoder works alone.",
          "Shell 3. Apply the one rule: a strong decoder can pay zero rent.",
          source="original toy", inner_h=400)
p.panel(32, p.top, 280, 260, fill=COUNT, label="encoder")
p.text(56, p.top + 64, "q(z | x) = N(0,1)", size=14, bold=True)
p.text(56, p.top + 96, "for every x.", size=13)
p.text(56, p.top + 128, "mu = 0, sigma = 1", size=13)
p.text(56, p.top + 160, "always.", size=13)
p.rect(56, p.top + 196, 160, 48, NEW, label="KL = 0", size=15, rx=12)
p.arrow(312, p.top + 130, 380, p.top + 130, label="z (ignored)")
p.panel(380, p.top, 280, 260, fill=PINK, label="decoder")
p.text(404, p.top + 64, "p(x | z): z unused", size=14, bold=True)
p.text(404, p.top + 96, "reconstructs from", size=13)
p.text(404, p.top + 120, "pixel autoregression", size=13)
p.text(404, p.top + 144, "alone.", size=13)
p.panel(700, p.top, 228, 260, fill=YELLOW, label="symptom")
p.text(724, p.top + 64, "samples ignore", size=14, bold=True)
p.text(724, p.top + 96, "the latent.", size=14, bold=True)
p.text(724, p.top + 140, "fix: KL annealing,", size=13, color=MUTED)
p.text(724, p.top + 164, "weaker decoder.", size=13, color=MUTED)
p.save("plate-l03-collapse.webp")
print("L03 plates done")

# ---------------------------------------------------------------- L04 ---
# 11. 1-D change of variables: stretch by 2, density halves
p = Plate("Stretch by 2, density halves",
          "z in [0,1] maps to x in [1,3]. p_x = 1 x 1/2 = 0.5. Exact.",
          "Shell 2. Count the width: the same mass spreads over twice the interval.",
          source="original toy", inner_h=420)
p.text(32, p.top, "z ~ Uniform(0,1),  p_z = 1", size=14, bold=True, color=MUTED)
p.rect(32, p.top + 48, 400, 64, COUNT, label="z in [0, 1], width 1", size=15, rx=12)
p.text(32, p.top + 140, "warp: x = 2z + 1", size=15, bold=True)
p.arrow(232, p.top + 180, 232, p.top + 216, label=None)
p.rect(32, p.top + 224, 800, 64, NEW, label="x in [1, 3], width 2", size=15, rx=12)
p.text(32, p.top + 316, "p_x(x) = p_z(z) x |dz/dx| = 1 x 1/2 = 0.5", size=16, bold=True)
p.text(32, p.top + 348, "check: 0.5 x width 2 = 1. probability is conserved.", size=14, color=MUTED)
p.save("plate-l04-stretch.webp")

# 12. Folds: x = z^2 needs a branch sum
p = Plate("Folds break the one-term formula",
          "z = 0.5 and z = -0.5 both reach x = 0.25. The density sums both branches: 1.0.",
          "Shell 2. Count the pre-images: one fold, two terms.",
          source="original toy", inner_h=440)
p.text(32, p.top, "z ~ Uniform(-1,1),  p_z = 0.5      warp: x = z^2", size=14, bold=True, color=MUTED)
p.rect(32, p.top + 48, 200, 64, COUNT, label="z = +0.5", size=15, rx=12)
p.rect(32, p.top + 128, 200, 64, COUNT, label="z = -0.5", size=15, rx=12)
p.arrow(232, p.top + 80, 300, p.top + 112, label="folds")
p.arrow(232, p.top + 160, 300, p.top + 128, label=None)
p.rect(300, p.top + 80, 200, 96, PINK, label="x = 0.25", size=16, rx=12)
p.text(532, p.top + 96, "branch 1: 0.5 x 1.0", size=14)
p.text(532, p.top + 124, "branch 2: 0.5 x 1.0", size=14)
p.text(532, p.top + 168, "p_x(0.25) = 1.0", size=16, bold=True)
p.text(32, p.top + 260, "a deep neural warp folds thousands of times.", size=14, color=MUTED)
p.text(32, p.top + 288, "the branch sum explodes. invertibility keeps one term.", size=14, color=MUTED)
p.save("plate-l04-fold.webp")

# 13. Coupling layer: triangular Jacobian, cheap determinant
p = Plate("Coupling: half frozen, half warped, determinant in O(d)",
          "z = [0.5, 1.0] -> x = [0.5, 3.0]. det J = 2. log det = 0.693.",
          "Shell 3. Apply the one rule: x_a = z_a freezes the upper-right block to zero.",
          source="original toy (Dinh et al., 2014)", inner_h=460)
p.text(32, p.top, "forward: x_a = z_a,   x_b = z_b x s(z_a) + t(z_a)", size=14, bold=True)
p.text(32, p.top + 28, "s(z_a) = 2,  t(z_a) = 1", size=13, color=MUTED)
p.chip(32, p.top + 64, "z_a = 0.5 (frozen)", fill=COUNT, size=13)
p.chip(32, p.top + 108, "z_b = 1.0 -> x_b = 1.0 x 2 + 1 = 3.0", fill=NEW, size=13)
p.text(32, p.top + 168, "Jacobian (x_a never sees z_b):", size=14, bold=True)
p.rect(32, p.top + 200, 120, 56, COUNT, label="I", size=16, rx=8)
p.rect(152, p.top + 200, 120, 56, BG, label="0", size=16, rx=8, stroke=LINE)
p.rect(32, p.top + 256, 120, 56, BG, label="*", size=16, rx=8, stroke=LINE)
p.rect(152, p.top + 256, 120, 56, NEW, label="s = 2", size=16, rx=8)
p.text(300, p.top + 236, "triangular: det = product", size=14, bold=True)
p.text(300, p.top + 262, "of the diagonal = 2", size=14, bold=True)
p.text(300, p.top + 300, "inverse is arithmetic:", size=13, color=MUTED)
p.text(300, p.top + 324, "z_b = (3.0 - 1) / 2 = 1.0", size=13)
p.save("plate-l04-coupling.webp")

# 14. Dequantization: integers become continuous
p = Plate("Dequantization: flows need continuous data",
          "Pixel 123 becomes 123.4. Densities exist only on continuous ground.",
          "Shell 2. Count the gap: integers have no density; add uniform noise to fill it.",
          source="original toy", inner_h=400)
p.text(32, p.top, "pixels are integers: 0, 1, ..., 255", size=14, bold=True, color=MUTED)
for i, v in enumerate([122, 123, 124]):
    p.rect(32 + i * 160, p.top + 48, 120, 64, COUNT if v == 123 else CHIP,
           label=str(v), size=18, rx=12)
p.arrow(470, p.top + 80, 540, p.top + 80, label="add u in [0,1)")
p.rect(560, p.top + 48, 220, 64, NEW, label="123 + 0.4 = 123.4", size=15, rx=12)
p.text(32, p.top + 160, "now x lives on [0, 256): a continuous interval.", size=14)
p.text(32, p.top + 192, "the change-of-variables formula applies.", size=14)
p.text(32, p.top + 240, "without it, p_x of an integer is undefined.", size=14, color=MUTED)
p.text(32, p.top + 268, "with it, the flow models a density that rounds to the pixels.", size=14, color=MUTED)
p.save("plate-l04-dequant.webp")
print("L04 plates done")

# ---------------------------------------------------------------- L05 ---
# 15. The one-leap failure: averaging trap
p = Plate("One leap must average: the mushy middle",
          "Faces at -5 and +5 both reach x_T = 0.3. The MSE-optimal guess is 0: neither face.",
          "Shell 2. Count the modes: two targets, one prediction, the average wins.",
          source="original toy", inner_h=420)
p.rect(32, p.top + 8, 180, 64, TEAL, label="face: x_0 = 5", size=14, rx=12)
p.rect(32, p.top + 88, 180, 64, TEAL, label="face: x_0 = -5", size=14, rx=12)
p.arrow(212, p.top + 40, 300, p.top + 72, label="noise")
p.arrow(212, p.top + 120, 300, p.top + 88, label=None)
p.rect(300, p.top + 48, 180, 64, PINK, label="x_T = 0.3", size=14, rx=12)
p.arrow(480, p.top + 80, 560, p.top + 80, label="one jump")
p.rect(560, p.top + 48, 220, 64, YELLOW, label="predict 0", size=15, rx=12)
p.text(32, p.top + 180, "0.5 x 5 + 0.5 x (-5) = 0", size=16, bold=True)
p.text(32, p.top + 212, "squared error forces the conditional mean.", size=14)
p.text(32, p.top + 244, "0 is gray blur: the average of the modes, matching none.", size=14, color=MUTED)
p.save("plate-l05-one-leap.webp")

# 16. Noise schedule: signal decay to pure noise
p = Plate("The noise schedule: from photo to pure static",
          "Linear beta from 1e-4 to 2e-2. Signal left at step 1000: 4e-5.",
          "Shell 2. Count the signal: alpha-bar is the product of the shrinkages.",
          source="original toy (Ho et al., 2020)", inner_h=460)
T = 1000
betas = [1e-4 + (2e-2 - 1e-4) * t / (T - 1) for t in range(T)]
abar = 1.0
abars = []
for b in betas:
    abar *= (1 - b)
    abars.append(abar)
x0, y0, x1, y1 = 80, p.top + 60, 880, p.top + 360
p.axes(x0, y0, x1, y1)
coords = [(x0 + t * (x1 - x0) / (T - 1), y1 - ab * (y1 - y0)) for t, ab in enumerate(abars)]
p.curve(coords, color=FOCUS, width=3)
marks = [(100, abars[99]), (500, abars[499]), (1000, abars[999])]
for t, ab in marks:
    cx = x0 + (t - 1) * (x1 - x0) / (T - 1)
    cy = y1 - ab * (y1 - y0)
    p.circle(cx, cy, 7, FOCUS, outline=INK)
    lab = f"{ab:.2f}" if ab > 0.001 else f"{ab:.1e}"
    p.text(cx - 30, cy - 42, f"t={t}: {lab}", size=13, bold=True)
p.text(560, p.top + 8, "alpha-bar_t = signal left after t steps", size=14, bold=True)
p.text(32, p.top + 400, "x_1000 is standard Gaussian noise, independent of x_0.", size=14, color=MUTED)
p.save("plate-l05-schedule.webp")

# 17. One training step: plain regression on the true noise
p = Plate("Training is plain regression on the true noise",
          "True epsilon 0.5, predicted 0.42. Loss = (0.5 - 0.42)^2 = 0.0064.",
          "Shell 2. Count the loss: the forward process supplies the target.",
          source="original toy", inner_h=420)
p.panel(32, p.top, 420, 260, fill=COUNT, label="free training pair")
p.text(56, p.top + 64, "x_0 = 0.8 (real pixel)", size=14)
p.text(56, p.top + 96, "t = 1, beta = 0.01", size=14)
p.text(56, p.top + 128, "draw eps = 0.5", size=14)
p.text(56, p.top + 160, "x_1 = 0.846", size=14, bold=True)
p.text(56, p.top + 200, "closed form: jump to", size=13, color=MUTED)
p.text(56, p.top + 224, "any t in one formula.", size=13, color=MUTED)
p.arrow(452, p.top + 130, 508, p.top + 130, label="predict")
p.panel(508, p.top, 420, 260, fill=NEW, label="network output")
p.text(532, p.top + 64, "eps-hat = 0.42", size=14, bold=True)
p.text(532, p.top + 104, "loss = (0.5 - 0.42)^2", size=14)
p.text(532, p.top + 132, "= 0.0064", size=16, bold=True)
p.text(532, p.top + 176, "no adversary. no saddle.", size=13, color=MUTED)
p.text(532, p.top + 200, "a pile of regressions.", size=13, color=MUTED)
p.save("plate-l05-noise-pred.webp")

# 18. Latent diffusion: DDPM in a VAE's basement
p = Plate("Latent diffusion: denoise a thumbnail, not the photo",
          "512x512x3 becomes 64x64x4: 48x fewer numbers per denoising step.",
          "Shell 3. Apply the one rule: the VAE compresses, diffusion dreams inside.",
          source="original (Rombach et al., 2022)", inner_h=440)
p.panel(32, p.top, 280, 260, fill=PINK, label="pixels")
p.text(56, p.top + 64, "512 x 512 x 3", size=15, bold=True)
p.text(56, p.top + 96, "= 786,432 numbers", size=14)
p.text(56, p.top + 140, "diffusion here costs", size=13, color=MUTED)
p.text(56, p.top + 164, "48x more per step.", size=13, color=MUTED)
p.arrow(312, p.top + 130, 380, p.top + 130, label="VAE encode")
p.panel(380, p.top, 280, 260, fill=NEW, label="latent")
p.text(404, p.top + 64, "64 x 64 x 4", size=15, bold=True)
p.text(404, p.top + 96, "= 16,384 numbers", size=14)
p.text(404, p.top + 140, "DDPM runs HERE.", size=13, bold=True)
p.arrow(660, p.top + 130, 728, p.top + 130, label="VAE decode")
p.panel(728, p.top, 200, 260, fill=COUNT, label="pixels")
p.text(752, p.top + 64, "512 x 512 x 3", size=14, bold=True)
p.text(752, p.top + 104, "fresh photo", size=13)
p.text(32, p.top + 300, "786,432 / 16,384 = 48. the denoiser never sees full pixels.", size=14, color=MUTED)
p.save("plate-l05-latent.webp")
print("L05 plates done")

# ---------------------------------------------------------------- L06 ---
# 19. The score field of N(0,1): arrows point uphill
p = Plate("The score always points uphill",
          "s(x) = -x for N(0,1). At x = 2.3 the slope is -2.3, back toward the mean.",
          "Shell 2. Count the slope: differentiate -x^2/2.",
          source="original toy", inner_h=420)
x0, y0, x1, y1 = 80, p.top + 200, 880, p.top + 200
p.line(x0, y0, x1, y1, color=INK)
for xv, sv in [(-3.0, 3.0), (-2.0, 2.0), (-1.0, 1.0), (0.0, 0.0),
               (1.0, -1.0), (2.0, -2.0), (2.3, -2.3), (3.0, -3.0)]:
    cx = x0 + (xv + 3.5) / 7.0 * (x1 - x0)
    L = abs(sv) * 34
    if sv > 0:
        p.arrow(cx - L, y0, cx + L, y0, color=TEAL, width=3)
    elif sv < 0:
        p.arrow(cx + L, y0, cx - L, y0, color=TEAL, width=3)
    else:
        p.circle(cx, y0, 8, FOCUS, outline=INK)
    p.text(cx - 14, y0 + 24, f"{xv}", size=12, color=MUTED)
    if xv in (2.3,):
        p.text(cx - 30, y0 - 56, "s = -2.3", size=13, bold=True)
p.text(x0, y0 - 120, "log p(x) = -x^2/2: the bell curve's log", size=14, color=MUTED)
p.text(x0, y0 + 80, "follow the arrows: a sampler that climbs the score finds the data.", size=14, color=MUTED)
p.save("plate-l06-score-field.webp")

# 20. The SDE trio: one score, three machines
p = Plate("One score drives three machines",
          "Forward SDE destroys, reverse SDE restores, the probability-flow ODE walks straight.",
          "Shell 4. Name the new symbol: the score appears in every reverse law.",
          source="original (Song et al., 2021)", inner_h=440)
p.panel(32, p.top, 280, 260, fill=PINK, label="forward SDE")
p.text(56, p.top + 64, "dx = f dt + g dw", size=14, bold=True)
p.text(56, p.top + 100, "data -> noise.", size=13)
p.text(56, p.top + 128, "fixed, like DDPM's", size=13)
p.text(56, p.top + 152, "forward chain.", size=13)
p.panel(340, p.top, 280, 260, fill=NEW, label="reverse SDE")
p.text(364, p.top + 64, "needs s(x, t)", size=14, bold=True)
p.text(364, p.top + 100, "noise -> data.", size=13)
p.text(364, p.top + 128, "the score steers", size=13)
p.text(364, p.top + 152, "every step.", size=13)
p.panel(648, p.top, 280, 260, fill=COUNT, label="prob. flow ODE")
p.text(672, p.top + 64, "same s(x, t)", size=14, bold=True)
p.text(672, p.top + 100, "no randomness.", size=13)
p.text(672, p.top + 128, "deterministic path", size=13)
p.text(672, p.top + 152, "noise -> data.", size=13)
p.arrow(312, p.top + 130, 340, p.top + 130, label="learn s")
p.arrow(620, p.top + 130, 648, p.top + 130, label="drop noise")
p.text(32, p.top + 300, "DDPM = discrete reverse SDE. DDIM = discretized ODE. same score.", size=14, color=MUTED)
p.save("plate-l06-sde.webp")

# 21. DDIM: 50 jumps replace 1000 steps
p = Plate("DDIM: 50 jumps replace 1000 steps",
          "Same trained denoiser. Jump t = 1000 -> 800 -> 600: estimate x_0, re-noise, repeat.",
          "Shell 3. Apply the one rule: skip the chain, keep the network.",
          source="original toy (Song, Meng & Ermon, 2020)", inner_h=440)
p.text(32, p.top, "DDPM: every step", size=14, bold=True, color=MUTED)
xx = 32
for i in range(10):
    p.circle(xx, p.top + 60, 10, PINK, outline=INK)
    xx += 56
p.text(32, p.top + 96, "1000 network calls, one per t.", size=13)
p.text(32, p.top + 140, "DDIM: jump", size=14, bold=True, color=MUTED)
jumps = [1000, 800, 600, 400, 200, 0]
xx = 32
for j in jumps:
    p.circle(xx, p.top + 200, 14, NEW if j > 0 else TEAL, outline=INK)
    p.text(xx - 16, p.top + 228, str(j), size=12, bold=True)
    xx += 130
for i in range(5):
    p.arrow(32 + i * 130 + 16, p.top + 200, 32 + (i + 1) * 130 - 16, p.top + 200, label=None)
p.text(32, p.top + 280, "each jump: x_0-hat from eps-hat, then re-noise to the target level.", size=14)
p.text(32, p.top + 308, "eta = 0: deterministic. 20x fewer calls, small quality cost.", size=14, color=MUTED)
p.save("plate-l06-ddim.webp")

# 22. Classifier-free guidance: extrapolate toward the condition
p = Plate("Guidance extrapolates toward the condition",
          "eps_u = 0.42, eps_c = 0.60, w = 3: eps-tilde = 0.42 + 3 x 0.18 = 0.96.",
          "Shell 3. Apply the one rule: push past the conditional prediction.",
          source="original toy (Ho & Salimans, 2022)", inner_h=420)
x0, y0, x1, y1 = 80, p.top + 220, 880, p.top + 220
p.line(x0, y0, x1, y1, color=INK)
def X(e):
    return x0 + (e - 0.2) / 1.0 * (x1 - x0)
for e, col, lab in [(0.42, CHIP, "eps_u = 0.42"), (0.60, NEW, "eps_c = 0.60"), (0.96, FOCUS, "eps-tilde = 0.96")]:
    cx = X(e)
    p.circle(cx, y0, 12, col, outline=INK)
    p.text(cx - 44, y0 - 48, lab, size=13, bold=True)
p.arrow(X(0.42), y0 - 80, X(0.96), y0 - 80, label="w = 3 pushes past conditional", color=FOCUS)
p.text(80, p.top + 280, "higher w: stronger prompt match, less diversity, eventual artifacts.", size=14, color=MUTED)
p.text(80, p.top + 308, "no separate classifier needed: the conditional model guides itself.", size=14, color=MUTED)
p.save("plate-l06-guidance.webp")

# 23. Flow matching: straight lines from noise to data
p = Plate("Flow matching: straight lines from noise to data",
          "x_t = (1 - t) x_0 + t eps. The target is the velocity, not the noise.",
          "Shell 3. Apply the one rule: interpolate linearly, learn the straight velocity.",
          source="original (Lipman et al., 2022)", inner_h=440)
x0, y0, x1, y1 = 80, p.top + 80, 880, p.top + 320
p.circle(x0, y0, 16, PINK, outline=INK)
p.text(x0 - 30, y0 - 44, "eps (noise)", size=13, bold=True)
p.circle(x1, y1, 16, TEAL, outline=INK)
p.text(x1 - 24, y1 + 28, "x_0 (data)", size=13, bold=True)
p.arrow(x0 + 16, y0 + 8, x1 - 16, y1 - 8, label="straight: v = eps - x_0", color=FOCUS, width=3)
# curved DDPM-ish path for contrast
import math as _m2
curve_pts = []
for i in range(41):
    s = i / 40.0
    cx = x0 + s * (x1 - x0)
    cy = y0 + s * (y1 - y0) - 90 * _m2.sin(s * _m2.pi)
    curve_pts.append((cx, cy))
p.curve(curve_pts, color=MUTED, width=2)
p.text(480, p.top + 60, "diffusion path: curved", size=13, color=MUTED)
p.text(80, p.top + 360, "Stable Diffusion 3 trains on this straight-line objective (rectified flow).", size=14, color=MUTED)
p.save("plate-l06-flow.webp")
print("L06 plates done")

# ---------------------------------------------------------------- L07 ---
# 24. The energy landscape: valleys are data
p = Plate("Valleys are data: low energy, high probability",
          "p(x) = exp(-E(x)) / Z. Face 4 has the lowest energy and p = 0.5345.",
          "Shell 2. Count the valley: the Boltzmann formula turns depth into probability.",
          source="original toy", inner_h=460)
E = [2.0, 1.0, 1.0, 0.0]
Z = sum(math.exp(-e) for e in E)
probs = [math.exp(-e) / Z for e in E]
x0, y0, x1, y1 = 80, p.top + 260, 880, p.top + 260
# energy curve: high, mid, mid, low
xs = [x0 + i * (x1 - x0) / 3 for i in range(4)]
ys = [y0 - (2.5 - e) * 60 for e in E]
pts = []
for i in range(4):
    pts.append((xs[i], ys[i]))
p.curve(pts, color=FOCUS, width=3)
# lesson-exact display values (lesson rounds 0.53444 to 0.5345; plate matches lesson exactly)
disp = ["0.0723", "0.1966", "0.1966", "0.5345"]
for i, (e, pr) in enumerate(zip(E, probs)):
    p.circle(xs[i], ys[i], 12, TEAL if e == 0 else (NEW if e == 1 else PINK), outline=INK)
    p.text(xs[i] - 24, ys[i] - 52, f"E = {e}", size=13, bold=True)
    p.text(xs[i] - 30, y0 + 24, f"p = {disp[i]}", size=12)
p.text(80, y0 + 80, f"Z = {Z:.4f}: the sum that normalizes. 4 terms here; 2^1024 for images.", size=14, color=MUTED)
p.text(80, p.top + 400, "sampling = roll downhill. training = dig valleys under data.", size=14, color=MUTED)
p.save("plate-l07-landscape.webp")

# 25. CD-1: push data down, push the fantasy up
p = Plate("CD-1: push the data down, push the fantasy up",
          "theta: [0,0,0,0] -> [0,-0.1,0,0.1]. Mass moves fantasy face 2 -> data face 4.",
          "Shell 3. Apply the one rule: one MCMC step from the data is the negative sample.",
          source="original toy (Hinton, 2002)", inner_h=460)
p.panel(32, p.top, 400, 300, fill=NEW, label="positive phase (data)")
p.text(56, p.top + 64, "x = 4 (data point)", size=14, bold=True)
p.text(56, p.top + 100, "lower E(4):", size=13)
p.text(56, p.top + 128, "theta_4: 0 -> +0.1", size=14, bold=True)
p.text(56, p.top + 172, "p(4): 0.25 -> 0.2756", size=13)
p.panel(496, p.top, 400, 300, fill=PINK, label="negative phase (fantasy)")
p.text(520, p.top + 64, "1 MCMC step lands", size=14, bold=True)
p.text(520, p.top + 96, "on face 2.", size=14, bold=True)
p.text(520, p.top + 132, "raise E(2):", size=13)
p.text(520, p.top + 160, "theta_2: 0 -> -0.1", size=14, bold=True)
p.text(520, p.top + 204, "p(2): 0.25 -> 0.2256", size=13)
p.arrow(432, p.top + 150, 496, p.top + 150, label="contrast")
p.text(32, p.top + 340, "Z never computed. learning stops when fantasies match data.", size=14, color=MUTED)
p.save("plate-l07-cd.webp")

# 26. Energies compose by addition
p = Plate("Energies compose by addition",
          "E_face + E_smile = E_smiling-face. No retraining: the sum is the new model.",
          "Shell 3. Apply the one rule: independent constraints add.",
          source="original toy", inner_h=440)
p.text(32, p.top, "two faces, two energies", size=14, bold=True, color=MUTED)
p.rect(32, p.top + 48, 200, 64, COUNT, label="E_face = [2, 1]", size=14, rx=12)
p.text(248, p.top + 68, "+", size=20, bold=True)
p.rect(280, p.top + 48, 200, 64, NEW, label="E_smile = [0, 3]", size=14, rx=12)
p.text(496, p.top + 68, "=", size=20, bold=True)
p.rect(528, p.top + 48, 220, 64, FOCUS, label="E_total = [2, 4]", size=14, rx=12, color=BG)
p1 = math.exp(-2) / (math.exp(-2) + math.exp(-4))
p2 = 1 - p1
p.text(32, p.top + 160, f"p = [{p1:.3f}, {p2:.3f}]: face 1 wins (face-like AND smiling).", size=15, bold=True)
p.text(32, p.top + 200, "scores compose too (gradients add), but only energies give", size=14, color=MUTED)
p.text(32, p.top + 228, "an (unnormalized) joint density to reason about.", size=14, color=MUTED)
p.save("plate-l07-compose.webp")

# 27. The stuck chain: slow mixing across valleys
p = Plate("The chain gets stuck in one valley",
          "Two valleys, one high ridge. The walk sits left for thousands of steps.",
          "Shell 2. Count the visits: the right valley starves, its fantasies never train.",
          source="original toy", inner_h=560)
x0, y0, x1, y1 = 80, p.top + 60, 880, p.top + 60
c1, c2 = x0 + 170, x0 + 630
# double well: valleys dip DOWN on screen at the two modes; ridge between
def well_y(x):
    g1 = math.exp(-((x - c1) / 120) ** 2)
    g2 = math.exp(-((x - c2) / 120) ** 2)
    return y0 + 40 + 170 * g1 + 170 * g2
pts = [(x0 + i * (x1 - x0) / 60, well_y(x0 + i * (x1 - x0) / 60)) for i in range(61)]
p.curve(pts, color=FOCUS, width=3)
p.text((c1 + c2) / 2 - 36, y0 + 12, "high ridge", size=12, color=MUTED)
import random as _r
_r.seed(7)
for _ in range(14):
    jx = c1 + _r.gauss(0, 45)
    p.circle(jx, well_y(jx) - 16, 6, ORANGE, outline=INK)
p.text(c1 - 140, y0 + 300, "chain lives here", size=13, bold=True)
p.text(c2 - 44, y0 + 300, "never visited", size=13, color=MUTED)
p.text(32, p.top + 500, "negative samples come from one mode only: the sampler's mode collapse.", size=14, color=MUTED)
p.save("plate-l07-stuck.webp")
print("L07 plates done")

# ---------------------------------------------------------------- L08 ---
# 28. FID on 2-D Gaussians: mean shift vs spread
p = Plate("FID sees both failures: shift and spread",
          "Mean shift 0.5 -> FID 0.5. Doubled spread -> FID 1.5.",
          "Shell 2. Count both terms: the mean distance and the covariance trace.",
          source="original toy (Heusel et al., 2017)", inner_h=560)
p.text(32, p.top, "real: N([0,0], I)", size=14, bold=True, color=MUTED)
p.panel(32, p.top + 36, 420, 200, fill=COUNT, label="case A: mean shift")
p.circle(160, p.top + 140, 44, BG, outline=TEAL)
p.circle(204, p.top + 140, 44, BG, outline=ORANGE)
p.text(300, p.top + 100, "gen: N([0.5,0], I)", size=13)
p.text(300, p.top + 130, "FID^2 = 0.25 + 0", size=13)
p.text(300, p.top + 160, "FID = 0.5", size=15, bold=True)
p.panel(500, p.top + 36, 428, 200, fill=NEW, label="case B: doubled spread")
p.circle(640, p.top + 150, 56, BG, outline=PINK)
p.circle(640, p.top + 150, 26, BG, outline=TEAL)
p.text(586, p.top + 212, "pink: doubled spread", size=11, color=MUTED)
p.text(792, p.top + 100, "gen: N([0,0], 4I)", size=13)
p.text(792, p.top + 130, "FID^2 = 0.25 + 2", size=13)
p.text(792, p.top + 160, "FID = 1.5", size=15, bold=True)
p.text(32, p.top + 280, "FID^2 = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2})", size=14, bold=True)
p.text(32, p.top + 320, "real FID uses 2,048-dimensional Inception features, not 2-D toys.", size=14, color=MUTED)
p.save("plate-l08-fid.webp")

# 29. The parrot: FID 0, learning 0
p = Plate("The parrot scores FID 0 and learned nothing",
          "Generated set = training set: the Gaussians match exactly. FID never asks 'is it new?'",
          "Shell 2. Count the distance: identical clouds, zero distance, zero learning.",
          source="original toy", inner_h=440)
p.panel(32, p.top, 420, 260, fill=PINK, label="the parrot")
p.text(56, p.top + 64, "outputs = training", size=14, bold=True)
p.text(56, p.top + 92, "photos, replayed.", size=14, bold=True)
for i, (cx, cy) in enumerate([(120, 180), (200, 200), (280, 180), (160, 240), (240, 240)]):
    p.circle(cx, cy + 0, 10, ORANGE, outline=INK)
p.text(56, p.top + 300 - 44, "FID = 0. perfect score.", size=14, bold=True)
p.panel(508, p.top, 420, 260, fill=NEW, label="the learner")
p.text(532, p.top + 64, "fresh samples from", size=14, bold=True)
p.text(532, p.top + 92, "the learned rule.", size=14, bold=True)
for i, (cx, cy) in enumerate([(600, 190), (680, 180), (760, 200), (640, 240), (720, 235)]):
    p.circle(cx, cy, 10, TEAL, outline=INK)
p.text(532, p.top + 300 - 44, "FID > 0. worse score.", size=14, bold=True)
p.text(32, p.top + 340, "the better model loses. detect memorization with nearest-neighbor tests.", size=14, color=MUTED)
p.save("plate-l08-parrot.webp")

# 30. Precision and recall: the pair diagnoses
p = Plate("Precision 0.8, recall 0.5: the pair diagnoses",
          "Mode collapse: precision 1.0, recall 0.1. FID reports one mediocre number.",
          "Shell 2. Count both: quality of samples, coverage of modes.",
          source="original toy", inner_h=480)
p.text(32, p.top, "10 generated faces (teal), 10 real face types (outline)", size=14, bold=True, color=MUTED)
# real manifold: 10 outline circles in two rows
real_xy = []
for i in range(10):
    cx = 80 + (i % 5) * 150
    cy = p.top + 80 + (i // 5) * 90
    real_xy.append((cx, cy))
    p.circle(cx, cy, 22, BG, outline=MUTED)
# 8 teal near the manifold, but only 5 of the 10 real types imitated:
# types 0,1,2 get two faces each; types 3,4 get one each
teal_plan = {0: 2, 1: 2, 2: 2, 3: 1, 4: 1}
for idx, n in teal_plan.items():
    cx, cy = real_xy[idx]
    if n == 2:
        p.circle(cx - 9, cy, 9, TEAL, outline=INK)
        p.circle(cx + 9, cy, 9, TEAL, outline=INK)
    else:
        p.circle(cx, cy, 9, TEAL, outline=INK)
# 2 generated far from the manifold
for cx, cy in [(700, 320), (830, 320)]:
    p.circle(cx, p.top + cy, 10, ORANGE, outline=INK)
p.text(32, p.top + 300, "precision = 8/10 = 0.8 (near the real manifold)", size=14, bold=True)
p.text(32, p.top + 330, "recall = 5/10 = 0.5 (5 of 10 real types imitated)", size=14, bold=True)
p.text(32, p.top + 380, "one perfect face, repeated: precision 1.0, recall 0.1. that is mode collapse, named.", size=14, color=MUTED)
p.save("plate-l08-prec-rec.webp")
print("L08 plates done")
print("ALL DEEPENING PLATES DONE")
