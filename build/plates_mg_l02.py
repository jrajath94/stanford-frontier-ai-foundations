#!/usr/bin/env python3
"""L02 plates: KL coin toy, forward vs reverse, KL split identity, MLE toy."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. KL coin toy: per-outcome terms
p = Plate("KL = 0.51 nats, counted term by term",
          "Tails surprise dominates: the model admits tails at 0.1, truth visits it at 0.5.",
          "Shell 2. Count the toy: 0.5*log(0.5/0.9) + 0.5*log(0.5/0.1) = 0.511.",
          source="original toy", inner_h=400)
p.text(32, p.top, "truth {0.5, 0.5} vs model {0.9, 0.1}", size=14, bold=True, color=MUTED)
p.bars(64, p.top + 300, [("heads", -0.294, PINK), ("tails", 0.805, TEAL), ("total KL", 0.511, FOCUS)],
       1.0, bar_w=120, gap=64, height=200, size=14)
# Term legend moved right of the bars: the old under-bar position collided
# with the heads bar (hangs below baseline) and its value label.
p.text(600, p.top + 320, "heads term: 0.5 x log(0.556) = -0.294", size=14)
p.text(600, p.top + 348, "tails term: 0.5 x log(5.0) = +0.805", size=14)
p.text(600, p.top + 376, "match would score log(1) = 0 on both terms", size=13, color=MUTED)
p.save("l02-kl-toy.webp")

# 2. Forward vs reverse: mode-covering vs mode-seeking
def hump(p, x0, ybase, w, h, color):
    pts = []
    n = 40
    for i in range(n + 1):
        x = x0 + i * w / n
        y = ybase - h * math.exp(-((i / n - 0.5) ** 2) * 18)
        pts.append((x, y))
    p.curve(pts, color=color, width=3)

p = Plate("Forward KL covers modes; reverse KL seeks one",
          "Same truth, same model family. The direction chooses the personality.",
          "Shell 3. Weight by the truth and you visit both humps; weight by the model and one hump suffices.",
          source="original toy", inner_h=420)
p.panel(32, p.top, 430, 380, label="forward KL(P_X || P_theta): mode-covering", fill=COUNT)
p.text(56, p.top + 56, "truth: two humps", size=14, bold=True)
hump(p, 80, p.top + 200, 120, 110, TEAL)
hump(p, 220, p.top + 200, 120, 110, TEAL)
p.text(56, p.top + 232, "model: one wide hump", size=14, bold=True)
hump(p, 130, p.top + 340, 220, 90, FOCUS)
p.text(56, p.top + 356, "stretches over both, fills the valley", size=13, color=MUTED)
p.panel(498, p.top, 430, 380, label="reverse KL(P_theta || P_X): mode-seeking", fill=PINK)
p.text(522, p.top + 56, "truth: two humps", size=14, bold=True)
hump(p, 546, p.top + 200, 120, 110, TEAL)
hump(p, 686, p.top + 200, 120, 110, TEAL)
p.text(522, p.top + 232, "model: one narrow hump", size=14, bold=True)
hump(p, 546, p.top + 340, 120, 130, ORANGE)
p.text(522, p.top + 356, "sits on one hump, ignores the other", size=13, color=MUTED)
p.save("l02-forward-reverse.webp")

# 3. KL split identity with numbers
p = Plate("KL splits: entropy minus expected log-likelihood",
          "Entropy has no theta. Dropping it turns min-KL into max-likelihood.",
          "Shell 2. 1.204 - 0.693 = 0.511 on the coin toy.",
          source="original toy", inner_h=360)
y = p.top + 40
p.rect(64, y, 240, 72, COUNT, label="cross-entropy 1.204", size=15, rx=12)
p.text(320, y + 28, "  =  ", size=22, bold=True)
p.rect(368, y, 240, 72, PANEL, label="entropy 0.693", size=15, rx=12)
p.text(624, y + 28, "  +  ", size=22, bold=True)
p.rect(672, y, 224, 72, FOCUS, label="KL 0.511", color=BG, size=15, rx=12)
p.text(64, y + 120, "-sum P_X log P_theta", size=14, color=MUTED)
p.text(368, y + 120, "-sum P_X log P_X", size=14, color=MUTED)
p.text(672, y + 120, "the distance", size=14, color=MUTED)
p.text(64, y + 176, "constant in theta: drop it", size=14, bold=True, color=TEAL)
p.arrow(64, y + 220, 560, y + 220, label="minimize over theta")
p.text(64, y + 248, "min KL  =  max (1/n) sum log P_theta(x_i)", size=16, bold=True)
p.text(64, y + 280, "the MLE objective: a sample average, no P_X formula needed", size=13, color=MUTED)
p.save("l02-kl-split.webp")

# 4. MLE toy: H, H, T
# 2*ln(0.7)+ln(0.3) = -1.9173227 -> -1.917 (precise intermediates, not rounded).
p = Plate("MLE listens to the data: H, H, T picks the 0.7 coin",
          "Less negative wins. Two heads out of three favor the heads-biased model.",
          "Shell 2. -1.917 beats -2.079.",
          source="original toy", inner_h=400)
p.bars(64, p.top + 280, [("fair 0.5", 2.079, TEAL, "-2.079"), ("biased 0.7", 1.917, FOCUS, "-1.917")],
       2.2, bar_w=140, gap=80, height=180, size=14)
p.text(64, p.top + 320, "log-likelihood = sum of log P(x_i), less negative is better", size=14)
p.text(64, p.top + 348, "fair: 3 x log 0.5 = -2.079", size=14)
p.text(64, p.top + 376, "biased: 2 x log 0.7 + log 0.3 = -1.917", size=14)
p.chip(560, p.top + 320, "winner: biased 0.7", fill=NEW)
p.save("l02-mle-toy.webp")
