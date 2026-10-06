#!/usr/bin/env python3
"""L06 plates: Jensen derivation chain, two-coin toy, EM staircase."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. The four-step Jensen chain
p = Plate("Four steps from an intractable log to a trainable bound",
          "Introduce q, read the integral as an expectation, apply Jensen.",
          "Shell 3. Each step is legal; the last one buys computability. Source: original.",
          source="original", inner_h=480)
y = p.top + 16
steps = [
    ("step 1", "log p(x) = log int p(x,z) dz", "the wall: intractable"),
    ("step 2", "x q(z|x) / q(z|x) inside", "legal: q >= 0"),
    ("step 3", "= log E_q[ p(x,z)/q(z|x) ]", "expectation form"),
    ("step 4", ">= E_q[ log p(x,z)/q(z|x) ]", "Jensen: concave log"),
]
yy = y
for label, formula, note in steps:
    p.rect(32, yy, 150, 72, COUNT, label=label, size=14, rx=12)
    p.text(206, yy + 16, formula, size=15, bold=True)
    p.text(206, yy + 44, note, size=13, color=MUTED)
    yy += 96
p.rect(640, p.top + 16, 288, 200, NEW, label="ELBO", size=18, rx=12)
p.text(664, p.top + 80, "= E_q[log p(x|z)]", size=15)
p.text(664, p.top + 108, "- KL(q(z|x) || p(z))", size=15)
p.text(664, p.top + 152, "gap = KL(q || true posterior)", size=14, bold=True, color=FOCUS)
p.text(664, p.top + 180, "q = posterior -> gap 0", size=13, color=TEAL)
p.save("l06-jensen-chain.webp")

# 2. Two-coin toy numbers
p = Plate("Two coins: ELBO -1.031, truth -0.968, gap 0.063",
          "The gap equals KL(q || posterior) to the digit. Set q right and it vanishes.",
          "Shell 2. Count the toy: 0.063 = 0.3*log(0.3/0.158) + 0.7*log(0.7/0.842). Source: original toy.",
          source="original toy", inner_h=460)
p.bars(64, p.top + 300, [("ELBO", 1.031, TEAL), ("true loglik", 0.968, FOCUS)],
       1.2, bar_w=120, gap=80, height=200, size=14)
p.text(64, p.top + 336, "magnitudes shown; both are negative (log of probability < 1)", size=13, color=MUTED)
p.text(64, p.top + 368, "gap: 1.031 - 0.968 = 0.063", size=15, bold=True)
p.text(64, p.top + 396, "KL(q || posterior) = 0.3*log(0.3/0.158) + 0.7*log(0.7/0.842)", size=14)
p.text(64, p.top + 424, "= 0.192 - 0.130 = 0.063. Exact match.", size=14, bold=True, color=TEAL)
p.panel(560, p.top + 40, 368, 300, label="two-term split", fill=YELLOW)
p.text(584, p.top + 96, "reconstruction: -0.847", size=15)
p.text(584, p.top + 132, "0.3*log 0.1 + 0.7*log 0.8", size=13, color=MUTED)
p.text(584, p.top + 180, "regularization: 0.184", size=15)
p.text(584, p.top + 216, "KL(q || prior)", size=13, color=MUTED)
p.text(584, p.top + 264, "ELBO = -0.847 - 0.184", size=15, bold=True)
p.save("l06-two-coin.webp")

# 3. EM staircase
p = Plate("EM: close the gap, then push the floor up",
          "E-step sets q to the exact posterior. M-step maximizes theta. Likelihood never drops.",
          "Shell 3. Coordinate ascent on the ELBO. Source: original.",
          source="original", inner_h=440)
x0, y0, x1, y1 = 80, 400, 880, 120
p.axes(x0, y0, x1, y1)
p.text(24, y1 - 8, "objective", size=13, color=MUTED)
p.text(x1 - 80, y0 + 24, "iterations", size=13, color=MUTED)
# staircase: likelihood flat-ish rising, ELBO touching at E-steps
like = [(100, 300), (300, 280), (500, 262), (700, 250), (860, 244)]
elbo = [(100, 340), (300, 280), (300, 320), (500, 262), (500, 300), (700, 250), (700, 286), (860, 244)]
p.curve(like, color=FOCUS, width=3)
pts = []
for xx, yy2 in elbo:
    pts.append((xx, yy2))
p.curve(pts, color=TEAL, width=2)
for xx, yy2 in [(300, 280), (500, 262), (700, 250)]:
    p.circle(xx, yy2, 7, TEAL)
    p.text(xx - 30, yy2 - 32, "E-step", size=12, color=TEAL, bold=True)
p.text(560, 150, "true log-likelihood", size=14, color=FOCUS, bold=True)
p.text(640, 330, "ELBO", size=14, color=TEAL, bold=True)
p.text(80, y0 + 60, "E-step: q = exact posterior -> gap 0, ELBO touches likelihood", size=14)
p.text(80, y0 + 88, "M-step: maximize ELBO over theta -> floor rises", size=14)
p.text(80, y0 + 116, "likelihood never decreases: each step is safe", size=14, bold=True, color=FOCUS)
p.save("l06-em-staircase.webp")
