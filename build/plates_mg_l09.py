#!/usr/bin/env python3
"""L09 plates: three moves, three faces, sampling step."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. Three algebraic moves
p = Plate("Three moves turn the ELBO monster into one MSE",
          "Condition on x_0. Reduce KL to squared error. Predict the noise.",
          "Shell 3. Each move is exact algebra until the last reweighting. Source: original.",
          source="original", inner_h=480)
y = p.top + 16
moves = [
    ("move 1", "q(x_t-1|x_t,x_0): tractable", "mean 3.944, var 0.0526"),
    ("move 2", "KL of Gaussians = squared error", "||mu_tilde - mu_theta||^2"),
    ("move 3", "predict eps, not the mean", "L = ||eps - eps_theta||^2"),
]
yy = y
for label, formula, note in moves:
    p.rect(32, yy, 150, 72, COUNT, label=label, size=14, rx=12)
    p.text(206, yy + 14, formula, size=15, bold=True)
    p.text(206, yy + 42, note, size=13, color=MUTED)
    yy += 96
# "L_simple" drawn as a header above the formula: the centered rect label
# stamped over the "toy: (0.688-0.5)^2" line.
p.rect(640, p.top + 16, 288, 220, NEW, size=18, rx=12)
p.text(664, p.top + 40, "L_simple", size=18, bold=True)
p.text(664, p.top + 88, "E ||eps - eps_theta||^2", size=15, bold=True)
p.text(664, p.top + 120, "toy: (0.688-0.5)^2", size=14)
p.text(664, p.top + 148, "= 0.0354", size=16, bold=True, color=TEAL)
p.text(664, p.top + 192, "one network, one MSE", size=13, color=MUTED)
p.text(32, yy + 32, "price of move 3: per-step weights dropped. Reweighted bound.", size=14, bold=True, color=ORANGE)
p.text(32, yy + 64, "slightly worse likelihood, much better samples. The field chose samples.", size=13, color=MUTED)
p.save("l09-three-moves.webp")

# 2. Three faces of one prediction
p = Plate("One prediction, three faces: noise, clean image, score",
          "x_t = sqrt(a_bar) x_0 + sqrt(1-a_bar) eps. Know one, know all three.",
          "Shell 3. Toy: eps = 0.688, score = -0.688/0.4359 = -1.578. Source: original toy.",
          source="original toy", inner_h=440)
p.rect(80, p.top + 60, 220, 120, TEAL, label="noise eps", color=BG, size=16, rx=12)
p.text(140, p.top + 140, "0.688", size=14, color=BG, bold=True)
p.rect(380, p.top + 60, 220, 120, FOCUS, label="clean x_0", color=BG, size=16, rx=12)
p.text(440, p.top + 140, "4.0", size=14, color=BG, bold=True)
p.rect(680, p.top + 60, 220, 120, ORANGE, label="score", color=BG, size=16, rx=12)
p.text(720, p.top + 140, "-1.578", size=14, color=BG, bold=True)
p.arrow(300, p.top + 120, 380, p.top + 120, label="solve")
p.arrow(600, p.top + 120, 680, p.top + 120, label="divide")
p.text(80, p.top + 240, "eps -> x_0: x_0 = (x_t - sqrt(1-a_bar) eps)/sqrt(a_bar)", size=14)
p.text(80, p.top + 272, "eps -> score: score = -eps/sqrt(1-a_bar) = -1.578", size=14)
p.text(80, p.top + 312, "same information, three costumes. Noise trains best:", size=14, bold=True)
p.text(80, p.top + 340, "fixed scale N(0,1) at every t. Friendlier regression.", size=14, color=TEAL)
p.save("l09-three-faces.webp")

# 3. Sampling step
p = Plate("One reverse step: from 3.9 back toward 4.0, plus a wobble",
          "Remove the predicted noise, then add controlled randomness for variety.",
          "Shell 2. x_1 = 3.990 + 0.2294 z. z = 0.3 gives 4.059. Source: original toy.",
          source="original toy", inner_h=440)
p.panel(32, p.top, 280, 300, label="input", fill=COUNT)
p.text(56, p.top + 80, "x_2 = 3.9", size=18, bold=True)
p.text(56, p.top + 130, "eps_theta = 0.5", size=14)
p.text(56, p.top + 160, "(network's guess)", size=13, color=MUTED)
p.arrow(328, p.top + 150, 392, p.top + 150, label="denoise")
p.panel(400, p.top, 280, 300, label="output", fill=NEW)
p.text(424, p.top + 80, "x_1 = 3.990", size=18, bold=True)
p.text(424, p.top + 130, "+ 0.2294 * z", size=14)
p.text(424, p.top + 160, "z = 0.3 -> 4.059", size=14, bold=True, color=TEAL)
p.text(424, p.top + 205, "toward the clean 4.0", size=13, color=MUTED)
p.text(424, p.top + 233, "wobble = variety", size=13, color=MUTED)
p.text(32, p.top + 340, "repeat T times: x_T (noise) -> ... -> x_0 (photo)", size=14, bold=True)
p.text(32, p.top + 368, "drop the z term and you get DDIM: deterministic, faster (Lesson 10)", size=13, color=FOCUS)
p.save("l09-sampling-step.webp")
