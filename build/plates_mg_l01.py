#!/usr/bin/env python3
"""L01 plates: recipe, memory-vs-gaussian, three approximations."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. The three-step recipe pipeline
p = Plate("The three-step recipe",
          "Samples in. New samples out. Three ingredients do the work.",
          "Shell 3. Apply the recipe: family, divergence, optimization.",
          source="original toy")
p.panel(32, p.top, 200, 240, label="1. family", fill=COUNT)
p.text(56, p.top + 64, "P_theta:", size=15, bold=True)
p.text(56, p.top + 92, "candidates", size=14)
p.text(56, p.top + 116, "with knobs theta", size=14)
p.chip(56, p.top + 152, "Gaussian(5, 2)", fill=NEW)
p.chip(56, p.top + 192, "neural net g_theta", fill=NEW)
p.arrow(248, p.top + 120, 312, p.top + 120, label="choose")
p.panel(320, p.top, 200, 240, label="2. divergence", fill=YELLOW)
p.text(344, p.top + 64, "D(P_X, P_theta):", size=15, bold=True)
p.text(344, p.top + 92, "one number for", size=14)
p.text(344, p.top + 116, "how far the", size=14)
p.text(344, p.top + 140, "model is", size=14)
p.chip(344, p.top + 176, "0 when they match", fill=ACTIVE)
p.arrow(536, p.top + 120, 600, p.top + 120, label="measure")
p.panel(608, p.top, 200, 240, label="3. optimize", fill=GREEN)
p.text(632, p.top + 64, "theta* = argmin", size=15, bold=True)
p.text(632, p.top + 92, "turn knobs to", size=14)
p.text(632, p.top + 116, "shrink D", size=14)
p.chip(632, p.top + 176, "gradient descent", fill=NEW)
p.arrow(824, p.top + 120, 896, p.top + 120, label="sample")
p.rect(824, p.top + 160, 112, 56, FOCUS, label="new x", color=BG, rx=12)
p.save("l01-recipe.webp")

# 2. Memory machine vs Gaussian on {2,4,6,8}
mu, sig = 5.0, 2.0
def gauss(x):
    return (1 / (sig * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sig * sig))
xs = [2, 3, 4, 5, 6, 7, 8]
mem = [0.25 if x in (2, 4, 6, 8) else 0.0 for x in xs]
gau = [round(gauss(x), 3) for x in xs]
p = Plate("Memory replays; a fitted family creates",
          "The memory machine gives probability 0 to 5. The Gaussian gives it 0.20.",
          "Shell 2. Count the toy: zeros on unseen outcomes vs smooth spread.",
          source="original toy", inner_h=780)
p.text(32, p.top, "memory machine P(x)", size=14, bold=True, color=MUTED)
p.bars(32, p.top + 300, [(str(x), v, PINK if v == 0 else TEAL) for x, v in zip(xs, mem)],
       0.25, bar_w=72, gap=32, height=220, size=13)
p.text(32, p.top + 336, "Gaussian(5, 2) P(x)", size=14, bold=True, color=MUTED)
p.bars(32, p.top + 620, [(str(x), v, FOCUS) for x, v in zip(xs, gau)],
       0.25, bar_w=72, gap=32, height=220, size=13)
p.save("l01-memory-vs-gaussian.webp")

# 3. The three approximations
p = Plate("Three gaps between P_theta* and P_X",
          "The trained model lands near the truth, never on it.",
          "Shell 3. Name each gap before trusting a sample.",
          source="original", inner_h=300)
p.panel(32, p.top, 280, 260, label="gap 1: wrong family", fill=PINK)
p.text(56, p.top + 64, "truth: two humps", size=14)
p.text(56, p.top + 92, "family: one Gaussian", size=14)
p.text(56, p.top + 124, "best fit still wrong", size=14, bold=True, color=ORANGE)
p.text(56, p.top + 172, "fix: richer family", size=13, color=MUTED)
p.panel(340, p.top, 280, 260, label="gap 2: finite samples", fill=YELLOW)
p.text(364, p.top + 64, "n = 4: noisy fit", size=14)
p.text(364, p.top + 92, "n = 10,000: tight fit", size=14)
p.text(364, p.top + 124, "law of large numbers", size=14, bold=True, color=FOCUS)
p.text(364, p.top + 172, "fix: more data", size=13, color=MUTED)
p.panel(648, p.top, 280, 260, label="gap 3: local minima", fill=ACTIVE)
p.text(672, p.top + 64, "loss has two valleys", size=14)
p.text(672, p.top + 92, "descent lands in one", size=14)
p.text(672, p.top + 124, "maybe the shallow one", size=14, bold=True, color=ORANGE)
p.text(672, p.top + 172, "fix: restarts, better opt", size=13, color=MUTED)
p.save("l01-three-gaps.webp")
