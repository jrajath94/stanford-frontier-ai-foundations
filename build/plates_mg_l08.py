#!/usr/bin/env python3
"""L08 plates: one step vs many, closed-form verification, schedule decay."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. One giant step vs many tiny steps
p = Plate("One giant leap is hopeless; a thousand small steps are easy",
          "Split the destruction and each reverse step becomes a small regression.",
          "Shell 3. Step size decides whether the reverse is learnable. Source: original.",
          source="original", inner_h=400)
p.panel(32, p.top, 430, 320, label="one step: destroy at once", fill=PINK)
p.text(56, p.top + 64, "x_0 -> x_1 ~ noise", size=15, bold=True)
p.text(56, p.top + 110, "reverse must rebuild the", size=14)
p.text(56, p.top + 138, "photo from nothing", size=14)
p.text(56, p.top + 184, "as hard as the original", size=14, bold=True, color=ORANGE)
p.text(56, p.top + 212, "generation problem", size=14, bold=True, color=ORANGE)
p.text(56, p.top + 258, "the fixed encoder bought nothing", size=13, color=MUTED)
p.arrow(478, p.top + 160, 542, p.top + 160, label="split")
p.panel(550, p.top, 378, 320, label="T steps: destroy slowly", fill=NEW)
p.text(574, p.top + 64, "x_0 -> x_1 -> ... -> x_T", size=15, bold=True)
p.text(574, p.top + 110, "each reverse step removes", size=14)
p.text(574, p.top + 138, "a little noise", size=14)
p.text(574, p.top + 184, "easy regression per step", size=14, bold=True, color=TEAL)
p.text(574, p.top + 230, "price: T network evals", size=13, color=MUTED)
p.save("l08-one-vs-many.webp")

# 2. Closed-form verification
p = Plate("The closed form agrees with chaining, to the digit",
          "x_2 = 0.9*4.0 + 0.4359*eps matches the chained 3.900 at eps = 0.688.",
          "Shell 2. Noise variance check: 0.3^2 = 0.09 = 1 - 0.81. Source: original toy.",
          source="original toy", inner_h=440)
p.panel(32, p.top, 440, 340, label="chained, step by step", fill=COUNT)
p.text(56, p.top + 64, "x_1 = 0.9487*4.0 + 0.3162*1.0", size=14)
p.text(56, p.top + 96, "    = 3.7947 + 0.3162 = 4.111", size=14, bold=True)
p.text(56, p.top + 140, "x_2 = 0.9487*4.111 + 0", size=14)
p.text(56, p.top + 172, "    = 3.900", size=14, bold=True)
p.text(56, p.top + 220, "noise added: 0.3 then 0", size=13, color=MUTED)
p.text(56, p.top + 248, "variance: 0.3^2 = 0.09", size=13, color=MUTED)
p.arrow(488, p.top + 170, 552, p.top + 170, label="equals")
p.panel(560, p.top, 368, 340, label="one jump, closed form", fill=NEW)
p.text(584, p.top + 64, "alpha_bar_2 = 0.81", size=14)
p.text(584, p.top + 100, "x_2 = 0.9*4.0 + 0.4359*eps", size=14)
p.text(584, p.top + 140, "3.900 = 3.6 + 0.4359*eps", size=14, bold=True)
p.text(584, p.top + 176, "eps = 0.688", size=14, bold=True, color=TEAL)
p.text(584, p.top + 220, "0.3/0.4359 = 0.688", size=13, color=MUTED)
p.text(584, p.top + 248, "1 - 0.81 = 0.09", size=13, color=MUTED)
p.text(32, p.top + 364, "training jumps straight to any t: no chaining, ever", size=14, bold=True, color=FOCUS)
p.save("l08-closed-form.webp")

# 3. Schedule decay: signal fraction over steps
p = Plate("The schedule decides how fast the signal dies",
          "alpha = 0.9: signal 0.9^t. At t = 100, only 0.003% remains: pure noise.",
          "Shell 2. alpha_bar_100 = 0.9^100 ~ 0.00003. Stationary: N(0,1). Source: original computation.",
          source="original computation", inner_h=440)
x0, y0, x1, y1 = 80, 400, 880, 120
p.axes(x0, y0, x1, y1)
p.text(24, y1 - 8, "signal frac", size=13, color=MUTED)
p.text(x1 - 40, y0 + 24, "t", size=13, color=MUTED)
def X(t): return x0 + t / 110 * (x1 - x0)
pts = [(X(t), y0 - (0.9 ** t) * (y0 - y1)) for t in range(0, 111)]
p.curve(pts, color=TEAL, width=3)
for t, lab in [(0, "t=0: 1.00"), (25, "t=25: 0.072"), (50, "t=50: 0.005"), (100, "t=100: 0.00003")]:
    xx, yy2 = X(t), y0 - (0.9 ** t) * (y0 - y1)
    p.circle(xx, yy2, 6, FOCUS)
    p.text(xx - 40, yy2 - 30, lab, size=12, color=INK)
p.text(80, y0 + 60, "too fast: signal dies in a few steps, most reverse steps learn nothing", size=14)
p.text(80, y0 + 88, "too slow: T must be huge to reach noise", size=14)
p.text(80, y0 + 116, "the schedule is tuned by experiment, not derived", size=14, bold=True, color=ORANGE)
p.save("l08-schedule.webp")
