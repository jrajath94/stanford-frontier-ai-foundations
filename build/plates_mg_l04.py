#!/usr/bin/env python3
"""L04 plates: saturation curves, mode collapse numbers."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. Saturation: loss curves near d=0
p = Plate("The original loss is flat where the generator starts",
          "Doubling the score 0.001 -> 0.002 moves the saturating loss 0.001, the fix 0.693.",
          "Shell 2. Count the curves: flat log(1-d) vs steep -log(d) at d near 0. Source: original toy.",
          source="original toy", inner_h=460)
x0, y0, x1, y1 = 80, 400, 440, 120
p.axes(x0, y0, x1, y1)
p.text(24, y1 - 8, "loss", size=13, color=MUTED)
p.text(x1 - 40, y0 + 24, "d = D(G(z))", size=13, color=MUTED)
# saturating loss L = -log(1-d), plotted scaled; non-saturating L = -log d
def X(d): return x0 + d / 0.01 * (x1 - x0)
pts_s, pts_n = [], []
for i in range(41):
    d = 0.0002 + i * (0.0098 / 40)
    pts_s.append((X(d), y0 - (-math.log(1 - d)) / 0.01005 * (y0 - y1)))
    pts_n.append((X(d), y0 - min((-math.log(d)), 8.5) / 8.5 * (y0 - y1)))
p.curve(pts_s, color=PINK, width=3)
p.curve(pts_n, color=TEAL, width=3)
p.text(x1 - 200, y0 - 60, "log(1-d): flat", size=14, color=ORANGE, bold=True)
p.text(x1 - 200, y1 + 60, "-log(d): steep", size=14, color=TEAL, bold=True)
p.text(80, y0 + 60, "d = 0.001: generator doubles its score to 0.002", size=14)
p.text(80, y0 + 88, "saturating loss changes by 0.001 (whisper)", size=14, color=ORANGE)
p.text(80, y0 + 116, "non-saturating loss changes by 0.693 (scream)", size=14, color=TEAL)
p.save("l04-saturation.webp")

# 2. Mode collapse numbers
p = Plate("Mode collapse: half the world missing, JS = 0.216",
          "Truth {0: 0.5, 10: 0.5}. Model {0: 1.0}. The divergence shrugs.",
          "Shell 2. 0.5*0.144 + 0.5*0.288 = 0.216. Generator loss stuck at 0.693. Source: original toy.",
          source="original toy", inner_h=520)
p.text(32, p.top, "truth P_X", size=14, bold=True, color=MUTED)
p.bars(32, p.top + 240, [("0", 0.5, TEAL), ("10", 0.5, TEAL)], 1.0,
       bar_w=100, gap=60, height=170, size=14)
p.text(32, p.top + 276, "collapsed model P_theta", size=14, bold=True, color=MUTED)
# Second bar row pushed down: the "1.0" value label sat on the heading above.
p.bars(32, p.top + 510, [("0", 1.0, ORANGE), ("10", 0.0, PINK)], 1.0,
       bar_w=100, gap=60, height=170, size=14)
p.panel(560, p.top + 40, 368, 400, label="the bill", fill=YELLOW)
p.text(584, p.top + 96, "JS = 0.216", size=20, bold=True)
p.text(584, p.top + 140, "small number for a model", size=14)
p.text(584, p.top + 168, "missing half the distribution", size=14)
p.text(584, p.top + 224, "generator loss at 0: -log(0.5)", size=14)
p.text(584, p.top + 252, "= 0.693, stable forever", size=14, bold=True)
p.text(584, p.top + 308, "moving toward 10: D ~ 0,", size=14)
p.text(584, p.top + 336, "loss explodes. So it stays.", size=14, color=ORANGE)
p.save("l04-mode-collapse.webp")
