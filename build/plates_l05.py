#!/usr/bin/env python3
"""CS329A L05 plates: planning with search."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. Six stages flow
p = Plate("LATS: six stages, one loop",
          "Select, expand, evaluate, simulate, backpropagate, reflect. Then repeat.",
          "Shell 3. Numbers update values. Words update understanding. Source: paper: LATS.",
          source="paper: LATS")
stages = ["select", "expand", "evaluate", "simulate", "backprop", "reflect"]
fills = [ACTIVE, COUNT, COUNT, COUNT, NEW, ACTIVE]
x = 32
for i, s in enumerate(stages):
    f = fills[i]
    p.rect(x, p.top + 80, 128, 112, f, label=None, rx=12)
    p.text(x + 64, p.top + 128, s, size=14, bold=True, anchor="middle")
    p.text(x + 64, p.top + 152, f"{i+1}", size=13, color=MUTED, anchor="middle")
    if i < 5:
        p.arrow(x + 128, p.top + 136, x + 152, p.top + 136, color=INK)
        x += 152
    else:
        x += 128
p.text(480, p.top + 232, "until success or the expansion budget runs out", size=14, color=MUTED, anchor="middle")
p.save("plate-l05-stages.svg")

# 2. UCT worked
p = Plate("UCT: exploit plus explore, worked",
          "Node A: value 1.26, 4 visits. Node B: value 1.10, 15 visits. Parent: 20 visits. c = 1.",
          "Shell 2. The less-visited node wins selection despite the lower value. Source: original toy.",
          source="original toy")
bonus_a = math.sqrt(math.log(20) / 4)
bonus_b = math.sqrt(math.log(20) / 15)
uct_a = 1.26 + bonus_a
uct_b = 1.10 + bonus_b
p.panel(32, p.top, 430, 240, label="node A: rarely visited", fill=NEW)
p.text(56, p.top + 72, "value 1.26 + explore 0.87", size=15)
p.text(56, p.top + 108, f"UCT = {uct_a:.2f}", size=20, bold=True, color=TEAL)
p.text(56, p.top + 148, "wins selection", size=14, color=MUTED)
p.panel(498, p.top, 430, 240, label="node B: often visited", fill=CHIP)
p.text(522, p.top + 72, "value 1.10 + explore 0.45", size=15)
p.text(522, p.top + 108, f"UCT = {uct_b:.2f}", size=20, bold=True, color=MUTED)
p.text(522, p.top + 148, "loses despite higher visits", size=14, color=MUTED)
p.save("plate-l05-uct.svg")

# 3. The maze tree
p = Plate("The tree remembers untried doors",
          "Greedy commits to the left door. The tree keeps the right door as a node.",
          "Shell 3. Selection can always go back. Source: original toy.",
          source="original toy")
p.circle(480, p.top + 60, 44, ACTIVE, label="room", size=14)
p.text(480, p.top + 16, "root", size=13, color=MUTED, anchor="middle")
kids = [("open left", "dead end", PINK, 200), ("open right", "lit hall", NEW, 480), ("inspect room", "map on wall", COUNT, 760)]
for label, state, fill, x in kids:
    p.circle(x, p.top + 200, 44, fill)
    p.text(x, p.top + 196, label, size=12, anchor="middle")
    p.text(x, p.top + 268, state, size=13, color=MUTED, anchor="middle")
    ang = math.atan2((p.top+200) - (p.top+60), x - 480)
    x1 = 480 + 46 * math.cos(ang); y1 = p.top + 60 + 46 * math.sin(ang)
    x2 = x - 46 * math.cos(ang); y2 = p.top + 200 - 46 * math.sin(ang)
    p.arrow(x1, y1, x2, y2, color=INK)
p.save("plate-l05-tree.svg")

# 4. Value: judge + self-consistency, then backprop
p = Plate("Two weak signals make one value",
          "Judge 0.6 + self-consistency 0.75 = 1.35. Backprop: (1.35 x 3 + 1) / 4 = 1.26.",
          "Shell 2. Values converge toward observed outcomes with visits. Source: original toy.",
          source="original toy")
p.panel(32, p.top, 420, 240, label="evaluation")
p.chip(56, p.top + 72, "judge: 0.6", fill=ACTIVE)
p.text(220, p.top + 96, "+", size=20, bold=True)
p.chip(248, p.top + 72, "consistency: 0.75", fill=COUNT)
p.text(56, p.top + 152, "value = 1.35", size=20, bold=True, color=FOCUS)
p.panel(508, p.top, 420, 240, label="backpropagation")
p.text(532, p.top + 72, "3 visits at 1.35, return 1", size=14)
p.text(532, p.top + 108, "(1.35 x 3 + 1) / 4", size=16, bold=True)
p.text(532, p.top + 148, "new value = 1.26", size=20, bold=True, color=TEAL)
p.arrow(452, p.top + 120, 508, p.top + 120, color=INK)
p.save("plate-l05-value.svg")

# 5. SPRINT vs LATS vs SWiRL
p = Plate("Three answers to multi-step reasoning",
          "LATS searches at test time. SPRINT parallelizes inside one model. SWiRL rewards the process.",
          "Shell 3. Same problem, three different places to put the smarts. Source: lecture-reported.",
          source="lecture-reported")
rows = [("LATS", "tree search over plans", "hundreds of calls per task", ACTIVE),
        ("SPRINT", "parallel plan and execute", "explore early, converge late", COUNT),
        ("SWiRL", "process reward before tools answer", "process-filtered data wins", NEW)]
y = p.top + 40
for name, what, note, fill in rows:
    p.text(48, y + 24, name, size=16, bold=True)
    p.chip(200, y, what, fill=fill)
    p.text(560, y + 24, note, size=13, color=MUTED)
    y += 60
p.save("plate-l05-family.svg")

print("L05 plates done")
