#!/usr/bin/env python3
"""CS329A cheatsheet/crash-course plates."""
import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. The whole course as one flywheel
p = Plate("The course on one plate",
          "Generate, verify, train. Each lecture strengthens one stage or names its price.",
          "Shell 4. Ten chapters, one engine. Source: original.",
          source="original")
segs = [("L01 loop", COUNT), ("L02 generate", COUNT), ("L03 act", ACTIVE), ("L04 train", NEW),
        ("L05 plan", ACTIVE), ("L06 scale", COUNT), ("L07 know", ACTIVE), ("L08 learn", NEW),
        ("L09 judge", PINK), ("L10 measure", CHIP)]
x, y = 40, p.top + 40
for i, (name, fill) in enumerate(segs):
    col = i % 5
    if i and col == 0:
        y += 64; x = 40
    p.chip(x, y, name, fill=fill, size=13)
    x += 190
p.text(40, y + 100, "engine: generate -> verify -> train -> repeat. stalls: verifier, diversity, cost.", size=14, color=FOCUS)
p.save("plate-crash-flywheel.svg")

# 2. Key numbers one-glance
p = Plate("The numbers that matter",
          "Every number on this plate is computed or lecture-reported in the lessons.",
          "Shell 2. Quote these in interviews. Source: lessons.",
          source="lessons")
rows = [("coverage: p=0.05, k=100", "99.4%", NEW), ("compounding: 0.9^10", "35%", PINK),
        ("10@k vs pass@k", "30% vs 40%", ACTIVE), ("horizon doubling", "7 months", COUNT),
        ("50% vs 80% horizon", "59min vs 15min", NEW), ("AlphaCode 2 vs 1", "100 vs 1M samples", NEW),
        ("IMO shortlist best-of-32", "42%", ACTIVE), ("DeepScholar top score", "<19%", PINK)]
y = p.top + 36
for i, (label, val, fill) in enumerate(rows):
    col = i % 2
    x = 40 + col * 470
    yy = y + (i // 2) * 56
    p.text(x, yy + 22, label, size=14)
    p.chip(x + 300, yy, val, fill=fill)
p.save("plate-crash-numbers.svg")

print("cheat/crash plates done")
