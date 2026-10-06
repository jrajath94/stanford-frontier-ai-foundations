#!/usr/bin/env python3
"""CS329A L10 plates: measuring agents."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. METR horizon trend
p = Plate("The 50% horizon doubles every 7 months",
          "2 seconds (2019) to 59 minutes (2025): six years, eleven doublings.",
          "Shell 2. Capability at 50% success is steep and predictable. Source: lecture-reported (METR).",
          source="lecture-reported (METR)")
pts = [("GPT-2\n2019", 2), ("GPT-4\n2023", 8*60), ("Claude 3.7\n2025", 59*60)]
logs = [math.log10(s) for _, s in pts]
mn, mx = 0, math.log10(59*60) + 0.2
x0, x1, y0, y1 = 140, 860, 300, 120
p.line(x0, y1, x0, y0, color=INK, width=2)
p.line(x0, y0, x1, y0, color=INK, width=2)
for i, ((label, s), l) in enumerate(zip(pts, logs)):
    x = x0 + i * (x1 - x0) / 2
    y = y0 - (l - mn) / (mx - mn) * (y0 - y1)
    p.circle(x, y, 10, FOCUS)
    p.text(x, y - 20, f"{s//60}:{s%60:02d}" if s >= 60 else f"{s}s", size=14, bold=True, anchor="middle")
    for j, line in enumerate(label.split("\n")):
        p.text(x, y0 + 24 + j*18, line, size=13, color=MUTED, anchor="middle")
    if i:
        px = x0 + (i-1) * (x1 - x0) / 2
        py = y0 - (logs[i-1] - mn) / (mx - mn) * (y0 - y1)
        p.line(px, py, x, y, color=FOCUS, width=3)
p.text(32, y1 - 8, "log horizon", size=13, color=MUTED)
p.save("plate-l10-horizon.svg")

# 2. Capability vs reliability gap
p = Plate("50% is not deployable",
          "Claude 3.7: 59 minutes at 50% success, about 15 minutes at 80%.",
          "Shell 2. The gap between the curves is years of headroom. Source: lecture-reported (METR).",
          source="lecture-reported (METR)")
p.panel(32, p.top, 430, 240, label="capability: 50% success", fill=ACTIVE)
p.text(56, p.top + 100, "59 minutes", size=24, bold=True, color=FOCUS)
p.text(56, p.top + 140, "research result", size=14, color=MUTED)
p.panel(498, p.top, 430, 240, label="reliability: 80% success", fill=NEW)
p.text(522, p.top + 100, "~15 minutes", size=24, bold=True, color=TEAL)
p.text(522, p.top + 140, "deployable tool", size=14, color=MUTED)
p.arrow(462, p.top + 120, 498, p.top + 120, label="4x gap", color=ORANGE)
p.save("plate-l10-gap.svg")

# 3. Four failure modes
p = Plate("Four ways long tasks fail",
          "Every failure mode compounds with trajectory length.",
          "Shell 2. Lost goal state is named repeatedly and never fully solved. Source: lecture-reported.",
          source="lecture-reported")
rows = [("poor planning", "no workable breakdown", PINK), ("poor tool choice", "wrong tool, wrong facts", PINK),
        ("no error recovery", "failures undetected", ACTIVE), ("lost goal state", "forgets the objective", ORANGE)]
y = p.top + 36
for name, what, fill in rows:
    p.text(48, y + 20, name, size=15, bold=True)
    p.chip(330, y, what, fill=fill)
    y += 56
p.save("plate-l10-failures.svg")

# 4. The whole course in one visual
p = Plate("The engine and its stalls",
          "Every open problem is a way the generate-verify-train engine stalls.",
          "Shell 4. Fix the stall, turn the crank again. Source: original.",
          source="original", inner_h=520)
rows = [("L01 loop", "needs fast honest verifiers", PINK), ("L02 generate", "pays per question", ACTIVE),
        ("L03 act", "0.9^10 = 0.35", PINK), ("L05 plan", "hundreds of calls", ACTIVE),
        ("L06 scale", "selection bottleneck", ACTIVE), ("L08 learn", "unfiltered rationales", PINK),
        ("L09 judge", "who judges the judge", ORANGE)]
y = p.top + 32
for name, stall, fill in rows:
    p.text(48, y + 18, name, size=14, bold=True)
    p.chip(240, y, stall, fill=fill, size=13)
    y += 52
p.save("plate-l10-engine.svg")

# 5. The cost of intelligence
p = Plate("Intelligence has a power bill",
          "Google served tokens: 160 trillion to 1.3 quadrillion in months.",
          "Shell 2. Test-time scaling burns compute per question. Efficiency is not optional. Source: lecture-reported.",
          source="lecture-reported")
p.panel(32, p.top, 420, 240, label="served tokens", fill=ACTIVE)
p.text(56, p.top + 88, "160T", size=20, bold=True)
p.text(56, p.top + 124, "then", size=14, color=MUTED)
p.text(56, p.top + 156, "1.3Q", size=28, bold=True, color=ORANGE)
p.text(56, p.top + 196, "8x in months", size=14, color=MUTED)
p.panel(508, p.top, 420, 240, label="demand", fill=PINK)
p.text(532, p.top + 88, "datacenter energy:", size=15)
p.text(532, p.top + 124, "hundreds of gigawatts", size=18, bold=True, color=ORANGE)
p.text(532, p.top + 172, "per-question cost compounds it", size=14, color=MUTED)
p.arrow(452, p.top + 120, 508, p.top + 120, color=INK)
p.save("plate-l10-cost.svg")

print("L10 plates done")
