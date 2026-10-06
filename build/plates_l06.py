#!/usr/bin/env python3
"""CS329A L06 plates: sampling at scale."""
import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. The 10@k metric
p = Plate("10@k: generate k, submit 10",
          "The metric measures search power and filtering together.",
          "Shell 2. The 10-submission limit models reality: you cannot try a million things against the world. Source: paper: AlphaCode.",
          source="paper: AlphaCode")
p.panel(32, p.top, 420, 240, label="generate: k samples")
p.text(56, p.top + 88, "k = up to 1,000,000", size=16, bold=True)
p.text(56, p.top + 124, "programs per problem", size=14, color=MUTED)
p.panel(508, p.top, 420, 240, label="submit: at most 10")
p.text(532, p.top + 88, "10 programs graded", size=16, bold=True)
p.text(532, p.top + 124, "on hidden tests", size=14, color=MUTED)
p.arrow(452, p.top + 120, 508, p.top + 120, label="filter picks", color=FOCUS)
p.save("plate-l06-10atk.svg")

# 2. AlphaCode pipeline
p = Plate("AlphaCode: generate, filter, cluster, submit",
          "Example tests remove 95%. Clustering buys diversity. One pick per cluster.",
          "Shell 3. Filtering is cheap. Clustering is the diversity heuristic. Source: lecture-reported.",
          source="lecture-reported")
stages = [("generate", "up to 1M programs", COUNT), ("filter", "example tests: -95%", ACTIVE),
          ("cluster", "group by behavior", ACTIVE), ("submit", "1 per cluster, max 10", NEW)]
x = 32
for i, (name, sub, fill) in enumerate(stages):
    p.rect(x, p.top + 72, 200, 128, fill, label=None, rx=12)
    p.text(x + 100, p.top + 118, name, size=16, bold=True, anchor="middle")
    p.text(x + 100, p.top + 146, sub, size=12, color=MUTED, anchor="middle")
    if i < 3:
        p.arrow(x + 200, p.top + 136, x + 232, p.top + 136, color=INK)
        x += 232
p.save("plate-l06-pipeline.svg")

# 3. The selection bottleneck
p = Plate("The bottleneck is selection, not generation",
          "Unlimited submissions pass 40%. Ten submissions stall near 30%.",
          "Shell 2. The missing 10 points were generated but not picked. Source: lecture-reported.",
          source="lecture-reported")
p.panel(32, p.top, 420, 240, label="unlimited submissions", fill=NEW)
p.chip(56, p.top + 88, "pass@k: 40%+", fill=NEW)
p.text(56, p.top + 148, "the winner is usually in the set", size=14, color=MUTED)
p.panel(508, p.top, 420, 240, label="10 submissions", fill=PINK)
p.chip(532, p.top + 88, "10@k: ~30%", fill=PINK)
p.text(532, p.top + 148, "the filter cannot always find it", size=14, color=MUTED)
p.arrow(452, p.top + 120, 508, p.top + 120, label="selection gap", color=ORANGE)
p.save("plate-l06-bottleneck.svg")

# 4. Diversity: copies vs clusters
p = Plate("Ten copies of one idea are one idea",
          "Blind sampling repeats favorites. Clustering submits one per behavioral group.",
          "Shell 3. Diversity must be engineered: it does not come free with samples. Source: original toy.",
          source="original toy")
p.panel(32, p.top, 420, 240, label="blind: 10 random picks", fill=PINK)
for i in range(5):
    p.chip(56 + (i % 3) * 120, p.top + 72 + (i // 3) * 48, "same idea", fill=PINK, size=12)
p.text(56, p.top + 196, "10 versions of one wrong approach", size=13, color=MUTED)
p.panel(508, p.top, 420, 240, label="clustered: 10 picks", fill=NEW)
for i in range(5):
    p.chip(532 + (i % 3) * 120, p.top + 72 + (i // 3) * 48, f"idea {i+1}", fill=NEW, size=12)
p.text(532, p.top + 196, "10 genuinely different approaches", size=13, color=MUTED)
p.save("plate-l06-diversity.svg")

# 5. AlphaCode 2's three changes
p = Plate("AlphaCode 2: learn the selector",
          "Better base. Diversity by design. A learned scoring model replaces the heuristic.",
          "Shell 3. 100 samples matched AlphaCode's million. Source: lecture-reported.",
          source="lecture-reported")
rows = [("better base", "fine-tune Gemini Pro, not from scratch", COUNT),
        ("diversity by design", "family of variants: different tags, mixes", ACTIVE),
        ("learned selector", "scoring model predicts correctness", NEW)]
y = p.top + 40
for name, what, fill in rows:
    p.text(48, y + 24, name, size=16, bold=True)
    p.chip(300, y, what, fill=fill)
    y += 60
p.text(48, y + 16, "result: ~85% of Codeforces participants beaten; 43% of problems in 10 attempts", size=14, color=MUTED)
p.save("plate-l06-ac2.svg")

print("L06 plates done")
