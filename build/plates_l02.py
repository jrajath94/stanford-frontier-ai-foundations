#!/usr/bin/env python3
"""CS329A L02 plates: test-time scaling."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. Coverage curve: computed log-linear points, p=0.05
p = Plate("Coverage rises log-linearly in samples",
          "p = 0.05 per try. Coverage = 1 - 0.95^k, computed at each k.",
          "Shell 2. Each tenfold increase in samples buys a fixed jump in coverage. Source: original computation.",
          source="original computation")
pts = [(1, 0.05), (10, 0.40), (100, 0.994), (1000, 1.0)]
x0, x1, y0, y1 = 120, 880, 300, 120
p.line(x0, y1, x0, y0, color=INK, width=2)
p.line(x0, y0, x1, y0, color=INK, width=2)
for i, (k, c) in enumerate(pts):
    x = x0 + i * (x1 - x0) / 3
    y = y0 - c * (y0 - y1)
    p.circle(x, y, 10, FOCUS)
    p.text(x, y - 20, f"{c*100:.1f}%" if c < 1 else "100%", size=14, bold=True, anchor="middle")
    p.text(x, y0 + 28, f"k={k}", size=14, color=MUTED, anchor="middle")
    if i:
        px = x0 + (i-1) * (x1 - x0) / 3
        py = y0 - pts[i-1][1] * (y0 - y1)
        p.line(px, py, x, y, color=FOCUS, width=3)
p.text(32, y1 - 8, "coverage", size=13, color=MUTED)
p.text(x0 + (x1 - x0) / 2 + 60, y0 + 52, "log samples", size=13, color=MUTED, anchor="middle")
p.save("plate-l02-coverage.svg")

# 2. Oracle vs imperfect judge
p = Plate("The verifier decides what sampling buys",
          "Oracle: every gain is real. A 90%-accurate judge approves 10 wrong answers per 100 samples.",
          "Shell 3. Sampling converts generation into verification. A weak judge converts it back into noise. Source: original toy.",
          source="original toy")
p.panel(32, p.top, 420, 240, label="oracle verifier", fill=NEW)
p.text(56, p.top + 72, "100 samples", size=15)
p.text(56, p.top + 100, "keeps only true wins", size=15)
p.chip(56, p.top + 128, "coverage up, trust up", fill=NEW)
p.panel(508, p.top, 420, 240, label="90%-accurate judge", fill=PINK)
p.text(532, p.top + 72, "100 samples on unsolvable problem", size=15)
p.text(532, p.top + 100, "judge approves ~10 wrong answers", size=15)
p.chip(532, p.top + 128, "coverage up, trust down", fill=PINK)
p.save("plate-l02-judge.svg")

# 3. The bill: k samples cost k times
p = Plate("Parallelism saves latency, not money",
          "100 samples in parallel finish as fast as 1. They cost 100x the compute.",
          "Shell 2. The bill is per question, paid every time. Source: original.",
          source="original")
p.panel(32, p.top, 420, 240, label="1 sample")
p.rect(56, p.top + 80, 120, 80, COUNT, label="1x", rx=8)
p.text(56, p.top + 196, "latency T, cost C", size=14, color=MUTED)
p.panel(508, p.top, 420, 240, label="100 samples, parallel")
for i in range(4):
    p.rect(532 + i*96, p.top + 80, 88, 80, COUNT, label="1x", rx=8)
p.text(532, p.top + 128, "...", size=18, bold=True)
p.text(532, p.top + 196, "latency T, cost 100xC", size=14, color=MUTED)
p.arrow(460, p.top + 120, 500, p.top + 120, color=INK)
p.save("plate-l02-bill.svg")

# 4. Archon pipeline: generators -> critic -> ranker -> fuser
p = Plate("Archon: spend inference compute as architecture",
          "Generators diversify. Critic judges. Ranker orders. Fuser writes the final answer.",
          "Shell 3. Fusion beats picking: the fuser synthesizes instead of selecting. Source: paper: Archon.",
          source="paper: Archon")
stages = [("generators", "10 open models, 1 sample each", COUNT),
          ("critic", "strengths and weaknesses", ACTIVE),
          ("ranker", "orders candidates", ACTIVE),
          ("fuser", "writes final answer", NEW)]
x = 40
for i, (name, sub, fill) in enumerate(stages):
    p.rect(x, p.top + 72, 200, 128, fill, label=None, rx=12)
    p.text(x + 100, p.top + 120, name, size=17, bold=True, anchor="middle")
    p.text(x + 100, p.top + 148, sub, size=12, color=MUTED, anchor="middle")
    if i < 3:
        p.arrow(x + 200, p.top + 136, x + 236, p.top + 136, color=INK)
        x += 236
    else:
        x += 200
p.text(480, p.top + 240, "+14.1% average pass@1 over GPT-4o and Claude 3.5 Sonnet, open models only", size=14, color=FOCUS, anchor="middle")
p.save("plate-l02-archon.svg")

# 5. Small beats big: bar comparison
p = Plate("Verification lets small models beat giants",
          "Llama 3 8B with thousands of verified samples outperforms GPT-4o-class models at one try.",
          "Shell 2. The small model did not get smarter. It got more chances. Source: paper: Large Language Monkeys.",
          source="paper: Large Language Monkeys")
p.rect(200, p.top + 40, 180, 48, CHIP, label="GPT-4o-class, 1 try", label_size=13, rx=8)
p.rect(200, p.top + 104, 420, 48, NEW, label="Llama 3 8B, thousands of tries + verifier", label_size=13, rx=8)
p.text(200, p.top + 188, "hard math and coding benchmarks, lecture-reported", size=13, color=MUTED)
p.save("plate-l02-small-beats-big.svg")

print("L02 plates done")
