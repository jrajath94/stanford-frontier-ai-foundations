#!/usr/bin/env python3
"""CS329A L08 plates: teaching the model to reason."""
import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. STaR loop rounds
p = Plate("STaR: each round converts failures into data",
          "Round 1: attempt 10,000, keep 3,000 winners, fine-tune. Round 2: the better model solves more.",
          "Shell 3. Bootstrapping: the frontier of solvable problems advances each round. Source: original toy.",
          source="original toy")
p.panel(32, p.top, 420, 240, label="round 1")
p.text(56, p.top + 72, "attempt 10,000 problems", size=15)
p.text(56, p.top + 104, "keep 3,000 with right answers", size=15)
p.chip(56, p.top + 136, "fine-tune on 3,000", fill=NEW)
p.panel(508, p.top, 420, 240, label="round 2")
p.text(532, p.top + 72, "better model tries the 7,000", size=15)
p.text(532, p.top + 104, "keep the new winners", size=15)
p.chip(532, p.top + 136, "fine-tune again", fill=NEW)
p.arrow(452, p.top + 196, 508, p.top + 196, label="improved model", color=TEAL)
p.save("plate-l08-rounds.svg")

# 2. Rationalization
p = Plate("Rationalization rescues the failures",
          "Hand the model the answer as a hint. Keep the rationale. Train without the hint.",
          "Shell 3. The hint is scaffolding, removed before training. Source: paper: STaR.",
          source="paper: STaR")
p.panel(32, p.top, 280, 240, label="failure")
p.text(56, p.top + 88, "model answered 37", size=15)
p.text(56, p.top + 120, "correct is 42", size=15, color=ORANGE)
p.panel(340, p.top, 280, 240, label="hint given", fill=ACTIVE)
p.text(364, p.top + 88, '"the answer is 42.', size=14)
p.text(364, p.top + 116, 'show your work."', size=14)
p.text(364, p.top + 156, "model writes rationale", size=14, color=MUTED)
p.panel(648, p.top, 280, 240, label="train", fill=NEW)
p.text(672, p.top + 88, "fine-tune on", size=15)
p.text(672, p.top + 116, "(problem, rationale, 42)", size=14)
p.text(672, p.top + 156, "WITHOUT the hint", size=14, bold=True, color=TEAL)
p.arrow(312, p.top + 120, 340, p.top + 120, color=INK)
p.arrow(620, p.top + 120, 648, p.top + 120, color=INK)
p.save("plate-l08-rationalize.svg")

# 3. Three assumptions
p = Plate("STaR stands on three assumptions",
          "Break any one and the loop stalls or teaches bad reasoning.",
          "Shell 2. The first assumption is the live wire. Source: lecture-reported.",
          source="lecture-reported")
rows = [("1. right answer => good rationale", "lucky guesses pass the filter", PINK),
        ("2. the model can rationalize", "too-hard problems get confabulation", ACTIVE),
        ("3. the base can bootstrap", "round 1 must solve something", COUNT)]
y = p.top + 40
for name, risk, fill in rows:
    p.text(48, y + 20, name, size=15, bold=True)
    p.chip(520, y, risk, fill=fill)
    y += 64
p.save("plate-l08-assumptions.svg")

# 4. GRPO: group comparison
p = Plate("GRPO: compare attempts in groups",
          "Sample a group per prompt. Normalize advantages inside the group. Push toward the better ones.",
          "Shell 3. Improves majority@K, not pass@K: more consistent, not smarter. Source: lecture-reported.",
          source="lecture-reported")
p.panel(32, p.top, 560, 240, label="one prompt, 4 sampled attempts")
scores = [("attempt A", "wrong", PINK), ("attempt B", "right", NEW), ("attempt C", "right", NEW), ("attempt D", "wrong", PINK)]
x = 56
for label, res, fill in scores:
    p.rect(x, p.top + 72, 120, 96, fill, label=None, rx=8)
    p.text(x + 60, p.top + 112, label, size=13, bold=True, anchor="middle")
    p.text(x + 60, p.top + 138, res, size=13, color=MUTED, anchor="middle")
    x += 132
p.panel(628, p.top, 300, 240, label="update", fill=ACTIVE)
p.text(652, p.top + 96, "advantage =", size=15)
p.text(652, p.top + 124, "score - group mean", size=15, bold=True, color=FOCUS)
p.text(652, p.top + 164, "push toward B, C", size=14, color=MUTED)
p.arrow(592, p.top + 120, 628, p.top + 120, color=INK)
p.save("plate-l08-grpo.svg")

# 5. The leak: right answer, broken steps
p = Plate("The leak: broken steps that reach right answers",
          "The filter keeps them. Fine-tuning bakes them in.",
          "Shell 3. Final-answer correctness is a proxy for reasoning quality, and proxies leak. Source: original.",
          source="original")
p.panel(32, p.top, 420, 240, label="rationalization output", fill=PINK)
p.text(56, p.top + 72, "step 1: ok", size=14)
p.text(56, p.top + 100, "step 2: does not follow", size=14, color=ORANGE)
p.text(56, p.top + 128, "step 3: lucky cancel", size=14, color=ORANGE)
p.text(56, p.top + 164, "final: 42, correct", size=15, bold=True)
p.panel(508, p.top, 420, 240, label="training", fill=PINK)
p.text(532, p.top + 88, "filter: answer right, keep", size=15)
p.text(532, p.top + 124, "model learns the gaps", size=15, bold=True, color=ORANGE)
p.text(532, p.top + 164, "bad reasoning, baked in", size=14, color=MUTED)
p.arrow(452, p.top + 120, 508, p.top + 120, label="no step filter", color=ORANGE)
p.save("plate-l08-leak.svg")

print("L08 plates done")
