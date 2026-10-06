#!/usr/bin/env python3
"""CS329A L03 plates: agents that act."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. The ReAct loop, with the calculator toy
p = Plate("Thought, action, observation, repeat",
          "Each thought picks a tool. Each observation grounds the next thought.",
          "Shell 3. The answer rests on two verified steps, not one leap. Source: original toy.",
          source="original toy")
nodes = [("thought", "15% of 240?", CHIP), ("action", "calculator(0.15*240)", ACTIVE), ("observation", "36.0", NEW)]
x = 48
for i, (name, sub, fill) in enumerate(nodes):
    p.rect(x, p.top + 72, 240, 128, fill, label=None, rx=12)
    p.text(x + 120, p.top + 118, name, size=18, bold=True, anchor="middle")
    p.text(x + 120, p.top + 148, sub, size=13, color=MUTED, anchor="middle")
    if i < 2:
        p.arrow(x + 240, p.top + 136, x + 288, p.top + 136, color=INK)
        x += 288
p.text(480, p.top + 240, "then again: thought -> calculator(36+7) -> 43. answer: 43", size=14, color=FOCUS, anchor="middle")
p.save("plate-l03-loop.svg")

# 2. Compounding error curve
p = Plate("Reliability decays exponentially in steps",
          "Each round 90% reliable. Task success = 0.9^n, computed.",
          "Shell 2. Ten steps at 90% give a 35% task. Twenty give 12%. Source: original computation.",
          source="original computation")
x0, x1, y0, y1 = 120, 880, 300, 120
p.line(x0, y1, x0, y0, color=INK, width=2)
p.line(x0, y0, x1, y0, color=INK, width=2)
pts = [(1, 0.9), (5, 0.9**5), (10, 0.9**10), (20, 0.9**20)]
for i, (n, v) in enumerate(pts):
    x = x0 + i * (x1 - x0) / 3
    y = y0 - v * (y0 - y1)
    p.circle(x, y, 10, ORANGE if v < 0.5 else TEAL)
    p.text(x, y - 22, f"{v*100:.0f}%", size=14, bold=True, anchor="middle")
    p.text(x, y0 + 28, f"n={n}", size=14, color=MUTED, anchor="middle")
    if i:
        px = x0 + (i-1) * (x1 - x0) / 3
        py = y0 - pts[i-1][1] * (y0 - y1)
        p.line(px, py, x, y, color=ORANGE, width=3)
p.text(32, y1 - 8, "task success", size=13, color=MUTED)
p.text(x0 + (x1 - x0) / 2 + 60, y0 + 52, "steps", size=13, color=MUTED, anchor="middle")
p.save("plate-l03-compound.svg")

# 3. Grounding: ungrounded vs grounded
p = Plate("Grounding pins each step to the world",
          "Ungrounded: the answer is a guess. Grounded: each claim has a source.",
          "Shell 3. Grounding moves failure from guessing to looking in the wrong place. Source: original.",
          source="original")
p.panel(32, p.top, 420, 240, label="closed box", fill=PINK)
p.text(56, p.top + 72, "Q: titles after 2024 French Open?", size=14)
p.text(56, p.top + 108, "model: 3 (from memory)", size=14)
p.text(56, p.top + 144, "no step touches a source", size=14, color=MUTED)
p.chip(56, p.top + 168, "guess dressed as fact", fill=PINK)
p.panel(508, p.top, 420, 240, label="ReAct", fill=NEW)
p.text(532, p.top + 72, "thought -> search -> observation", size=14)
p.text(532, p.top + 108, "observation: third Grand Slam title", size=14)
p.text(532, p.top + 144, "every claim has a source", size=14, color=MUTED)
p.chip(532, p.top + 168, "answer: 3, sourced", fill=NEW)
p.save("plate-l03-grounding.svg")

# 4. The thinking dial
p = Plate("Thinking is a dial, not a virtue",
          "Too little thought acts blind. Too much stalls. The sweet spot moves per task.",
          "Shell 3. Current models overthink simple tasks: the lecture's observation. Source: lecture-reported.",
          source="lecture-reported")
p.panel(32, p.top, 896, 200, label="thought per action")
p.rect(64, p.top + 72, 220, 64, PINK, label="too little: blind action", label_size=13, rx=8)
p.rect(370, p.top + 72, 220, 64, NEW, label="sweet spot: task-dependent", label_size=13, rx=8)
p.rect(676, p.top + 72, 220, 64, PINK, label="too much: stalls, burns", label_size=13, rx=8)
p.arrow(284, p.top + 104, 362, p.top + 104, color=INK)
p.arrow(590, p.top + 104, 668, p.top + 104, color=INK)
p.text(64, p.top + 168, "the field has not settled how to set the dial", size=13, color=MUTED)
p.save("plate-l03-think-dial.svg")

# 5. What the paper measured
p = Plate("ReAct's measured wins",
          "1-2 in-context examples. Absolute gains over imitation and RL baselines.",
          "Shell 2. The traces are legible: you can read what the model believed at step 4. Source: paper: ReAct.",
          source="paper: ReAct")
rows = [("ALFWorld", "+34% absolute", "best trial 71% avg"), ("WebShop", "+10% absolute", "vs baselines")]
y = p.top + 40
for task, gain, note in rows:
    p.text(64, y + 24, task, size=16, bold=True)
    p.chip(240, y, gain, fill=NEW)
    p.text(440, y + 24, note, size=14, color=MUTED)
    y += 56
p.text(64, y + 16, "HotpotQA and Fever: cuts hallucination by letting the model go check", size=14, color=MUTED)
p.save("plate-l03-paper.svg")

print("L03 plates done")
