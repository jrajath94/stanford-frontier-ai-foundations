#!/usr/bin/env python3
"""CS329A L01 plates: the self-improving agent."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, hbar_bars, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. Scaling curve: params across eras (log-scaled bars)
p = Plate("Bigger models, predictable gains",
          "Parameters rose 1,500x from BERT to PaLM. Test loss fell on a smooth curve.",
          "Shell 2. Three orders of magnitude of scale, one predictable loss curve. Source: public model cards.",
          source="public model cards")
models = [("BERT (2018)", 0.34e9), ("GPT-2 (2019)", 1.5e9), ("GPT-3 (2020)", 175e9), ("PaLM (2022)", 540e9)]
logs = [(n, math.log10(v)) for n, v in models]
mn = min(l for _, l in logs) - 0.3
mx = max(l for _, l in logs)
x0, bw = 200, 620
y = p.top + 24
for n, l in logs:
    w = bw * (l - mn) / (mx - mn)
    p.text(32, y + 26, n, size=15, bold=True)
    p.rect(x0, y, w, 36, ACTIVE if "PaLM" in n else COUNT, label=f"{l:.1f} (log10)", label_size=13, rx=8)
    y += 52
p.text(32, y + 8, "log10 of parameter count; each step right is 10x", size=13, color=MUTED)
p.save("plate-l01-scaling.svg")

# 2. CoT: one leap vs three steps
p = Plate("Thinking out loud beats one leap",
          "The tennis toy: 5 + 2 x 3. One leap guesses. Steps check each other.",
          "Shell 3. Three easy steps replace one hard leap. Source: original toy.")
p.panel(32, p.top, 420, 240, label="one shot: 5 + 2 x 3 = ?")
p.text(56, p.top + 72, "model guesses:", size=14, color=MUTED)
p.chip(56, p.top + 88, "11?", fill=PINK)
p.text(56, p.top + 148, "no step to check", size=14, color=MUTED)
p.text(56, p.top + 172, "a wrong guess hides", size=14, color=MUTED)
p.panel(508, p.top, 420, 240, label="chain of thought")
steps = ["5 start", "2 cans x 3 = 6", "5 + 6 = 11"]
sx = 532
for s in steps:
    w = p.chip(sx, p.top + 88, s, fill=NEW)
    sx += w + 44
    if sx < 900:
        p.arrow(sx - 44, p.top + 104, sx - 8, p.top + 104, color=TEAL)
p.text(532, p.top + 148, "each step is easy alone", size=14, color=MUTED)
p.text(532, p.top + 172, "each step is checkable", size=14, color=MUTED)
p.save("plate-l01-cot.svg")

# 3. Monkeys: 10k samples funnel
p = Plate("Ten thousand tries surface tail knowledge",
          "Llama 3 8B, F2F benchmark, oracle verifier: one try fails, 10k tries solve.",
          "Shell 2. The knowledge was in the weights. One shot could not reach it. Source: Brown et al., Large Language Monkeys (2024).",
          source="paper: Large Language Monkeys")
p.panel(32, p.top, 300, 240, label="1 attempt")
p.chip(56, p.top + 88, "1 draw", fill=CHIP)
p.text(56, p.top + 148, "5% per try: fails", size=14, color=MUTED)
p.panel(628, p.top, 300, 240, label="10,000 attempts + oracle")
p.chip(652, p.top + 88, "IMO-level solved", fill=NEW)
p.text(652, p.top + 148, "verifier keeps the win", size=14, color=MUTED)
p.arrow(340, p.top + 104, 620, p.top + 104, label="10,000x samples", color=FOCUS)
p.save("plate-l01-monkeys.svg")

# 4. The loop: generate -> verify -> train
p = Plate("The self-improvement loop",
          "Generate many tries. Verify automatically. Train on winners. Repeat.",
          "Shell 3. Test-time compute makes data. Train-time compute absorbs it. Source: original.")
stages = [("generate", "many attempts", COUNT), ("verify", "keep winners", ACTIVE), ("train", "fine-tune", NEW)]
x = 56
for i, (name, sub, fill) in enumerate(stages):
    p.rect(x, p.top + 80, 240, 120, fill, label=None, rx=12)
    p.text(x + 120, p.top + 128, name, size=20, bold=True, anchor="middle")
    p.text(x + 120, p.top + 156, sub, size=14, color=MUTED, anchor="middle")
    x += 240
    if i < 2:
        p.arrow(x - 8, p.top + 140, x + 64, p.top + 140, color=INK)
        x += 72
# repeat arrow back
p.text(480, p.top + 232, "repeat: the better model generates better attempts", size=14, color=FOCUS, anchor="middle")
p.save("plate-l01-loop.svg")

# 5. Test-time vs train-time compute
p = Plate("Two kinds of compute, one flywheel",
          "Test-time: weights fixed, answers improve. Train-time: weights change, answers get cheaper.",
          "Shell 3. Joining them is the DeepSeek/o1 breakthrough. Source: original.")
p.panel(32, p.top, 440, 240, label="test-time compute", fill=COUNT)
p.text(56, p.top + 72, "more samples", size=15)
p.text(56, p.top + 100, "longer traces", size=15)
p.text(56, p.top + 128, "tool calls", size=15)
p.text(56, p.top + 172, "weights: FIXED", size=15, bold=True, color=FOCUS)
p.panel(488, p.top, 440, 240, label="train-time compute", fill=NEW)
p.text(512, p.top + 72, "fine-tune on winners", size=15)
p.text(512, p.top + 100, "RL on verifiers", size=15)
p.text(512, p.top + 128, "distill traces", size=15)
p.text(512, p.top + 172, "weights: CHANGE", size=15, bold=True, color=TEAL)
p.arrow(472, p.top + 120, 488, p.top + 120, color=INK)
p.save("plate-l01-test-train.svg")

print("L01 plates done")
