#!/usr/bin/env python3
"""CS329A L07 plates: search that reads."""
import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. Three approaches, one chemistry question
p = Plate("Dumping documents is not reading them",
          "Guess: 14, wrong. Dump 10 docs: still wrong. Search, read, extract: 10, right.",
          "Shell 3. The difference is not the search. It is the reading. Source: original toy.",
          source="original toy")
rows = [("pure reasoning", "gap -> guess -> cascade", "14 carbons: wrong", PINK),
        ("single-shot RAG", "10 docs dumped -> noise", "still wrong", PINK),
        ("agentic search", "gap -> search -> read -> extract", "10 carbons: right", NEW)]
y = p.top + 36
for name, mech, out, fill in rows:
    p.text(48, y + 20, name, size=15, bold=True)
    p.text(300, y + 20, mech, size=13, color=MUTED)
    p.chip(680, y, out, fill=fill)
    y += 64
p.save("plate-l07-three.svg")

# 2. The trigger: uncertainty -> search
p = Plate("Uncertainty is the search trigger",
          "'Perhaps', 'alternatively', 'wait': hedging words mark the knowledge gap.",
          "Shell 2. The trigger decides what the model does not know. It is heuristic. Source: lecture-reported.",
          source="lecture-reported")
p.panel(32, p.top, 280, 240, label="trace signal")
p.chip(56, p.top + 72, '"perhaps"', fill=ACTIVE)
p.chip(56, p.top + 120, '"wait..."', fill=ACTIVE)
p.text(56, p.top + 184, "uncertainty spikes", size=14, color=MUTED)
p.panel(340, p.top, 280, 240, label="trigger fires")
p.text(364, p.top + 88, "emit search token", size=15, bold=True)
p.text(364, p.top + 124, "query the gap", size=14, color=MUTED)
p.text(364, p.top + 152, "not the question", size=14, color=MUTED)
p.panel(648, p.top, 280, 240, label="reasoning continues")
p.text(672, p.top + 88, "insert result", size=15, bold=True)
p.text(672, p.top + 124, "continue the chain", size=14, color=MUTED)
p.arrow(312, p.top + 120, 340, p.top + 120, color=INK)
p.arrow(620, p.top + 120, 648, p.top + 120, color=INK)
p.save("plate-l07-trigger.svg")

# 3. Reason in documents
p = Plate("File notes, not documents",
          "A reading step extracts the relevant span. Only the notes join the prompt.",
          "Shell 3. Context length is not reasoning capacity. Source: paper: Search-o1.",
          source="paper: Search-o1")
p.panel(32, p.top, 360, 240, label="retrieved: long, noisy")
p.text(56, p.top + 72, "document 1: 8,000 words", size=14, color=MUTED)
p.text(56, p.top + 100, "document 2: 5,000 words", size=14, color=MUTED)
p.text(56, p.top + 128, "document 3: 12,000 words", size=14, color=MUTED)
p.text(56, p.top + 172, "relevant paragraph drowns", size=14, color=ORANGE)
p.panel(568, p.top, 360, 240, label="extracted: tight notes", fill=NEW)
p.text(592, p.top + 72, "compound structure: ...", size=14)
p.text(592, p.top + 100, "carbon count rule: ...", size=14)
p.text(592, p.top + 128, "two chunks, 120 words", size=14, color=TEAL)
p.arrow(400, p.top + 120, 560, p.top + 120, label="read + extract", color=FOCUS)
p.save("plate-l07-read.svg")

# 4. The context budget
p = Plate("Every search spends from a budget",
          "Searches cost latency. Chunks cost context. Stop when chunks stop helping.",
          "Shell 2. The binding constraint is what the model reasons over well. Source: original.",
          source="original")
p.panel(32, p.top, 440, 240, label="costs", fill=ACTIVE)
p.text(56, p.top + 72, "1 search = latency + 1 tool call", size=15)
p.text(56, p.top + 108, "1 chunk = context the model must reason over", size=15)
p.text(56, p.top + 148, "20 searches = 20 calls + long prompt", size=15)
p.panel(488, p.top, 440, 240, label="budget rule", fill=NEW)
p.text(512, p.top + 72, "search where the gap is load-bearing", size=15)
p.text(512, p.top + 108, "extract tightly", size=15)
p.text(512, p.top + 148, "stop when answers stop changing", size=15)
p.save("plate-l07-budget.svg")

# 5. Gap cascade
p = Plate("One guessed fact infects the chain",
          "A wrong guess at step 3 makes steps 4-7 wrong. No later step can recover.",
          "Shell 2. Fill the gap when met, before reasoning continues. Source: original toy.",
          source="original toy")
steps = [("step 1-2", "facts", NEW), ("step 3", "GUESS", PINK), ("step 4", "built on guess", PINK),
         ("step 5", "built on guess", PINK), ("step 6-7", "wrong answer", PINK)]
x = 40
for i, (name, sub, fill) in enumerate(steps):
    p.rect(x, p.top + 72, 160, 128, fill, label=None, rx=12)
    p.text(x + 80, p.top + 118, name, size=14, bold=True, anchor="middle")
    p.text(x + 80, p.top + 146, sub, size=12, color=MUTED, anchor="middle")
    if i < 4:
        p.arrow(x + 160, p.top + 136, x + 184, p.top + 136, color=ORANGE)
        x += 184
p.save("plate-l07-cascade.svg")

print("L07 plates done")
