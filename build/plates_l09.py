#!/usr/bin/env python3
"""CS329A L09 plates: the verifier bottleneck."""
import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. Outcome vs process
p = Plate("Outcome checks miss broken reasoning",
          "The final line matches. Step 4 does not follow from step 3. The outcome verifier approves.",
          "Shell 3. Saturating benchmarks on outcome rewards does not mean the reasoning got better. Source: original toy.",
          source="original toy")
p.panel(32, p.top, 420, 240, label="outcome reward", fill=PINK)
p.text(56, p.top + 72, "proof steps 1-6", size=14)
p.text(56, p.top + 104, "final line: matches", size=14)
p.chip(56, p.top + 136, "APPROVED", fill=PINK)
p.text(56, p.top + 184, "gap at step 4 unseen", size=14, color=ORANGE)
p.panel(508, p.top, 420, 240, label="process reward", fill=NEW)
p.text(532, p.top + 72, "step 1: ok ... step 3: ok", size=14)
p.text(532, p.top + 104, "step 4: does not follow", size=14, color=ORANGE)
p.chip(532, p.top + 136, "FLAGGED at step 4", fill=NEW)
p.text(532, p.top + 184, "scores the steps, not the ending", size=14, color=TEAL)
p.save("plate-l09-outcome-process.svg")

# 2. Meta-verifier loop
p = Plate("The meta-verifier judges the judge",
          "Experts seed issue-finding. The meta-verifier audits the verifier's analysis.",
          "Shell 3. Each side lifts the other: harder proofs, sharper judging. Source: lecture-reported.",
          source="lecture-reported")
stages = [("generator", "writes proofs", COUNT), ("verifier", "finds issues, scores", ACTIVE),
          ("meta-verifier", "audits the analysis", NEW)]
x = 40
for i, (name, sub, fill) in enumerate(stages):
    p.rect(x, p.top + 72, 260, 128, fill, label=None, rx=12)
    p.text(x + 130, p.top + 118, name, size=16, bold=True, anchor="middle")
    p.text(x + 130, p.top + 146, sub, size=13, color=MUTED, anchor="middle")
    if i < 2:
        p.arrow(x + 260, p.top + 136, x + 300, p.top + 136, color=INK)
        x += 300
p.text(480, p.top + 232, "8 iterations climbing; best-of-32 reaches 42% on the IMO 2024 shortlist", size=14, color=FOCUS, anchor="middle")
p.save("plate-l09-meta.svg")

# 3. Diversity: single-agent collapse vs multi-agent debate
p = Plate("Debate maintains what fine-tuning collapses",
          "One agent converges on favorites. Generators plus a critic keep disagreeing productively.",
          "Shell 3. Diversity is not a starting condition. The debate structure maintains it. Source: lecture-reported.",
          source="lecture-reported")
p.panel(32, p.top, 420, 240, label="single-agent fine-tuning", fill=PINK)
p.text(56, p.top + 88, "round 1: diverse chains", size=14)
p.text(56, p.top + 120, "round 3: favorite patterns", size=14)
p.text(56, p.top + 152, "round 5: accuracy collapses", size=14, color=ORANGE)
p.panel(508, p.top, 420, 240, label="multi-agent debate", fill=NEW)
p.text(532, p.top + 88, "generators answer in parallel", size=14)
p.text(532, p.top + 120, "critic critiques the set", size=14)
p.text(532, p.top + 152, "revise, vote; chains stay diverse", size=14, color=TEAL)
p.save("plate-l09-debate.svg")

# 4. The speed limit
p = Plate("Verification has a speed limit",
          "RL loops need thousands of fast judgments. Some domains have no fast judge at all.",
          "Shell 2. The workaround, learned stand-ins, invites reward hacking. Source: lecture-reported.",
          source="lecture-reported")
rows = [("math, code", "milliseconds", "inside the loop", NEW),
        ("chip design sims", "days per score", "outside the loop", ACTIVE),
        ("wet-lab chemistry", "weeks per experiment", "outside the loop", ACTIVE),
        ("creative work", "no objective judge", "no loop at all", PINK)]
y = p.top + 36
for domain, speed, verdict, fill in rows:
    p.text(48, y + 20, domain, size=15, bold=True)
    p.text(330, y + 20, speed, size=14, color=MUTED)
    p.chip(560, y, verdict, fill=fill)
    y += 56
p.save("plate-l09-speed.svg")

# 5. Weaver: filter then weight weak verifiers
p = Plate("Weaver: combine weak verifiers, carefully",
          "Filter the weak ones. Weight the rest with weak supervision on ~1% labels.",
          "Shell 3. Many imperfect judges beat one, if the bad ones are filtered first. Source: lecture-reported.",
          source="lecture-reported")
p.panel(32, p.top, 280, 240, label="pool of verifiers")
for i, t in enumerate(["v1: weak", "v2: ok", "v3: weak", "v4: ok"]):
    p.chip(56, p.top + 56 + i * 44, t, fill=CHIP if "weak" in t else ACTIVE, size=13)
p.panel(340, p.top, 280, 240, label="filter", fill=ACTIVE)
p.text(364, p.top + 100, "drop v1, v3", size=15, bold=True)
p.text(364, p.top + 132, "keep v2, v4", size=14, color=MUTED)
p.panel(648, p.top, 280, 240, label="weight", fill=NEW)
p.text(672, p.top + 100, "weak supervision", size=15, bold=True)
p.text(672, p.top + 132, "on ~1% labels", size=14, color=MUTED)
p.arrow(312, p.top + 120, 340, p.top + 120, color=INK)
p.arrow(620, p.top + 120, 648, p.top + 120, color=INK)
p.save("plate-l09-weaver.svg")

print("L09 plates done")
