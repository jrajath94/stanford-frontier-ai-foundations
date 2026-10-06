#!/usr/bin/env python3
"""CS329A L04 plates: learning from execution."""
import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_cs329a import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, GREEN, PINK

# 1. Two loops: inference-time vs train-time
p = Plate("Two loops: exploit now, learn later",
          "Inference-time: try, run, retry. Train-time: PPO on the outcomes.",
          "Shell 3. The first loop uses the policy. The second loop improves it. Source: original.",
          source="original")
p.panel(32, p.top, 440, 240, label="inference-time loop", fill=COUNT)
p.text(56, p.top + 72, "write program", size=15)
p.text(56, p.top + 100, "run public tests", size=15)
p.text(56, p.top + 128, "read the failure", size=15)
p.text(56, p.top + 156, "retry until pass or limit", size=15)
p.text(56, p.top + 200, "weights: FIXED", size=15, bold=True, color=FOCUS)
p.panel(488, p.top, 440, 240, label="train-time loop", fill=NEW)
p.text(512, p.top + 72, "collect passing trajectories", size=15)
p.text(512, p.top + 100, "reward 1 on private tests", size=15)
p.text(512, p.top + 128, "PPO update toward winners", size=15)
p.text(512, p.top + 156, "policy improves", size=15)
p.text(512, p.top + 200, "weights: CHANGE", size=15, bold=True, color=TEAL)
p.arrow(472, p.top + 120, 488, p.top + 120, color=INK)
p.save("plate-l04-two-loops.svg")

# 2. Two tiers of tests
p = Plate("Public tests guide. Private tests grade.",
          "The model sees public tests during generation. Private tests decide the reward, unseen.",
          "Shell 3. The only reliable way to earn reward 1 is genuinely correct code. Source: paper: RLEF.",
          source="paper: RLEF")
p.panel(32, p.top, 420, 240, label="public tests: visible", fill=ACTIVE)
p.text(56, p.top + 72, "small set, fast", size=15)
p.text(56, p.top + 100, "model sees failures", size=15)
p.text(56, p.top + 128, "guides retries", size=15)
p.chip(56, p.top + 160, "iteration", fill=ACTIVE)
p.panel(508, p.top, 420, 240, label="private tests: hidden", fill=NEW)
p.text(532, p.top + 72, "hidden until grading", size=15)
p.text(532, p.top + 100, "decide reward 1 or 0", size=15)
p.text(532, p.top + 128, "cannot be memorized", size=15)
p.chip(532, p.top + 160, "honest reward", fill=NEW)
p.save("plate-l04-two-tiers.svg")

# 3. Sparse binary reward
p = Plate("Binary reward is honest and sparse",
          "9 of 10 tests pass: reward 0. Syntax error on line 1: reward 0. All pass: reward 1.",
          "Shell 2. Near misses teach nothing. The model must already pass sometimes. Source: original toy.",
          source="original toy")
cases = [("9/10 tests pass", "0", PINK), ("syntax error", "0", PINK), ("all tests pass", "1", NEW)]
y = p.top + 56
for label, r, fill in cases:
    p.text(64, y + 24, label, size=16)
    p.chip(360, y, f"reward {r}", fill=fill)
    y += 60
p.text(64, y + 8, "the gradient has nothing to climb until something passes", size=14, color=MUTED)
p.save("plate-l04-sparse.svg")

# 4. Turn-level credit
p = Plate("One advantage for the whole program",
          "The reward exists per turn. Every token in the program shares one advantage value.",
          "Shell 3. Coarse, but honest about what the world reports. Source: paper: RLEF.",
          source="paper: RLEF")
p.panel(32, p.top, 560, 240, label="one program, many tokens")
toks = ["def", "solve", "(", "):", "...", "return"]
x = 56
for t in toks:
    w = p.chip(x, p.top + 88, t, fill=CHIP)
    x += w + 12
p.text(56, p.top + 160, "no test says which token failed", size=14, color=MUTED)
p.panel(628, p.top, 300, 240, label="credit", fill=ACTIVE)
p.text(652, p.top + 100, "advantage A", size=18, bold=True)
p.text(652, p.top + 132, "shared by all tokens", size=14, color=MUTED)
p.arrow(592, p.top + 120, 628, p.top + 120, label="one number", color=INK)
p.save("plate-l04-credit.svg")

# 5. The execution feedback template
p = Plate("The feedback the model actually reads",
          "Failures arrive as assertion errors with runtime values, not as a bare 0.",
          "Shell 2. Informative failures make retries targeted. Source: paper: RLEF.",
          source="paper: RLEF")
p.panel(32, p.top, 896, 220, label="execution feedback template", fill=PANEL)
p.text(56, p.top + 64, "Your code failed some test cases:", size=15, bold=True)
p.text(56, p.top + 96, "- Failure: test_3: AssertionError: got 14, expected 10", size=14, color=INK)
p.text(56, p.top + 124, "- Failure: test_7: Execution took too long.", size=14, color=INK)
p.text(56, p.top + 152, "- Success: test_1, test_2, test_4, test_5, test_6", size=14, color=TEAL)
p.text(56, p.top + 188, "Give it another try.", size=15, bold=True, color=FOCUS)
p.save("plate-l04-feedback.svg")

print("L04 plates done")
