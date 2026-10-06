#!/usr/bin/env python3
"""Chapter plates for cs229 l17 (RL for LLMs). All numbers code-computed with asserts."""
import math, os

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/cs229/assets")
SANS = "Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------- number computations (asserted) ----------------
# advantage toy: V=0.5
assert (1 - 0.5) == 0.5 and (0 - 0.5) == -0.5
# GAE toy: deltas [+0.5, -0.2, +0.3], gamma*lambda = 0.9
A0 = 0.5 + 0.9 * (-0.2) + 0.81 * 0.3
assert abs(A0 - 0.563) < 1e-9, A0
# PPO four cases, A = -2, eps = 0.2
A = -2.0
def clipped_obj(r):
    return min(r * A, max(1 - 0.2, min(1 + 0.2, r)) * A)
assert abs(clipped_obj(1.5) - (-3.0)) < 1e-9   # unclipped, full correction
assert abs(clipped_obj(0.5) - (-1.6)) < 1e-9   # clipped flat
# A > 0 cases
A = 2.0
def clipped_obj_pos(r):
    return min(r * A, max(1 - 0.2, min(1 + 0.2, r)) * A)
assert abs(clipped_obj_pos(1.5) - 2.4) < 1e-9  # = 1.2*A, clipped flat
assert abs(clipped_obj_pos(0.5) - 1.0) < 1e-9  # = 0.5*A, unclipped
# Bradley-Terry toy: scores 2.0 vs 0.5
p_win = 1 / (1 + math.exp(-1.5))
assert abs(p_win - 0.81757) < 1e-4, p_win
loss_bt = -math.log(p_win)
assert abs(loss_bt - 0.20141) < 1e-4, loss_bt
# DPO toy: beta=0.1, margin 1.0
sig = 1 / (1 + math.exp(-0.1))
assert abs(sig - 0.52498) < 1e-4, sig
loss_dpo = -math.log(sig)
assert abs(loss_dpo - 0.64440) < 1e-4, loss_dpo
# GRPO toy: G=4, rewards [1,1,0,0]
rs = [1, 1, 0, 0]
mean = sum(rs) / len(rs)
var = sum((x - mean) ** 2 for x in rs) / len(rs)
std = math.sqrt(var)
assert abs(mean - 0.5) < 1e-9 and abs(std - 0.5) < 1e-9
advs = [(x - mean) / std for x in rs]
assert advs == [1.0, 1.0, -1.0, -1.0], advs
# R1 numbers
assert abs(71.0 - 15.6 - 55.4) < 1e-9  # +55.4 pts
assert abs(79.8 - 71.0 - 8.8) < 1e-9
# group-size dial: G=8, 10% solve rate -> P(all wrong) = 0.9^8
p_all_wrong_8 = 0.9 ** 8
p_all_wrong_64 = 0.9 ** 64
assert abs(p_all_wrong_8 - 0.4305) < 1e-3, p_all_wrong_8
assert p_all_wrong_64 < 0.002, p_all_wrong_64
# verifier blind spot: 5% of wins from broken reasoning (lesson number, kept)

# ---------------- svg builder ----------------
def plate(path, title, subtitle, left_label, center_label, right_label,
          left_lines, center_lines, right_lines, bottom_lines, footer,
          panel_h=300, bottom_h=72):
    W = 960
    y_panels = 140
    y_bottom = y_panels + panel_h + 16
    H = y_bottom + bottom_h + 56
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{SANS}">')
    parts.append(f'<rect width="{W}" height="{H}" fill="#F7F4EE"/>')
    parts.append(f'<text x="48" y="56" font-size="30" font-weight="600" fill="#1B2838">{esc(title)}</text>')
    parts.append(f'<text x="48" y="86" font-size="17" fill="#5C6B7A">{esc(subtitle)}</text>')
    parts.append('<g font-size="18" font-weight="600" fill="#1B2838">')
    parts.append(f'<text x="48" y="124">{esc(left_label)}</text>')
    parts.append(f'<text x="328" y="124">{esc(center_label)}</text>')
    parts.append(f'<text x="648" y="124">{esc(right_label)}</text>')
    parts.append('</g>')
    def region(x, w, fill, stroke, sw, lines, cx):
        parts.append(f'<rect x="{x}" y="{y_panels}" width="{w}" height="{panel_h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        y = y_panels + 44
        for (t, size, color, weight) in lines:
            parts.append(f'<text x="{cx}" y="{y}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{color}">{esc(t)}</text>')
            y += 30 if size >= 16 else 26
    region(48, 264, "#FFFDF8", "#1B2838", 1.5, left_lines, 180)
    region(328, 304, "#E7F1F8", "#1E4D8C", 1.5, center_lines, 480)
    region(648, 264, "#E7F4EF", "#1F7A72", 2, right_lines, 780)
    parts.append(f'<rect x="48" y="{y_bottom}" width="864" height="{bottom_h}" rx="12" fill="#FFFDF8" stroke="#1B2838" stroke-width="1.5"/>')
    y = y_bottom + 30
    for (t, color) in bottom_lines:
        parts.append(f'<text x="480" y="{y}" text-anchor="middle" font-size="15" fill="{color}">{esc(t)}</text>')
        y += 26
    parts.append(f'<text x="48" y="{H - 22}" font-size="16" font-weight="500" fill="#1B2838">{esc(footer)}</text>')
    parts.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", path)

DK, MU, TE, BL = "#1B2838", "#5C6B7A", "#1F7A72", "#1E4D8C"

plate(os.path.join(ASSETS, "plate-l17-chap-advantages.svg"),
    "Chapter plate: advantages subtract the luck",
    "Chapter plate. The baseline is half the algorithm. Source: original synthesis of the session.",
    "WITHOUT the rule: raw reward", "the stored object: the advantage", "WITH the rule: GAE",
    [("one bit at the end", 15, DK, 600), ("500 thinking tokens", 15, MU, 500),
     ("490 good steps punished", 15, MU, 500), ("lucky guess reinforced", 15, MU, 500),
     ("too coarse for 500 decisions", 15, MU, 500)],
    [("A = return - baseline", 16, DK, 600), ("V = 0.5 from here", 15, MU, 500),
     ("correct: 1 - 0.5 = +0.5", 15, TE, 600), ("wrong: 0 - 0.5 = -0.5", 15, MU, 500),
     ("better vs worse than usual", 15, DK, 600), ("expectation unchanged", 15, MU, 500)],
    [("deltas +0.5, -0.2, +0.3", 15, DK, 600), ("lambda 0.95 standard", 15, MU, 500),
     ("A_0 = 0.563", 16, TE, 600), ("near deltas count most", 15, MU, 500),
     ("group mean: free baseline", 15, MU, 500), ("critic: doubles forward cost", 15, MU, 500)],
    [("Tradeoff: the expectation is unchanged, the variance drops. Baselines do not bias; they calm.", DK),
     ("Baseline zoo: constant (crude), V(s) (pricey, best), group mean (free, binary rewards).", MU)],
    "One connection: advantage turns 'right or wrong' into 'better or worse than usual', which is the learnable signal.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l17-chap-ppo.svg"),
    "Chapter plate: PPO clips the jump",
    "Chapter plate. Trust old data a little, briefly. Source: original synthesis of the session.",
    "WITHOUT the rule: on-policy", "the stored object: the clipped ratio", "WITH the rule: the plumbing",
    [("one step per fresh batch", 15, DK, 600), ("data stale after update", 15, MU, 500),
     ("single-use samples", 15, MU, 500), ("r = 3: one trajectory triple", 15, MU, 500),
     ("millions of trajectories", 15, MU, 500)],
    [("r = pi_new / pi_old", 15, DK, 600), ("clip to [0.8, 1.2]", 16, DK, 600),
     ("A>0, r=1.5: clipped flat", 15, MU, 500), ("A>0, r=0.5: full push", 15, MU, 500),
     ("A<0, r=1.5: full correction", 15, TE, 600), ("A<0, r=0.5: clipped flat", 15, MU, 500)],
    [("eps 0.2, 4 epochs", 15, DK, 600), ("LR 1e-5 to 3e-6", 15, MU, 500),
     ("GAE 0.95, adv norm on", 15, MU, 500), ("KL leash to the SFT ref", 15, MU, 500),
     ("10% objective, 90% plumbing", 15, TE, 600), ("ratio histogram is the dashboard", 15, MU, 500)],
    [("Tradeoff: clipping throws away legitimate signal to buy stability. Watch the ratio histogram, not the loss.", DK),
     ("Clip the direction the objective pulls, not the direction the ratio sits.", MU)],
    "One connection: the clip is TRPO's trust region as a first-order hack, and the hack is what ships.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l17-chap-rlhf.svg"),
    "Chapter plate: RLHF, preferences as reward",
    "Chapter plate. The reward model is the task specification. Source: original synthesis of the session.",
    "WITHOUT the rule: no verifier", "the stored object: the reward model", "WITH the rule: the recipe",
    [("open-ended writing", 15, DK, 600), ("no right answer", 15, MU, 500),
     ("math's 0/1 does not apply", 15, MU, 500), ("taste is not checkable", 15, MU, 500)],
    [("A scores 2.0, B 0.5", 15, DK, 600), ("P(A beats B) = 0.818", 16, DK, 600),
     ("loss: 0.20", 16, DK, 600), ("4-9 responses per prompt", 15, MU, 500),
     ("annotators agree ~70%", 15, MU, 500), ("the data is the moat", 15, TE, 600)],
    [("SFT -> reward model -> PPO", 15, DK, 600), ("KL leash to the SFT ref", 15, MU, 500),
     ("Goodhart: past the peak", 15, TE, 600), ("DPO: loss 0.644, no RM", 15, MU, 500),
     ("offline: never explores", 15, MU, 500), ("length bias, staleness", 15, MU, 500)],
    [("Tradeoff: RLHF trains the model to please the reward model, which approximates pleasing humans.", DK),
     ("The approximation is the whole game. Hold out human evals; the peak is the operating point.", MU)],
    "One connection: the KL leash is the SFT checkpoint's ghost, guarding every RL update against the proxy.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l17-chap-grpo.svg"),
    "Chapter plate: GRPO deletes the critic",
    "Chapter plate. The group's average is the baseline. Source: original synthesis of the session.",
    "WITHOUT the rule: the critic", "the stored object: the group", "WITH the rule: R1",
    [("value head on the policy", 15, DK, 600), ("second forward pass", 15, MU, 500),
     ("doubles the RL step cost", 15, MU, 500), ("bias infects advantages", 15, MU, 500),
     ("value loss, GAE, tuning", 15, MU, 500)],
    [("G = 4, rewards [1,1,0,0]", 15, DK, 600), ("mean 0.5, std 0.5", 15, MU, 500),
     ("advantages +1,+1,-1,-1", 16, TE, 600), ("no critic, no GAE", 15, DK, 600),
     ("G = 8 standard", 15, MU, 500), ("group mean is free", 15, MU, 500)],
    [("R1-Zero: no SFT at all", 15, DK, 600), ("AIME 15.6% -> 71.0%", 16, TE, 600),
     ("R1: 79.8%, matches o1", 15, MU, 500), ("aha: self-corrects", 15, MU, 500),
     ("rule rewards, no neural RM", 15, MU, 500), ("rejection sampling: the baseline", 15, MU, 500)],
    [("Tradeoff: crude baseline, G samples per prompt. Scale G with difficulty, not habit.", DK),
     ("Blind spot: 5% of wins come from broken reasoning. Never trust a rising reward curve alone.", MU)],
    "One connection: verifiable 0/1 rewards need no learned judge, and the group mean is the cheapest baseline.",
    panel_h=300, bottom_h=72)
print("l17 plates done")
