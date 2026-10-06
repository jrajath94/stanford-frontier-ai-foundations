#!/usr/bin/env python3
"""Chapter plates for cs229 l16 (reinforcement learning). All numbers code-computed with asserts."""
import math, os

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/cs229/assets")
SANS = "Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------- number computations (asserted) ----------------
# UCB toy: t=15, pulls A=10 (0.3), B=4 (0.5), C=1 (0.0)
t = 15
ln_t = math.log(t)
bonus = lambda n: math.sqrt(2 * ln_t / n)
bA, bB, bC = bonus(10), bonus(4), bonus(1)
assert abs(bA - 0.736) < 0.01, bA
assert abs(bB - 1.164) < 0.01, bB
assert abs(bC - 2.328) < 0.01, bC
assert abs(0.3 + bA - 1.036) < 0.01
assert abs(0.5 + bB - 1.664) < 0.01
assert abs(0.0 + bC - 2.328) < 0.01
# Bellman chain: always-right, gamma=0.9, step -1, dock value V(10)=10
# (lesson convention: V(9) = -1 + 0.9*10 = 8.0)
g = 0.9
V = {10: 10.0}
for s in range(9, 0, -1):
    V[s] = -1.0 + g * V[s + 1]
assert abs(V[9] - 8.0) < 1e-6
assert abs(V[8] - 6.2) < 1e-6
assert abs(V[7] - 4.58) < 1e-6
assert abs(V[6] - 3.122) < 1e-6
assert abs(V[5] - 1.8098) < 1e-4
assert abs(V[4] - 0.62882) < 1e-4
assert abs(V[3] + 0.43406) < 1e-4, V[3]
# cross-check V(3) directly
direct = -(1 - g ** 7) / (1 - g) + g ** 7 * 10
assert abs(direct - V[3]) < 1e-9, (direct, V[3])
# Q(3,left): -1 + 0.9 * V(2)
q_left = -1 + g * V[2]
assert abs(q_left + 2.2515) < 1e-3, q_left  # -2.25
q_right = -1 + g * V[4]
assert abs(q_right - V[3]) < 1e-9
# greedy forever: -1/(1-g)
assert abs(-1 / (1 - g) - (-10.0)) < 1e-9
# discount priced
assert abs(g ** 10 - 0.3487) < 1e-3
assert abs(1 / (1 - 0.9) - 10.0) < 1e-9
assert abs(1 / (1 - 0.99) - 100.0) < 1e-9
# contraction: 0.9^44 ~ 0.01
assert abs(0.9 ** 44 - 0.00974) < 1e-3
# TD toy: V(9) <- 0 + 0.1*(9 + 0.9*0 - 0)
assert abs(0.1 * 9 - 0.9) < 1e-9
# Q-learning toy
q = 0.0
q = q + 0.5 * (9 + 0.9 * 0 - q)
assert abs(q - 4.5) < 1e-9
q = q + 0.5 * (9 + 0.9 * 0 - q)
assert abs(q - 6.75) < 1e-9
# REINFORCE toy: total 3, alpha=0.1, pi(right)=0.6
dtheta = 0.1 * 3 * (1 - 0.6)
assert abs(dtheta - 0.12) < 1e-9
total_bad = -12
# baseline -4.5
assert abs((3 - (-4.5)) - 7.5) < 1e-9
assert abs((total_bad - (-4.5)) - (-7.5)) < 1e-9
# actor-critic toy
theta = math.log(0.7 / 0.3)
assert abs(theta - 0.84730) < 1e-4, theta
delta = -1 + 0.9 * 8.0 - 5.0
assert abs(delta - 1.2) < 1e-9, delta
theta2 = theta + 0.1 * delta * 0.3
assert abs(theta2 - 0.88330) < 1e-4, theta2
pi2 = 1 / (1 + math.exp(-theta2))
assert abs(pi2 - 0.70752) < 1e-4, pi2
v8_new = 5.0 + 0.1 * delta
assert abs(v8_new - 5.12) < 1e-9

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

plate(os.path.join(ASSETS, "plate-l16-chap-bandits.svg"),
    "Chapter plate: exploration vs exploitation",
    "Chapter plate. Uncertainty is the criterion. Source: original synthesis of the session.",
    "WITHOUT the rule: exploit only", "the stored object: the dial", "WITH the rule: UCB",
    [("after 10 pulls: A 3/10", 15, DK, 600), ("greedy: pull A forever", 15, MU, 500),
     ("0.3 per pull", 15, MU, 500), ("never finds C's 0.7", 15, MU, 500),
     ("the unknown stays unknown", 15, MU, 500)],
    [("epsilon 0.1: 10% random", 15, DK, 600), ("decay 1.0 -> 0.01", 15, MU, 500),
     ("UCB: value + bonus", 15, DK, 600), ("bonus = sqrt(2 ln t / n)", 15, MU, 500),
     ("Thompson: sample beliefs", 15, MU, 500), ("explore early, exploit late", 15, TE, 600)],
    [("t = 15: bonuses", 15, DK, 600), ("A 0.74, B 1.16, C 2.32", 16, DK, 600),
     ("scores: 1.04, 1.66, 2.32", 15, MU, 500), ("pull C: optimism wins", 15, TE, 600),
     ("no random pulls", 15, MU, 500), ("regret is logarithmic", 15, MU, 500)],
    [("Tradeoff: exploration costs reward now for knowledge later. Epsilon is practical; UCB is principled.", DK),
     ("Name the exploration strategy before the algorithm: no plan is a plan to get stuck.", MU)],
    "One connection: the uncertain arm wins by optimism, and the bonus is how optimism is priced.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l16-chap-values.svg"),
    "Chapter plate: values price the future",
    "Chapter plate. Greedy is blind; values see. Source: original synthesis of the session.",
    "WITHOUT the rule: greedy", "the stored object: the Bellman chain", "WITH the rule: act on V*",
    [("at 3: left -1, right -1", 15, DK, 600), ("tie: wander forever", 15, MU, 500),
     ("never reasons 9 rights", 15, MU, 500), ("discounted total: -10", 16, MU, 500),
     ("optimal now != optimal later", 15, MU, 500)],
    [("V(9)=8.0, V(8)=6.2", 15, DK, 600), ("V(7)=4.58, V(6)=3.12", 15, DK, 600),
     ("V(5)=1.81, V(4)=0.63", 15, DK, 600), ("V(3) = -0.43", 16, TE, 600),
     ("value now = reward + gamma x value later", 14, MU, 500), ("contraction: error x 0.9 per sweep", 14, MU, 500)],
    [("Q(3,right) = -0.43", 16, DK, 600), ("Q(3,left) = -2.25", 15, MU, 500),
     ("right wins by the numbers", 15, TE, 600), ("not profitable: least loss", 15, MU, 500),
     ("pi* = argmax Q*", 15, DK, 600), ("greedy on V* is optimal", 15, MU, 500)],
    [("Tradeoff: values turn delayed rewards into present prices, including bad news.", DK),
     ("Greedy on immediate reward is ruin; greedy on optimal values is optimal.", MU)],
    "One connection: the value at 3 is negative, and acting on that honest number still picks right.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l16-chap-experience.svg"),
    "Chapter plate: learn without the model",
    "Chapter plate. DP needs P and R; the world rarely obliges. Source: original synthesis of the session.",
    "WITHOUT the rule: DP", "the stored object: the estimators", "WITH the rule: Q-learning",
    [("model given: P, R known", 15, DK, 600), ("policy iteration: 2 rounds", 15, MU, 500),
     ("exact and unusable", 15, MU, 500), ("slippery floors unmodeled", 15, MU, 500),
     ("no learning from experience", 15, MU, 500)],
    [("MC: wait, average returns", 15, DK, 600), ("unbiased, high variance", 15, MU, 500),
     ("TD: bootstrap every step", 15, DK, 600), ("V(9): 0 -> 0.9 in one step", 15, TE, 600),
     ("biased, fast, online", 15, MU, 500), ("n-step: the dial between", 15, MU, 500)],
    [("Q(9,right): 0 -> 4.5", 15, DK, 600), ("next visit: 6.75 -> 9", 15, MU, 500),
     ("max = off-policy", 15, TE, 600), ("SARSA: honest about exploration", 15, MU, 500),
     ("DQN: replay + target net", 15, MU, 500), ("old data stays usable", 15, MU, 500)],
    [("Tradeoff: MC is unbiased and slow; TD is biased and fast; off-policy reuses data at the price of mismatch.", DK),
     ("Modern RL is TD almost everywhere. Tables converge; networks negotiate.", MU)],
    "One connection: every method is an operation on the return G_t, and REINFORCE weights by it.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l16-chap-reinforce.svg"),
    "Chapter plate: REINFORCE and the variance bill",
    "Chapter plate. Unbiased, wild, single-use. Source: original synthesis of the session.",
    "WITHOUT the rule: no gradient", "the stored object: the trick", "WITH the rule: calm it",
    [("values cannot improve pi_theta", 15, DK, 600), ("reward not differentiable", 15, MU, 500),
     ("future depends on own actions", 15, MU, 500), ("the gradient seems impossible", 15, MU, 500)],
    [("grad E[R] = E[R grad log pi]", 16, DK, 600), ("d p = p d log p: p cancels", 15, MU, 500),
     ("total 3: right up 0.12", 15, TE, 600), ("total -12: left falls", 15, MU, 500),
     ("reinforce the good actions", 15, DK, 600), ("stochastic policies only", 15, MU, 500)],
    [("baseline -4.5: 7.5, -7.5", 16, TE, 600), ("reward-to-go: cut past luck", 15, MU, 500),
     ("actor-critic: TD error 1.2", 15, DK, 600), ("theta 0.847 -> 0.883", 15, MU, 500),
     ("V(8): 5.0 -> 5.12", 15, MU, 500), ("advantage = return minus par", 15, MU, 500)],
    [("Tradeoff: the identity is exact, the estimator is wild. Subtract the predictable; learn from the surprise.", DK),
     ("On-policy: every gradient needs fresh trajectories. Old data is stale the moment theta moves.", MU)],
    "One connection: REINFORCE is the precursor because it never bootstraps, and everything after is variance repair.",
    panel_h=300, bottom_h=72)
print("l16 plates done")
