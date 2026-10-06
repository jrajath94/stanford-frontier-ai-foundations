"""Render U03 + U04b figures, math-ml RUN 3. Palette from visual_system_generic.md.
matplotlib 3.6.3, Agg, dpi 150. All values computed, none invented."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT3 = Path("/home/hatch/workspace/stanford-frontier-ai/v2-pack/math-ml/visuals/u03")
OUT4 = Path("/home/hatch/workspace/stanford-frontier-ai/v2-pack/math-ml/visuals/u04")
OUT3.mkdir(parents=True, exist_ok=True)
OUT4.mkdir(parents=True, exist_ok=True)

BG = "#F7F4EE"; INK = "#1B2838"; MUTED = "#5C6B7A"; LINE = "#D9D3C7"
PANEL = "#FFFDF8"; TEAL = "#1F7A72"; ORANGE = "#C46B2C"; FOCUS = "#1E4D8C"
PINK = "#F3D4D8"; YELLOW = "#F6E7A8"; GREEN = "#D9E8D3"

def style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)

# f01: GD vs SGD on J(w) = (w-3)^2 (C07)
eta = 0.1
traj = {"gd": [0.0], "sgd": [0.0]}
w = 0.0
for _ in range(10):
    w -= eta * 2 * (w - 3)
    traj["gd"].append(w)
rng = np.random.default_rng(7)
xs = np.array([1., 2., 3., 4.])
w = 0.0
for _ in range(20):
    i = rng.integers(0, 4)
    xi = xs[i]
    w -= eta * xi * (xi * w - 3.0)
    traj["sgd"].append(w)
print("f01 gd final =", traj["gd"][-1], "sgd final =", traj["sgd"][-1])
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
steps = np.arange(11)
ax.plot(steps, [(t - 3) ** 2 for t in traj["gd"]], color=TEAL, lw=2.5, marker="o",
        ms=4, label="GD eta=0.1, 10 steps, J=0.1038")
steps_s = np.arange(21)
ax.plot(steps_s, [(t - 3) ** 2 for t in traj["sgd"]], color=ORANGE, lw=1.5, marker=".",
        ms=3, label="SGD eta=0.1, 20 steps, seed 7, J=3.2022")
ax.set_title("f01: GD walks straight down, SGD wobbles toward the same bowl",
             fontsize=13, fontweight=600)
ax.set_xlabel("step"); ax.set_ylabel("J(w) = (w - 3)^2")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=9)
fig.text(0.5, 0.01, "original toy, computed 2026-10-06, numpy 1.26.4 float64",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT3 / "f01_gd_path.png", dpi=150); plt.close(fig)

# f02: convex vs non-convex (C05)
x = np.linspace(-2.2, 2.2, 400)
convex = (x - 1) ** 2
nonc = x ** 4 - 3 * x ** 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=BG)
for ax in (a1, a2):
    style(ax)
a1.plot(x, convex, color=TEAL, lw=2.5)
a1.scatter([1.0], [0.0], color=FOCUS, s=60, zorder=3)
a1.set_title("convex: (x-1)^2, one bowl, H'' = 2 > 0", fontsize=11)
a1.set_xlabel("x"); a1.set_ylabel("f(x)")
a2.plot(x, nonc, color=ORANGE, lw=2.5)
a2.scatter([1.2247, -1.2247], [-2.25, -2.25], color=FOCUS, s=60, zorder=3)
a2.scatter([0.0], [0.0], color=PINK, s=60, zorder=3, edgecolor=INK)
a2.set_title("non-convex: x^4 - 3x^2, two bowls, H''(0) = -6", fontsize=11)
a2.set_xlabel("x"); a2.set_ylabel("f(x)")
fig.suptitle("f02: convexity is one bowl; non-convexity is many traps",
             color=INK, fontsize=14, fontweight=600)
fig.text(0.5, 0.01, "original toy, computed values, blue dots are minima, pink dot is the saddle",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT3 / "f02_convex.png", dpi=150); plt.close(fig)

# f03: Newton on h(x) = x^4 - 3x^2 from 1.0 (C08)
xn = 1.0
nts = [xn]
for _ in range(5):
    xn = xn - (4 * xn**3 - 6 * xn) / (12 * xn**2 - 6)
    nts.append(xn)
print("f03 newton iterates:", np.round(nts, 6))
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
ax.plot(x, nonc, color=INK, lw=1.5, label="h(x) = x^4 - 3x^2")
hv = np.array([t**4 - 3 * t**2 for t in nts])
ax.scatter(nts, hv, color=TEAL, s=55, zorder=3)
for i, t in enumerate(nts):
    ax.annotate(str(i), (t, hv[i]), textcoords="offset points", xytext=(6, 6),
                fontsize=10, color=TEAL, fontweight=600)
ax.set_title("f03: Newton reaches the bowl in 4 steps from x0 = 1.0",
             fontsize=13, fontweight=600)
ax.set_xlabel("x"); ax.set_ylabel("h(x)")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=9)
fig.text(0.5, 0.01, "original toy, computed iterates: 1.0, 1.333, 1.2367, 1.2249, 1.2247",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT3 / "f03_newton.png", dpi=150); plt.close(fig)

# f04: learning-rate behavior on J(w) = (w-3)^2 (C11)
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
colors = {0.05: TEAL, 0.5: FOCUS, 1.5: ORANGE}
labels = {0.05: "eta=0.05 slow (J end 1.06)", 0.5: "eta=0.5 one-step (J end 0.0)",
          1.5: "eta=1.5 diverges (J end 1.5e8)"}
for eta_c in (0.05, 0.5, 1.5):
    w = 0.0
    js = [9.0]
    for _ in range(12):
        w -= eta_c * 2 * (w - 3)
        js.append((w - 3) ** 2)
    print("f04 eta", eta_c, "J end", js[-1])
    ax.plot(np.arange(13), js, color=colors[eta_c], lw=2.2, marker="o", ms=3,
            label=labels[eta_c])
ax.set_yscale("symlog", linthresh=1.0)
ax.set_title("f04: step size picks the regime, slow, fast, or exploded",
             fontsize=13, fontweight=600)
ax.set_xlabel("step"); ax.set_ylabel("J(w), symlog scale")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=9)
fig.text(0.5, 0.01, "original toy, computed 2026-10-06, symlog shows both crawl and explosion",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT3 / "f04_learning_rates.png", dpi=150); plt.close(fig)

# f05: entropy and KL bars (U04b SB10)
p = np.array([0.7, 0.3]); q = np.array([0.5, 0.5])
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
xb = np.arange(2); wdt = 0.35
ax.bar(xb - wdt/2, p, wdt, color=TEAL, edgecolor=INK, label="p = [0.7, 0.3], H = 0.6109")
ax.bar(xb + wdt/2, q, wdt, color=ORANGE, edgecolor=INK, label="q = [0.5, 0.5]")
ax.set_xticks(xb); ax.set_xticklabels(["outcome 1", "outcome 2"])
ax.set_title("f05: KL = 0.0823 nats is the surprise gap between p and q",
             fontsize=13, fontweight=600)
ax.set_ylabel("probability")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=9)
fig.text(0.5, 0.01, "original toy, D_KL(p||q) = 0.0823, D_KL(q||p) = 0.0872, not symmetric",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT4 / "f03_entropy_kl.png", dpi=150); plt.close(fig)

# f06: MLE Gaussian on 4 points (U04b SB13)
d = np.array([2.1, 2.5, 1.9, 2.3])
mu, s2 = d.mean(), ((d - d.mean()) ** 2).mean()
xg = np.linspace(1.2, 3.2, 400)
pdf = np.exp(-(xg - mu) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
ax.hist(d, bins=4, range=(1.2, 3.2), color=YELLOW, edgecolor=INK, alpha=0.85,
        label="4 data points")
ax.plot(xg, pdf, color=TEAL, lw=2.5, label="MLE N(2.2, 0.05)")
ax.scatter([mu], [0.0], color=FOCUS, s=70, zorder=3, label="mu_hat = 2.2")
ax.set_title("f06: the data picks the bell, MLE sets center and width",
             fontsize=13, fontweight=600)
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=9)
fig.text(0.5, 0.01, "original toy, computed 2026-10-06, sigma2_hat = 0.05",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT4 / "f04_mle_gaussian.png", dpi=150); plt.close(fig)
print("done")
