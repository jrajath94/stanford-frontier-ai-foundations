"""Render U04a figures, math-ml RUN 2. Palette from visual_system_generic.md.
matplotlib 3.6.3, Agg, dpi 150. All values computed, none invented."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/home/hatch/workspace/stanford-frontier-ai/v2-pack/math-ml/visuals/u04")
OUT.mkdir(parents=True, exist_ok=True)

BG = "#F7F4EE"
INK = "#1B2838"
MUTED = "#5C6B7A"
LINE = "#D9D3C7"
PANEL = "#FFFDF8"
TEAL = "#1F7A72"
ORANGE = "#C46B2C"
FOCUS = "#1E4D8C"
PINK = "#F3D4D8"

def style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.label.set_color(INK), ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)

rng = np.random.default_rng(7)
draws = rng.binomial(1, 0.3, size=8)  # [0 1 1 0 0 1 0 1]

# f01: sample vs truth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=BG)
fig.suptitle("f01: 8 draws (seed 7) vs the true Bernoulli(0.3) PMF", color=INK,
             fontsize=14, fontweight=600)
style(ax1)
ax1.bar([0, 1], [0.7, 0.3], color=TEAL, edgecolor=LINE, linewidth=1.5, width=0.6)
ax1.set_ylim(0, 1), ax1.set_xticks([0, 1])
ax1.set_title("truth: PMF(0) = 0.7, PMF(1) = 0.3", color=INK, fontsize=11)
style(ax2)
ax2.bar([0, 1], [0.5, 0.5], color=PINK, edgecolor=LINE, linewidth=1.5, width=0.6)
ax2.set_ylim(0, 1), ax2.set_xticks([0, 1])
ax2.set_title("sample: mean 0.5, miss by 0.2", color=INK, fontsize=11)
fig.text(0.5, 0.02, "original toy, computed values, draws [0 1 1 0 0 1 0 1]",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f01_iid_sample.png", dpi=150), plt.close(fig)

# f02: histogram density
xs = np.array([0.2, 0.5, 0.7, 1.1, 1.4, 1.6, 2.0, 2.3])
counts, edges = np.histogram(xs, bins=4, range=(0, 2.5))
width = edges[1] - edges[0]
density = counts / (len(xs) * width)
fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=BG)
style(ax)
bars = ax.bar(range(4), density, color=TEAL, edgecolor=LINE, linewidth=1.5, width=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels([f"[{edges[i]:.3g}, {edges[i+1]:.3g})" for i in range(4)])
ax.set_ylabel("density (count / (n * width))", fontsize=11)
ax.set_title("f02: histogram density of 8 points, counts [2, 2, 2, 2]",
             color=INK, fontsize=13, fontweight=600)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={c}",
            ha="center", color=INK, fontsize=11)
fig.text(0.5, 0.02, "original toy, computed values, counts sum to 8 = n",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f02_histogram_density.png", dpi=150), plt.close(fig)

print("rendered:", sorted(p.name for p in OUT.glob("*.png")))
