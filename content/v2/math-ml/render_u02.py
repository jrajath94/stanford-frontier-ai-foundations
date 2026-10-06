"""Render U02 figures, math-ml RUN 2. Palette from visual_system_generic.md.
matplotlib 3.6.3, Agg, dpi 150. All values computed, none invented."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/home/hatch/workspace/stanford-frontier-ai/v2-pack/math-ml/visuals/u02")
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

# f04: projection. b=[3,4], a=[1,0], p=[3,0], r=[0,4]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=BG)
fig.suptitle("f04: projection of b onto the line of a", color=INK, fontsize=15, fontweight=600)
for ax, ttl in ((ax1, "before: b = [3, 4], line a = [1, 0]"),
                (ax2, "after: shadow p = [3, 0], residual r = [0, 4]")):
    style(ax), ax.set_xlim(-0.5, 4.5), ax.set_ylim(-0.5, 4.5), ax.set_aspect("equal")
    ax.set_title(ttl, color=INK, fontsize=11), ax.grid(color=LINE, linewidth=0.5)
ax1.arrow(0, 0, 1, 0, head_width=0.12, color=TEAL, length_includes_head=True, linewidth=2)
ax1.text(1.05, 0.15, "a", color=TEAL, fontsize=12)
ax1.arrow(0, 0, 3, 4, head_width=0.12, color=FOCUS, length_includes_head=True, linewidth=2)
ax1.text(3.05, 4.0, "b", color=FOCUS, fontsize=12)
ax2.arrow(0, 0, 1, 0, head_width=0.12, color=TEAL, length_includes_head=True, linewidth=2)
ax2.arrow(0, 0, 3, 0, head_width=0.12, color=ORANGE, length_includes_head=True, linewidth=2)
ax2.text(3.05, 0.15, "p", color=ORANGE, fontsize=12)
ax2.arrow(3, 0, 0, 4, head_width=0.12, color=PINK, length_includes_head=True, linewidth=2, linestyle="--")
ax2.text(3.15, 2.0, "r, r.a = 0", color=INK, fontsize=10)
fig.text(0.5, 0.02, "original toy, computed values, r.a = 0.0 measured", ha="center",
         color=MUTED, fontsize=10)
fig.savefig(OUT / "f04_projection.png", dpi=150), plt.close(fig)

# f05: 90-degree rotation of the unit square grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=BG)
fig.suptitle("f05: rotation R (90 deg) applied to the unit square", color=INK,
             fontsize=15, fontweight=600)
R = np.array([[0., -1.], [1., 0.]])
corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
rot = (R @ corners.T).T
for ax, pts, ttl in ((ax1, corners, "before: unit square"),
                     (ax2, rot, "after: R @ square")):
    style(ax), ax.set_xlim(-1.5, 1.5), ax.set_ylim(-1.5, 1.5), ax.set_aspect("equal")
    ax.set_title(ttl, color=INK, fontsize=11), ax.grid(color=LINE, linewidth=0.5)
    ax.plot(pts[:, 0], pts[:, 1], color=FOCUS, linewidth=2)
ax2.plot([0, 0], [0, 1], color=TEAL, linewidth=2)
ax2.text(0.05, 1.05, "R@[1,0] = [0,1]", color=TEAL, fontsize=10)
fig.text(0.5, 0.02, "original toy, computed values, det(R) = 1.0, area preserved",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f05_transform.png", dpi=150), plt.close(fig)

# f08: SVD singular values [5, 3], kept vs dropped
fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=BG)
style(ax)
vals = [5.0, 3.0]
bars = ax.bar(["sigma 1 (kept)", "sigma 2 (dropped)"], vals,
              color=[TEAL, PINK], edgecolor=LINE, linewidth=1.5)
ax.set_ylim(0, 6), ax.set_ylabel("singular value", fontsize=12)
ax.set_title("f08: singular values of M, rank-1 truncation", color=INK,
             fontsize=13, fontweight=600)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v}",
            ha="center", color=INK, fontsize=12)
ax.text(1, 4.4, "rank-1 error = 3.0 = sigma 2", ha="center", color=INK, fontsize=11)
fig.text(0.5, 0.02, "original toy M = [[3,2,2],[2,3,-2]], computed values",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f08_svd_values.png", dpi=150), plt.close(fig)

# f10: covariance scatter + eigendirections
X = np.array([[1., 2.], [2., 3.], [3., 5.], [4., 6.]])
mean = X.mean(axis=0)
Xc = X - mean
C = Xc.T @ Xc / 3
eigvals, eigvecs = np.linalg.eigh(C)
fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
style(ax)
ax.set_title("f10: covariance eigendirections of 4 points", color=INK,
             fontsize=13, fontweight=600)
ax.scatter(X[:, 0], X[:, 1], color=FOCUS, s=60, zorder=3)
ax.set_xlabel("feature 1"), ax.set_ylabel("feature 2")
for i in range(2):
    d = eigvecs[:, i] * 2 * np.sqrt(eigvals[i])
    ax.arrow(mean[0], mean[1], d[0], d[1], head_width=0.12,
             color=TEAL if i == 1 else ORANGE, length_includes_head=True, linewidth=2)
    ax.text(mean[0] + d[0] * 1.05, mean[1] + d[1] * 1.05,
            f"eig {eigvals[i]:.2f}", color=INK, fontsize=10)
ax.text(1.1, 5.6, "eigvals 4.98, 0.02: data nearly 1-D", color=INK, fontsize=11)
fig.text(0.5, 0.02, "original toy, computed values, trace 5.0 = 4.9777 + 0.0223",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f10_covariance.png", dpi=150), plt.close(fig)

print("rendered:", sorted(p.name for p in OUT.glob("*.png")))
