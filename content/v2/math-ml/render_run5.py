#!/usr/bin/env python3
"""Render RUN 5 figures, math-ml. Palette from visual_system_generic.md.
matplotlib 3.6.3, Agg, dpi 150. All values computed, none invented.
Also executes the interview-u06 debug-task premise (T1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

U06 = Path("/home/hatch/workspace/stanford-frontier-ai/v2-pack/math-ml/visuals/u06")
U09 = Path("/home/hatch/workspace/stanford-frontier-ai/v2-pack/math-ml/visuals/u09")
U06.mkdir(parents=True, exist_ok=True); U09.mkdir(parents=True, exist_ok=True)

BG = "#F7F4EE"; INK = "#1B2838"; MUTED = "#5C6B7A"; LINE = "#D9D3C7"
PANEL = "#FFFDF8"; TEAL = "#1F7A72"; ORANGE = "#C46B2C"; FOCUS = "#1E4D8C"
PINK = "#F3D4D8"; GREEN = "#D9E8D3"; PURP = "#6A4C93"

def style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)

FOOT = "original toy, computed 2026-10-06, numpy 1.26.4 float64, seed 7"

# u06 f01: OLS fit, x=[0,1,2,3], y=[0,1,3,2], w_hat=[0.3,0.8]
x = np.array([0., 1., 2., 3.]); y = np.array([0., 1., 3., 2.])
b_, w_ = 0.3, 0.8
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG); style(ax)
ax.scatter(x, y, s=90, color=FOCUS, edgecolors=INK, zorder=3, label="data (4 points)")
xg = np.linspace(-0.2, 3.2, 50)
ax.plot(xg, b_ + w_ * xg, color=ORANGE, lw=2.5, label="OLS line: 0.3 + 0.8 x, RSS 1.8")
for xi, yi_ in zip(x, y):
    ax.plot([xi, xi], [yi_, b_ + w_ * xi], color=PINK, lw=2)
ax.set_title("f01: the OLS line minimizes the sum of squared residuals", fontsize=13, fontweight=600)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(U06 / "f01_ols_fit.png", dpi=150); plt.close(fig)

# u06 f02: logistic step on x=[0,1], y=[0,1]; w 0 -> 0.25, loglik -1.3863 -> -1.2691
sig = lambda z: 1 / (1 + np.exp(-z))
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG); style(ax)
zg = np.linspace(-3, 3, 200)
ax.plot(zg, sig(zg), color=TEAL, lw=2.5, label="sigmoid")
ax.scatter([0.], [0.5], s=110, color=MUTED, edgecolors=INK, zorder=3,
           label="w=0: p=[0.5, 0.5], loglik -1.3863")
ax.scatter([0., 0.25], [0.5, 0.562177], s=110, color=ORANGE, edgecolors=INK, zorder=3,
           label="w=0.25: p=[0.5, 0.5622], loglik -1.2691 (dots sit on the curve)")
ax.set_title("f02: one gradient step lifts the likelihood", fontsize=13, fontweight=600)
ax.set_xlabel("z = w x"); ax.set_ylabel("P(y=1)")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(U06 / "f02_logistic_step.png", dpi=150); plt.close(fig)

# u06 f03: margin plate. Points, w=[1,1], b=0, margin 2/||w|| = 1.4142
Xc = np.array([[-1., 0.], [0., -1.], [1., 1.], [2., 2.]])
yc = np.array([-1., -1., 1., 1.])
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG); style(ax)
neg = Xc[yc == -1]; pos = Xc[yc == 1]
ax.scatter(neg[:, 0], neg[:, 1], s=120, color=TEAL, edgecolors=INK, zorder=3, label="y = -1")
ax.scatter(pos[:, 0], pos[:, 1], s=120, color=ORANGE, edgecolors=INK, zorder=3, label="y = +1")
xg = np.linspace(-1.6, 2.6, 50)
ax.plot(xg, -xg, color=INK, lw=2, label="boundary: x1 + x2 = 0")
ax.plot(xg, -xg + 1, color=MUTED, lw=1.5, ls="--", label="margin lines, width 1.4142")
ax.plot(xg, -xg - 1, color=MUTED, lw=1.5, ls="--")
ax.set_title("f03: the max-margin boundary sits halfway between the two classes",
             fontsize=13, fontweight=600)
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(U06 / "f03_margin.png", dpi=150); plt.close(fig)

# u06 f04: Gram heatmap, gamma=0.5, eigenvalues 0.3911 / 0.9656 / 1.6433
K = np.array([[1., 0.606531, 0.135335], [0.606531, 1., 0.082085],
              [0.135335, 0.082085, 1.]])
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG); style(ax)
im = ax.imshow(K, cmap="Greens", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{K[i, j]:.3f}", ha="center", va="center",
                color=INK if K[i, j] < 0.6 else "white", fontsize=11)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_title("f04: the Gram matrix is PSD, eigenvalues all positive",
             fontsize=13, fontweight=600)
ax.set_xlabel("point index"); ax.set_ylabel("point index")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.text(0.5, 0.01, FOOT + ", eig = 0.3911, 0.9656, 1.6433",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(U06 / "f04_kernel_gram.png", dpi=150); plt.close(fig)

# u09 f01: k-means, 6 points, Lloyd path to obj 0.1333
Xk = np.array([[0., 0.], [0.2, 0.1], [-0.1, 0.2], [5., 5.], [5.2, 4.9], [4.9, 5.1]])
mu_hist = [Xk[[0, 3]].copy()]
mu = Xk[[0, 3]].copy()
for _ in range(3):
    d = np.sum((Xk[:, None, :] - mu[None, :, :]) ** 2, axis=2)
    a = d.argmin(axis=1)
    mu = np.array([Xk[a == k_].mean(axis=0) for k_ in range(2)])
    mu_hist.append(mu.copy())
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG); style(ax)
ax.scatter(Xk[:3, 0], Xk[:3, 1], s=120, color=TEAL, edgecolors=INK, zorder=3, label="cluster 0")
ax.scatter(Xk[3:, 0], Xk[3:, 1], s=120, color=ORANGE, edgecolors=INK, zorder=3, label="cluster 1")
for k_ in range(2):
    path = np.array([m[k_] for m in mu_hist])
    ax.plot(path[:, 0], path[:, 1], color=PURP, lw=1.5, ls="--", marker="x", ms=8)
    ax.scatter([mu[k_, 0]], [mu[k_, 1]], s=200, color=PURP, edgecolors=INK,
               marker="D", zorder=4)
ax.text(mu[0, 0] - 0.7, mu[0, 1] + 0.25, "center 0", color=PURP, fontsize=10)
ax.text(mu[1, 0] - 0.9, mu[1, 1] + 0.35, "center 1", color=PURP, fontsize=10)
ax.set_title("f01: Lloyd iterations move the centers to the cluster means",
             fontsize=13, fontweight=600)
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", objective 0.1700 -> 0.1333 in 1 step",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(U09 / "f01_kmeans.png", dpi=150); plt.close(fig)

# u09 f02: PCA, PC1 [0.5512, 0.8344], recon of one point with k=1
Xp = np.array([[1., 2.], [2., 3.], [3., 5.], [4., 6.], [5., 8.]])
m = Xp.mean(0)
v1 = np.array([0.551163, 0.834398])
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG); style(ax)
ax.scatter(Xp[:, 0], Xp[:, 1], s=110, color=FOCUS, edgecolors=INK, zorder=3, label="data")
t = np.linspace(-3.5, 3.5, 50)
ax.plot(m[0] + t * v1[0], m[1] + t * v1[1], color=ORANGE, lw=2.5,
        label="PC1: 99.72% of variance")
z = (Xp[0] - m) @ v1
pr = m + z * v1
ax.plot([Xp[0, 0], pr[0]], [Xp[0, 1], pr[1]], color=PINK, lw=3, label="one dropped error")
ax.scatter([pr[0]], [pr[1]], s=130, color=GREEN, edgecolors=INK, zorder=4,
           label="reconstruction (k=1)")
# zoom on the first point so the tiny dropped error is visible
cx, cy = (Xp[0, 0] + pr[0]) / 2, (Xp[0, 1] + pr[1]) / 2
ax.set_xlim(cx - 0.35, cx + 0.35); ax.set_ylim(cy - 0.35, cy + 0.35)
ax.set_title("f02: PCA drops the error, here zoomed on one point",
             fontsize=13, fontweight=600)
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", recon MSE 0.009172 = dropped eigval / 2",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(U09 / "f02_pca.png", dpi=150); plt.close(fig)

# u09 f03: held-out inertia by k
ks = [1, 2, 3, 5]
tr = [7.6417, 0.3698, 0.3161, 0.1789]
te = [14.1768, 0.4439, 0.3360, 0.2939]
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG); style(ax)
xa = np.arange(len(ks))
ax.bar(xa - 0.2, tr, 0.4, color=TEAL, edgecolor=INK, label="train inertia (falls)")
ax.bar(xa + 0.2, te, 0.4, color=ORANGE, edgecolor=INK, label="held-out inertia (flattens)")
ax.set_xticks(xa); ax.set_xticklabels([str(k) for k in ks])
for i in range(len(ks)):
    ax.text(i + 0.2, te[i] + 0.35, f"{te[i]:.2f}", ha="center", color=INK, fontsize=9)
ax.set_title("f03: the elbow is at k=2, extra centers buy little on held-out data",
             fontsize=13, fontweight=600)
ax.set_xlabel("k"); ax.set_ylabel("inertia")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", 60 train / 20 held-out points, seed 7",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(U09 / "f03_heldout.png", dpi=150); plt.close(fig)

print("rendered 7 PNGs")
# ---- interview-u06 debug T1 premise execution (measured 2026-10-06) ----
p3 = np.array([[0., 0.], [1., 0.], [0., 2.]])
D2t = np.sum((p3[:, None, :] - p3[None, :, :]) ** 2, axis=2)
b3 = np.array([1., 2., 3.])
for g in (1e-6, 0.5):
    K_ = np.exp(-g * D2t)
    a_ = np.linalg.solve(K_, b3)
    print(f"T1 premise gamma={g}: cond={np.linalg.cond(K_):.3e} "
          f"alpha={np.round(a_, 4).tolist()} residual={np.linalg.norm(K_ @ a_ - b3):.2e}")
Ktiny = np.exp(-1e-6 * D2t)
a_jit = np.linalg.solve(Ktiny + 1e-6 * np.eye(3), b3)
a_raw = np.linalg.solve(Ktiny, b3)
print("T1 premise jitter: ||alpha|| =", round(float(np.linalg.norm(a_jit)), 1),
      "vs no-jitter ||alpha|| =", round(float(np.linalg.norm(a_raw)), 1))
