#!/usr/bin/env python3
"""Render completion-run figures, math-ml U07/U08/U10.
Palette from visual_system_generic.md. matplotlib 3.6.3, Agg, dpi 150.
All values computed, none invented."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
from math import log, erf, sqrt
from pathlib import Path

BASE = Path("/home/hatch/workspace/stanford-frontier-ai/v2-pack/math-ml/visuals")
for u in ["u07", "u08", "u10"]:
    (BASE / u).mkdir(parents=True, exist_ok=True)

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

# ---- u07 f01: decision regions of a depth-2 tree ----
pts = np.array([[0.2,0.2,0],[0.3,0.7,0],[0.2,0.8,0],[0.7,0.3,1],[0.8,0.8,1],[0.7,0.7,0]])
gx, gy = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
pred = np.where(gx <= 0.5, 0, np.where(gy <= 0.5, 1, 1))
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
ax.pcolormesh(gx, gy, pred, cmap=matplotlib.colors.ListedColormap([GREEN, PINK]),
              shading="auto", alpha=0.7, vmin=0, vmax=1)
ax.scatter(pts[pts[:,2]==0,0], pts[pts[:,2]==0,1], s=140, color=TEAL, edgecolors=INK,
           zorder=3, label="class 0")
ax.scatter(pts[pts[:,2]==1,0], pts[pts[:,2]==1,1], s=140, color=ORANGE, edgecolors=INK,
           zorder=3, label="class 1")
ax.plot([0.5,0.5],[0,1], color=INK, lw=2.5)
ax.plot([0.5,1.0],[0.5,0.5], color=INK, lw=2.5)
ax.set_title("f01: a depth-2 tree cuts the plane into three rectangles", fontsize=13, fontweight=600)
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", noisy point absorbed by region x1>0.5,x2>0.5",
         ha="center", color=MUTED, fontsize=9)
fig.savefig(BASE/"u07"/"f01_regions.png", dpi=150); plt.close(fig)

# ---- u07 f02: impurity curves ----
p = np.linspace(0.001, 0.999, 300)
gini_c = 2*p*(1-p); ent_c = -(p*np.log(p)+(1-p)*np.log(1-p)); mis_c = np.minimum(p, 1-p)
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
ax.plot(p, gini_c, color=TEAL, lw=2.5, label="Gini 2p(1-p)")
ax.plot(p, ent_c, color=ORANGE, lw=2.5, label="entropy (nats)")
ax.plot(p, mis_c, color=PURP, lw=2.5, label="misclassification")
for pv, gv, ev, mv in [(0.2, 0.32, 0.5004, 0.2)]:
    ax.scatter([pv]*3, [gv, ev, mv], s=80, color=INK, zorder=3)
    ax.text(pv+0.02, gv+0.02, "Gini 0.32", color=TEAL, fontsize=10)
    ax.text(pv+0.02, ev+0.02, "entropy 0.5004", color=ORANGE, fontsize=10)
    ax.text(pv+0.02, mv-0.06, "miscls 0.2", color=PURP, fontsize=10)
ax.set_title("f02: three impurity scores agree on pure nodes, differ in the middle",
             fontsize=13, fontweight=600)
ax.set_xlabel("p (class-1 share)"); ax.set_ylabel("impurity")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u07"/"f02_impurity.png", dpi=150); plt.close(fig)

# ---- u07 f03: AdaBoost weights round 0 -> 1 ----
w0 = [0.25]*4; w1 = [1/6, 1/6, 1/6, 0.5]
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
xs = np.arange(4); wd = 0.35
ax.bar(xs-wd/2, w0, wd, color=MUTED, edgecolor=INK, label="round 0: 0.25 each")
ax.bar(xs+wd/2, w1, wd, color=ORANGE, edgecolor=INK, label="round 1: wrong point 0.5, rest 1/6")
ax.set_xticks(xs); ax.set_xticklabels(["x=0", "x=1", "x=2", "x=3 (missed)"])
ax.set_title("f03: AdaBoost moves half the weight onto the missed point",
             fontsize=13, fontweight=600)
ax.set_ylabel("weight"); ax.set_xlabel("point")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", alpha 0.5493", ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u07"/"f03_adaboost_weights.png", dpi=150); plt.close(fig)

# ---- u07 f04: bagged stump val MSE vs B ----
Bv = [1, 5, 25, 100]; mse = [0.0513, 0.0278, 0.0344, 0.0335]
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
ax.plot(Bv, mse, color=TEAL, lw=2.5, marker="o", ms=8, label="bagged stumps, val MSE")
ax.axhline(0.0421, color=ORANGE, lw=2, ls="--", label="single stump on full train: 0.0421")
ax.set_xscale("log"); ax.set_xticks(Bv); ax.set_xticklabels([str(b) for b in Bv])
ax.set_title("f04: averaging stumps cuts validation error, then flattens",
             fontsize=13, fontweight=600)
ax.set_xlabel("B (trees)"); ax.set_ylabel("val MSE")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", curve is not monotone on 6 val points",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u07"/"f04_bagging_mse.png", dpi=150); plt.close(fig)

# ---- u08 f01: attention heatmap ----
A = np.array([[0.3837, 0.2327, 0.3837],[0.2327, 0.3837, 0.3837]])
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=0.5)
for i in range(2):
    for j in range(3):
        ax.text(j, i, f"{A[i,j]:.3f}", ha="center", va="center",
                color="white" if A[i,j] > 0.3 else INK, fontsize=12)
ax.set_xticks(range(3)); ax.set_xticklabels(["key 0", "key 1", "key 2"])
ax.set_yticks(range(2)); ax.set_yticklabels(["query 0", "query 1"])
ax.set_title("f01: attention weights are a soft lookup over keys", fontsize=13, fontweight=600)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.text(0.5, 0.01, FOOT + ", rows sum to 1.0", ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u08"/"f01_attention.png", dpi=150); plt.close(fig)

# ---- u08 f02: gradient powers ----
t = np.arange(0, 11)
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
ax.plot(t, 0.5**t, color=TEAL, lw=2.5, marker="o", ms=6, label="w=0.5: 0.5^10 = 9.8e-4 (dies)")
ax.plot(t, 1.5**t, color=ORANGE, lw=2.5, marker="o", ms=6, label="w=1.5: 1.5^10 = 57.67 (blows up)")
ax.set_yscale("log")
ax.set_title("f02: one scalar decides whether the gradient dies or explodes",
             fontsize=13, fontweight=600)
ax.set_xlabel("time steps"); ax.set_ylabel("gradient factor")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u08"/"f02_grad_powers.png", dpi=150); plt.close(fig)

# ---- u08 f03: MLP forward schematic with values ----
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG)
ax.set_facecolor(PANEL); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, wdt, hgt, txt, fc):
    r = mp.FancyBboxPatch((x, y), wdt, hgt, boxstyle="round,pad=0.08,rounding_size=0.25",
                          facecolor=fc, edgecolor=INK, lw=1.5)
    ax.add_patch(r); ax.text(x+wdt/2, y+hgt/2, txt, ha="center", va="center",
                             color=INK, fontsize=10, fontweight=500)
box(0.4, 2.4, 1.7, 1.4, "x\n[1, 2]", "#E7F1F8")
box(3.6, 0.7, 2.2, 4.8, "z1 = W1 x + b1\n[1.0, 1.0, 1.5]\na1 = relu\n[1.0, 1.0, 1.5]", "#E7F4EF")
box(7.6, 2.4, 1.9, 1.4, "out\nW2 a1 + b2\n0.0", "#F4E6D4")
ax.annotate("", xy=(3.6, 3.1), xytext=(2.1, 3.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
ax.annotate("", xy=(7.6, 3.1), xytext=(5.8, 3.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
ax.text(2.85, 3.45, "W1 (3,2)", color=MUTED, fontsize=10, ha="center")
ax.text(6.7, 3.45, "W2 (1,3)", color=MUTED, fontsize=10, ha="center")
ax.set_title("f03: one forward pass, every shape and value on the plate",
             fontsize=13, fontweight=600, color=INK)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u08"/"f03_mlp_forward.png", dpi=150); plt.close(fig)

# ---- u08 f04: Adam vs SGD one step on a quadratic ----
g1, g2 = np.linspace(-0.2, 1.4, 120), np.linspace(-0.2, 1.4, 120)
G1, G2 = np.meshgrid(g1, g2); J = 0.5*(G1**2 + 4*G2**2)
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
ax.contour(G1, G2, J, levels=12, colors=LINE)
ax.scatter([1], [1], s=120, color=INK, zorder=3, label="start (1,1), grad (1,4)")
ax.annotate("", xy=(0.9, 0.6), xytext=(1, 1), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5))
ax.text(0.86, 0.62, "SGD eta .1: (-0.1,-0.4)", color=TEAL, fontsize=10)
ax.annotate("", xy=(0.9, 0.9), xytext=(1, 1), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5))
ax.text(0.62, 0.93, "Adam eta .1: (-0.1,-0.1)", color=ORANGE, fontsize=10)
ax.set_title("f04: Adam normalizes each coordinate, SGD follows the raw slope",
             fontsize=13, fontweight=600)
ax.set_xlabel("w1"); ax.set_ylabel("w2"); ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", first update from zero state",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u08"/"f04_adam_sgd.png", dpi=150); plt.close(fig)

# ---- u10 f01: ELBO gap bars ----
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
bars = ax.bar(["log p(x)", "ELBO"], [-0.7340, -0.8779], color=[FOCUS, ORANGE],
       edgecolor=INK, width=0.5)
ax.text(0, -0.68, "-0.7340", ha="center", color="white", fontsize=11, fontweight=600)
ax.text(1, -0.82, "-0.8779", ha="center", color="white", fontsize=11, fontweight=600)
ax.annotate("", xy=(1, -0.75), xytext=(1, -0.86),
            arrowprops=dict(arrowstyle="<->", color=PURP, lw=2))
ax.text(1.06, -0.81, "gap 0.1438\n= KL(q||posterior)", color=PURP, fontsize=10)
ax.set_title("f01: the ELBO sits below the true log-likelihood by the KL gap",
             fontsize=13, fontweight=600)
ax.set_ylabel("nats")
fig.text(0.5, 0.01, FOOT + ", toy: p(x=1)=0.48, q=[0.5,0.5]",
         ha="center", color=MUTED, fontsize=9)
fig.savefig(BASE/"u10"/"f01_elbo_gap.png", dpi=150); plt.close(fig)

# ---- u10 f02: mixture density + ancestral samples ----
Phi = lambda tt: 0.5*(1+erf(tt/sqrt(2)))
xs = [0.0084, 1.0601, 2.3402, 0.5078, 0.3795, 1.4898, 1.3569, -0.8946, -1.9305, -1.0293]
xg = np.linspace(-4, 4, 300)
dens = 0.5*np.exp(-0.5*(xg+1)**2)/sqrt(2*np.pi) + 0.5*np.exp(-0.5*(xg-1)**2)/sqrt(2*np.pi)
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
ax.plot(xg, dens, color=TEAL, lw=2.5, label="true density 0.5 N(-1,1) + 0.5 N(1,1)")
ax.scatter(xs, [0.01]*10, s=90, color=ORANGE, edgecolors=INK, zorder=3,
           label="10 ancestral samples (seed 7)")
ax.set_title("f02: ancestral sampling draws z first, then x given z",
             fontsize=13, fontweight=600)
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u10"/"f02_mixture_samples.png", dpi=150); plt.close(fig)

# ---- u10 f03: empirical vs true CDF ----
xs_s = np.sort(xs)
emp = np.arange(1, 11)/10
xt = np.linspace(-4, 4, 300)
true_cdf = 0.5*0.5*(1+np.vectorize(lambda tt: erf(tt/sqrt(2)))(xt+1)) + \
           0.5*0.5*(1+np.vectorize(lambda tt: erf(tt/sqrt(2)))(xt-1))
fig, ax = plt.subplots(figsize=(8.4,4.6), facecolor=BG); style(ax)
ax.plot(xt, true_cdf, color=TEAL, lw=2.5, label="true CDF")
ax.step(np.concatenate([[-4], xs_s]), np.concatenate([[0], emp]), where="post",
        color=ORANGE, lw=2.5, label="empirical CDF, 10 samples")
ax.set_title("f03: ten samples give a coarse CDF, the truth is smooth",
             fontsize=13, fontweight=600)
ax.set_xlabel("x"); ax.set_ylabel("P(X <= x)")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", empirical CDF at 0: 0.3 vs true 0.5",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(BASE/"u10"/"f03_cdf.png", dpi=150); plt.close(fig)

print("rendered 11 figures")
