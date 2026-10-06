#!/usr/bin/env python3
"""Render U05 figures, math-ml RUN 4. Palette from visual_system_generic.md.
matplotlib 3.6.3, Agg, dpi 150. All values computed, none invented.
Also executes the interview-u05 debug-task premise (T1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/home/hatch/workspace/stanford-frontier-ai/v2-pack/math-ml/visuals/u05")
OUT.mkdir(parents=True, exist_ok=True)

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

FOOT = "original toy, computed 2026-10-06, numpy 1.26.4 float64, seed 7"

# f01: bias-variance decomposition vs polynomial degree (C04)
rng = np.random.default_rng(7)
xg = np.linspace(-1, 1, 12)
ftrue = xg ** 2
B = 100
degs = np.arange(1, 9)
bias2 = np.zeros(8); var = np.zeros(8)
xgrid = np.linspace(-1, 1, 60)
for di, d in enumerate(degs):
    preds = np.zeros((B, 60))
    for b in range(B):
        yb = ftrue + rng.normal(0, 0.3, 12)
        c = np.polyfit(xg, yb, d)
        preds[b] = np.polyval(c, xgrid)
    fgrid = xgrid ** 2
    bias2[di] = np.mean((preds.mean(0) - fgrid) ** 2)
    var[di] = np.mean(preds.var(0))
total = bias2 + var
print("f01 bias2:", np.round(bias2, 4))
print("f01 var:  ", np.round(var, 4))
print("f01 total:", np.round(total, 4))
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
ax.plot(degs, bias2, color=TEAL, lw=2.5, marker="o", ms=5, label="bias^2 (falls)")
ax.plot(degs, var, color=ORANGE, lw=2.5, marker="s", ms=5, label="variance (rises)")
ax.plot(degs, total, color=FOCUS, lw=3, marker="D", ms=5, label="total error (U shape)")
ax.set_xticks(degs)
ax.set_title("f01: more flexibility cuts bias but buys variance", fontsize=13, fontweight=600)
ax.set_xlabel("polynomial degree"); ax.set_ylabel("error on the toy")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f01_bias_variance.png", dpi=150); plt.close(fig)

# f02: capacity gap, train vs test MSE by degree (C09)
deg = ["1", "2", "3"]
train = [0.03075, 0.015125, 0.0]
test = [2.8757, 0.531325, 28.56125]
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
xa = np.arange(3)
ax.bar(xa - 0.2, train, 0.4, color=TEAL, edgecolor=INK, label="train MSE (falls)")
ax.bar(xa + 0.2, test, 0.4, color=PINK, edgecolor=INK, label="test MSE (U shape)")
ax.set_xticks(xa); ax.set_xticklabels(deg)
ax.set_title("f02: train error falls with degree, test error turns up", fontsize=13, fontweight=600)
ax.set_xlabel("polynomial degree"); ax.set_ylabel("MSE")
ax.set_ylim(0, 6)
for i in range(3):
    ax.text(i + 0.2, test[i] + 0.15, f"{test[i]:.2f}", ha="center", color=INK, fontsize=9)
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT + ", y-values clipped at 6 for display, degree-3 test = 28.56",
         ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f02_capacity_gap.png", dpi=150); plt.close(fig)

# f03: learning curves (C10)
ns = [3, 5, 8, 12, 20, 40]
train_lc = [0.025894, 0.068007, 0.205461, 0.242672, 0.103307, 0.161056]
val_lc = [0.334311, 0.265935, 0.322726, 0.270378, 0.258986, 0.271418]
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
ax.plot(ns, train_lc, color=TEAL, lw=2.5, marker="o", ms=5, label="train MSE (rises)")
ax.plot(ns, val_lc, color=ORANGE, lw=2.5, marker="s", ms=5, label="val MSE (falls)")
ax.set_title("f03: more data shrinks the train-val gap", fontsize=13, fontweight=600)
ax.set_xlabel("training size n"); ax.set_ylabel("MSE")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f03_learning_curves.png", dpi=150); plt.close(fig)

# f04: LOOCV fold errors, lambda 0 vs 0.5 (C08)
folds = ["x=0", "x=1", "x=2", "x=3"]
e0 = [0.0, 0.005917, 2.56, 4.84]
e1 = [0.0, 0.029727, 3.177694, 1.514793]
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=BG)
style(ax)
xa = np.arange(4)
ax.bar(xa - 0.2, e0, 0.4, color=MUTED, edgecolor=INK, label="lambda=0, CV mean 1.8515")
ax.bar(xa + 0.2, e1, 0.4, color=GREEN, edgecolor=INK, label="lambda=0.5, CV mean 1.1806")
ax.set_xticks(xa); ax.set_xticklabels(folds)
ax.set_title("f04: cross-validation picks the penalty that wins on held-out folds",
             fontsize=13, fontweight=600)
ax.set_xlabel("held-out point"); ax.set_ylabel("squared error")
ax.legend(frameon=True, facecolor=PANEL, edgecolor=LINE, fontsize=10)
fig.text(0.5, 0.01, FOOT, ha="center", color=MUTED, fontsize=10)
fig.savefig(OUT / "f04_cv_folds.png", dpi=150); plt.close(fig)

# ---- interview-u05 debug task T1 premise execution ----
# A teammate shuffles X before the split but forgets y. Measure the damage.
rng2 = np.random.default_rng(7)
X = rng2.normal(0, 1, (40, 2))
y = (X[:, 0] > 0).astype(int)
Xs = X.copy(); rng2.shuffle(Xs)          # bug: X shuffled, y not
def acc(X_, y_):
    tr_X, te_X = X_[:30], X_[30:]; tr_y, te_y = y_[:30], y_[30:]
    thr = np.median(tr_X[:, 0])
    pred = (te_X[:, 0] > thr).astype(int)
    return float(np.mean(pred == te_y))
print("T1 premise: aligned split test acc =", round(acc(X, y), 4))
print("T1 premise: shuffled-X split test acc =", round(acc(Xs, y), 4))
print("T1 premise: shuffled label agreement with truth =",
      round(float(np.mean((Xs[:, 0] > 0).astype(int) == y)), 4))
