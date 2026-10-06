#!/usr/bin/env python3
"""Compute all completion-run toy numbers, math-ml U07/U08/U10.
numpy 1.26.4, float64, seed 7. Every printed number may appear in a lesson."""
import numpy as np
rng = np.random.default_rng(7)
print("numpy", np.__version__)

def gini(counts):
    p = np.asarray(counts, float); p = p / p.sum()
    return float(1 - (p ** 2).sum())

def entropy_nats(counts):
    p = np.asarray(counts, float); p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def miscls(counts):
    p = np.asarray(counts, float); p = p / p.sum()
    return float(1 - p.max())

print("=== U07-C02 impurity, node [4,1] ===")
print("gini", gini([4, 1]), "entropy", entropy_nats([4, 1]), "miscls", miscls([4, 1]))
print("node [2,3]: gini", gini([2, 3]), "entropy", entropy_nats([2, 3]))

print("=== U07-C03 best split, 1D toy x=[.1,.3,.4,.6,.8,.9] y=[0,0,0,1,1,1] ===")
x = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9]); y = np.array([0, 0, 0, 1, 1, 1])
root = gini([3, 3])
for t in [0.2, 0.35, 0.5, 0.7, 0.85]:
    L = y[x <= t]; R = y[x > t]
    wg = (len(L) / 6) * gini([np.sum(L == 0), np.sum(L == 1)]) + \
         (len(R) / 6) * gini([np.sum(R == 0), np.sum(R == 1)])
    print(f"t={t}: weighted gini {wg:.6f} gain {root - wg:.6f} nL={len(L)} nR={len(R)}")

print("=== U07-C04 depth vs error (tiny CART, 1D) ===")
def stump_fit(xs, ys):
    best = (1e9, None)
    for t in (xs[:-1] + xs[1:]) / 2:
        pred = np.where(xs <= t, np.bincount(ys[xs <= t], minlength=2).argmax(),
                        np.bincount(ys[xs > t], minlength=2).argmax())
        err = np.mean(pred != ys)
        if err < best[0]: best = (err, t)
    return best
xtr = np.array([0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 1.1])
ytr = np.array([0, 0, 0, 1, 0, 1, 1, 1])  # one noisy flip at 0.7
xva = np.array([0.2, 0.5, 0.8, 1.0]); yva = np.array([0, 0, 1, 1])
err, t = stump_fit(xtr, ytr)
pred_tr = np.where(xtr <= t, 0, 1); pred_va = np.where(xva <= t, 0, 1)
print("depth1 t=", t, "train err", np.mean(pred_tr != ytr), "val err", np.mean(pred_va != yva))
# memorize: each train point its own leaf
print("depth-full train err 0.0, val err of memorized rule:",
      np.mean(np.array([ytr[np.argmin(np.abs(xtr - q))] for q in xva]) != yva))
# cost complexity: depth1 tree (2 leaves) err 0.125 vs depth2 (4 leaves) err 0.0 on train
print("alpha 0.05: depth1 cost", 0.125 + 0.05 * 2, "depth2 cost", 0.0 + 0.05 * 4)

print("=== U07-C05 bootstrap sets, n=4, B=3, seed 7 ===")
rb = np.random.default_rng(7)
for b in range(3):
    print("rep", b, sorted(rb.integers(0, 4, size=4).tolist()))
print("OOB prob approx (1-1/6)^6 =", (1 - 1/6) ** 6)

print("=== U07-C06 variance reduction ===")
rho, s2, B = 0.3, 1.0, 10
print("single", s2, "bagged", rho * s2 + (1 - rho) * s2 / B)
rho2, B2 = 0.6, 100
print("trees rho .6 B100:", rho2 * s2 + (1 - rho2) * s2 / B2)
print("rf rho .2 B100:", 0.2 * s2 + 0.8 * s2 / B2)

print("=== U07-C08 AdaBoost round 1, x=[0,1,2,3] y=[-1,-1,+1,-1] ===")
xa = np.array([0., 1., 2., 3.]); ya = np.array([-1., -1., 1., -1.])
w = np.full(4, 0.25); h = np.where(xa <= 1.5, -1., 1.)
eps = float(w[h != ya].sum()); alpha = 0.5 * np.log((1 - eps) / eps)
w2 = w * np.exp(-alpha * ya * h); w2 = w2 / w2.sum()
print("eps", eps, "alpha", alpha, "w after", w2)

print("=== U07-C09 gradient boost first step, y=[1,2,3] ===")
yg = np.array([1., 2., 3.]); F0 = yg.mean(); r = yg - F0
print("F0", F0, "residuals", r, "sq loss before", float(((yg - F0) ** 2).sum()),
      "after adding mean of r (=0)? residuals sum", r.sum())

print("=== U07-C10 imbalance ===")
print("all-neg: acc", 95/100, "recall 0.0 bal-acc", 0.5)
tp, fp, fn = 4, 5, 1
prec = tp / (tp + fp); rec = tp / (tp + fn)
print("model: prec", prec, "rec", rec, "F1", 2 * prec * rec / (prec + rec))

print("=== U08-C01 MLP forward, x=[1,2] ===")
W1 = np.array([[1., 0.], [0., 1.], [-1., 1.]]); b1 = np.array([0., -1., 0.5])
W2 = np.array([[1., -2., 0.5]]); b2 = np.array([0.25])
xx = np.array([1., 2.])
z1 = W1 @ xx + b1; a1 = np.maximum(z1, 0); out = W2 @ a1 + b2
print("z1", z1, "a1", a1, "out", out)

print("=== U08-C02 backprop vs finite diff, scalar net ===")
w1, b1s, w2, b2s, xs, yt = 0.5, -0.2, 1.5, 0.1, 2.0, 3.0
def fwd(w1, b1s, w2, b2s):
    z = w1 * xs + b1s; a = z if z > 0 else 0.0
    o = w2 * a + b2s; return o, 0.5 * (o - yt) ** 2
o, L = fwd(w1, b1s, w2, b2s)
do = o - yt; dw2 = do * (w1 * xs + b1s); db2 = do
da = do * w2; dw1 = da * xs; db1 = da
print("analytic dw1", dw1, "dw2", dw2, "db1", db1, "db2", db2, "loss", L)
h = 1e-7
for name, args, i in [("w1", [w1, b1s, w2, b2s], 0), ("w2", [w1, b1s, w2, b2s], 2)]:
    ap = list(args); am = list(args); ap[i] += h; am[i] -= h
    fd = (fwd(*ap)[1] - fwd(*am)[1]) / (2 * h)
    print("fd", name, fd)

print("=== U08-C03 1D conv, signal [1,2,3,4], kernel [1,0,-1] ===")
s = np.array([1., 2., 3., 4.]); k = np.array([1., 0., -1.])
print("valid:", [float((s[i:i+3] * k).sum()) for i in range(2)])

print("=== U08-C04 param counts ===")
print("dense 32x32->100:", 32*32*100 + 100, "conv 5x5x8:", 5*5*8 + 8)

print("=== U08-C05 CNN shape, 28 in, k5 s1 p0, pool2 s2 ===")
print("conv out", (28 - 5) // 1 + 1, "pool out", (24 - 2) // 2 + 1)

print("=== U08-C06 RNN 2 steps ===")
Wh = np.array([[0.8, -0.2], [0.1, 0.5]]); Wx = np.array([[1.0], [0.5]])
h_ = np.zeros(2)
for t_, xt in enumerate([1.0, -0.5]):
    h_ = np.tanh(Wh @ h_ + (Wx * xt).ravel())
    print("t", t_, "h", h_)

print("=== U08-C07 gradient powers T=10 ===")
print("0.5^10", 0.5 ** 10, "1.5^10", 1.5 ** 10)

print("=== U08-C08 LSTM one step ===")
sig = lambda z: 1 / (1 + np.exp(-z))
x8 = 0.7; h8 = np.array([0.2]); c8 = np.array([0.3])
Wf, Wi, Wg, Wo = 0.5, -0.4, 0.9, 0.6
bf, bi, bg, bo = 0.1, -0.1, 0.0, 0.2
f = sig(Wf * x8 + bf); i = sig(Wi * x8 + bi); g = np.tanh(Wg * x8 + bg)
o8 = sig(Wo * x8 + bo); c9 = f * c8 + i * g; h9 = o8 * np.tanh(c9)
print("f", f, "i", i, "g", g, "o", o8, "c", c9, "h", h9)

print("=== U08-C09 attention, 2 queries 3 keys d=4 ===")
Q = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
K = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 1., 0., 0.]])
V = np.array([[1., 2.], [3., 4.], [5., 6.]])
S = Q @ K.T / np.sqrt(4)
S = S - S.max(axis=1, keepdims=True)
A = np.exp(S); A = A / A.sum(axis=1, keepdims=True)
print("scores\n", Q @ K.T / 2, "weights\n", A, "out\n", A @ V)

print("=== U08-C11 layer norm [1,2,3] ===")
v = np.array([1., 2., 3.]); m = v.mean(); sd = v.std()
print("mean", m, "std", sd, "normed", (v - m) / sd)

print("=== U08-C12 Adam one update, g=[0.5,-0.3] ===")
g = np.array([0.5, -0.3]); b1a, b2a, eta, epsa = 0.9, 0.999, 0.01, 1e-8
m_ = (1 - b1a) * g; v_ = (1 - b2a) * g ** 2
mh = m_ / (1 - b1a); vh = v_ / (1 - b2a)
upd = eta * mh / (np.sqrt(vh) + epsa)
print("m", m_, "v", v_, "update", upd, "sgd update", -eta * g)

print("=== U10-C01 ancestral sample, mixture 0.5 N(-1,1)+0.5 N(1,1), 5 draws ===")
r10 = np.random.default_rng(7)
z = r10.integers(0, 2, 5); xs10 = np.where(z == 0, -1, 1) + r10.standard_normal(5)
print("z", z, "x", xs10)
from math import erf, sqrt, pi, log
Phi = lambda t: 0.5 * (1 + erf(t / sqrt(2)))
print("true CDF at 0:", 0.5 * Phi(1) + 0.5 * Phi(-1))

print("=== U10-C02 VAE KL, q=N(0.4,0.5) p=N(0,1) ===")
mu, sg = 0.4, 0.5
kl = 0.5 * (mu ** 2 + sg ** 2 - 1 - log(sg ** 2))
print("KL", kl)
z1s = 0.55; x1s = 0.7
recon = -0.5 * (x1s - z1s) ** 2 - 0.5 * log(2 * pi)
print("recon(1 sample)", recon, "ELBO(1 sample)", recon - kl)

print("=== U10-C03 GAN toy V ===")
print("V start", log(0.8) + log(1 - 0.3), "V after G step", log(0.8) + log(1 - 0.5))

print("=== U10-C04 JSD p=[0.7,0.3] q=[0.4,0.6] ===")
p = np.array([0.7, 0.3]); q = np.array([0.4, 0.6]); m_ = (p + q) / 2
jsd = 0.5 * (p * np.log(p / m_)).sum() + 0.5 * (q * np.log(q / m_)).sum()
print("JSD nats", jsd)

print("=== U10-C07 discrete ELBO check ===")
lp = log(0.48)
qz = np.array([0.5, 0.5]); pxz = np.array([0.12, 0.36])
elbo = float((qz * np.log(pxz)).sum() + log(2))
print("log p(x)", lp, "ELBO", elbo, "gap", lp - elbo)

print("=== U10-C09 batch noise: grad var B=4 vs B=16 ===")
r9 = np.random.default_rng(7)
true_w = 2.0; Xb = r9.uniform(-1, 1, 64); noise = r9.standard_normal(64)
yb = true_w * Xb + noise
def gvar(B, trials=4000):
    rr = np.random.default_rng(11); gs = []
    for _ in range(trials):
        idx = rr.integers(0, 64, B)
        gs.append(float(np.mean(2 * (0.0 * Xb[idx] - yb[idx]) * Xb[idx])))
    return float(np.var(gs))
v4, v16 = gvar(4), gvar(16)
print("var B4", v4, "var B16", v16, "ratio", v4 / v16)

print("=== U07-C04b bagged stumps val MSE vs B (for f04) ===")
r7 = np.random.default_rng(7)
xt = r7.uniform(0, 1, 12); yt7 = xt ** 2 + r7.standard_normal(12) * 0.1
xv = np.linspace(0, 1, 6); yv = xv ** 2
def fit_stump(xs, ys):
    order = np.argsort(xs); xs, ys = xs[order], ys[order]
    best = (1e9, None, None, None)
    for j in range(1, len(xs)):
        t = (xs[j-1] + xs[j]) / 2
        pl, pr = ys[:j].mean(), ys[j:].mean()
        e = float((((ys[:j] - pl) ** 2).sum() + ((ys[j:] - pr) ** 2).sum()))
        if e < best[0]: best = (e, t, pl, pr)
    return best[1:]
rb7 = np.random.default_rng(7)
stumps = []
for b in range(100):
    idx = rb7.integers(0, 12, 12)
    stumps.append(fit_stump(xt[idx], yt7[idx]))
def bag_mse(B):
    preds = np.zeros(6)
    for t_, pl, pr in stumps[:B]:
        preds += np.where(xv <= t_, pl, pr)
    preds /= B
    return float(np.mean((preds - yv) ** 2))
for B in [1, 5, 25, 100]:
    print("B", B, "val MSE", bag_mse(B))
print("single stump on full train:",
      (lambda tp: float(np.mean((np.where(xv <= tp[0], tp[1], tp[2]) - yv) ** 2)))(fit_stump(xt, yt7)))
