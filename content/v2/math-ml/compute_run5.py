#!/usr/bin/env python3
"""compute_run5.py -- executed toy numbers for math-ml RUN 5.
U06 lesson (linear models, kernels, margins), U09 remaining leaves,
lab-06, interview-u06, transfer sets, oral defenses.
numpy 1.26.4, float64, seeds stated. All output is measured, not predicted.
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)
print("numpy", np.__version__, "float64, seeds stated per block")

def show(name, v):
    print(f"{name} = {v}")

# ================= U06 =================
print("==== U06 linear models, kernels, margins ====")
# C01 OLS with intercept: x=[0,1,2,3], y=[0,1,3,2]
x = np.array([0., 1., 2., 3.]); y = np.array([0., 1., 3., 2.])
X = np.column_stack([np.ones(4), x])
XtX = X.T @ X; Xty = X.T @ y
det = XtX[0, 0] * XtX[1, 1] - XtX[0, 1] * XtX[1, 0]
inv = np.array([[XtX[1, 1], -XtX[0, 1]], [-XtX[1, 0], XtX[0, 0]]]) / det
w_hat = inv @ Xty
pred = X @ w_hat; resid = y - pred; rss = resid @ resid
show("C01 XtX det", det); show("C01 w_hat [b, w]", w_hat)
show("C01 preds", pred); show("C01 resid", resid); show("C01 RSS", rss)

# C02 probabilistic interpretation: Gaussian noise MLE = OLS
n = 4
sig2_hat = rss / n
lp = -n / 2 * np.log(2 * np.pi * sig2_hat) - rss / (2 * sig2_hat)
show("C02 sigma2_hat", sig2_hat); show("C02 max loglik", lp)
# predictive variance at x*=5: sig2*(1 + x*' XtX^{-1} x*)
xs = np.array([1., 5.]); pv = sig2_hat * (1 + xs @ inv @ xs)
show("C02 predictive var at x=5", pv)

# C03 logistic model: x=[0,1], y=[0,1], w start 0, one GD step eta=1
xl = np.array([0., 1.]); yl = np.array([0., 1.])
sig = lambda z: 1 / (1 + np.exp(-z))
w0 = 0.0
p0 = sig(w0 * xl)
ll0 = np.sum(yl * np.log(np.clip(p0, 1e-12, 1)) + (1 - yl) * np.log(np.clip(1 - p0, 1e-12, 1)))
grad0 = np.mean((p0 - yl) * xl)
w1 = w0 - 1.0 * grad0
p1 = sig(w1 * xl)
ll1 = np.sum(yl * np.log(np.clip(p1, 1e-12, 1)) + (1 - yl) * np.log(np.clip(1 - p1, 1e-12, 1)))
show("C03 p at w=0", p0); show("C03 loglik at w=0", ll0)
show("C03 grad", grad0); show("C03 w after 1 GD step", w1)
show("C03 p at w1", p1); show("C03 loglik at w1", ll1)

# C04 softmax: logits [2, 1, 0], true class 0
z = np.array([2., 1., 0.]); z -= z.max()
e = np.exp(z); probs = e / e.sum()
ce = -np.log(probs[0])
show("C04 probs", probs); show("C04 sum", probs.sum())
show("C04 cross-entropy class 0", ce)

# C05 GLM bridge: Bernoulli canonical link table values
p_ = 0.7
logit = np.log(p_ / (1 - p_))
show("C05 logit(0.7)", logit); show("C05 sigmoid(logit)", sig(logit))

# C06 kernel feature map: phi(x)=[1,x,x^2], poly kernel (1+xz)^2
def phi(u): return np.array([1., u, u ** 2])
x1, x2 = 2., 3.
kphi = phi(x1) @ phi(x2)
kpoly = (1 + x1 * x2) ** 2
show("C06 phi(2).phi(3)", kphi); show("C06 (1+2*3)^2", kpoly)
# explicit degree-2 map match
def phi2(u): return np.array([1., np.sqrt(2) * u, u ** 2])
show("C06 phi2(2).phi2(3)", phi2(x1) @ phi2(x2))

# C07 Gram/PSD: RBF kernel gamma=0.5 on 3 points
pts = np.array([[0., 0.], [1., 0.], [0., 2.]])
g = 0.5
D2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
K = np.exp(-g * D2)
eig = np.linalg.eigvalsh(K)
show("C07 Gram K", K); show("C07 eigenvalues", eig)
show("C07 min eig >= 0", eig.min() >= -1e-12)

# C08 hard/soft margins: 4 points, separable
Xc = np.array([[-1., 0.], [0., -1.], [1., 1.], [2., 2.]])
yc = np.array([-1., -1., 1., 1.])
# separable rule w=[1,1], b=0: margin = 2/||w||
w = np.array([1., 1.]); b = 0.0
margins = yc * (Xc @ w + b)
show("C08 margins", margins); show("C08 geometric margin", 2 / np.linalg.norm(w))
# soft margin: add noisy point (0.2, 0.2, y=-1), C=1 vs C=100 hinge totals
Xc2 = np.vstack([Xc, [0.2, 0.2]]); yc2 = np.append(yc, -1.)
hinge = np.maximum(0, 1 - yc2 * (Xc2 @ w + b))
show("C08 hinge losses", hinge); show("C08 slack sum", hinge.sum())
show("C08 total obj C=1", 0.5 * w @ w + 1 * hinge.sum())
show("C08 total obj C=100", 0.5 * w @ w + 100 * hinge.sum())

# C09 SVM dual: x+=[1,1] y=+1, x-=[-1,-1] y=-1. Symmetry gives a+=a-=a.
# full dual: 2a - 0.5*(4 a^2 * 2); optimum a* = 0.25
a = 0.25
pair = a * a * 2.0  # a_i a_j y_i y_j (x_i.x_j) per pair, all four pairs equal
dual_obj = 2 * a - 0.5 * (4 * pair)
w_dual = a * np.array([1., 1.]) + a * np.array([1., 1.])
show("C09 a*", a); show("C09 dual objective", dual_obj)
show("C09 w from dual", w_dual); show("C09 primal obj", 0.5 * w_dual @ w_dual)
show("C09 y_i(w.x_i+b) (must equal 1)", np.array([w_dual @ [1., 1.], -w_dual @ [-1., -1.]]))

# C10 optimization: subgradient step on hinge for the noisy point
eta = 0.5
gi = -(-1.) * np.array([0.2, 0.2])  # -y*x since margin < 1
w_new = w - eta * gi
show("C10 subgradient at noisy point", gi); show("C10 w after step", w_new)

# C11 numerical conditioning: raw (one feature x1000) vs standardized
z = np.array([1., 0., 1., 0.])
Xr = np.column_stack([np.ones(4), x, 1000 * z])
Gr = Xr.T @ Xr
k_raw = np.linalg.cond(Gr)
Xs = np.column_stack([np.ones(4), (x - x.mean()) / x.std(), (z - z.mean()) / z.std()])
Gs = Xs.T @ Xs
k_std = np.linalg.cond(Gs)
show("C11 cond raw Gram", k_raw); show("C11 cond standardized Gram", k_std)
# ridge stabilization on raw
k_ridge = np.linalg.cond(Gr + 1.0 * np.eye(3))
show("C11 cond raw Gram + ridge 1.0", k_ridge)

# C12 model choice: noisy-label toy, linear SVM acc vs logistic acc (toy protocol)
rng = np.random.default_rng(7)
Xn = rng.normal(0, 1, (60, 2)); yn = np.sign(Xn[:, 0] + 0.1 * rng.normal(0, 1, 60))
flip = rng.random(60) < 0.1
yn[flip] *= -1
# protocol: linear score s = x0; logistic (Newton 5 iters) vs hinge subgradient
def fit_logistic(X_, y_, iters=50, eta=0.5):
    w_ = np.zeros(2)
    for _ in range(iters):
        p_ = sig(X_ @ w_)
        g_ = X_.T @ (p_ - (y_ + 1) / 2) / len(y_)
        w_ -= eta * g_
    return w_
def fit_svm(X_, y_, iters=200, eta=0.05, C=1.0):
    w_ = np.zeros(2)
    for _ in range(iters):
        m_ = y_ * (X_ @ w_)
        sub = w_ - C * np.mean((m_ < 1)[:, None] * y_[:, None] * X_, axis=0)
        w_ -= eta * sub
    return w_
wl = fit_logistic(Xn, yn); ws = fit_svm(Xn, yn)
al = float(np.mean(np.sign(Xn @ wl) == yn)); as_ = float(np.mean(np.sign(Xn @ ws) == yn))
show("C12 logistic train acc (noisy labels)", al); show("C12 linear SVM train acc (noisy labels)", as_)
show("C12 flips", int(flip.sum()))

print("==== U09 remaining leaves ====")
# C01 k-means: 6 points, 2 clusters, Lloyd iterations
Xk = np.array([[0., 0.], [0.2, 0.1], [-0.1, 0.2], [5., 5.], [5.2, 4.9], [4.9, 5.1]])
def kmeans_obj(X_, assign, mu):
    return float(np.sum((X_ - mu[assign]) ** 2))
mu = Xk[[0, 3]].copy()
for it in range(4):
    d = np.sum((Xk[:, None, :] - mu[None, :, :]) ** 2, axis=2)
    a = d.argmin(axis=1)
    obj = kmeans_obj(Xk, a, mu)
    print(f"U09C01 iter {it}: assign={a.tolist()} obj={obj:.6f}")
    mu = np.array([Xk[a == k_].mean(axis=0) for k_ in range(2)])
# bad init: both centers near left cluster
mu_bad = np.array([[0., 0.], [0.1, 0.1]])
for it in range(3):
    d = np.sum((Xk[:, None, :] - mu_bad[None, :, :]) ** 2, axis=2)
    a = d.argmin(axis=1)
    print(f"U09C01 bad-init iter {it}: assign={a.tolist()} obj={kmeans_obj(Xk, a, mu_bad):.6f}")
    mus = []
    for k_ in range(2):
        mus.append(Xk[a == k_].mean(axis=0) if np.any(a == k_) else mu_bad[k_])
    mu_bad = np.array(mus)
print("U09C01 note: empty-cluster guard keeps the old center when a cluster is empty")

# C02 distance/scaling: one feature rescaled x5 flips nearest neighbor
A = np.array([0., 0.]); B = np.array([1., 2.]); Cc = np.array([2.2, 1.])
P2 = np.vstack([A, B, Cc])
def nn_excl(i, P_):
    d = np.sum((P_ - P_[i]) ** 2, axis=1); d[i] = np.inf
    return int(np.argmin(d))
show("U09C02 nearest of A raw", nn_excl(0, P2))
P2s = P2 * np.array([1., 5.])
show("U09C02 nearest of A with y x5", nn_excl(0, P2s))
P2z = (P2s - P2s.mean(0)) / P2s.std(0)
show("U09C02 nearest of A standardized", nn_excl(0, P2z))

# C03 PCA: 2-D toy
Xp = np.array([[1., 2.], [2., 3.], [3., 5.], [4., 6.], [5., 8.]])
Xc_ = Xp - Xp.mean(0)
C_ = Xc_.T @ Xc_ / len(Xp)
eigvals, eigvecs = np.linalg.eigh(C_)
order = np.argsort(eigvals)[::-1]; eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
show("U09C03 covariance", C_); show("U09C03 eigenvalues", eigvals)
show("U09C03 PC1", eigvecs[:, 0])
show("U09C03 variance explained PC1", eigvals[0] / eigvals.sum())

# C04 reconstruction: error from k PCs = sum of dropped eigenvalues
Z = Xc_ @ eigvecs[:, :1]
Xrec = Z @ eigvecs[:, :1].T + Xp.mean(0)
rec_err = np.mean((Xp - Xrec) ** 2)          # mean over all n*d entries
d_dim = Xp.shape[1]
show("U09C04 recon MSE k=1", rec_err)
show("U09C04 dropped eigenvalue / d", eigvals[1] / d_dim)
show("U09C04 match", abs(rec_err - eigvals[1] / d_dim) < 1e-12)

# C09 factor models bridge: per-dimension noise; FA vs PCA on one noisy dim
rng9 = np.random.default_rng(7)
z = rng9.normal(0, 1, 200)
W = np.array([[2.0], [1.0], [0.5]])
Xf = (W @ z[None, :]).T + rng9.normal(0, [0.1, 0.1, 3.0], (200, 3))
Xfc = Xf - Xf.mean(0)
Cf = Xfc.T @ Xfc / 200
ef, Vf = np.linalg.eigh(Cf); o = np.argsort(ef)[::-1]; ef = ef[o]; Vf = Vf[:, o]
# PCA k=1 recon MSE vs FA-ish: PCA on data with noisy dim dropped
Zp = Xfc @ Vf[:, :1]; Xpr = Zp @ Vf[:, :1].T
mse_pca_all = np.mean((Xf - (Xpr + Xf.mean(0))) ** 2)
# oracle: use true direction, noise floor = per-dim noise variance
noise_floor = np.mean([0.01, 0.01, 9.0])
show("U09C09 PCA k=1 recon MSE (all dims)", mse_pca_all)
show("U09C09 noise floor mean", noise_floor)
show("U09C09 PCA eigvals", ef)

# C12 held-out evaluation: k-means inertia train vs held-out; choose k
rng12 = np.random.default_rng(7)
Xh = np.vstack([rng12.normal(0, 0.5, (40, 2)), rng12.normal(4, 0.5, (40, 2))])
tr, te = Xh[:60], Xh[60:]
def kmeans_fit(X_, k_, iters=20, seed=0):
    r_ = np.random.default_rng(seed)
    mu_ = X_[r_.choice(len(X_), k_, replace=False)]
    for _ in range(iters):
        d_ = np.sum((X_[:, None, :] - mu_[None, :, :]) ** 2, axis=2)
        a_ = d_.argmin(1)
        mu_ = np.array([X_[a_ == j].mean(0) if np.any(a_ == j) else mu_[j] for j in range(k_)])
    d_ = np.sum((X_[:, None, :] - mu_[None, :, :]) ** 2, axis=2)
    return mu_, d_.min(1).mean()
for k_ in (1, 2, 3, 5):
    mu_, tr_in = kmeans_fit(tr, k_, seed=7)
    dte = np.sum((te[:, None, :] - mu_[None, :, :]) ** 2, axis=2)
    print(f"U09C12 k={k_}: train inertia={tr_in:.4f} held-out inertia={dte.min(1).mean():.4f}")

print("==== lab-06 ====")
# T1 OLS audit dataset
xa = np.array([1., 2., 3.]); ya = np.array([1., 2., 2.])
Xa = np.column_stack([np.ones(3), xa])
wa = np.linalg.solve(Xa.T @ Xa, Xa.T @ ya)
show("lab06 T1 w_hat", wa)
show("lab06 T1 residual sum (must be ~0)", (ya - Xa @ wa).sum())
# T2 Gram PSD check on new 3 points + classify one point via kernel
P = np.array([[0., 0.], [2., 0.], [0., 2.]])
D2p = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
Kp = np.exp(-0.5 * D2p)
ep = np.linalg.eigvalsh(Kp)
show("lab06 T2 eigvals", ep)
# kernel nearest-centroid in feature space: classify q=(1.6,0.2) vs q=(0.2,1.6)
def krow(q_):
    return np.exp(-0.5 * np.sum((P - q_) ** 2, axis=1))
q1 = np.array([1.6, 0.2]); q2 = np.array([0.2, 1.6])
show("lab06 T2 k(q1)", krow(q1)); show("lab06 T2 k(q2)", krow(q2))
# T3 C sweep numbers: use C08 toy, scan C in {0.01, 0.1, 1, 10}
for C_ in (0.01, 0.1, 1.0, 10.0):
    obj = 0.5 * w @ w + C_ * hinge.sum()
    show(f"lab06 T3 total C={C_}", obj)

print("==== interview-u06 premise checks ====")
# Debug T1 premise: RBF gamma too small -> K ~ ones -> ill-conditioned solve.
# 3 points, gamma 1e-6 vs 0.5, solve K alpha = b. Jitter is a patch, not the fix.
p3 = np.array([[0., 0.], [1., 0.], [0., 2.]])
D2t = np.sum((p3[:, None, :] - p3[None, :, :]) ** 2, axis=2)
b3 = np.array([1., 2., 3.])
for g in (1e-6, 0.5):
    K_ = np.exp(-g * D2t)
    a_ = np.linalg.solve(K_, b3)
    print(f"U06-T1 gamma={g}: cond={np.linalg.cond(K_):.3e} "
          f"alpha={np.round(a_, 4).tolist()} residual={np.linalg.norm(K_ @ a_ - b3):.2e}")
Ktiny = np.exp(-1e-6 * D2t)
a_jit = np.linalg.solve(Ktiny + 1e-6 * np.eye(3), b3)
a_raw = np.linalg.solve(Ktiny, b3)
print("U06-T1 jitter: ||alpha|| =", round(float(np.linalg.norm(a_jit)), 1),
      "vs no-jitter ||alpha|| =", round(float(np.linalg.norm(a_raw)), 1))
