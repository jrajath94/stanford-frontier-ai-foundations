#!/usr/bin/env python3
"""compute_run4.py -- executed toy numbers for math-ml RUN 4.
U05 lesson, third source block (04c), lab-05, interview-u05.
numpy 1.26.4, float64, seeds stated. All output is measured.
"""
import numpy as np

rng = np.random.default_rng(7)
np.set_printoptions(precision=6, suppress=True)
print("numpy", np.__version__, "seed default_rng(7), float64")

def show(name, v):
    print(f"{name} = {v}")

# ---------- U05-C01 empirical vs population risk ----------
x = np.array([0., 1., 2., 3.])
y = np.array([0, 0, 1, 1])
pred_h = (x >= 2.5).astype(int)          # h: 1 iff x >= 2.5
R_emp = np.mean(pred_h != y)
show("C01 R_emp h(x>=2.5)", R_emp)
# population: x uniform on [0,4], y = 1[x>=1.5], error region [1.5,2.5)
R_pop = (2.5 - 1.5) / 4.0
show("C01 R_pop", R_pop)

# ---------- U05-C02 loss selection ----------
yy = np.array([0., 1., 1.])
c_sq = yy.mean()
risk_sq = np.mean((yy - c_sq) ** 2)
risk_01 = np.mean((np.ones(3) != yy))   # majority rule c=1
risk_abs = np.mean(np.abs(yy - 1.0))    # median c=1
p = 2.0 / 3.0
risk_log = -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
show("C02 c*(sq)", c_sq)
show("C02 risk sq", risk_sq)
show("C02 risk 0-1", risk_01)
show("C02 risk abs", risk_abs)
show("C02 risk logloss", risk_log)

# ---------- U05-C03 consistency ----------
p_true = 0.7
mse_iid = p_true * (1 - p_true) / 4      # mean of 4 IID
mse_dep = p_true * (1 - p_true)          # mean of 4 perfect copies
show("C03 MSE IID n=4", mse_iid)
show("C03 MSE dependent", mse_dep)

# ---------- U05-C04 bias/variance ----------
n = 4
k = 3
p_hat = k / n
bias_lap = (n * p_true + 1) / (n + 2) - p_true
var_lap = n * p_true * (1 - p_true) / (n + 2) ** 2
mse_lap = bias_lap ** 2 + var_lap
mse_mle = p_true * (1 - p_true) / n
show("C04 p_hat", p_hat)
show("C04 laplace", (k + 1) / (n + 2))
show("C04 bias laplace", bias_lap)
show("C04 var laplace", var_lap)
show("C04 MSE laplace", mse_lap)
show("C04 MSE mle", mse_mle)

# ---------- U05-C05 penalties ----------
xx = np.array([0., 1., 2., 3.])
yy2 = np.array([0., 1., 3., 2.])
Sxx = np.sum(xx ** 2)
Sxy = np.sum(xx * yy2)
w_ols = Sxy / Sxx
nl = 2.0  # n*lambda, lambda=0.5
w_r = Sxy / (Sxx + nl)
rss_ols = np.sum((yy2 - w_ols * xx) ** 2)
rss_r = np.sum((yy2 - w_r * xx) ** 2)
pen_r = nl * w_r ** 2
obj_ols = rss_ols + nl * w_ols ** 2
obj_r = rss_r + pen_r
show("C05 w_ols", w_ols)
show("C05 w_ridge", w_r)
show("C05 RSS ols", rss_ols)
show("C05 RSS ridge", rss_r)
show("C05 penalty ridge", pen_r)
show("C05 objective at ols", obj_ols)
show("C05 objective at ridge", obj_r)

# ---------- U05-C06 priors ----------
tau2 = 0.5
sig2 = 1.0
w_map = Sxy / (Sxx + sig2 / tau2)
lam_equiv = sig2 / (4 * tau2)
show("C06 w_map", w_map)
show("C06 lambda equiv", lam_equiv)

# ---------- U05-C07 train/test split ----------
w = w_ols
train_mse = rss_ols / 4
xt = np.array([4., 5.])
yt = np.array([4.8, 3.9])
test_mse = np.mean((yt - w * xt) ** 2)
show("C07 train MSE", train_mse)
show("C07 test MSE", test_mse)

# ---------- U05-C08 LOOCV ----------
def loocv(nl_):
    errs = []
    for i in range(4):
        m = np.ones(4, bool)
        m[i] = False
        w_ = np.sum(xx[m] * yy2[m]) / (np.sum(xx[m] ** 2) + (3.0 / 4.0) * nl_)
        errs.append((yy2[i] - w_ * xx[i]) ** 2)
    return np.mean(errs), errs
for nl_ in (0.0, 2.0):
    m_, e_ = loocv(nl_)
    show(f"C08 LOOCV nl={nl_}", m_)
    print("   fold errs:", [round(v, 6) for v in e_])

# ---------- U05-C09 capacity ----------
xc = np.array([0., 1., 2., 3.])
yc = np.array([0.1, 1.9, 2.9, 4.2])
xte = np.array([4., 5.])
yte = np.array([4.05, 5.1])
for d in (1, 2, 3):
    coef = np.polyfit(xc, yc, d)
    tr = np.mean((yc - np.polyval(coef, xc)) ** 2)
    te = np.mean((yte - np.polyval(coef, xte)) ** 2)
    show(f"C09 degree {d} train MSE", tr)
    show(f"C09 degree {d} test MSE", te)
    print("   coef:", np.round(coef, 4))

# ---------- U05-C10 learning curves ----------
rng10 = np.random.default_rng(7)
rngv = np.random.default_rng(11)
Xv = rngv.uniform(0, 4, 200)
Yv = Xv + rngv.normal(0, 0.5, 200)
for nn in (3, 5, 8, 12, 20, 40):
    X = rng10.uniform(0, 4, nn)
    Y = X + rng10.normal(0, 0.5, nn)
    w_ = np.sum(X * Y) / np.sum(X ** 2)
    tr = np.mean((Y - w_ * X) ** 2)
    va = np.mean((Yv - w_ * Xv) ** 2)
    show(f"C10 n={nn} train", tr)
    show(f"C10 n={nn} val", va)

# ---------- U05-C11 leakage ----------
ytr = np.array([1]*12 + [0]*8)
acc_leak_train = 1.0
acc_deploy = max(np.mean(ytr), 1 - np.mean(ytr))
show("C11 train acc with leak", acc_leak_train)
show("C11 deploy acc without leak", acc_deploy)

# ---------- U05-C12 covariate shift ----------
xs = np.array([0., 1., 2., 3.])
ys = np.array([-0.5, 0.8, 2.3, 3.6])
ws = np.sum(xs * ys) / np.sum(xs ** 2)
trm = np.mean((ys - ws * xs) ** 2)
xsh = np.array([4., 5., 6.])
ysh = np.array([4.1, 4.8, 6.3])
tem = np.mean((ysh - ws * xsh) ** 2)
show("C12 w", ws)
show("C12 train MSE", trm)
show("C12 shift test MSE", tem)

print("==== 04c third source block ====")
# ---------- SB17 ERM ----------
x6 = np.array([0., 1., 2., 3., 4., 5.])
y6 = np.array([0, 0, 0, 0, 1, 1])
h1 = (x6 >= 2.5).astype(int)
h2 = (x6 >= 3.5).astype(int)
show("SB17 R_emp h1", np.mean(h1 != y6))
show("SB17 R_emp h2", np.mean(h2 != y6))

# ---------- SB18 Bayes ----------
from math import erf, sqrt, exp, pi, log
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
bayes_err = 1 - Phi(1.0)
t15 = 1.5
err15 = 0.5 * (1 - Phi(t15)) + 0.5 * Phi(1 - t15)
show("SB18 Bayes error", bayes_err)
show("SB18 err threshold 1.5", err15)

# ---------- SB20 NP/ROC ----------
s = np.array([0.1, 0.35, 0.4, 0.6, 0.8])
yl = np.array([0, 0, 1, 1, 1])
for t in (0.3, 0.5, 0.7):
    pr = (s >= t).astype(int)
    tp = np.sum((pr == 1) & (yl == 1))
    fp = np.sum((pr == 1) & (yl == 0))
    tpr = tp / 3
    fpr = fp / 2
    fnr = 1 - tpr
    print(f"SB20 t={t}: TPR={tpr:.4f} FPR={fpr:.4f} FNR={fnr:.4f} "
          f"max(FPR,FNR)={max(fpr,fnr):.4f}")
pts = [(0, 0), (0, 1/3), (0, 2/3), (0.5, 1.0), (1, 1)]
auc = sum((pts[i+1][0]-pts[i][0])*(pts[i+1][1]+pts[i][1])/2
          for i in range(len(pts)-1))
show("SB20 AUC", auc)

# ---------- SB21 latent mixture incomplete likelihood ----------
xm = np.array([0.2, -0.3, 0.1, 5.1, 4.8, 5.3])
def gmix(x_, mu):
    N = lambda z, m: np.exp(-0.5*(z-m)**2) / sqrt(2*pi)
    return 0.5*N(x_, mu[0]) + 0.5*N(x_, mu[1])
ll = np.sum(np.log(gmix(xm, [0.0, 5.0])))
show("SB21 incomplete loglik mu=[0,5]", ll)
ll_swap = np.sum(np.log(gmix(xm, [5.0, 0.0])))
show("SB24 loglik swapped labels", ll_swap)

# ---------- SB22 EM two iterations ----------
mu = np.array([1.0, 4.0])
sig2 = 1.0
pi_ = 0.5
def em_step(mu, sig2, pi_):
    N = lambda z, m: np.exp(-0.5*(z-m)**2/sig2)/sqrt(2*pi*sig2)
    g1 = pi_*N(xm, mu[0])
    g2 = (1-pi_)*N(xm, mu[1])
    r = g1/(g1+g2)
    n1 = r.sum()
    n2 = (1-r).sum()
    mu1 = np.sum(r*xm)/n1
    mu2 = np.sum((1-r)*xm)/n2
    pi1 = n1/len(xm)
    s2 = (np.sum(r*(xm-mu1)**2)+np.sum((1-r)*(xm-mu2)**2))/len(xm)
    ll_ = np.sum(np.log(g1+g2))
    return np.array([mu1, mu2]), s2, pi1, r, ll_
print("SB22 init loglik:", round(em_step(mu, sig2, pi_)[4], 6))
for it in (1, 2):
    mu, sig2, pi_, r, ll_ = em_step(mu, sig2, pi_)
    print(f"SB22 iter {it}: mu={np.round(mu,4)} sig2={round(sig2,4)} "
          f"pi={round(pi_,4)} loglik={round(ll_,6)}")
print("SB22 responsibilities after iter2:", np.round(r, 4))

# ---------- SB23 EM GMM init sensitivity ----------
def em_run(mu0, iters=20):
    mu_, s2_, p_ = np.array(mu0, float), 1.0, 0.5
    lls = []
    for _ in range(iters):
        mu_, s2_, p_, _, ll_ = em_step(mu_, s2_, p_)
        lls.append(ll_)
    return mu_, s2_, p_, lls
muA, sA, pA, llsA = em_run([1.0, 4.0])
muB, sB, pB, llsB = em_run([2.5, 2.6])
print("SB23 init A final mu:", np.round(muA, 4), "loglik:", round(llsA[-1], 6))
print("SB23 init B final mu:", np.round(muB, 4), "loglik:", round(llsB[-1], 6))
print("SB23 init B iters 1..3 loglik:", [round(v, 4) for v in llsB[:3]])

# ---------- SB24 MAP ----------
k_, n_ = 7, 10
map_ = (k_ + 2 - 1) / (n_ + 2 + 2 - 2)
mle_ = k_ / n_
post_mean = (k_ + 2) / (n_ + 4)
show("SB24 MAP", map_)
show("SB24 MLE", mle_)
show("SB24 post mean", post_mean)

# ---------- SB25 Parzen ----------
xp = np.array([0.5, 1.5, 2.5, 4.0])
h = 0.5
xq = 1.5
K = lambda u: np.exp(-0.5*u*u)/sqrt(2*pi)
terms = K((xq-xp)/h)
phat = terms.sum()/(len(xp)*h)
print("SB25 kernel terms:", np.round(terms, 6))
show("SB25 p_hat(1.5)", phat)

# ---------- SB26 1-NN ----------
xtr = np.array([0., 1., 2., 3.])
ytr2 = np.array([0, 0, 1, 1])
train_err_1nn = np.mean(ytr2 != ytr2)  # each point is its own nearest neighbor
xq2 = 2.4
nn = xtr[np.argmin(np.abs(xtr - xq2))]
show("SB26 1-NN train error", train_err_1nn)
show("SB26 NN of 2.4", nn)
# 3-NN at 1.5
d = np.abs(xtr - 1.5)
idx = np.argsort(d)[:3]
show("SB26 3-NN vote at 1.5", np.mean(ytr2[idx]))

print("==== lab-05 ====")
# Task 1
rng1 = np.random.default_rng(7)
flips = rng1.random(50) < 0.6
p_emp = flips.mean()
show("lab05 T1 n=50 heads", int(flips.sum()))
show("lab05 T1 p_emp", p_emp)
show("lab05 T1 gap |p_emp-0.6|", abs(p_emp - 0.6))
# Task 2 ridge path
for lam in (0.0, 0.1, 0.5, 2.0, 10.0):
    nl_ = 4 * lam
    w_ = Sxy / (Sxx + nl_)
    mse = np.mean((yy2 - w_ * xx) ** 2)
    print(f"lab05 T2 lambda={lam}: w={w_:.6f} in-sample MSE={mse:.6f}")
# Task 3 bias/variance repeated sampling
rng3 = np.random.default_rng(7)
xg = np.linspace(0, 3, 10)
B = 200
pred_line = np.zeros(B)
pred_const = np.zeros(B)
for b in range(B):
    Yb = 2 * xg + rng3.normal(0, 1, 10)
    wl = np.sum(xg * Yb) / np.sum(xg ** 2)
    pred_line[b] = wl * 1.0
    pred_const[b] = Yb.mean()
f_true = 2.0
for name, pr in (("line", pred_line), ("const", pred_const)):
    bias2 = (pr.mean() - f_true) ** 2
    var = pr.var()
    mse = np.mean((pr - f_true) ** 2)
    print(f"lab05 T3 {name}: bias^2={bias2:.6f} var={var:.6f} mse={mse:.6f} "
          f"sum={bias2+var:.6f}")
# Task 3(c): break it, truth y = 2x + 5, estimator A only.
# Fresh seed-7 stream (same noise draws as (a)), f(1) = 2*1 + 5 = 7.0.
rng3c = np.random.default_rng(7)
pred_c = np.zeros(B)
for b in range(B):
    Yb = (2 * xg + 5) + rng3c.normal(0, 1, 10)
    wl = np.sum(xg * Yb) / np.sum(xg ** 2)
    pred_c[b] = wl * 1.0
f_true_c = 7.0
Ew_c = pred_c.mean()
bias2_c = (Ew_c - f_true_c) ** 2
var_c = pred_c.var()
mse_c = np.mean((pred_c - f_true_c) ** 2)
print(f"lab05 T3(c) line under 2x+5: E[w]={Ew_c:.6f} "
      f"bias^2={bias2_c:.6f} var={var_c:.6f} mse={mse_c:.6f} "
      f"sum={bias2_c+var_c:.6f}")
np.savez("/tmp/figdata_run4.npz",
         xc=xc, yc=yc, xte=xte, yte=yte,
         Xv=Xv, Yv=Yv,
         xx=xx, yy2=yy2,
         xm=xm, xp=xp)
print("saved /tmp/figdata_run4.npz")
