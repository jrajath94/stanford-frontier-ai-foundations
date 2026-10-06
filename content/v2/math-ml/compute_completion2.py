#!/usr/bin/env python3
"""Second compute pass: refined toys + interview debug premises.
numpy 1.26.4, float64."""
import numpy as np
from math import log

print("=== U10-C01b mixture draws, n=10, seed 7 ===")
r = np.random.default_rng(7)
z = r.integers(0, 2, 10); x = np.where(z == 0, -1, 1) + r.standard_normal(10)
print("z", z.tolist()); print("x", [round(v, 4) for v in x])
print("empirical CDF at 0:", float(np.mean(x <= 0)), "true 0.5")

def gini(c):
    p = np.asarray(c, float); p /= p.sum(); return float(1 - (p ** 2).sum())

print("=== U07-T1 debug: minimized sum-p^2 picks wrong threshold ===")
x = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9]); y = np.array([0, 0, 0, 1, 1, 1])
def sump2(counts):
    p = np.asarray(counts, float); p /= p.sum(); return float((p ** 2).sum())
for t in [0.2, 0.35, 0.5, 0.7, 0.85]:
    L = y[x <= t]; R = y[x > t]
    w = (len(L)/6)*sump2([np.sum(L==0), np.sum(L==1)]) + \
        (len(R)/6)*sump2([np.sum(R==0), np.sum(R==1)])
    print(f"t={t}: weighted sum-p^2 {w:.4f}")
print("minimizing sum-p^2 picks t=0.2 or 0.85 (worst splits). correct: minimize 1-sum-p^2 -> t=0.5")

print("=== U07-C07 OOB toy ===")
# x=[0,1,2,3] y=[-1,-1,+1,-1]; stumps s0/s1/s2; bootstrap sets from C05
xa = np.array([0., 1., 2., 3.]); ya = np.array([-1, -1, 1, -1])
s0 = np.where(xa <= 1.5, -1, 1); s1 = np.where(xa <= 2.5, -1, 1); s2 = np.where(xa <= 0.5, -1, 1)
sets = {0: [2,2,3,3], 1: [0,2,3,3], 2: [0,1,1,3]}
stumps = {0: s0, 1: s1, 2: s2}
for i in range(4):
    oob = [b for b, s in sets.items() if i not in s]
    preds = [stumps[b][i] for b in oob]
    print(f"point {i}: OOB reps {oob}, preds {preds}, true {ya[i]}")
print("OOB error 0.0 on 3 predicted points; point 3 never OOB")
print("s0 in-bag err", float(np.mean(s0 != ya)))

print("=== U07-C11 correlated features ===")
print("gain on x1:", 0.5 - 0.0, "importance x1 0.5, x2 0.0 (x1==x2 exactly)")
print("permute x1 -> acc 0.5; permute x2 -> acc 1.0 (tree never reads x2)")

print("=== U07-C12 baselines on 6-pt toy ===")
print("majority acc 0.5, stratified-random expected 0.5, stump acc 1.0")

print("=== U08-T1 debug: gradient check at relu kink ===")
w1, b1s, w2, b2s, xs, yt = 0.5, -1.0, 1.5, 0.1, 2.0, 3.0  # z = 0.5*2-1 = 0
z = w1*xs + b1s; a = max(z, 0.0); o = w2*a + b2s; L = 0.5*(o-yt)**2
analytic_dw1 = (o-yt)*w2*0.0*xs  # subgradient 0 at kink
h = 1e-7
def loss(w1_):
    zz = w1_*xs + b1s; aa = max(zz, 0.0); oo = w2*aa + b2s; return 0.5*(oo-yt)**2
fd = (loss(w1+h) - loss(w1-h))/(2*h)
print("z=0: analytic dw1 (subgrad 0)", analytic_dw1, "two-sided fd", fd)

print("=== U10-T1 debug: KL collapse vs informative, x=0.7 ===")
r2 = np.random.default_rng(7)
def recon_mc(mu_, sg_, trials=20000):
    zz = r2.normal(mu_, sg_, trials)
    return float(np.mean(-0.5*(0.7-zz)**2 - 0.5*log(2*np.pi)))
ri = recon_mc(0.4, 0.5); rc = recon_mc(0.0, 1.0)
print("informative recon", ri, "KL 0.3981 ELBO", ri-0.3981)
print("collapsed recon", rc, "KL 0.0 ELBO", rc)
