"""Compute lab-01 and lab-02 key values, math-ml RUN 2 (fix builder 2026-10-06).

Reproduces every numeric claim in labs/keys-lab-01.md and
labs/keys-lab-02.md. numpy 1.26.4, float64. Seed 7 where the lab
uses RNG (lab-02 Task 1). Run: python3 compute_run2_labs.py.
Exit 0 and matching printout = keys verified.
"""
import warnings
import numpy as np

print("numpy", np.__version__)

def show(name, v):
    print(f"{name}: {v!r}")

# ---------- lab-01 Task 1: rank audit ----------
A = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9]])
U, s, Vh = np.linalg.svd(A)
show("lab01 T1a singular values", s)
n = Vh[-1, :]
if n[0] > 0:  # sign convention: match the keyed nullspace direction
    n = -n
show("lab01 T1c nullspace direction", n)
show("lab01 T1c A@n residual (2-norm)", np.linalg.norm(A @ n))

# ---------- lab-01 Task 2: projection ----------
a = np.array([2., 0])
b = np.array([1., 1])
p = (b @ a) / (a @ a) * a
r = b - p
show("lab01 T2a p", p)
show("lab01 T2a r", r)
show("lab01 T2b r.a", r @ a)
show("lab01 T2b p+r", p + r)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    a0 = np.array([0., 0])
    p0 = (b @ a0) / (a0 @ a0) * a0
show("lab01 T2c a=0 result", p0)

# ---------- lab-01 Task 3: conditioning shake ----------
K = np.array([[1., 1], [1, 1.001]])
b3 = np.array([2., 2])
kappa = np.linalg.cond(K)
x = np.linalg.solve(K, b3)
show("lab01 T3a kappa", kappa)
show("lab01 T3a x", x)
b3p = np.array([2., 2.000001])
xp = np.linalg.solve(K, b3p)
rin = np.linalg.norm(b3p - b3) / np.linalg.norm(b3)
rout = np.linalg.norm(xp - x) / np.linalg.norm(x)
show("lab01 T3b rel input move", rin)
show("lab01 T3b rel output move", rout)
show("lab01 T3c kappa*rin", kappa * rin)
show("lab01 T3c bound holds", rout <= kappa * rin)

# ---------- lab-02 Task 1: 8-draw rerun ----------
# Bernoulli draws via Generator.binomial (the canonical Bernoulli
# sampler). This is the stream the keys were computed from.
rng = np.random.default_rng(7)
draws = rng.binomial(1, 0.3, 8)
show("lab02 T1a draws", draws)
show("lab02 T1a mean", draws.mean())
show("lab02 T1b predicted typical error", np.sqrt(0.3 * 0.7 / 8))
show("lab02 T1b observed |mean-0.3|", abs(draws.mean() - 0.3))
rng800 = np.random.default_rng(7)
d800 = rng800.binomial(1, 0.3, 800)
show("lab02 T1c n=800 mean", d800.mean())
show("lab02 T1c n=800 error", abs(d800.mean() - 0.3))

# ---------- lab-02 Task 2: likelihood grid ----------
flips = np.array([1, 1, 0, 1])
grid = np.arange(0.1, 1.0, 0.1)
ll = flips.sum() * np.log(grid) + (len(flips) - flips.sum()) * np.log(1 - grid)
i = int(np.argmax(ll))
show("lab02 T2a grid argmax p", grid[i])
show("lab02 T2a loglik", ll[i])
show("lab02 T2b analytic MLE 0.75 loglik",
     3 * np.log(0.75) + np.log(0.25))
flips1 = np.array([1])
ll1 = flips1.sum() * np.log(grid) + (len(flips1) - flips1.sum()) * np.log(1 - grid)
i1 = int(np.argmax(ll1))
show("lab02 T2c n=1 grid max p", grid[i1])
show("lab02 T2c n=1 loglik log(0.9)", np.log(0.9))

# ---------- lab-02 Task 3: histogram honesty ----------
xs = np.array([0.2, 0.5, 0.7, 1.1, 1.4, 1.6, 2.0, 2.3])
counts4, edges4 = np.histogram(xs, bins=4, range=(0, 2.5))
show("lab02 T3a 4-bin counts", counts4)
show("lab02 T3a edges", edges4)
width = edges4[1] - edges4[0]
dens = counts4 / (len(xs) * width)
show("lab02 T3b width", width)
show("lab02 T3b density", dens[0])
show("lab02 T3b total area", float((dens * width).sum()))
counts8, _ = np.histogram(xs, bins=8, range=(0, 2.5))
show("lab02 T3c 8-bin counts", counts8)
print("done")
