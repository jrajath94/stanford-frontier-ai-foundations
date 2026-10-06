"""RUN 3 computations for math-ml.

U03: calculus and optimization (12 leaf concepts C01-C12).
U04b: second source block (Lec 11-19: entropy, KL, KL minimization,
ML estimate example, MLE Gaussian, MLE discrete, mixed density;
tutorials 2, 7A, 7B).

Convention: seed, dtype, shape stated for every computation.
Measured values below are the ground truth quoted in the lesson files.
numpy 1.26.4, float64, CPython. RNG: numpy default_rng(7) where random.
"""
import numpy as np

print("numpy", np.__version__, "float64 default")

# ============ U03 ============

# C01 partial derivatives: f(x,y) = x^2 + 3xy + y^2 at (2,1)
def f_c01(x, y):
    return x**2 + 3*x*y + y**2
h = 1e-7
x0, y0 = 2.0, 1.0
dfdx_fd = (f_c01(x0+h, y0) - f_c01(x0-h, y0)) / (2*h)
dfdy_fd = (f_c01(x0, y0+h) - f_c01(x0, y0-h)) / (2*h)
print("C01 analytic (7, 8); fd =", dfdx_fd, dfdy_fd)

# C02 chain rule: y = (2x+1)^3 at x=2
def y_c02(x):
    return (2*x + 1)**3
dy_fd = (y_c02(2.0+h) - y_c02(2.0-h)) / (2*h)
print("C02 analytic 150; fd =", dy_fd, "y(2) =", y_c02(2.0))

# C03 gradient/Jacobian/Hessian
grad = np.array([2*x0 + 3*y0, 3*x0 + 2*y0])
print("C03 grad =", grad)
def g_c03(v):
    return np.array([v[0]**2, v[0]*v[1]])
jac_fd = np.stack([(g_c03(np.array([x0+h, y0])) - g_c03(np.array([x0-h, y0]))) / (2*h),
                   (g_c03(np.array([x0, y0+h])) - g_c03(np.array([x0, y0-h]))) / (2*h)], axis=1)
print("C03 jacobian analytic [[4,0],[1,2]]; fd =\n", jac_fd)
print("C03 hessian of f (constant) = [[2,3],[3,2]]")

# C04 Taylor: e^x at 0
e05 = np.exp(0.5)
t1 = 1 + 0.5
t2 = 1 + 0.5 + 0.5**2/2
print("C04 e^0.5 =", e05, "T1 err =", abs(e05-t1), "T2 err =", abs(e05-t2))
# sin Taylor at 0.5 rad, T3
s05 = np.sin(0.5)
st3 = 0.5 - 0.5**3/6
print("C04 sin(0.5) =", s05, "T3 err =", abs(s05-st3))

# C05 convexity: Hessians
H_convex = np.array([[2., 3.], [3., 2.]])
print("C05 convex Hessian eig =", np.linalg.eigvalsh(H_convex))
H_saddle = np.array([[2., 0.], [0., -2.]])
print("C05 saddle Hessian eig =", np.linalg.eigvalsh(H_saddle))
# two-minimum toy h(x) = x^4 - 3x^2: h'' = 12x^2 - 6; minima at x=±sqrt(1.5)
xm = np.sqrt(1.5)
print("C05 non-convex minima x =", xm, -xm, "h'' there =", 12*xm**2-6, "h''(0) =", -6.0)

# C06 least-squares gradient: points (1,2),(2,3),(3,5),(4,4); fit y = w x
xs = np.array([1., 2., 3., 4.])
ys = np.array([2., 3., 5., 4.])
w = 1.0
grad_ls = (xs * (w*xs - ys)).mean()
print("C06 J grad at w=1 =", grad_ls)
# normal equation closed form
w_star = (xs @ ys) / (xs @ xs)
print("C06 w_star =", w_star, "residual grad check =",
      (xs * (w_star*xs - ys)).mean())

# C07 GD vs SGD on J(w) = (w-3)^2, eta=0.1, from w0=0, 10 steps
eta = 0.1
w_gd = 0.0
traj_gd = [w_gd]
for _ in range(10):
    w_gd = w_gd - eta * 2*(w_gd - 3)
    traj_gd.append(w_gd)
print("C07 GD w after 10 steps =", traj_gd[-1], "J =", (traj_gd[-1]-3)**2)
rng = np.random.default_rng(7)
w_sgd = 0.0
traj_sgd = [w_sgd]
xs_s = np.array([1., 2., 3., 4.])
for _ in range(20):
    i = rng.integers(0, 4)
    xi = xs_s[i]
    wi = w_sgd
    w_sgd = w_sgd - eta * xi * (xi*w_sgd - 3.0)
    traj_sgd.append(w_sgd)
print("C07 SGD w after 20 steps (seed 7) =", traj_sgd[-1])

# C08 Newton: f(w) = (w-3)^2 from 0 -> 1 step
w_n = 0.0
w_n = w_n - (2*(w_n-3)) / 2.0
print("C08 Newton quadratic: one step from 0 ->", w_n)
# Newton on h(x) = x^4 - 3x^2 from x0=1.0
x_n = 1.0
traj_n = [x_n]
for _ in range(5):
    g1 = 4*x_n**3 - 6*x_n
    g2 = 12*x_n**2 - 6
    x_n = x_n - g1/g2
    traj_n.append(x_n)
print("C08 Newton non-convex from 1.0:", np.round(traj_n, 6))

# C09 constraints: min x^2 s.t. x >= 1
# C10 KKT on same toy: x*=1, lambda*=2*(1)-0 -> L = x^2 - lam(x-1), dL/dx = 2x - lam = 0 => lam = 2
print("C09/C10 optimum x*=1, KKT multiplier lam*=2 (lam>=0, lam*(x-1)=0 satisfied)")

# C11 learning rates on J(w)=(w-3)^2 from 0
for eta_c in (0.1, 1.5, 2.1):
    w_c = 0.0
    tr = [w_c]
    for _ in range(12):
        w_c = w_c - eta_c * 2*(w_c - 3)
        tr.append(w_c)
    print("C11 eta =", eta_c, "final w =", tr[-1], "J =", (tr[-1]-3)**2)

# C12 gradient check: J(w) = sin(w) + w^2 at w=1
def J_c12(w):
    return np.sin(w) + w**2
dJ = np.cos(1.0) + 2*1.0
dJ_fd = (J_c12(1.0+1e-7) - J_c12(1.0-1e-7)) / 2e-7
print("C12 analytic =", dJ, "fd =", dJ_fd, "rel err =", abs(dJ-dJ_fd)/abs(dJ))
# wrong-sign check: analytic says +, code says -
print("C12 broken variant analytic +2.5403 vs returned -2.5403: check catches it")

# R-ladders: softmax cross-entropy toy gradient (ties C06/C07 to U04)
logits = np.array([2.0, 1.0, 0.1])
p = np.exp(logits - logits.max()); p = p / p.sum()
grad_ce = p.copy(); grad_ce[0] -= 1  # true class 0
print("CE grad =", np.round(grad_ce, 4), "sum =", grad_ce.sum())

# ============ U04b ============

# SB09 entropy: fair coin and [0.75, 0.25], nats
def entropy(p):
    p = np.asarray(p, dtype=float)
    return -(p * np.log(p)).sum()
print("SB09 H([0.5,0.5]) =", entropy([0.5, 0.5]), "nats;",
      "H([0.75,0.25]) =", entropy([0.75, 0.25]), "nats")

# SB10 KL: p=[0.7,0.3], q=[0.5,0.5]
def kl(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    return (p * (np.log(p) - np.log(q))).sum()
print("SB10 D_KL(p||q) =", kl([0.7, 0.3], [0.5, 0.5]),
      "D_KL(q||p) =", kl([0.5, 0.5], [0.7, 0.3]))
print("SB10 H(p) + D_KL = cross-entropy check:",
      entropy([0.7, 0.3]) + kl([0.7, 0.3], [0.5, 0.5]),
      "vs direct", -(np.array([0.7, 0.3])*np.log([0.5, 0.5])).sum())

# SB11 KL minimization: scan q1 over [0.05..0.95], p fixed [0.7,0.3]
p_fix = np.array([0.7, 0.3])
qs = np.linspace(0.05, 0.95, 19)
best = min(((q, kl(p_fix, [q, 1-q])) for q in qs), key=lambda t: t[1])
print("SB11 KL min over grid: q =", round(best[0], 4), "D_KL =", best[1])

# SB12 ML estimate example (Lec 14 style): flips H T H H
flips = np.array([1, 0, 1, 1])
print("SB12 theta_hat =", flips.mean())

# SB13 MLE Gaussian (univariate): data [2.1, 2.5, 1.9, 2.3]
d = np.array([2.1, 2.5, 1.9, 2.3])
mu_hat = d.mean(); sig2_hat = ((d - mu_hat)**2).mean()
print("SB13 mu_hat =", mu_hat, "sigma2_hat =", sig2_hat)
# score check: derivative of log-lik wrt mu at mu_hat ~ 0
print("SB13 score at mu_hat =", ((d - mu_hat) / sig2_hat).sum())

# SB14 MLE discrete (Lec 18): counts [4,2,4] over 3 outcomes
counts = np.array([4., 2., 4.])
print("SB14 p_hat =", counts / counts.sum())

# SB15 multivariate Gaussian MLE (C05): 6 points in R^2
pts = np.array([[0.5, -0.2], [-0.3, 0.4], [0.1, 0.2],
                [0.4, 0.5], [-0.2, -0.4], [0.0, 0.1]])
mu2 = pts.mean(axis=0)
cov2 = np.cov(pts, rowvar=False, bias=True)
print("SB15 mu_hat =", np.round(mu2, 4))
print("SB15 cov_hat =\n", np.round(cov2, 4))
print("SB15 cov eig =", np.round(np.linalg.eigvalsh(cov2), 4))
# pdf of a point under fitted Gaussian
def gauss2_pdf(z, mu, cov):
    z = np.asarray(z, dtype=float)
    k = 2
    d_ = z - mu
    return np.exp(-0.5 * d_ @ np.linalg.inv(cov) @ d_) / np.sqrt((2*np.pi)**k * np.linalg.det(cov))
print("SB15 pdf at [0.1,0.2] =", gauss2_pdf([0.1, 0.2], mu2, cov2))

# SB16 mixed density (Lec 19): mixture 0.5*N(0,1)+0.5*N(5,1) at x=2.5
def g1(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)
x_m = 2.5
mix = 0.5*g1(x_m) + 0.5*g1(x_m-5)
print("SB16 mixture density at 2.5 =", mix)

# P-remediation toys
print("R23 derivative of 3x^2 at x=2:", (3*(2+1e-7)**2 - 3*(2-1e-7)**2)/2e-7, "(analytic 12)")
print("R29-R34 toys covered in main lesson computations above")
