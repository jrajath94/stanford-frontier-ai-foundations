"""RUN 2 computations for math-ml U02 (linear geometry) and the first
source block (Lec 02-10: probability recap, IID, estimation, density).

Convention: seed, dtype, shape stated for every computation.
Measured values below are the ground truth quoted in the lesson files.
"""
import numpy as np

print("numpy", np.__version__, "python float64 default")

# --- U02-C01 subspaces ---
A1 = np.array([[1., 2.], [2., 4.]])
print("C01 rank of [[1,2],[2,4]] =", np.linalg.matrix_rank(A1))

# --- U02-C02 rank/nullspace ---
A2 = np.array([[1., 2.], [2., 4.]])
u2, s2, vt2 = np.linalg.svd(A2)
print("C02 svd s =", s2)
n2 = vt2[-1]
print("C02 nullspace dir =", n2, "residual =", np.linalg.norm(A2 @ n2))

# --- U02-C03 dot/norm ---
v = np.array([3., 4.])
w = np.array([1., 0.])
dot = v @ w
norm = np.linalg.norm(v)
print("C03 dot =", dot, "norm =", norm, "cos =", dot / norm)

# --- U02-C04 projection ---
a = np.array([1., 0.])
b = np.array([3., 4.])
proj = (b @ a) / (a @ a) * a
resid = b - proj
print("C04 proj =", proj, "resid =", resid, "resid norm =", np.linalg.norm(resid))

# --- U02-C05 matrix transforms ---
R = np.array([[0., -1.], [1., 0.]])
x0 = np.array([1., 0.])
print("C05 rot90([1,0]) =", R @ x0)
S = np.array([[1., 1.], [0., 1.]])
print("C05 shear([1,1]) =", S @ np.array([1., 1.]))

# --- U02-C06 determinant ---
D = np.array([[2., 1.], [1., 2.]])
print("C06 det =", np.linalg.det(D))
Z = np.array([[1., 2.], [2., 4.]])
print("C06 det singular =", np.linalg.det(Z))

# --- U02-C07 inverse/pseudoinverse ---
Ai = np.linalg.inv(D)
print("C07 inv =", Ai.ravel())
print("C07 inv*D residual =", np.linalg.norm(Ai @ D - np.eye(2)))
Ap = np.linalg.pinv(Z)
print("C07 pinv of rank-1 =", Ap.ravel())
print("C07 pinv roundtrip A@pinv@A-A =", np.linalg.norm(Z @ Ap @ Z - Z))

# --- U02-C08 eigenvectors ---
E = np.array([[2., 1.], [1., 2.]])
vals, vecs = np.linalg.eigh(E)
print("C08 eigenvalues =", vals)
print("C08 eigenvector for 3 =", vecs[:, 1])

# --- U02-C09 SVD ---
M = np.array([[3., 2., 2.], [2., 3., -2.]])
u9, s9, vt9 = np.linalg.svd(M)
print("C09 singular values =", s9)
M1 = s9[0] * np.outer(u9[:, 0], vt9[0])
print("C09 rank-1 recon error Fro =", np.linalg.norm(M - M1))
print("C09 sigma2 (expected err) =", s9[1])

# --- U02-C10 PSD ---
P = np.array([[2., 1.], [1., 2.]])
Q = np.array([[1., 2.], [2., 1.]])
print("C10 eig P =", np.linalg.eigvalsh(P))
print("C10 eig Q =", np.linalg.eigvalsh(Q))
print("C10 quadratic form Q at [1,-1]:", np.array([1., -1.]) @ Q @ np.array([1., -1.]))

# --- U02-C11 covariance ---
X = np.array([[1., 2.], [2., 3.], [3., 5.], [4., 6.]])
Xc = X - X.mean(axis=0)
C = Xc.T @ Xc / (X.shape[0] - 1)
print("C11 covariance =\n", C)
print("C11 eig =", np.linalg.eigvalsh(C))

# --- U02-C12 conditioning ---
K = np.array([[1., 1.], [1., 1.001]])
uk, sk, vkt = np.linalg.svd(K)
print("C12 singular values =", sk, "kappa =", sk[0] / sk[-1])
b12 = np.array([2., 2.])
x12 = np.linalg.solve(K, b12)
b12p = np.array([2., 2.000001])
x12p = np.linalg.solve(K, b12p)
print("C12 x =", x12, "x perturbed =", x12p)
print("C12 relative input move =", np.linalg.norm(b12p - b12) / np.linalg.norm(b12))
print("C12 relative output move =", np.linalg.norm(x12p - x12) / np.linalg.norm(x12))

# --- Source block SB1 (Lec 02-10) ---
# SB-C01 Bernoulli PMF/CDF
p = 0.3
print("SB PMF(1) =", p, "PMF(0) =", 1 - p)

# SB expectation/variance of fair die
die = np.arange(1., 7.)
E = die.mean()
Var = ((die - E) ** 2).mean()
print("SB die E =", E, "Var =", Var, "= 35/12?", 35 / 12)

# SB joint/marginal/conditional 2x2 table
T = np.array([[0.3, 0.2], [0.1, 0.4]])
pm_x = T.sum(axis=1)
pm_y = T.sum(axis=0)
print("SB marginal X =", pm_x, "marginal Y =", pm_y)
cond = T[0] / pm_x[0]
print("SB P(Y|X=0) =", cond, "check P(X=0|Y=1) =", T[0, 1] / pm_y[1])

# SB MLE coin: H H T H -> 3/4
flips = np.array([1., 1., 0., 1.])
phat = flips.mean()
ll = np.log(phat) * flips.sum() + np.log(1 - phat) * (len(flips) - flips.sum())
print("SB p_hat =", phat, "log-likelihood =", ll)

# SB histogram toy: 8 draws from Bernoulli(0.3) seeded
rng = np.random.default_rng(7)
draws = rng.binomial(1, 0.3, size=8)
print("SB draws (seed 7, n=8) =", draws, "mean =", draws.mean())

# SB density histogram counts on tiny sample
xs = np.array([0.2, 0.5, 0.7, 1.1, 1.4, 1.6, 2.0, 2.3])
counts, edges = np.histogram(xs, bins=4, range=(0, 2.5))
print("SB hist counts =", counts, "edges =", edges)
