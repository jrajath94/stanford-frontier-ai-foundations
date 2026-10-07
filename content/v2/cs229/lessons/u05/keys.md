# keys.md, U05 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate.

## E01

phi(x) = [1, x, x^2, x^3]^T.

## E02

theta = sum_i beta_i phi(x^{(i)}).

## E03

beta := beta + alpha (y - K beta), K_ij =
K(x^{(i)}, x^{(j)}).

## E04

Symmetric: K(x, z) = K(z, x). PSD: every Gram matrix has
nonnegative eigenvalues.

## E05

K is a valid (Mercer) kernel iff every finite Gram matrix
is symmetric positive semidefinite.

## E06

Sum of valid kernels is valid. Product of valid kernels
is valid. Positive scaling is valid.

## E07

x^T z = 3 + 2 = 5. K = 25. phi(x) = [1, 2, 2, 4],
phi(z) = [9, 3, 3, 1]. Inner product: 9 + 6 + 6 + 4 =
25. Matches.

## E08

z^T (K1 + K2) z = z^T K1 z + z^T K2 z >= 0 + 0. Symmetry
is entrywise. So K1 + K2 is symmetric PSD: valid.

## E09

Causes: (1) alpha too large for the dual dynamics (K can
have large eigenvalues. The stable range differs from
the primal). (2) K is not PSD (invalid kernel), so the
dual objective is not convex.

## E10

Team A wins: the Gaussian kernel draws smooth closed
curves and can trace the circle. Degree-2 polynomial
draws conic sections, and a circle needs the right
parameter tuning but is representable. Honestly: both
can represent a circle ((x^T z + c)^2 expands to
quadratic terms), so validate both. The Gaussian is the
safer default for blob-like geometry.

## E11

Setup: ring data (class 1 on a circle, class 0 inside
and outside), Gaussian kernel, sigma on a log grid,
ridge fixed. Claim: test error is U-shaped in sigma:
tiny sigma memorizes, huge sigma underfits, interior
sigma wins. Falsified if the curve is monotone.

## E12

```python
import numpy as np

def is_valid_kernel(K, tol=1e-8):
    if np.max(np.abs(K - K.T)) > tol:
        return False
    return bool(np.all(np.linalg.eigvalsh(K) >= -tol))

rng = np.random.default_rng(0)
X = rng.normal(size=(20, 3))
D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
K = np.exp(-D) + (X @ X.T + 1.0) ** 2
print(is_valid_kernel(K))
assert is_valid_kernel(K)
```

## L01 key

(1) K(x, z) = phi(x)^T phi(z).
(2) <x,z> = 5, K = 1 + 5 + 25 + 125 = 156.
(3) Substitute theta = sum beta_j phi(x^{(j)}) into the
primal update and read off the beta update.
(4) Dual code in SL-02. Check predictions match primal
LMS on degree-2 explicit features.
(5) d = 1000, degree 3: explicit needs 1e9 numbers per
update. Kernelized needs O(d) per kernel eval. Dual
wins by orders of magnitude.
(6) Causes: beta initialized nonzero (primal started at
0). Kernel computed with a different degree or offset.
(7) The dual is wrong when n >> p: Gram is n^2 and
primal is cheaper.
(8) n = 1e6, d = 10: primal. The Gram never fits in
memory.

## L02 key

(1) K_ij = K(x^{(i)}, x^{(j)}).
(2) See SL-03 computed example. Eigenvalues positive.
(3) z^T K z = sum_k (sum_i z_i phi_k(x^{(i)}))^2 >= 0.
(4) Test: symmetry then eigvalsh >= -tol.
(5) On a ring: Gaussian traces the ring. Polynomial of
low degree cannot. Check test errors.
(6) Cause: invalid kernel or numerical noise. Check
symmetry first, then the kernel formula.
(7) PSD is needed for the convex-dual story. Non-PSD
"kernels" can still predict via other algorithms, but
the SVM/kernel-LMS guarantees void.
(8) Form Gram matrices on samples, test PSD, test on a
held-out task vs a known-valid kernel.
