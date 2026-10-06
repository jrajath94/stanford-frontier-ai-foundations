# Lab 01, linear geometry computations

Unit: math-ml-U02. Date: 2026-10-06. numpy 1.26.4, float64.
Keys in labs/keys-lab-01.md. Test-mode: solve closed-book, then check.

## Task 1, rank audit of a data matrix

Build A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
(a) Compute the SVD singular values.
(b) State the rank and name the evidence (the sigma gap).
(c) Find one nonzero vector in the nullspace and verify A @ n = 0
within 1e-12.

## Task 2, projection check on paper then in code

Take a = [2, 0] and b = [1, 1].
(a) By hand: compute p = (b dot a)/(a dot a) * a and r = b - p.
(b) In code: verify r dot a = 0 and p + r = b.
(c) Break it: run the same code with a = [0, 0] and record the exact
failure.

## Task 3, conditioning shake test

Take K = [[1, 1], [1, 1.001]] and b = [2, 2].
(a) Compute kappa = cond(K) and solve for x.
(b) Perturb b to [2, 2.000001]. Compute the relative input move and
the relative output move.
(c) Check the bound: is the output move <= kappa * input move?
Record both numbers.
(d) Write one sentence: what would you tell a teammate who trusts the
solver output because "the residual is zero"?
