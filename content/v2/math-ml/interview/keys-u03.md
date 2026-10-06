# Interview keys, U03 calculus and optimization

Date: 2026-10-06. Computed values, numpy 1.26.4, float64.

## Breadth

B1. A partial derivative is the slope of a function when every input
but one is frozen. df/dx = 2x + 3y = 7 at (2, 1).
B2. dy/dx = 3(2x+1)^2 * 2. At x = 2: outer slope 75, inner slope 2,
product 150.
B3. Gradient: shape (3,). Jacobian: not defined for scalar output.
for f: R^3 -> R the Jacobian is the 1-by-3 row of the gradient.
Hessian: shape (3, 3).
B4. The function is strictly convex: one bowl, and any stationary
point is the unique global minimum. The name is convexity.
B5. w = w - eta * grad. The number is eta, the learning rate.
B6. (1) x >= 1. (2) lam >= 0. (3) 2x - lam = 0. (4) lam(x-1) = 0.
At x = 1, lam = 2: all pass.

## Deep ladders

D1.1. The boundary is eta < 2/L where L is the gradient Lipschitz
constant (the curvature). Beyond it the distance multiplier exceeds
1.
D1.2. eta = 0.05: J = 0.7178979876918524. eta = 0.5: J = 0.0.
eta = 1.5: J = 150994944.0.
D1.3. The update is w_new - 3 = (1 - 2 eta)(w - 3): subtract 3 from
both sides of w_new = w - 2 eta (w - 3) and factor. The absolute
value is the per-step distance multiplier.
D1.4. Mechanism: the valley narrowed along the path, so L grew and
1/L fell below the fixed eta: late detonation. Diagnostics: (1)
plot the loss curve and check for oscillation onset before the
explosion. (2) Log gradient norms per step and compare the
divergence step against the |1 - 2 eta| arithmetic.
D1.5. New boundary: eta < 2/10 = 0.2. eta = 0.5 gives multiplier
|1 - 10*0.5| = 4: divergence, distance quadruples per step.

D2.1. The Newton step jumps to the bottom of the local second-order
parabola: w_new = w - f'(w)/f''(w).
D2.2. w_new = 0 - (-6)/2 = 3. One step lands exactly.
D2.3. Near the answer the portrait matches two derivatives, so the
error of the new point is proportional to the square of the old
error: quadratic convergence.
D2.4. At x0 = 0.1 the Hessian is 12*0.01 - 6 = -5.88, negative. The
local parabola opens downward, so Newton jumps to its peak: the
saddle at x = 0. The arithmetic is right. The target is wrong.
D2.5. Attack: (1) large n: forming and inverting the Hessian costs
O(n^3). Gradient descent costs O(n). Replacement: L-BFGS or
Hessian-free methods. (2) Non-convex or saddle regions: the Hessian
is indefinite and Newton aims at saddles. Gradient descent at least
follows the slope down. Replacement: trust-region or damped Newton.
Red flags: "always", "more information" without a cost model.

## Analytical/quantitative

Q1. w* = 39/30 = 1.3. Gradient at w = 1: -2.25. One GD step:
w = 1.225.
Q2. Error: positive diagonal entries do not imply a bowl. The
eigenvalues decide. Eigenvalues -1 and 5 mean indefinite: the
surface is a saddle, curving up in one direction and down in
another.

## Implementation/debug

T1. Bug: forward difference (first-order, error O(h)) instead of
central difference (second-order, error O(h^2)). Fix: fd = (J(1+h)
- J(1-h)) / (2*h). Expected absolute error after the fix: about
4.2e-10 (measured 4.1829527e-10), relative 1.6e-10.

## Changed-constraint scenarios

S1. Replacement: L-BFGS (limited-memory quasi-Newton), which builds
a Hessian approximation from gradient history in O(n) memory per
step. Assumption: the objective is smooth enough that past
gradients inform the curvature.
S2. Feasible set [1, 2]. x* = 1. Multipliers: lam1 = 2 for x >= 1
(binds), lam2 = 0 for x <= 2 (slack). Slackness: 2*(1-1) = 0 and
0*(1-2) = 0.

## Research critique

R1. The reasoning is unsound. The missing quantity is the compute
budget: GD's exact steps cost O(nd) each while SGD's cost O(d).
Final quality per unit wall-clock is what matters, and SGD's noise
also acts as a regularizer. Experiment: fix the wall-clock budget,
run both, and compare validation loss. The winner is decided by
budget, not by step exactness.

## Scoring rubric (per ladder)

- Define (1 pt): one crisp sentence with shapes where relevant.
- Toy (1 pt): correct numbers, no hand-waving.
- Derive/justify (2 pts): the key step named, not skipped.
- Implement/debug (2 pts): the mechanism or the bug named exactly.
- Compare/break/transfer (2 pts): selection boundary or failure
  stated with a concrete case.
- Red flags: "obvious", "simply", inventing numbers, confusing the
  multiplier with the step, claiming Newton always wins.
- Remediation: miss on D1 -> reread C11 and lab-03 task 2. Miss on
  D2 -> reread C08 and the saddle failure. Miss on T1 -> reread C12.
