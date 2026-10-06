# Lesson 03 keys, calculus and optimization

Unit: math-ml-U03. Date: 2026-10-06. Do not read before attempting.
All values computed 2026-10-06, numpy 1.26.4, float64.

## Exercises

E01. df/dx = 2x + 3y = 7. df/dy = 3x + 2y = 8. Code returned
6.99999999298484 and 7.999999995789153.
E02. df/dx = 2xy = 12, df/dy = x^2 = 9 at (3, 2). The x input moves
the output faster (12 > 9).
E03. Error: |x - 2| has a kink at x = 2, so no derivative exists
there. The left slope is -1 and the right slope is +1. "0 by
symmetry" invents a slope where the definition fails.
E04. u = 2x + 1, du/dx = 2, dy/du = 3u^2 = 75 at x = 2. Product:
150.
E05. dy/dx = cos(u) * 3x^2. At x = 1: u = 1, cos(1) * 3 =
1.6209069176044193.
E06. The outer link |u| at u = 0 has no slope: dy/dx does not exist
at x = 2. The chain rule needs a slope at every link.
E07. Column 1 = d[x^2, xy]/dx = [2x, y] = [4, 1] at (2, 1).
Column 2 = d[x^2, xy]/dy = [0, x] = [0, 2]. J = [[4, 0], [1, 2]].
E08. Hessian [[2, 3], [3, 2]], eigenvalues -1 and 5 (measured).
Indefinite: the surface is a saddle, not a bowl. The lesson's bowl
example is (x-1)^2, not this f.
E09. grad f is undefined: the gradient is defined only for scalar
outputs (f: R^n -> R). J has shape (2, 3). H is undefined for the
same reason as the gradient.
E10. T1(0.5) = 1.5, error 0.1487212707001282. T2(0.5) = 1.625,
error 0.023721270700128194.
E11. T1(x) = x. At x = 0.5 the true value is 0.479425538604203 and
the T3 error is 0.0002588719375363202, so the T1 error is bounded
above by the T1-T3 gap plus the T3 error: roughly 0.0208 + tiny.
(Minimum sufficient: the measured T3 error anchors the scale. The
T1 error must be much larger, on the order of the x^3/6 term.)
E12. Chord: f(0) = 1, f(2) = 1, midpoint value f(1) = 0. Rule needs
f(1) <= 0.5*1 + 0.5*1 = 1. 0 <= 1 holds.
E13. Error: convexity needs the Hessian positive everywhere, not at
the minima. h''(0) = -6 < 0, so the chord rule fails across the
hill. Minima-only curvature proves nothing.
E14. w* = (1*2 + 2*3)/(1 + 4) = 8/5 = 1.6. Gradient at w = 0:
(1/2)[1*(0-2) + 2*(0-3)] = -4.
E15. New w = 1 - 0.1*(-2.25) = 1.225. The gradient was negative, so
w rises toward w* = 1.3, and the cost falls.
E16. Subgradient at w = 1: (1/4) sum x_i sign(w x_i - y_i) =
(1/4)[1*(-1) + 2*(-1) + 3*(-1) + 4*0] = -1.5. Code change: replace
the residual (w x_i - y_i) with its sign. Guard the exact-zero case.
E17. Step 1: w = 0 - 0.1*2*(0-3) = 0.6. J = (0.6-3)^2 = 5.76.
Code confirms.
E18. Each SGD step uses one sample, so 20 steps see 20 samples with
noise, while each GD step uses the exact full gradient. The price
per step is O(d) versus O(nd). The lag is the noise tax.
E19. The wobble shrinks over time: early steps explore, late steps
settle. The added assumption is that the noise variance is bounded
so the shrinking steps can average it out.
E20. w_new = 0 - (-6)/2 = 3. It lands exactly because the function
IS the second-order portrait: for a quadratic, the Taylor portrait
is the function itself.
E21. x1 = 1 - (4-6)/(12-6) = 1 + 2/6 = 1.333333.
x2 = 1.333333 - (4*2.3704-8)/(12*1.7778-6) = 1.236715. The error
digits roughly double per step: quadratic convergence.
E22. The free bottom x = 4 is forbidden (x <= 1). Answer x* = 1.
Lagrangian L = (x-4)^2 + lam*(x-1), lam >= 0. dL/dx = 2(x-4) + lam
= 0 at x = 1 gives lam* = 6.
E23. Fails: (1) allowed fails, 0 < 1. (4) slackness fails,
0*(0-1) = 0 passes numerically but the point is infeasible, so the
whole certificate is void. (3) 2*0 - 0 = 0 passes but is meaningless
off the feasible set. The first failure is the allowed check.
E24. The free bottom x = 0 is feasible (1 is not a bound here.
bounds are x >= 1 and x <= 2. Note: 0 violates x >= 1). Feasible set
[1, 2]. x* = 1 (closest to 0). Multipliers: for x >= 1, lam1 = 2
from 2*1 - lam1 = 0. For x <= 2, lam2 = 0 since the fence is slack.
Slackness: lam1*(1-1) = 0 pass. Lam2*(1-2) = 0 pass.
E25. (1) 1 >= 1 pass. (2) 2 >= 0 pass. (3) 2*1 - 2 = 0 pass.
(4) 2*(1-1) = 0 pass.
E26. g(0) = min_x x^2 + 0 = 0 at x = 0. Bound: primal >= 0.
g(4) = min_x [x^2 - 4x + 4] = 0 at x = 2. Bound: primal >= 0.
Both give the weak bound 0 <= 1. The tight dual lam = 2 gives 1.
E27. The multiplier is |1 - 2 eta|. It stops shrinking at
|1 - 2 eta| = 1, so eta = 1. The walk bounces between 0 and 6
forever.
E28. New boundary: eta < 2/10 = 0.2 (L = 10). eta = 0.5 gives
multiplier |1 - 10*0.5| = 4 > 1: divergence, distance quadruples
per step.
E29. The six lines are in figure f12. Measured relative error
1.6466354233040405e-10, far below 1e-7: pass.
E30. Relative error 2.0 means the code returns the negative of the
truth (or a same-magnitude wrong value). Most likely a sign flip.
First test: negate the returned gradient and re-run the check.

## Deep ladders, strong answers

L01. df/dx is the slope of f when every input but x is frozen.
Toy: 7 and 8. Justification: freezing turns f into a one-variable
function of x, whose slope is the directional rate. The probe adds
independence from the formula: it catches derivation errors.
Partial versus total: partials assume independent dials. Total
derivatives follow a path. Break: no slope at the kink. Transfer:
probe a random subset of coordinates, not all 1e6. Cost is 2
evaluations per coordinate.

L02. The total slope is the product of the link slopes. Toy:
2 * 75 = 150. Derivation: dy/dx = lim of ratio products, each ratio
tending to its link slope. Five-line backward pass: store u, then
dy_du = 3u^2, du_dx = 2, return dy_du * du_dx. Reverse mode wins for
huge input, tiny output. Break: kink in outer link. Transfer:
3 links, 2 multiplies per output scalar per backward pass, O(L).

L03. Gradient: steepest-uphill arrow, shape (n,). Jacobian: local
road map, shape (m, n). Hessian: curvature table, shape (n, n).
Toy tables as in the lesson. Symmetry: mixed partials agree for
smooth f (Clairaut). Column 1 of J reads: "one unit more of input 1
adds 4 to output 1 and 1 to output 2". Forming H costs O(n^2)
memory. A Hessian-vector product costs O(n) extra gradient work:
use products at scale. Break: no Hessian at a kink. Transfer: the
blocker is the O(n^3) inversion. Use L-BFGS or Hessian-free.

L04. T2(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)^2/2. Toy errors as in
the lesson. The remainder is proportional to (x-a)^3 because the
first unmatched term of the smooth function is cubic. The e^x
errors are positive because all dropped terms are positive at
x > 0. Taylor owns derivatives at one point. Interpolation owns
values at many points. Break: the step has zero derivatives but a
cliff. Transfer: the central difference IS the T2 portrait with the
quadratic term cancelling: the error is the cubic remainder.

L05. Chord rule: the graph sits below every chord. Toy: one bowl
versus two. Hessian PSD everywhere implies the second-order
portrait opens upward at every point, so no hill can hide a second
valley: any stationary point is the global bottom. Check:
4x^3 - 6x = 0 gives x = 0, ±sqrt(1.5). Quasi-convex allows flat
plateaus but keeps one valley. Break: the min of two bowls keeps
two valleys. Transfer: you lose the global certificate. Practice
becomes restarts, schedules, and validation, not proofs.

L06. The gradient is the weighted ballot of the points. Toy: -2.25.
Derivation: pull d/dw inside the mean of the sum, apply the chain
rule to each (wx_i - y_i)^2/2 term. Check: 4.44e-16 is float dust.
Normal equations solve directly, O(d^3). Descent steps, O(nd) each.
Break: collinear columns make X^T X singular. The valley becomes a
trench. Transfer: SGD, O(d) per step, one pass is 1e9 steps of
cheap updates.

L07. GD: full-gradient step. SGD: one-sample step. Toy numbers as in
the lesson. The 0.8 factor comes from 1 - eta * curvature = 1 - 0.2.
Step 1 by hand: 0.6, J = 5.76. L-BFGS when n is small and steps are
precious. Break: fixed eta bounces forever around the bottom.
Transfer: with 3 epochs, use a decaying schedule and the largest
batch that fits, so the noise averages down within budget.

L08. Step: jump to the bottom of the local parabola. Toy: 3.0 in
one step. Then 1.224745 in four. The error squares each step because the
portrait matches two derivatives. Check: sqrt(1.5) = 1.22474487139.
BFGS when forming H is refused. Break: negative curvature aims at
a hill. Transfer: d = 50 is tiny: Newton wins, O(50^3) per step is
nothing, and logistic loss is convex so the certificate holds.

L09. The Lagrangian is the objective plus the fine for fence
violation. Toy: x* = 1, lam* = 2. lam >= 0 because the fine must
punish, not reward, violation. Lam*(x-1) = 0 because a slack fence
charges nothing. Four checks in code: allowed, non-negative,
stationary, slackness. Penalties when the fence is soft. Break:
degenerate gradients admit no finite lam. Non-convex KKT can certify
a trap. Transfer: lam* is the shadow price: it tells the stakeholder
the cost of tightening the budget by one unit.

L10. The boundary is eta < 2/L. Beyond it the distance multiplier
exceeds 1. Toy: 0.1, 0.5, 1.5. The 2^12 * 3 arithmetic: after 12
steps the distance is 3 * 2^12 = 12288, matching the measured
12285. eta = 0.5: multiplier 0, one-step landing. Line search when
evaluations are dear. Break: curvature grows along the path and a
fixed eta crosses the boundary mid-run. Transfer: first check the
loss curve shape (oscillation onset) and the gradient norms. Then
cut eta and check whether the divergence step matches the boundary
arithmetic.
