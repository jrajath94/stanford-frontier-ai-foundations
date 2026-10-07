# Interview bank: U04 (questions)

Unit: math-genmodels-U04. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice questions. Not actual employer
questions. Keys in interview/keys/interview-u04.md.

## Breadth (6)

Q1. Why must a flow layer be invertible? What breaks first if it is
not?
Q2. What does the Jacobian determinant correct for in the density
transform?
Q3. Why is the Jacobian of a coupling layer triangular, and why
does that matter?
Q4. Write the exact log-likelihood formula for a flow. Where does
the minus sign come from?
Q5. MAF versus IAF: which direction is cheap in each, and why?
Q6. Why do flow implementations accumulate log-determinants instead
of determinants?

## Deep ladders (2 x 5)

L1 (exact likelihood, end to end):
1. State the change-of-variables formula.
2. Compute the coupling Jacobian at (0.5, -1) by hand.
3. Derive the triangular log-det shortcut.
4. Evaluate the exact log-likelihood at the toy point.
5. A colleague's implementation reports -2.2398 instead of -2.6860.
   Diagnose from the numbers alone.

L2 (architecture and numerics):
1. Define coupling versus autoregressive layers.
2. State the per-direction serial cost of each at D = 256.
3. Explain why a never-alternated split fails on joint structure.
4. Prove one affine layer cannot model two modes.
5. A 100-layer stack NaNs at step 3. Name the two most likely
   numerical causes and the guard for each.

## Analytical exercises (2)

A1. Derive the change-of-variables formula from the local volume
fact P(x in A) = P(z in f^-1(A)).
A2. A flow stacks L coupling layers with diagonal log-dets
l_1..l_L. Write the total log-likelihood and show the log-det
computation is O(L x D).

## Implementation and debug (1)

D1. Round-trips pass but held-out likelihoods are worse than a
Gaussian baseline. The splits alternate and the log-dets are
finite. Name the next three things to check, in order.

## Changed-constraint scenarios (2)

T1. Inference must run at 10k points/second on one CPU core. The
current MAF scores 2k/second. Redesign the architecture choice and
the layer count tradeoff, and state what you lose.
T2. The data are known to be multimodal with 5 modes. The current
single-coupling-layer flow underfits. Propose the minimal
architectural change and the experiment that proves it worked.

## Research critique (1)

R1. "Exact likelihood makes flows the right choice for anomaly
detection." Give two reasons exactness alone does not settle it,
and design the experiment that would.
