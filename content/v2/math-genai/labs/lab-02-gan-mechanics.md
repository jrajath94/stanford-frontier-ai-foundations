# Lab 02, GAN mechanics on the 1-D toy

Unit: math-genai-U03. Date: 2026-10-06. numpy 1.26.4,
float64, seed 0 where RNG is used. Keys in
labs/keys-lab-02.md. Test-mode: solve closed-book, then
check. The toy: p_data = N(0,1), p_g = N(mu_g, sig_g)
with the bad start mu_g = 1.0, sig_g = 0.5. Log base 2 in
bits. Ground truth: compute_run3.py.

## Task 1, optimal discriminator by hand then in code

(a) By hand: write D*(x) and evaluate it at x = 0 and x
= 1. Show a and b at each point.
(b) In code: implement D_star(x) and reproduce the five
marks: D*(-2) = 1.0000, D*(0) = 0.7870, D*(0.5) =
0.4211, D*(1) = 0.2327, D*(2) = 0.3333, each to 1e-4.
(c) Verify the identity max_D V = 2 JS - 2 numerically:
trapezoid grid, clip D* to [1e-15, 1-1e-15]. Record V,
JS, and the identity error.
(d) Break it: drop the clip. Record exactly what numpy
does and which line produces it.

## Task 2, saturation: predict first, measure second

(a) Predict: the minimax and non-saturating generator
gradients in t at D = 0.5 and at D = 0.001. Write all
four numbers before computing.
(b) Measure: implement sig(t) and both gradient formulas.
Record the four numbers and the 500x collapse ratio of
the minimax gradient.
(c) Mirror check: at D = 0.999, which loss is dead now?
Record both gradients.
(d) Write one sentence: what would you tell a teammate
whose minimax GAN loss is flat while D accuracy is 1.0?

## Task 3, mode collapse diagnosis

(a) Predict: the left-mode mass P(x < 0) for the
collapsed generator N(2,1). Write the number before
running code.
(b) Measure: seed 0, N = 10000 draws. Record the
measured mass and its SE. Judge the prediction.
(c) Measure the healthy mixture's left-mode mass the
same way. Record both numbers.
(d) Compute D*(-2) on the collapse toy. State in one
sentence what the critic knows that the samples hide.

## Task 4, the non-saturating identity

(a) By hand: derive E_{p_g}[-log2 D*] = KL(p_g ||
p_data) - KL(p_g || m) + 1 from D* = pd/(pd+pg).
(b) In code: evaluate both sides on the toy by
trapezoid integration. Record both numbers and their
difference.
(c) Interpret: the minus sign rewards what behavior?
Name the failure mode it encourages.
(d) Break it: set p_g = p_data exactly (mu_g = 0,
sig_g = 1). What are both sides? What does the identity
say at equilibrium?

## Task 5, generator gradient: finite differences versus analytic

(a) Fix D* at the bad start. Define loss(mu) =
E_z[log2(1 - D*(mu + 0.5 z))], z ~ N(0,1), seed 1, N =
20000.
(b) Compute d loss/d mu at mu = 1.0 by central finite
differences with eps = 1e-6. Record the number.
(c) Compute it analytically: d/dm log2(1-D) =
(1/ln2)(-D'/(1-D)) via np.gradient of D* on a grid, then
the sample mean. Record the number.
(d) Compare: do they agree to 1e-5? State what a
disagreement would indicate (bug in which part?).
