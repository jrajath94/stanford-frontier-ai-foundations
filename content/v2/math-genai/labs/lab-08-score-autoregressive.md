# Lab 08, score and autoregressive mechanics

Unit: math-genai-U09. Date: 2026-10-06. numpy 1.26.4,
float64, seed 0 where RNG is used. Keys in
labs/keys-lab-08.md. Test-mode: solve closed-book, then
check. Toys: the two-mode mixture (score), the bigram P
(AR), the tiny transformer (d = 4). Ground truth:
compute_run5b.py.

## Task 1, the score field: predict first, measure second

(a) Predict: s(0), s(1.5), and the sign of s(1.0).
Write all three before computing.
(b) Measure: implement score_mix. Record s(0),
s(1.0), s(1.5), s(-1.0).
(c) Explain in one sentence why s(0) = 0 does not
mean x = 0 is a mode.
(d) Break it: evaluate the score with the wrong
variance (0.5 instead of 0.25). Record s(1.0) and
explain the direction of the error.

## Task 2, DSM and Langevin

(a) By hand: the DSM target for x = 1.5, x_tilde =
1.7, sigma^2 = 0.04. Verify -5.0.
(b) In code: run 5 Langevin steps from x = 0, a =
0.1, seed 0. Record the trajectory.
(c) Run 20000 steps at a = 0.1 and a = 0.5 (seed
1). Record mean, std, frac(x > 0), crossings for
both.
(d) Write one paragraph: is the a = 0.1 chain
broken? Use the numbers.

## Task 3, the AR chain rule and teacher forcing

(a) By hand: NLL of [a, b, c] in bits. Verify
1.73697. Compute perplexity.
(b) In code: the free-run expected NLL (both
tokens). Verify 3.2675 bits. Compute the
exposure gap per token.
(c) State in one sentence why the one-token
free-run number (1.6966) differs from the
two-token number.
(d) Break it: drop the causal mask in the
attention. Compute row 0 of A and explain the
training consequence.

## Task 4, the tiny transformer

(a) By hand: X = E + P for the 3-token toy.
Write X.
(b) In code: the full forward pass with identity
QKV. Verify O[0] == X[0] and record O rows.
(c) Remove P and re-run. Record the max |diff|
in row 1.
(d) Explain in two sentences what P buys and
what the mask buys. Name which one enforces
the chain rule.

## Task 5, sampling and budgets

(a) Compute softmax(z/T) for T = 0.5, 1.0, 2.0,
z = [2.0, 1.0, 0.5, 0.1]. Record entropies.
(b) Apply top-p 0.9 with the lesson's
convention. Record kept indices and renorm
probs.
(c) Compute attention bytes for n = 1024, 2048,
4096 (12 heads, 12 layers, fp32). Verify the
16x rule.
(d) Write one paragraph: pick score vs AR for a
new modality (audio waveforms). Justify with
two toy numbers.
