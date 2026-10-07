# Answer keys: U07

Unit: math-genmodels-U07. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers below. Red flags name the common failure. Rubrics say
what earns full credit.

## B1

Strong: p(x_1..x_L) = product over t of p(x_t | x_<t). Toy:
0.5 x 0.4 x 0.25 = 0.05. Red flag: writing the product without
the conditioning. Rubric: the formula and the number.

## B2

Strong: perplexity = exp(NLL / L), the effective branching
factor. Toy: exp(2.9957/3) = 2.7144. Red flag: "average
probability." Rubric: the definition and the number.

## B3

Strong: training feeds true prefixes. Sampling feeds model
outputs. Failure: exposure bias, errors compound on unseen
contexts. Red flag: "teacher forcing is just faster."
Rubric: the mechanism plus the named failure.

## B4

Strong: position 2 sees positions 1 and 2 only. The mask row is
[1, 1, 0]. Red flag: "it sees the future too." Rubric: the row
read correctly.

## B5

Strong: weights (0.3302, 0.6698), output (3.3395, 4.3395).
Red flag: forgetting the mask on row 1. Rubric: both vectors.

## B6

Strong: L = E[||v_theta(x_t, t) - (x_1 - x_0)||^2]. Toy:
(2 - 1.8)^2 = 0.04. Red flag: calling it a score. Rubric: the
loss and the number.

## L1 ladder

1. Product of conditionals, one per position.
2. log 0.05 = -2.9957 nats.
3. exp(2.9957/3) = 2.7144.
4. Exact because the factorization is an identity: no bound,
   no sampling.
5. VAE bound -3.4 sits below the truth. It could hide a true
   -2.9. The bound proves nothing about the ranking.

## L2 ladder

1. Lower triangular: output t reads inputs up to t.
2. Row 2: (0.3302, 0.6698), output (3.3395, 4.3395).
3. 512^2 x 64 = 16.8M mults per head. Scores 512x512.
4. Store K, V: 256 KB per layer per sequence in fp32.
   Generation drops from O(L^3) to O(L^2).
5. Without cache each token recomputes all KVs: throughput
   dies. The cache is not optional at scale.

## A1

Strong: uniform: NLL = L log V, perplexity = exp(log V) = V.
Counterexample: tokenize "abc" as one token versus three
bytes. One-token NLL = -log p(abc). Byte NLL = sum of 3
terms. Different scales, same data: the comparison is void.
Red flag: "perplexity is comparable anyway." Rubric: the
uniform proof plus the constructed counterexample.

## A2

Strong: scores row 2 = [0, 0.7071] after QK^T/sqrt(2). Mask
adds nothing on row 2 (no future). Softmax: [1, e^0.7071] /
(1 + e^0.7071) = [0.3302, 0.6698]. Times V rows: (3.3395,
4.3395). Red flag: applying the mask after softmax. Rubric:
scores, mask, softmax, product in order.

## D1

Strong: bug: the model was trained with the mask but the data
pipeline leaks: e.g. the labels are shifted wrong (predicting
x_t from x_t instead of x_{t-1}), or position embeddings are
shared wrongly, or the sampler uses a different temperature or
truncation than assumed. Most likely: an off-by-one in the
shift: the model predicts the current token from the current
token, loss looks great (copying), samples are garbage. Fix:
align labels to inputs shifted by one. Test: the shift test:
shift inputs by one position and confirm outputs shift by one.
also confirm the loss on a constant sequence is not
suspiciously near zero. Red flag: blaming temperature first.
Rubric: the off-by-one mechanism, the fix, the shift test.

## T1

Strong: sliding-window attention (O(L w)), or linear attention
(O(L d)), or a recurrent/state-space layer. Lose: full
pairwise context (window), or exact softmax weighting
(linear), or parallel training (recurrent). At 32768 with
fixed memory, sliding window with w = 4096 plus a few global
tokens is the standard compromise. Red flag: "just use
fp16." Rubric: the redesign plus the named loss.

## T2

Strong: conflict: exact sampling needs a valid factorization
(AR order). Bidirectional context conditions on the future,
which has no factorization. Closest: XLNet-style permutation
LM (AR over random orders: exact, sees varied contexts), or a
masked model for understanding plus an AR model for
generation. Red flag: "iterative unmasking is exact."
Rubric: the conflict stated plus the closest design.

## R1

Strong: attack 1: pseudo-perplexity is not a likelihood: the
masked conditionals do not factor a joint, so the number has
no probabilistic meaning (C10). Attack 2: the tokenizations
and masking rates differ, so the scales are incomparable
anyway. Fair experiment: compare on AR NLL with one shared
tokenization, or compare on a downstream task both models can
do. Red flag: accepting pseudo-perplexity as a likelihood.
Rubric: both attacks plus the redesigned comparison.
