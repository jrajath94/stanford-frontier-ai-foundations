# Interview keys: U07

Unit: math-genmodels-U07. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice. Strong answers, red flags, rubrics,
remediation.

## Q1

Strong: the chain rule is an identity: the product of true
conditionals is the joint by definition. The VAE bound replaces
an intractable integral with a variational lower bound. the gap
is KL(q||posterior) and is nonzero in general. Red flag:
"both are approximations." Rubric: identity versus bound with
the gap named. Remediation: U07-C01, C08.

## Q2

Strong: teacher forcing trains on true prefixes but sampling
uses model prefixes. the model never learns recovery, so errors
compound. Red flag: blaming temperature. Rubric: the
train/sample mismatch named. Remediation: U07-C04.

## Q3

Strong: one unmasked layer gives the future a gradient path to
every output through the stack. The guarantee must hold at
every layer or it holds nowhere. Red flag: "the last layer is
enough." Rubric: the gradient-path argument. Remediation:
U07-C05.

## Q4

Strong: dot products grow with d. the softmax saturates at
init. gradients vanish and training never starts. The scale
keeps scores O(1). Red flag: "it is just a convention."
Rubric: the saturation mechanism. Remediation: U07-C06.

## Q5

Strong: T divides the logits before softmax: same model, new
distribution. Low T sharpens (exploits), high T flattens
(explores). The checkpoint is unchanged. Red flag: "higher T
is always more creative." Rubric: the mechanism plus the
tradeoff. Remediation: U07-C07.

## Q6

Strong: perplexity is per token. finer tokenization means more
terms over the same text. The scales differ by construction.
Normalize to bits per character first. Red flag: comparing raw
perplexities anyway. Rubric: the scale argument plus the fix.
Remediation: U07-C03, C08.

## L1 ladder

1. p(abc) = p(a) p(b|a) p(c|a,b).
2. log 0.05 = -2.9957 nats.
3. exp(2.9957/3) = 2.7144: effective branching factor.
4. Exact: the factorization is an identity, no bound, no
   sampling.
5. Nothing about the ranking: -3.4 is a lower bound and could
   hide a true -2.9 above the AR number.

## L2 ladder

1. Lower triangular 3x3, ones on and below the diagonal.
2. Scores [[0.7071,-inf],[0,0.7071]]. weights [1,0] and
   [0.3302, 0.6698]. outputs [2,3] and [3.3395, 4.3395].
3. 512^2 x 64 = 16.8M mults per head. 512x512 scores.
4. Store K, V per layer: 256 KB per sequence in fp32.
   generation O(L^3) becomes O(L^2).
5. Fixes: sliding-window attention (loses long range) or
   linear attention (loses exact weighting). Both trade
   quality for the quadratic wall.

## A1

Strong: sum over x_1..x_L of the product. Sum over x_L first:
the last conditional sums to 1, leaving the L-1 product.
Induct down to p(x_1), which sums to 1. Red flag: assuming it
without the induction. Rubric: the induction with the base
case.

## A2

Strong: row 2 scores [0, 0.7071]. mask adds [0, 0]. softmax
[1, e^0.7071]/(1+e^0.7071) = [0.3302, 0.6698]. The -inf sits
on row 1 column 2: exp(-inf) = 0, so row 1 is [1, 0]. Red
flag: masking after softmax. Rubric: the order with the
-inf handled.

## D1

Strong: off-by-one label shift: the model predicts x_t from
x_t (copying) instead of x_{t-1}. Loss looks great, samples
are garbage. Fix: shift labels by one. Test: the shift test
(shift inputs, outputs must shift) and the constant-sequence
check (loss near zero on constants is the tell). Red flag:
tuning temperature. Rubric: the mechanism, the fix, the
test.

## T1

Strong: sliding window (O(L w), loses distant context) or
linear attention (O(L d), loses exact softmax) or recurrent
state-space layers (O(L d), serial training). At 32768 with
fixed memory: window 4096 plus global tokens. Red flag: "use
fp16 and hope." Rubric: the redesign plus the named loss.

## T2

Strong: exact sampling needs a factorization. bidirectional
context has none: the conditionals need not come from one
joint. Closest: permutation LM (AR over random orders) or a
masked encoder paired with a separate AR decoder. Red flag:
"iterative unmasking is exact." Rubric: the conflict plus the
design.

## R1

Strong: attack 1: pseudo-perplexity is unnormalized (sums to
1.0639 on the toy, not 1.0): not a likelihood. Attack 2:
masking rates and tokenizations differ: incomparable scales.
Fair test: one shared tokenization, AR NLL for both, or a
shared downstream task. Red flag: accepting the
pseudo-number. Rubric: both attacks plus the fair test.
