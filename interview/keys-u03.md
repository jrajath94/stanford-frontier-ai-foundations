# keys-u03.md: interview answer keys, U03

Date: 2026-10-06. Minimum sufficient explanation, strong
answer, red flags, rubric, remediation per item.

## B1

Minimum: C = 6PD + 12LnTD. Attention dominates past
T ~ 6n (at T=32768, n=4096 it exceeds dense).
Strong: gives the ratio derivation.
Red flags: quoting 6PD alone for long context.
Rubric: 2 points. Remediation: C01.

## B2

Minimum: 2P + 4LTn per token. Bandwidth-bound because
each weight is read once per token (intensity ~1).
Strong: computes the time floor from bandwidth.
Red flags: "low FLOPs means fast."
Rubric: 2 points. Remediation: C02, C05.

## B3

Minimum: 2*B*T*L*n*2 bytes = 17179869184 = 16.0 GiB.
Strong: notes this exceeds the 14 GB weights.
Red flags: forgetting the factor of 2 for K+V.
Rubric: 2 points. Remediation: C03.

## B4

Minimum: parallel forward pass over the prompt that
builds the KV cache. Long prompts make it
compute-heavy (O(T) dense + O(T^2) attention).
Strong: contrasts with decode ticks.
Red flags: "prefill is free."
Rubric: 2 points. Remediation: C04.

## B5

Minimum: prefill ~T, decode ~B (batch 1: ~1). At ridge
150: prefill T=2048 compute-bound, decode memory-bound.
Strong: places both on the roofline.
Red flags: one intensity for both phases.
Rubric: 2 points. Remediation: C05.

## B6

Minimum: speedup = E[k]/(gamma*c+1), E[k] =
(1-a^{gamma+1})/(1-a).
Strong: defines c, gamma, a.
Red flags: missing the +1 (bonus token).
Rubric: 2 points. Remediation: C07, C12.

## B7

Minimum: accept with min(1, p(x)/q(x)). Resample from
norm(max(0, p-q)).
Strong: explains why the residual completes p.
Red flags: "resample from q."
Rubric: 2 points. Remediation: C09.

## B8

Minimum: (1) same tokenizer/vocabulary, (2) exact
resample distribution.
Strong: states the consequence (output = p exactly).
Red flags: "approximately the same."
Rubric: 2 points. Remediation: C10.

## Deep ladder 1

L1a. C: draft/target step-time ratio. Gamma: drafts per
round. A: per-token acceptance probability.
L1b. E[k] = 3.689, speedup 2.95x.
L1c. E[k] = sum_{i=0}^{gamma} a^i (i drafts accepted
with prob a^i, plus the bonus token).
L1d. O(gamma) per evaluation, the sweep is trivial.
L1e. Good case 2.95x, bad case E[k]=1.42, cost 3.5,
0.41x (2.4x slowdown). Good wins by 7.2x relative.
L1f. Verification slower than 1 step, or measured a
below assumed a.
L1g. Assumes constant a across positions, real a_i
declines with position (conditioning on drafts).
L1h. Grid over gamma at fixed workload, controls: same
prompts, warm cache, fixed batch. Success: peak gamma
matches argmax of the formula within 1-2 steps.
Scoring: 1 point per rung.

## Deep ladder 2

L2a. Every emitted token is distributed exactly as the
target model p, not approximately.
L2b. 0.2/0.7 = 0.286.
L2c. Accept mass: q*min(1,p/q) = min(q,p). Residual:
p - min(q,p) = max(0,p-q), covered by resample. Sum
= p.
L2d. O(V) for the residual, negligible vs model FLOPs.
L2e. Sampling preserves the distribution, greedy only
the argmax path. Use sampling for quality parity,
greedy for speed at some drift.
L2f. Wrong resample distribution or mishandled bonus
token.
L2g. Same tokenizer required, different tokenizers put
p and q over different spaces and the identity fails.
L2h. N=50K rounds, tolerance ~0.01 per bin, failure =
systematic bin deviation beyond sampling noise.
Scoring: 1 point per rung.

## A1

Dense: 4.2e22. T=2048: attn 3.2e21, share 7.1%.
T=32768: attn 5.2e22, share 55.3%.

## A2

E[k] = (1-0.9^8)/0.1 = 5.695. Speedup = 5.695/1.7 =
3.35x. Break-even: E[k] = 1.7. Solve: a ~ 0.55
(check: (1-0.55^8)/0.45 = 2.16... Refine: a=0.42:
(1-0.42^8)/0.58 = 1.72, a=0.41: 1.69. So a ~ 0.415).

## D1

Bugs: (1) accept probability p[x]/q[x] is not capped
at 1 (must be min(1, p/q)), (2) on reject there is no
resample from norm(max(0,p-q)), the residual mass is
dropped. Invariant: over many rounds the output
histogram equals p.

## S1

New speedup = E[k]/(1+0.1*gamma). At a=0.8:
gamma=4: 3.362/1.4 = 2.40x, gamma=5: 3.689/1.5 =
2.46x, gamma=6: 3.951/1.6 = 2.47x, gamma=7:
4.161/1.7 = 2.45x. Best at gamma=6.

## S2

At batch 256 decode is compute-bound: the target step
is no longer bandwidth-idle, so the "free"
verification assumption breaks. The cost model gains a
compute term per verification FLOP, speedup shrinks
and can flip. Speculative decoding is a
latency/batch-1 technique.

## R1

(1) Same tokenizer for draft and target? Invalid: no.
(2) Measured a on what workload? Invalid: cherry-picked
easy prompts. (3) Is the baseline the fastest
non-speculative serving? Invalid: naive baseline.
(4) Quality metric proving no loss? Invalid: only
perplexity, no task eval. (5) End-to-end latency with
the draft's own serving overhead? Invalid: draft time
excluded.
