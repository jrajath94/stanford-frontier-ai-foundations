# Keys, interview U08 neural and sequence architectures

Date: 2026-10-06. Each answer: minimum sufficient
explanation, strong answer, red flags, rubric, remediation.

## B1

Minimum: (5,4)(4,) -> (5,). (2,5)(5,) -> (2,).
Output shape (2,).
Strong: names each intermediate shape.
Red flags: "(5,2)" or dropped biases.
Rubric: 2. Remediation: C01.

## B2

Minimum: dw1 = do w2 [z>0] x. Reverse mode gives
all parameter gradients in one backward pass.
forward mode needs one pass per parameter.
Strong: states the O(1) vs O(n) pass count.
Red flags: "backprop is just the chain rule"
with no cost statement.
Rubric: 2. Remediation: C02.

## B3

Minimum: 9 - 4 + 1 = 6. Kernel 10 > signal 9:
empty output, shape 0.
Strong: cites the n - k + 1 rule and the pad-or-
shrink fix.
Red flags: "it still works with zeros".
Rubric: 2. Remediation: C03.

## B4

Minimum: dense 64(64)(50) + 50 = 204,850. Conv
3(3)(1)(8) + 8 = 80 (one input channel).
Strong: shows the bias terms explicitly.
Red flags: missing biases. "conv has more".
Rubric: 1 per count. Remediation: C04.

## B5

Minimum: A = softmax(QK^T/sqrt(d)). Without the
scale at d = 512 the scores are tens of units,
the softmax saturates to one-hot, gradients die.
Strong: quantifies typical score growth as
sqrt(d).
Red flags: "the scale is a minor detail".
Rubric: formula 1, mechanism 2. Remediation:
C09.

## B6

Minimum: [0.01, -0.01] = eta sign(g).
Strong: derives the bias-correction canceling at
step 1.
Red flags: "[-0.005, 0.003]" (that is SGD).
Rubric: 2. Remediation: C12.

## D1

D1.1. One backward walk computes every parameter
gradient via the chain rule.
D1.2. dw1 -5.1, dw2 -1.36, db1 -2.55, db2 -1.7.
finite differences agree to 8 digits.
D1.3. do = o - y. da = do w2. dz = da (z > 0).
dw1 = dz x.
D1.4. See T1.
D1.5. The dz line: local slope becomes (1 - a^2)
instead of [z > 0].
Rubric: 2 per follow-up. Remediation: C02.

## D2

D2.1. h_t = tanh(W_h h_{t-1} + W_x x_t): the
state mixes the old note with the new input.
D2.2. h_0 = [0.7616, 0.4621], h_1 = [0.0169,
0.0572]. The first input's trace nearly
vanished in one step.
D2.3. dh_T/dh_0 = prod dh_t/dh_{t-1} = w^T in
the scalar case. the spectral radius replaces
|w| for matrices.
D2.4. Gating treats the cause (the repeated
multiplication). clipping treats the symptom
(the norm).
D2.5. Attack: attention costs O(n^2 d) memory
vs O(d) state. streaming inference at
unbounded n is recurrence's surviving niche.
Rubric: 2 per follow-up. Remediation:
C06-C08.

## Q1

z = 1, a = 1, o = 2, L = 2. do = 2. dw2 = 2,
db2 = 2, da = 4, dz = 4 (z > 0), dw1 = 4,
db1 = 4. Finite difference: (L(1+h) -
L(1-h))/2h = 4.0000000001 (executed
2026-10-06).

## Q2

Scores [0.7071, 0.7071, 1.4142]. Weights
[0.2483, 0.2483, 0.5034]: exp([-0.7071,
-0.7071, 0]) = [0.4931, 0.4931, 1.0], sum
1.9862, each over the sum. total 1.0 by
construction. Output [1.0, 1.0].

## T1

(a) z = 0.5(2.0) - 1.0 = 0. Analytic dw1 with
subgradient 0: -0.0. Two-sided fd: -4.35
(executed 2026-10-06: -4.3499997782).
(b) Diagnosis: the check sits exactly on the
relu kink, where the two-sided difference
averages the left slope (0) and the right
slope (do w2 x). the gradient is not defined
there.
(c) Fix: never gradient-check at z = 0.
Correct check at a kink: one-sided
differences, or check at a nearby smooth
point, or accept any subgradient in [0,
do w2 x].
Strong: also notes the analytic 0.0 is a valid
subgradient choice, so the code may be right
and the check wrong.
Red flags: "the backward pass has a bug".
Rubric: reproduce 2, diagnose 2, fix 1.
Remediation: C02.

## S1

Minimum: measure the gradient norm per step
first. Candidates: exploding gradients through
time (w^T law) vs saturating tanh killing the
signal then NaN from a later op. Cheapest fix:
gradient clipping + lower learning rate before
touching the architecture.
Strong: adds measuring per-step slope norms to
separate the two causes.
Red flags: "just use LSTM" as the first move.
Rubric: measure 1, causes 2, cheap fix 1.
Remediation: C06-C08.

## S2

Minimum: the n x n attention weights: O(n^2)
memory. 32k^2 = 1e9 entries = 4 GB per head in
float32. Exact fix: flash attention (same math,
tiled memory). Approximate: linear attention or
a sliding window, trading exactness for O(n).
Strong: quantifies 4 GB/head and names KV
cache as the decode-side twin.
Red flags: "buy more GPUs" as the plan.
Rubric: term+scaling 2, fixes 2. Remediation:
C09-C10.

## R1

Minimum: attack with the collapse proof: two
linear layers compose to one (C01), so depth
without nonlinearity is decoration. and deep
nets need init/norm/optimizer care (C11, C12).
The two mechanisms that let depth train:
nonlinearity plus residual paths with
normalization, and sane init/optimization.
Strong: names the exact failure (vanishing
signal, shattered gradients) the mechanisms
address.
Red flags: "depth just works now".
Rubric: collapse 2, mechanisms 2.
Remediation: C01, C11, C12.
