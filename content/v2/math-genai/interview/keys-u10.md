# Interview keys, U10 LLM inference, quantization, and alignment

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4,
float64. Ground truth: compute_run5b.py. Interview
provenance: role-derived practice, not employer material.
C04 numbers are authored arithmetic. Format per answer:
strong answer, red flags, rubric, remediation.

## Breadth

B1. T=0.5: [0.8282, 0.1121, 0.0412, 0.0185], ent
0.8754. T=2.0: [0.4056, 0.2460, 0.1916, 0.1569],
ent 1.9017. Entropy rises in T. Strong answer: both
distributions, both entropies, and the trend. Red flags:
"temperature sharpens the distribution". Rubric: 2/2
numbers plus trend. 1/2 numbers only. Remediation:
U10-C01.
B2. 12582912 bytes = 12.00 MiB. Per token 24576
bytes. Strong answer: the total plus the per-token
figure. Red flags: forgetting the K and V factor of 2.
Rubric: 2/2 both numbers. 1/2 one. Remediation: U10-C02.
B3. q = 123, w_hat = 0.494118, err 0.005882. Strong answer: all three numbers. Red flags: "err exceeds s/2".
Rubric: 2/2 all numbers. 1/2 two. Remediation: U10-C03.
B4. 14e9/2e12 = 7.00 ms -> 142.9 tok/s. INT4:
3.5e9 bytes -> 1.75 ms -> 571.4 tok/s. Strong answer:
both precisions with the full chains. Red flags: "INT4 is
4x faster for free". Rubric: 2/2 both chains. 1/2 one.
Remediation: U10-C03, C04.
B5. BT: -log sigmoid(r_c - r_r). PPO: min(rho A,
clip(rho,1-e,1+e) A). DPO: -log sigmoid(beta
(lr_c - lr_r)). Strong answer: all three losses. Red flags: mixing up the DPO margin. Rubric: 2/2 three
losses. 1/2 two. Remediation: U10-C06, C07, C08.
B6. 4 title-level (C06-C09), 7 authored bridges
(C01-C05, C10, C11), 1 audit (C12). Strong answer: the
three counts with their concept sets. Red flags: "the
unit is fully titled". Rubric: 2/2 counts plus sets. 1/2
counts only. Remediation: U10-C12.

## Deep ladder D1

D1.1. The stored K and V of past positions,
reused so decode never recomputes the prefix. Strong answer: what is stored and why. Red flags: "the cache
stores queries". Rubric: 2/2 what plus why. 1/2 what
only. Remediation: U10-C02.
D1.2. 2 12 8 64 512 2 = 12582912, per token 2
12 8 64 2 = 24576. Strong answer: both products. Red flags: dropping a factor. Rubric: 2/2 both. 1/2 one.
Remediation: U10-C02.
D1.3. Prefill attends n queries over n keys:
n^2. Decode: one query over the cached n keys:
n per step. Strong answer: both regimes with the
complexities. Red flags: "decode is n^2 per step". Rubric:
2/2 both regimes. 1/2 one. Remediation: U10-C02, C04.
D1.4. The factor-2 (K and V) forgotten: budget
6 MiB vs true 12 MiB. Strong answer: the forgotten factor
plus the budget numbers. Red flags: "6 MiB is enough".
Rubric: 2/2 factor plus numbers. 1/2 factor only.
Remediation: U10-C02.
D1.5. fp16 cache (halves), fewer layers, sliding
window. Concrete: 12 layers fp16 window 2048 at
n=8192: full 201326592 bytes = 192 MiB, windowed
~48 MiB. All fit, quality cost is the window. Strong answer: the three levers plus the concrete numbers. Red flags: "windowing is free". Rubric: 2/2 levers plus
numbers. 1/2 levers only. Remediation: U10-C02, C04.

## Deep ladder D2

D2.1. SFT: CE on demo responses. BT: pairwise
preference as sigmoid of reward margin. PPO:
clipped online policy climb. DPO: offline BT
loss on policy log-ratios. Strong answer: all four in
one line each. Red flags: "DPO needs rollouts". Rubric:
2/2 four methods. 1/2 three. Remediation: U10-C05, C06,
C07, C08.
D2.2. BT: -log sigmoid(0.9) = 0.3412. DPO: -log
sigmoid(0.07) = 0.6588. Strong answer: both numbers.
Red flags: sign errors. Rubric: 2/2 both. 1/2 one.
Remediation: U10-C06, C08.
D2.3. Sketch: the KL-regularized objective yields
pi* propto pi_ref exp(r/beta) in closed form, solve
for r =
beta log pi*/pi_ref + const, plug into BT. Strong answer: the three sketch steps. Red flags: "DPO trains
a reward model". Rubric: 2/2 three steps. 1/2 two.
Remediation: U10-C08, C09.
D2.4. The known DPO pathology: the margin rises
while both log-ratios fall. Diagnose: track
absolute log-ratios, not just the loss, the
model unlearns both, just the rejected
faster. Strong answer: the pathology plus the diagnostic.
Red flags: "a falling loss means learning". Rubric: 2/2
pathology plus diagnostic. 1/2 pathology only.
Remediation: U10-C08, C10.
D2.5. Attack: the loss measures the training
margin, quality can fall while it falls (both
log-ratios dropping, C10 hacking). Decide with
held-out win-rate + humans vs the loss curve. Strong answer: the attack plus the decision rule. Red flags:
"pick the lowest loss". Rubric: 2/2 attack plus rule.
1/2 attack only. Remediation: U10-C10, C11.

## Analytical/quantitative

A1. (1.3, 0.5): min(0.65, 0.6) = 0.6, clip
binds. (0.5, -0.4): min(-0.2, -0.32) = -0.32,
pessimism wins. Strong answer: both cases with the
verdicts. Red flags: "the clip never binds". Rubric: 2/2
both cases. 1/2 one. Remediation: U10-C07.
A2. Without bonus: A 0.9 > B 0.7, A wins.
Proxy without length: quality + 0.001
formatting-score? Any proxy has an exploit,
residual risk: the policy finds the next
loophole (Goodhart). Name it, monitor it. Strong answer:
the winner plus the Goodhart point. Red flags: "the proxy
is safe now". Rubric: 2/2 winner plus point. 1/2 winner
only. Remediation: U10-C10, C11.

## Implementation/debug

I1. Symptom: at p=0.9 the kept set is the tail
(d, c, ...) instead of the head, samples are
incoherent at any temperature. Fix: sort
descending. Assert: kept[0] == argmax and
kept set == the head indices on a fixed logit
vector. Strong answer: symptom, fix, and assert. Red flags: "raise the temperature". Rubric: 2/2 all three.
1/2 two. Remediation: U10-C01.

## Changed-constraint scenarios

S1. 1B params INT4 = 0.5e9 bytes = 0.5 GB.
KV (toy shape, int8): 6 MiB at n=512.
Total ~0.51 GB < 2 GB. Feasible. Justify:
weights dominate, pick the smallest model
that passes the quality gate. Strong answer: the budget
arithmetic plus the feasibility verdict. Red flags:
"KV dominates". Rubric: 2/2 arithmetic plus verdict. 1/2
arithmetic only. Remediation: U10-C02, C03, C04.
S2. PPO degrades faster: rollouts chase the
mislabeled rewards (exploration amplifies
noise). DPO fits the noise but stays inside
its pairs. Justify with the C06 extension:
accuracy falls with noise rate, exploration
multiplies the damage. Strong answer: the verdict plus
the mechanism. Red flags: "offline methods suffer more".
Rubric: 2/2 verdict plus mechanism. 1/2 verdict only.
Remediation: U10-C07, C08, C10.

## Research-critique

R1. Attack: 8 titles cover the RL arc, but the
unit has 12 concepts and only 4 are
title-level, inference, quantization,
hardware, SFT demos, hacking, and eval have
no titles. "Complete" needs the denominator:
4/12 title-level. Re-audit trigger: any
transcript inspection (G2) or repo content
inspection (G6) re-opens every row. Strong answer: the
denominator argument plus the re-audit trigger. Red flags: "8 titles means complete". Rubric: 2/2 denominator
plus trigger. 1/2 denominator only. Remediation: U10-C12.
