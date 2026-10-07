# keys.md, U06 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

b = (log10 L2 - log10 L1) / (log10 N2 - log10 N1).
a = log10 L1 - b * log10 N1. L(N) = 10^(a + b log10 N).

## E02

b = -0.0748 per decade, a = 1.0534. L(1e11) = 1.70.

## E03

The recipe changed between runs (data mix, optimizer),
or the small runs sat above the floor and the big run
approaches it. Also seed luck on a single big run.

## E04

The N-only law predicts no change. It is wrong because
loss depends on data scale and quality too. The law held
N fixed but the true driver moved.

## E05

Train three toy scales on fixed data, fit on two,
predict the third. Hypothesis: prediction within 5% of
measured. Report all three numbers.

## E06

k-shot: k examples placed in the prompt, no weight
update. The model conditions on the examples. The
weights stay frozen.

## E07

Deltas: 0.09, 0.07, 0.03. Saturation: the last delta
is one third of the first (0.03/0.09 = 0.33).

## E08

Noisy examples dilute the format signal. The context
fills with contradictions, so the conditioning gets
worse, not better.

## E09

Large-k few-shot dies first. The maximum usable k
halves, and the saturation point moves to smaller k.

## E10

Hypothesis: A(k) saturates. The marginal gain of the
last 5 examples is under 0.02. Report the curve.

## E11

The threshold artifact: a step metric on a smooth
skill curve manufactures a jump at the crossing
point.

## E12

N* = 1e12. Continuous gap from 1e9 to 1e12: 0.70 -
0.55 = 0.15, while the metric jumps 1.0.

## E13

Check the continuous score alongside the threshold
metric. If the score climbs smoothly while the
metric jumps, the jump is a measurement artifact.

## E14

The jump disappears. The metric becomes the ramp
itself.

## E15

Hypothesis: with the continuous score, the jump
flattens into a ramp. Report both curves.

## E16

Loss = -mean over response tokens of log P(y_t | x,
y_<t). Instruction tokens carry zero weight.

## E17

Loss = 0.3583. Response perplexity = e^0.3583 =
1.4309 (observed, the lesson asserts within 0.001).

## E18

The mask leaked: instruction tokens entered the
loss, so the model learned to predict instructions.

## E19

Instruction prediction improves, response quality
drops. The model spends capacity on the wrong
target.

## E20

Hypothesis: mask-off loss is lower on instructions
but response quality is worse. Report both.

## E21

P(A beats B) = sigmoid(r_A - r_B). RL objective:
E[r] - beta * KL(policy || SFT).

## E22

P(A wins) = sigmoid(0.4) = 0.5987. Objective = 1.0 -
0.1 * 2.5 = 0.75.

## E23

Reward hacking: the policy games the reward model
while true quality falls.

## E24

Beta = 0 removes the leash. The policy drifts far
from SFT and exploits reward-model blind spots.

## E25

Hypothesis: reward rises then plateaus while KL
keeps growing. The knee marks hacking onset.
Report the curve.

## E26

The model critiques its draft against the written
constitution, revises it, and the (revised, draft)
pair trains the reward model.

## E27

Weights (0.6, 0.4): original 0.74, revised 0.86,
margin 0.12. Weights (0.4, 0.6): original 0.66,
revised 0.84, margin 0.18.

## E28

The judge rewards verbosity, or no principle
penalizes length. The weights moved toward
verbosity, or the constitution never priced it.

## E29

The policy learns to ignore helpfulness. It
maximizes harmlessness only, since that is all the
judge scores.

## E30

Hypothesis: mean response length falls
monotonically as the verbosity principle weight
rises. Report the curve.

## E31

W = W0 + (alpha/r) B A. Trainable params: r(d+k)
vs full dk.

## E32

LoRA: 16 * 8192 = 131,072. Full: 4096^2 =
16,777,216. Ratio: 128x.

## E33

Rank 8 is too low for the task. The needed update
is not near a rank-8 subspace.

## E34

LoRA with r < d cannot represent the full-rank
update. It keeps only the top-r directions and
drops the rest.

## E35

Hypothesis: the r-sweep saturates. R = 4 reaches
95% of the r = 16 score. Report the curve.

## E36

Step time = max(t_compute, t_fetch + t_assemble)
with full prefetch overlap.

## E37

Fast disk: 2.0 s, 100%. Slow disk: 2.8 s, 71.4%.
Doubled workers: 2.0 s, 100%.

## E38

The disk or the metadata server is the bottleneck.
Worker parallelism cannot beat the storage
ceiling.

## E39

No overlap: step time = 2.0 + 0.5 + 0.3 = 2.8 s,
always data-bound.

## E40

Hypothesis: step time falls with worker count then
plateaus. The plateau names the true bottleneck.
Report the sweep.

## E41

Yield = product of keep rates. Tokens out = tokens
in * yield.

## E42

Yield 0.68, tokens out 340,000. With q2 = 0.5:
yield 0.425, tokens out 212,500.

## E43

The filter dropped code and math. Odd punctuation
looked like low quality to the classifier.

## E44

Dedup runs on the smaller filtered set, so it
costs less. The yield product is about the same. 
the risk is the filter was tuned on duplicated
data.

## E45

Hypothesis: filtered wins per token at fixed
budget, but the gap shrinks as tokens grow.
Report both curves.

## E46

Packing concatenates short sequences into full
bins. A document mask blocks cross-document
attention inside each bin.

## E47

Naive: 2400/4800 = 50%. Packed: 2400/3072 =
78.1%. Pad tokens: 2400 -> 672 (3.6x fewer).

## E48

The document mask is broken. Tokens attend across
document boundaries, so long-range coherence is
trained on false bigrams.

## E49

The same documents co-occur in the same bins every
epoch. The model learns bin-level correlations:
an order bias.

## E50

Hypothesis: packing reaches the same loss in fewer
steps because each step carries more real tokens.
Report both curves.

## E51

R = B * T / t_step tokens per second. Cost per
million = 1e6 / R * dollars per second.

## E52

R = 8192 tok/s. Cost = $0.81 per million tokens at
the toy price. Packing lift 1.56x: 12,779 real
tok/s, $0.52 per million.

## E53

Pad tokens inflate R but teach nothing. The metric
to watch is time-to-loss on real tokens, not raw
throughput.

## E54

Cost per token doubles. Throughput halves at fixed
price.

## E55

Hypothesis: R rises by the factor the step-time
model predicts. Report both.

## E56

Fix code and data order per seed, vary only the
seed over n runs, report mean plus spread. A
recipe wins only if the gap clears the spread.

## E57

Mean 2.31, std 0.0158. Recipe B at 2.295: gap
0.015 < 1 std. Call it a tie. Keep the simpler
recipe.

## E58

Nondeterministic GPU kernels (atomic adds), or the
data loader order was not seeded.

## E59

Almost nothing. One run is a smoke test, not a
recipe comparison.

## E60

Hypothesis: same-seed losses match to 1e-4 across
GPU types if kernels are deterministic, else they
drift. Report both.
