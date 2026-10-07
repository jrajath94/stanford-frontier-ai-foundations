# Keys: Lesson 13, Foundation models and representations

## Breadth recall

E01: L_pre(theta) = (1/n) sum_i l_pre(x^(i),
theta). Two phases: pretraining on broad
(usually unlabeled) data, then adaptation to
downstream tasks.

E02: min_w (1/n_task) sum_i l_task(y_task^(i),
w^T phi_theta-hat(x_task^(i))), theta-hat
frozen.

E03: Delta W = B A with B in R^{d_out x r},
A in R^{r x d_in}, scaled by alpha / r.
Trainable count: r (d_out + d_in).

E04: Positive pair: two augmentations of the
same image. Negative (random) pair: an
augmentation of a different image.

E05: L_pre = -sum_i log(exp(pos_i) /
(exp(pos_i) + sum_{j != i} exp(neg_{ij}))).
The loss decreases when a positive inner
product grows and increases when a negative
inner product grows.

E06: Recall@k = |R(q) intersect top-k| /
|R(q)|. NDCG@k = DCG@k / IDCG@k with DCG@k =
sum_{j=1}^{k} s*(q, d_{ij}) / log_2(j + 1)
and IDCG@k the ideal-ordering value.

## Deep oral ladders

L01: (1) (15.2) and (15.3). (2) phi(x_i) =
[i, 1], y = [1..6]: w = [1.0, 0.0], MSE 0.
(3) Normal equations: w = (Phi^T Phi)^{-1}
Phi^T y. (4) Closed form vs gradient descent
on theta and w. (5) 200 examples: probe is
safe and convex, finetuning can win but risks
distortion. LP-FT is the hedge. (6) Feature
distortion from too large a learning rate on
too few examples. (7) More parameters help
only with enough data and the right
regularization, otherwise they hurt. (8)
Nearby task, 500 labels: LP-FT or LoRA first,
full finetuning only if they underfit.

L02: (1) The SIMCLR variant of SL-04. (2)
B = 3, d = 4, seed 5: loss 3.4603, then
3.3709 after pulling positives closer. (3)
-log(p/(p+q)): derivative in p is negative,
in q is positive, for p, q > 0. (4) The
loss function plus the closer-positives
check. (5) Contrastive needs no labels and
large batches, supervised needs labels and
is more sample-efficient per example. (6)
Collapse: no negative pressure or a symmetric
architecture with no stop-gradient. (7)
Random pairs can be false negatives, the
loss then pushes apart same-class items.
(8) Fix k, fix the grading rubric, report
Recall@k and NDCG@k per query, average.

## Analytical exercises

E07: f(p) = -log(p / (p + q)) = -log p + log
(p + q). df/dp = -1/p + 1/(p+q) = -q /
(p(p+q)) < 0 for p, q > 0. df/dq = 1/(p+q)
> 0.

E08: DCG@4 = 3/1 + 0/log2(3) + 2/2 +
1/log2(5) = 4.4307. IDCG@4 = 3/1 +
2/log2(3) + 1/2 + 0 = 4.7619. NDCG@4 =
4.4307 / 4.7619 = 0.9305.

## Failure diagnosis

E09: The retriever misses but the generator
answers anyway. Diagnose: retrieval quality
is unmonitored (no Recall@k on a judgment
set), and the generator is not trained to
say "not in the passages". Fix: add a
retrieval eval, surface scores, and require
abstention or citation grounding.

## Counterfactual comparison

E10: Team A: 40 full copies of the weights,
40x serving memory, no sharing. Team B: one
base plus 40 small adapters, fast swapping,
near-simultaneous serving. A wins when
tasks are far from pretraining (large
shifts need full-rank updates) or when
latency forbids even adapter swapping.

## Research question

E11: Falsifiable claim: as the distribution
shift grows, the OOD gap between LP-FT and
plain finetuning shrinks and vanishes past a
stated shift level, because the probe
initialization no longer constrains the
needed feature change.

## Implementation task

E12: Verified by the numbers: SIMCLR loss
3.4603 -> 3.3709 when positives move closer,
LoRA counts match r(d_out + d_in) for the
rank table. Recall@k and NDCG@k match the
hand computation. See lab-07 keys.
