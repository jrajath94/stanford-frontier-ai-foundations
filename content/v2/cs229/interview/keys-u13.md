# Interview keys, U13

Date: 2026-10-06. Each answer gives the minimum sufficient
explanation, a strong answer, common red flags, a rubric, and
remediation. Computed values: numpy 1.26.x, float64.

## B1

Minimum: L_pre(theta) = (1/n) sum_i l_pre(x^(i), theta).
Phases: pretraining on broad (usually unlabeled) data, then
adaptation to downstream tasks.
Strong: adds that l_pre is self-supervised (built from x alone)
and that n_task is much smaller than n.
Red flags: calling pretraining supervised by default. Missing
the adaptation phase.
Rubric: 1 pt loss, 1 pt two phases, 1 pt self-supervision.
Remediation: lesson C01, SL-01.

## B2

Minimum: min_w (1/n_task) sum_i l_task(y^(i), w^T
phi_theta-hat(x^(i))). The backbone theta-hat is frozen, only
the head w trains.
Strong: notes the regression special case (15.2) and the
convexity (closed form for squared loss).
Red flags: training the backbone and calling it probing.
Rubric: 1 pt objective, 1 pt frozen backbone, 1 pt head only.
Remediation: lesson C02, SL-02.

## B3

Minimum: Delta W = B A, B in R^{d_out x r}, A in R^{r x d_in},
h = W_0 x + (alpha / r) B A x. Trainable count r(d_out + d_in).
Strong: adds B = 0 initialization (starts as the pretrained
model) and the fact that only the change is low rank.
Red flags: "the adapted matrix is low rank". Forgetting the
alpha / r scale.
Rubric: 1 pt update, 1 pt scaling, 1 pt count.
Remediation: lesson C04, SL-03.

## B4

Minimum: positive pair = two augmentations (crop, flip, color)
of the same image. Negative (random) pair = an augmentation of a
different image.
Strong: adds the false-negative caveat (a random pair can be
semantically related) and that augmentation defines the
similarity.
Red flags: "negatives are labeled as different". Missing the
augmentation step.
Rubric: 1 pt positives, 1 pt negatives, 1 pt caveat.
Remediation: lesson C06, C07, SL-04.

## B5

Minimum: L_pre = -sum_i log(exp(pos_i) / (exp(pos_i) +
sum_{j != i} exp(neg_{ij}))). Growing a positive inner product
lowers the loss, growing a negative inner product raises it.
Strong: connects to the scalar monotonicity fact and the
B-way classification reading.
Red flags: sign errors. Including j = i in the negative sum.
Rubric: 1 pt loss, 1 pt monotonicity, 1 pt j != i.
Remediation: lesson C06, SL-04.

## B6

Minimum: Recall@k = |R(q) intersect top-k| / |R(q)|, NDCG@k =
DCG@k / IDCG@k with logarithmic discount. Recall@k ignores
ranking.
Strong: writes DCG@k and IDCG@k and notes the IDCG@k = 0 edge
case.
Red flags: reporting NDCG without k. Confusing DCG with NDCG.
Rubric: 1 pt each definition, 1 pt ranking statement.
Remediation: lesson C10, SL-06.

## D1

D1.1: (15.2) with theta-hat frozen and w random, (15.3) with
both trained, theta initialized at theta-hat, w random.
D1.2: w = [1.0, 0.0], train MSE 0.0.
D1.3: the probe is linear in w with squared loss: a convex
quadratic with the normal-equation closed form. Finetuning
optimizes a nonconvex neural loss over theta with no closed
form.
D1.4: the backbone is not frozen (gradients reach theta). The
comparison is void because the "probe" is really a finetune,
any win may come from representation change, not from the
readout. Fix: freeze the backbone (no grad) and rerun.
D1.5: far shift + 40k labels: full finetuning first (data and
distance justify it), then LP-FT, then LoRA r = 8, then probe
last. The probe cannot fix wrong features. LoRA's rank
bottleneck binds under large shifts.
Rubric: 1 pt each, 2 pts for D1.5.
Remediation: SL-02.

## D2

D2.1: the SIMCLR variant of SL-04, numerator = positive pair,
denominator = positive + negatives (j != i), batch of B gives
B such terms.
D2.2: loss 3.3532, after the 10% closer step, 3.2695.
D2.3: -log(p/(p+q)) has negative derivative in p and positive
in q (E07), p = exp(positive inner product), q = sum of
exp(negative inner products).
D2.4: collapse: the loss has no pressure keeping embeddings
apart (negatives too weak, batch too small, or a symmetric
tower with no stop-gradient). Fixes: larger batches / harder
negatives, or an asymmetric architecture (predictor +
stop-gradient, BYOL-style).
D2.5: the benchmark measures its own task mix, not legal-search
quality. Decide with one evaluation: build a judgment set of
real legal queries with graded relevance on the firm's corpus
and report Recall@k and NDCG@k for both models at the k the
lawyers actually read.
Rubric: 1 pt each, 2 pts for D2.5.
Remediation: SL-04, SL-06.

## Q1

f = -log p + log(p + q). df/dp = -1/p + 1/(p+q) = -q/(p(p+q))
< 0. df/dq = 1/(p+q) > 0. Holds for p, q > 0.

## Q2

Recall@3 = 1/3 = 0.3333. DCG@5 = 3/log2(3) + 2/log2(5) +
1/log2(6) = 1.8928 + 0.8614 + 0.3869 = 3.1410. IDCG@5 = 3 +
2/log2(3) + 1/2 = 4.7619. NDCG@5 = 0.6596.

## T1

Bug 1: the denominator sums over all j including i, so the
positive is counted twice, the notes' variant sums j != i.
Fix: exclude j = i from the negative sum.
Bug 2: embeddings are not normalized, unnormalized inner
products mix direction with magnitude and one long vector can
dominate. Fix: divide each embedding by its norm first.
Check: after the fix, moving each positive pair closer must
lower the loss (the lesson's 3.4603 -> 3.3709 check), and
permuting the batch must leave the loss unchanged.
Rubric: 1 pt per bug, 1 pt per fix, 1 pt check.
Remediation: lesson C06, SL-04.

## S1

Use an approximate nearest-neighbor index: HNSW graph search,
quantization, or inverted-file/centroid indexes. The
approximation: accept a small chance of missing the exact
nearest neighbor for large speedups. Monitor: recall of the
index against brute force on a sample (index recall), and tail
latency, also rebuild on corpus change.

## S2

Positive pair: the image and its paired text (two views of one
underlying item). Negative pairs: the image with other items'
texts (and symmetrically). The SIMCLR argument that survives
unchanged: the loss still pulls the paired views together and
pushes the unpaired ones apart, with the same monotonicity,
only the definition of "two views" changes.

## R1

Hidden condition: the 40 tasks are all near the pretraining
distribution, where a low-rank update has enough capacity.
It fails under large distribution shifts or when the task
needs high-rank changes (then full finetuning wins). Boundary
experiment: sort tasks by a shift measure (e.g., probe-vs-
finetune gap), sweep the rank r in {4, 8, 32, 128} plus full
finetuning, and find the shift level where LoRA r = 8 first
underperforms full finetuning by a stated margin.
