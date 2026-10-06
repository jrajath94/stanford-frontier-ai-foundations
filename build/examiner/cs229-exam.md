# EXAMINER REPORT — CS229
Gate: E1 (25+ questions, 100% answerable from lesson text alone)
Course: CS229, 17 lessons, l01 through l17
Date: 2026-10-06
Examiner role: per build/PIPELINE.md. Builder and auditor are different
agents; the examiner never fixes. This file only tests.

## Method

30 interview questions written first, then answered using ONLY the
lesson text. Each answer carries the exact lesson and line range it
comes from. A question is PASS if every claim in the answer is
derivable from the cited lines. A question is FAIL if any part of
the answer needs knowledge outside the text.

Coverage: 17 per-lesson questions (one per lesson) + 13 depth
follow-ups on the hardest topics (EM, backprop, transformers,
diffusion, RL, PPO/GRPO, GMM, PCA, contrastive, MoE, quantization,
bias-variance, GLMs).

---

## Part A: one question per lesson (17)

### Q1 (l01). Mitchell's definition says a program learns when P
improves with E. Why must P be measured on data the machine has
never trained on?

**Answer.** Because memorization scores high without generalizing.
The lesson runs the experiment: 10,000 labeled emails split into
8,000 for training and 2,000 held out. Before any training the
machine guesses at 50 percent on the held-out 2,000. After training
on the 8,000, held-out accuracy is 94 percent. After 72,000 more
labeled emails, held-out accuracy is 99 percent. P rose with E on
unseen data, so learning happened. The trap: measuring P on the
8,000 training emails gives 100 percent from a lookup table that
learned nothing. Mitchell's P must be measured on new data, or the
test cannot tell learning from memorization.

**Source:** l01-introduction.md, L97 (Mitchell, 1997, the testable
definition) and L166 (the testable experiment, worked).
**Status: PASS.**

### Q2 (l02). Derive the gradient descent update for linear
regression and name the pattern in it.

**Answer.** Start from J(theta) = 1/(2m) sum over i of
(h_theta(x^(i)) - y^(i))^2 with h = theta^T x. The update steps
each knob opposite its gradient: theta_j := theta_j - alpha *
(partial derivative of J / partial theta_j). Differentiating gives
theta_j := theta_j - alpha * (1/m) * sum over i of
(h_theta(x^(i)) - y^(i)) * x_j^(i). The shape is error times
feature, summed over examples: the error (h - y) says how wrong
the prediction was and in which direction, and the feature x_j
says how much knob j contributed. This error-times-something
pattern recurs in every learning rule in the course, including
logistic regression and backprop.

**Source:** l02-linear-regression.md, L158 (Gradient descent, by
hand) and L179 (the linear-regression update rule).
**Status: PASS.**

### Q3 (l03). Why must the sigmoid pair with cross-entropy, and what
goes wrong if you pair it with squared loss?

**Answer.** With squared loss the gradient keeps the sigmoid's
h(1-h) factor, so confident wrong predictions get near-zero
gradient. The lesson works it: one tumor, true label y = 0,
model confidently wrong at h = 0.99. Squared-loss gradient:
(h - y) * x * h(1-h) = 0.99 * 0.0099 * x = 0.0098x. The flat
tail of the sigmoid shrinks the update to 1 percent of the
error. The model is confidently wrong and barely moves.
Cross-entropy gradient on the same example: (h - y) * x =
0.99x, no shrinkage, 100 times larger. The log loss cancels
the h(1-h) factor. The loss and the output squashing are a
matched pair. Sigmoid plus squared loss is the flat-gradient
trap. Sigmoid plus cross-entropy is the design.

**Source:** l03-logistic-regression.md, L269 (the flat-gradient
trap, worked).
**Status: PASS.**

### Q4 (l04). State the three-step GLM recipe and show it
producing logistic regression.

**Answer.** Step 1: pick the exponential-family distribution for
the target. For yes-or-no targets, the Bernoulli. Step 2: set
the natural parameter linearly, eta = theta^T x. Step 3: read
off the prediction as the mean, E[y] = d/d(eta) a(eta). For the
Bernoulli, a(eta) = log(1+e^eta), so a'(eta) = e^eta/(1+e^eta)
= 1/(1+e^-eta), the sigmoid. Logistic regression drops out
with no extra assumptions: h_theta(x) = g(theta^T x) fit by
MLE. Run the same recipe on the Gaussian (a(eta) = eta^2/2,
derivative eta) and least squares appears instead.

**Source:** l04-glms-softmax.md, L151 (The GLM recipe: three
steps) and L83 (Bernoulli in the family, the algebra).
**Status: PASS.**

### Q5 (l05). GDA and logistic regression can draw the same linear
boundary. When does each win, and what is the classical result
behind it?

**Answer.** GDA wins on small data: closed-form fits squeeze more
from few examples because the Gaussian assumption does half the
work. Logistic regression wins on big data: it optimizes the
boundary directly instead of hoping the boundary falls out of
two class models. The classical result is Ng and Jordan
("On Discriminative vs. Generative Classifiers," 2001): GDA
reaches its asymptotic (higher) error with O(log n) examples,
while logistic regression needs O(n) examples to reach its
asymptotic (lower) error. With infinite data, the
discriminative model wins, because it optimizes the boundary
directly. Small data, generative. Big data, discriminative.

**Source:** l05-gda-naive-bayes.md, L387 (Ng and Jordan, the
sample-complexity duel).
**Status: PASS.**

### Q6 (l06). Ridge or Lasso: give the decision rule and the
geometry behind it.

**Answer.** Lasso when you want feature selection: its L1
penalty is a diamond whose corners sit on the axes, and the
loss contours hit those corners first, setting knobs exactly
to zero. Ridge when features correlate and should share
credit: its L2 penalty is a circle that shrinks every knob
toward zero but never quite to zero. The geometry: the
optimum sits where a loss contour first touches the penalty
shape. The diamond's corners stick out along the axes, so
first touch lands at a corner where all but one coordinate
are zero. The circle has no corners, so first touch is
almost never exactly on an axis. Decision rule: Lasso on
1,000 candidate features keeps 17 and names them. Ridge on
50 correlated sensors keeps all 50, calmed.

**Source:** l06-bias-variance.md, L321 (Lasso, the cousin that
deletes).
**Status: PASS.**

### Q7 (l07). How do residual connections fix the vanishing
gradient?

**Answer.** The block computes out = x + F(x). During
backpropagation the gradient splits into two paths. One path
goes through F, getting multiplied by the layers' small
slopes. The other path goes through the skip connection,
multiplied by exactly 1, untouched. Even if F's path
vanishes, the skip carries the full learning signal back to
every earlier layer, so gradients flow through 100+ layers.
Blocks can also default to identity (F = 0), so adding more
blocks cannot hurt: each block can choose to pass the input
through unchanged. The price: x and F(x) must have the same
shape, so the block preserves dimension.

**Source:** l07-neural-networks-1.md, L496 (Residual
connections).
**Status: PASS.**

### Q8 (l08). Why is backpropagation O(N) and not O(N^2)?

**Answer.** The naive fear was that each of the N knobs needs
its own forward pass. The chain rule shares the work. Each
module's backward function converts the gradient at its
output to the gradient at its input using only local
information: a matrix-vector product shaped like the module
itself. Chaining these in reverse visits each module once,
so the total work is proportional to the number of
operations N, the same as the forward pass. The gradient
for 1 billion knobs costs about as much as 2 or 3 forward
passes, not 1 billion. This single fact is what makes
training large models possible.

**Source:** l08-backpropagation.md, L138 (The fundamental
theorem).
**Status: PASS.**

### Q9 (l09). Prove that k-means is the zero-variance limit of a
Gaussian mixture model.

**Answer.** Take a GMM with spherical components sharing one
variance sigma^2 and uniform mixing weights. The complete-data
likelihood is a sum of Gaussians. As sigma^2 -> 0, the sum
inside the log is dominated by the nearest center: log of the
sum approaches -min_j (||x_i - mu_j||^2) / 2sigma^2. Soft
responsibilities harden into winner-take-all. Maximizing the
likelihood then becomes minimizing sum over i of min over j
||x_i - mu_j||^2, which is exactly the k-means distortion.
The geometry: k-means' hard circles are what GMMs look like
when the variance shrinks to zero.

**Source:** l09-kmeans-gmm.md, L409 (the distortion-likelihood
bridge).
**Status: PASS.**

### Q10 (l10). Prove that the likelihood never falls after one EM
round.

**Answer.** The E-step sets the new auxiliary distribution
Q_new to the posterior at the old parameters, which makes the
bound tight: B(theta_old, Q_new) = L(theta_old). The M-step
maximizes the bound over theta, so B(theta_new, Q_new) >=
B(theta_old, Q_new). The bound is always a lower bound, so
L(theta_new) >= B(theta_new, Q_new). Chaining:
L(theta_new) >= B(theta_new, Q_new) >= B(theta_old, Q_new) =
L(theta_old). Each E-step touches, each M-step climbs,
each bound chains. The guarantee is likelihood, not
convergence to a global optimum: EM is a climber, not an
oracle.

**Source:** l10-em-pca.md, L113 (why the likelihood cannot
fall).
**Status: PASS.**

### Q11 (l11). In diffusion training, why predict the noise instead
of the clean image?

**Answer.** The training loss simplifies from an ELBO over
latents to a noise-prediction MSE per level, and the network
is asked to predict each step's added noise epsilon. Noise
prediction is better conditioned: the target epsilon is
always standard Gaussian, the same scale at every timestep,
while x_0's scale varies. Predict x_0 and the network must
output a full image with fine structure at every step,
fighting different scales. Predict epsilon and the target
is always the same well-behaved distribution. Same training
distribution, better conditioning.

**Source:** l11-diffusion-models.md, L161 (The reverse process:
the learned denoiser).
**Status: PASS.**

### Q12 (l12). Compute the LoRA savings at d = 1,000, r = 10 and
explain what made it work.

**Answer.** A full weight update is d^2 = 1,000,000 degrees of
freedom. LoRA writes the update as a product of two thin
matrices, W + AB with A d-by-r and B r-by-d, costing
2*d*r = 20,000 parameters: fifty times fewer. At d = 4,096,
r = 8 the ratio is 256x. Only AB are trained; W stays
frozen. What made it work: fine-tuning moves through a
low-intrinsic-dimensional subspace, so the update's rank is
bounded and the low-rank restriction loses little. Drop-in
at inference: merge AB into W, so inference costs nothing
extra.

**Source:** l12-foundation-models.md, L215 (LoRA: adapt without
moving billions).
**Status: PASS.**

### Q13 (l13). Why do easy negatives teach the model almost nothing,
and what makes a negative "hard"?

**Answer.** The InfoNCE loss only rewards negatives that compete
with the positive. Worked: positive scores 0.9, 100 negatives
score 0.1, tau = 0.1. The fraction is 0.9676 and the loss is
0.033, barely above zero. Those 100 easy negatives are
saturated: the model already ranks them correctly, so the
gradient is near zero. Move one negative to 0.85 and the
fraction drops toward 0.62, the loss jumps to 0.494, fifteen
times hotter. The lesson's hard negative: anchor a photo of
a cat playing soccer, negative text about the FIFA World Cup.
It is wrong but semantically adjacent, so the model must learn
finer structure to rank the true pair above it. Easy
negatives teach nothing because the gradient only flows from
the ones that are already winning.

**Source:** l13-contrastive-rag.md, L480 (Hard negatives) and
L105 (the loss, audited).
**Status: PASS.**

### Q14 (l14). Why do we divide attention scores by sqrt(d)?

**Answer.** Because dot products grow with dimension and the
softmax dies. Worked: d = 64, random query and key vectors
with unit-variance entries. Dot products have variance 64,
standard deviation 8, so scores spread over roughly +-16.
Softmax of that: e^16 vs e^-16, a ratio of e^32, about
8x10^13. Attention becomes one-hot and gradients vanish,
exactly the sigmoid saturation problem from earlier
lessons. Divide by sqrt(64) = 8: the standard deviation
becomes 1, scores spread +-2, the ratio is e^4 = 54.6, and
the softmax stays soft and differentiable. The 1/sqrt(d)
is a variance stabilizer: it keeps the softmax inputs at
constant scale no matter the dimension.

**Source:** l14-transformers.md, L253 (the 1/sqrt(d) audit).
**Status: PASS.**

### Q15 (l15). Audit the KV cache: how do we get 2,730x less work
per token?

**Answer.** Without the cache, decoding position t attends to
all t previous tokens and recomputes their keys and values,
costing O(t*d) per token. Summed over a 4,096-token run at
d = 1,024: total ~ 2.29e10 units (22.9 billion), quadratic.
With the cache, each token's K and V are stored once after
its first computation; position t only computes its own
query and reuses the stored rows. Per token O(d), total ~
8.4 million units, linear. Ratio: 2.29e10 / 8.4e6 = 2,730x.
The price is memory: 512 KB per token, 2 GB for the full
4,096-token sequence.

**Source:** l15-efficient-icl-sft.md, L115 (The KV cache) and
L89 (the cubic, audited).
**Status: PASS.**

### Q16 (l16). Why is subtracting a baseline from the return
unbiased in policy gradients?

**Answer.** The estimator is E[(total - b) * grad log pi].
Expanding: E[total * grad log pi] - b * E[grad log pi]. Any
b that does not depend on the action leaves the second term
as b times the expected score, and the expected score is the
gradient of 1, which is 0. So the baseline term vanishes and
the estimator is unbiased. The best baseline is the state's
value V(s), turning raw totals into the advantage: the toy
shows raw totals of 3 and -12 collapsing to scaled updates of
7.5 and -7.5 against a baseline of -4.5, so the gradient
answers "better than expected?" instead of "was it good?"

**Source:** l16-reinforcement-learning.md, L630 (subtract the
luck) and the Q&A at L830.
**Status: PASS.**

### Q17 (l17). PPO clips the importance ratio in [0.8, 1.2].
Work all four cases and say which ones clip.

**Answer.** The objective takes min(r * A, clip(r, 0.8, 1.2) * A).
With epsilon = 0.2: A > 0, r = 1.5 (good move, jump up):
min(1.5A, 1.2A) = 1.2A, clipped, the upside is capped so a
single lucky batch cannot take the policy far. A > 0, r = 0.5
(good move, step back): min(0.5A, 0.8A) = 0.5A, unclipped,
moving toward good actions is always allowed. A < 0, r = 1.5
(bad move, jump up): min(1.5A, 1.2A) = 1.5A since A < 0,
unclipped, moving away from bad actions is never blocked.
A < 0, r = 0.5 (bad move, step back): min(0.5A, 0.8A) = 0.8A
since A < 0, clipped, the step away from a bad action is
capped too. The asymmetry: exploration toward good actions
is free, all other jumps are capped.

**Source:** l17-rl-for-llms.md, L211 (PPO: clip the jump) and
L258 (the four cases, corrected).
**Status: PASS.**

---

## Part B: depth follow-ups on the hardest topics (13)

### Q18 (EM depth, l10). The ELBO is a lower bound on the
log-likelihood. When exactly is it tight, and why?

**Answer.** The bound is tight, meaning equality holds, when the
auxiliary distribution Q(z) equals the posterior p(z|x, Theta).
Jensen's inequality is an equality when the chord collapses,
which happens for a concave function like log when the
distribution matches the one the log is taken over. This is
what makes the E-step work: setting Q to the posterior at
the old parameters makes the bound touch the likelihood,
B(theta_old, Q_new) = L(theta_old), which is the "touch"
step in the monotonicity proof.

**Source:** l10-em-pca.md, L88 (Jensen builds a lower bound).
**Status: PASS.**

### Q19 (backprop depth, l08). For one training example, why is
dL/dW always rank one, and what caps the rank for a batch?

**Answer.** Because the gradient is an outer product: dL/dW =
(backward signal) outer (forward input), a column times a row.
One example contributes one direction in activation space
times one direction in signal space, so the rank is exactly
one. In the lesson's toy, dL/dW2 = [0, -3.75] is (-2.5) times
[0, 1.5]. This is why it matters: the batch gradient is a sum
of such rank-1 terms, so its rank cannot exceed the batch
size. Full-rank updates need large batches, which is part of
why distributed training matters and why second-order
methods struggle: they invert a matrix whose effective rank
grows only with the batch.

**Source:** l08-backpropagation.md, L306 (The rank-1 gradient).
**Status: PASS.**

### Q20 (transformers depth, l14). Does causal masking save any
compute?

**Answer.** No. The causal mask zeroes the upper triangle of the
score matrix, but the standard implementation still computes
all N^2 scores. It saves nothing computationally; it only
enforces the no-peeking rule that stops position t from
attending to positions after t. True savings need different
algorithms, such as sparse attention. The quadratic price
remains: a T-by-T score matrix with inner products in d_h,
O(T^2 d_h), is the bottleneck of every transformer.

**Source:** l14-transformers.md, L304 (Causal masking: no
peeking) and L908 (The honest price: quadratic).
**Status: PASS.**

### Q21 (diffusion depth, l11). Work the classifier-free guidance
dial at w = 7.5 and say what breaks past w ~ 15.

**Answer.** The guided score is uncond + w * (cond - uncond):
the prompt's direction, amplified w times. With uncond
(0.1, 0.2) and cond (0.3, 0.1), the direction is (0.2, -0.1),
amplified 7.5x to (1.5, -0.75), added back to uncond gives
(1.6, -0.55). At w = 1 there is no amplification. Default w =
7.5 gives strong prompt following. Past about 15 the
amplified direction saturates and distorts: oversaturated
colors, warped structure, the image follows the prompt at
the cost of looking wrong. The dial trades prompt fidelity
against image quality.

**Source:** l11-diffusion-models.md, L237 (Classifier-free
guidance: the steering dial) and L248 (the dial, worked).
**Status: PASS.**

### Q22 (RL depth, l16). Derive the REINFORCE update and work the
lesson toy's single trajectory.

**Answer.** The policy gradient theorem gives
grad_theta J = E over trajectories from the current policy
of [sum over t of grad log pi(a_t | s_t) * R(tau)], where
R(tau) is the trajectory's total return. The log-derivative
trick, grad log p = grad p / p, turns the gradient of an
expectation into an expectation of a gradient weighted by
return. The update: sample trajectories, weight each step's
score by its total return, move up on good totals, down on
bad ones. Toy: at state 3 the policy has pi(right) = 0.6,
one knob theta = log-odds. One rollout: go right each time,
7 steps, total = -1*7 + 10 = 3. The update is
theta += alpha * 3 * (1 - 0.6) = theta + 0.12 at
alpha = 0.1: a good trajectory nudges the log-odds toward
right. Failure modes: high variance (one trajectory's
opinion swings wildly) and on-policy staleness (trajectories
go stale when theta moves, so every update needs fresh
rollouts).

**Source:** l16-reinforcement-learning.md, L500 (REINFORCE:
the log-derivative trick).
**Status: PASS.**

### Q23 (PPO/GRPO depth, l17). How does GRPO get advantages without
a critic, and what does the group-normalization buy?

**Answer.** Sample G responses per prompt (G = 8 to 64), score
each with the reward, and set the advantage as the
group-normalized reward: A_i = (r_i - mean(r)) / std(r).
No critic, no value head, no GAE. The lesson toy: G = 4,
rewards [1, 1, 0, 0], mean 0.5, std 0.5, advantages
[+1, +1, -1, -1]. The group mean replaces the learned
baseline, and dividing by the standard deviation puts every
prompt's advantages on the same scale: a prompt where
rewards range over [0, 100] and one where they range over
[0, 1] contribute equally. The price: everything is on-policy
and relative, so if all G samples fail, the signal is noise
among zeros and the policy has nothing to climb toward.

**Source:** l17-rl-for-llms.md, L526 (GRPO: the critic-free
answer).
**Status: PASS.**

### Q24 (GMM depth, l09). A GMM's covariance shape is a dial.
Price all four types at d = 10, k = 5 and state the rule for
choosing.

**Answer.** Full covariances: each component gets its own
d(d+1)/2 parameters, so 5 * 55 = 275. Diagonal: axis-aligned
ellipsoids, k*d = 50. Spherical: one variance per component,
k = 5. Tied: one full covariance shared by all components,
d(d+1)/2 = 55. The rule: start diagonal, go full only with
abundant data, tie when clusters share a shape. The dial
controls parameters, and with 275 free parameters on small
data the full GMM fits noise instead of clusters.

**Source:** l09-kmeans-gmm.md, L388 (GMM covariance types: the
shape dial).
**Status: PASS.**

### Q25 (PCA depth, l10). Why does PCA ship as SVD(X) and not as
the eigendecomposition of X^T X?

**Answer.** Forming X^T X squares the condition number: small
eigenvalues drown in floating-point error. SVD works on X
directly, so it keeps the small directions recoverable. The
link is exact: for centered X, the eigendecomposition of
X^T X gives eigenvalues lambda_i, and the singular values of
X are s_i = sqrt(n * lambda_i). The lesson's check:
lambda_1 = 2.5, n = 4, so s_1 = sqrt(4 * 2.5) = sqrt(10) =
3.162, matching the SVD. The PC directions are the right
singular vectors, identical to the eigenvectors. Same
answer, better arithmetic.

**Source:** l10-em-pca.md, L359 (PCA via SVD: the computation
that ships).
**Status: PASS.**

### Q26 (contrastive depth, l13). InfoNCE with N negatives can
certify at most log(N) nats. Where does the ceiling come from,
and what follows from it?

**Answer.** The InfoNCE bound on mutual information is tight to
within log(N): with N negatives the loss can certify at most
log(N) nats of shared information between the views. With
100 negatives, log(100) = 4.6 nats is the ceiling, no matter
how perfect the embeddings are. Consequence: more negatives
raise the ceiling, which is why SimCLR needed batches of
4,096 and why the field moved to memory banks and momentum
encoders to get more negatives without bigger batches. The
ceiling is the reason batch size is a hyperparameter that
directly controls how much structure the model is allowed to
certify.

**Source:** l13-contrastive-rag.md, L142 (InfoNCE, the
mutual-information origin).
**Status: PASS.**

### Q27 (MoE depth, l15). Work the router's decision for one token
and explain the load-balancing fix.

**Answer.** With E = 4 experts and top-2 routing, a token's
router produces logits [2.0, 1.0, 0.5, -1.0]. Softmax:
[0.61, 0.22, 0.14, 0.03]. Top-2 picks experts 1 and 2,
renormalized to [0.735, 0.265]. The output is the weighted
sum of the two experts' outputs. The failure mode: the
router collapses and sends every token to expert 1, wasting
the other three. The fix is an auxiliary loss,
L_aux = E * sum over i of (fraction of tokens sent to expert
i) * (mean router probability for expert i), which is
minimized when tokens spread evenly. DeepSeek-V3's
aux-loss-free alternative adds a per-expert bias that nudges
overloaded experts down without a separate loss term.

**Source:** l15-efficient-icl-sft.md, L345 (Mixture of experts:
scale parameters, not compute).
**Status: PASS.**

### Q28 (quantization depth, l15). Compare GPTQ and AWQ for 4-bit
serving and name the price of 4-bit.

**Answer.** Both store weights in 4 bits for roughly 4x smaller
weights with small quality loss, and both quantize after
training: 4-bit training is not a thing. GPTQ quantizes
layer-wise to minimize the reconstruction error, weighting
by the Hessian so directions the layer is sensitive to are
protected. AWQ protects the 1 percent of salient weights,
the ones with large activations, in higher precision and
quantizes the rest. With 4-bit, a 70B model fits on one 80GB
GPU at 35 GB of weights. The price: slower dequantization
kernels (mitigated by fused kernels), and both need
calibration data to set the scales.

**Source:** l15-efficient-icl-sft.md, L471 (GPTQ and AWQ, the
4-bit answers).
**Status: PASS.**

### Q29 (bias-variance depth, l06). Where does the double-descent
peak sit, and why?

**Answer.** The interpolation threshold is where the parameter
count roughly equals the data count: p ~ n. Below the
threshold, more parameters mean a better fit and lower test
error, the classical U. At the threshold, the model has just
enough knobs to fit every training point exactly, and there
is exactly one such interpolating function: it is jagged,
brittle, and swings wildly between points, so test error
peaks. The lesson's 12-point toy: degree 11 gives 12 knobs
for 12 points, the threshold, and test error climbs to its
maximum there (about 8). Past the threshold the model has
spare capacity, so among all interpolating functions the
optimization finds the smoothest, simplest one, and test
error descends again: degree 100 gives 0.4, degree 1,000
gives 0.25. The peak sits where capacity equals data, and
the second descent comes from implicit regularization of
the optimizer among interpolators.

**Source:** l06-bias-variance.md, L108 (Where the canon breaks:
double descent) and L132 (where the peak sits).
**Status: PASS.**

### Q30 (GLMs depth, l04). Why softmax and not something else for
K classes?

**Answer.** The lesson gives four reasons. First, it is the GLM
answer: the multinomial's canonical link produces exactly
exp(eta_k) / sum_j exp(eta_j). Second, it is smooth: every
class gets some probability, so every class gets gradient,
unlike hard argmax. Third, it is the maximum-entropy
distribution over the K classes under a mean constraint, so
it assumes nothing beyond what the model learned. Fourth,
it is numerically convenient: the log-softmax plus
cross-entropy simplifies to p - y in the gradient, the same
error-times-feature pattern as the binary case. As a check,
K = 2 reduces exactly to the sigmoid.

**Source:** l04-glms-softmax.md, L250 (Why softmax and not
something else?) and L271 (K=2 softmax is the sigmoid,
proved).
**Status: PASS.**

---

## Verdict: PASS (30/30)

All 30 questions are answerable from the lesson text alone. Every
claim in every answer traces to the cited lesson and line range.
No question required knowledge outside the text; no lesson gaps
were found during examination.

Gate E1 result: 30 questions drafted, 30 answerable, 0 unanswerable.
Requirement met (25+ at 100 percent).
