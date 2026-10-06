# Oral defense keys, math-genai U01-U10

Date: 2026-10-06. Computed 2026-10-06 where
numbers appear. Ground truth: lessons U01-U10,
compute_run5b.py, verify_transfer_keys.py.
Interview provenance: role-derived practice, not
employer material. Format per step: strong answer,
red flags, rubric, remediation.

## O-U01, the likelihood principle

O1.1. Strong: likelihood is P(data | params)
as a function of params. probability is P(data)
as a function of data with params fixed. Red
flags: using the two interchangeably. Rubric:
2/2 both directions right. 1/2 one. Remediation:
U01-C03.
O1.2. Strong: L(0.5) = 0.125, L(0.7) = 0.7^2 *
0.3 = 0.147. p = 0.7 wins by a factor 1.176.
Red flags: picking 0.5 "because it is fair".
Rubric: 2/2 numbers plus comparison. 1/2 one
number. Remediation: U01-C03.
O1.3. Strong: l = k ln p + (n-k) ln(1-p),
dl/dp = k/p - (n-k)/(1-p) = 0 gives p = k/n.
Red flags: maximizing the likelihood instead
of the log without noting monotonicity.
Rubric: 2/2 derivative plus solution. 1/2
setup only. Remediation: U01-C11, P05.
O1.4. Strong: O(n) to count heads. At p = 0 or
1 the log-likelihood hits log 0 = -inf. use
clipping or work in logit space. Red flags:
"O(1)". Remediation: U01-C03, P02.
O1.5. Strong: MLE = 2/3 = 0.6667. MAP with
Beta(2,2): (2 + 2 - 1)/(3 + 2 + 2 - 2) = 3/5 =
0.6. The prior pulls toward 0.5. Red flags:
prior added to the MLE value. Rubric: 2/2
both numbers. 1/2 one. Remediation: U01-C11,
P07.
O1.6. Strong: a zero-probability event under
the model (unseen outcome), or a log of a
non-positive density from numerical
underflow. Red flags: "more data fixes it".
Remediation: U01-C04.
O1.7. Strong: the Bernoulli model fails first:
tickets are not binary, and IID is the lesser
sin next to a wrong outcome space. Red flags:
reciting "IID" with no reason. Remediation:
U01-C11.
O1.8. Strong: split future flips by time,
predict with the MLE, compare predicted vs
empirical frequency in bins. miscalibration
shows as systematic gaps. Red flags: "check
accuracy". Remediation: U01-C12.

## O-U02, Jensen to the ELBO

O2.1. Strong: a convex function lies below the
chord between any two of its points. its
epigraph curves upward. Red flags: "curves
up" with no chord statement. Remediation:
U02-C01.
O2.2. Strong: E[X] = 1, f(E) = 1, E[f] = 2,
gap = 1. Red flags: gap = 0 "by symmetry".
Remediation: U02-C01.
O2.3. Strong: log p(x) = log E_q[p(x,z)/q]
>= E_q[log p(x,z)/q] by Jensen (log is
concave), which is the ELBO. Red flags:
inequality flipped. Rubric: 2/2 direction
plus concavity named. 1/2 otherwise.
Remediation: U02-C03.
O2.4. Strong: draw S latents from q, average
log p(x,z) - log q(z|x). O(S) with one
decoder pass per sample. Red flags: "O(1)".
Remediation: U02-C08.
O2.5. Strong: the f-divergence dual estimates
a divergence between two fixed distributions
via a critic. the ELBO lower-bounds a
marginal likelihood for learning. Use the dual
to measure, the ELBO to train. Red flags:
"they are the same bound". Remediation:
U02-C05, C06.
O2.6. Strong: log of a non-positive number,
usually a density ratio or a variance that
went negative/zero. check the variance
parameterization first. Red flags: "increase
the learning rate". Remediation: U02-C09.
O2.7. Strong: tight when q equals the true
posterior. a loose bound hides posterior
mismatch, so ELBO gains can come from a
better q rather than a better model. Red
flags: "tight means the model is good".
Remediation: U02-C10.
O2.8. Strong: fix data and model, enrich q
(diagonal -> full -> flow), plot the ELBO
gap estimate vs family size with seeds.
predict diminishing returns. Red flags: no
controls named. Remediation: U02-C10.

## O-U03, the minimax game

O3.1. Strong: V = E_P[ln D] + E_Q[ln(1-D)],
the discriminator maximizes it, the generator
minimizes it. Red flags: swapped min/max.
Remediation: U03-C02.
O3.2. Strong: D* = [0.6, 0.47368, 0.375],
V = -1.35179. Red flags: D* = 0.5
everywhere. Remediation: U03-C03.
O3.3. Strong: pointwise maximize p ln D +
q ln(1 - D). derivative p/D - q/(1-D) = 0
gives D = p/(p+q). Red flags: missing the
pointwise argument. Remediation: U03-C03.
O3.4. Strong: one D step plus one G step
costs about 2*B*F forward/backward. the
ratio is set by n_critic. Red flags: "free".
Remediation: U03-C11.
O3.5. Strong: at D near 1 the minimax
generator gradient vanishes (log(1-D)
saturates) while the non-saturating -ln D
gradient stays alive. that is why training
uses the non-saturating form. Red flags:
"they are equivalent". Remediation: U03-C05,
C07.
O3.6. Strong: the discriminator is too strong
too early (saturation), or the generator
learning rate is effectively zero. check D's
accuracy and the gradient norms. Red flags:
"train longer". Remediation: U03-C07.
O3.7. Strong: with finite capacity D != D*,
so the JS identity is approximate. the
generator chases the discriminator's
mistakes, not the divergence. Red flags:
quoting the identity as exact in practice.
Remediation: U03-C04.
O3.8. Strong: fix the generator, train
discriminators to convergence on checkpoints,
plot the JS estimate vs blind human rankings
with several seeds. predict weak correlation.
Red flags: "FID proves quality".
Remediation: U03-C12.

## O-U04, the Wasserstein dual

O4.1. Strong: a coupling is a joint
distribution with the two given marginals. W1
is the minimum expected transport cost over
couplings. Red flags: "W1 is the mean
difference". Remediation: U04-C01, C02.
O4.2. Strong: move 0.1 mass over distance 10:
W1 = 1.0. Red flags: 10. Remediation:
U04-C02.
O4.3. Strong: W1 = sup over 1-Lipschitz f of
E_P[f] - E_Q[f]. the Lipschitz cap keeps the
sup finite and attained, without it the critic
is unbounded. Red flags: "Lipschitz speeds
training". Remediation: U04-C03, C04.
O4.4. Strong: 5 critic updates per generator
update, so the critic costs about 5x the
generator's step. the ratio is the price of a
fresh distance estimate. [Authored: n_critic =
5 is the lesson's choice.] Red flags: "1:1 is
fine". Remediation: U04-C08.
O4.5. Strong: clipping boxes the weights,
cheap but cripples capacity (the 1.0 vs 2.0
gap). the gradient penalty targets
||grad f|| = 1 on interpolations, softer and
costlier per step. Red flags: "clipping is
exact". Remediation: U04-C05, C06.
O4.6. Strong: the Lipschitz constraint
broke (weights escaped the clip, or the
penalty weight is too small). the critic is
unbounded below. Check the constraint, not
the learning rate. Red flags: "lower the
LR". Remediation: U04-C07.
O4.7. Strong: W1 on pixels prices
translations and brightness shifts that
humans ignore, and ignores texture swaps
humans notice. it is a geometric, not
perceptual, metric. Red flags: "lower W1 =
better looking". Remediation: U04-C11.
O4.8. Strong: train WGAN variants, record the
final critic gap and FID across seeds, test
correlation. predict weak-to-moderate.
Controls: same net budget. Red flags: no
budget control. Remediation: U04-C12.

## O-U05, the VAE

O5.1. Strong: the ELBO is a tractable lower
bound on log p(x): expected reconstruction
minus KL(q || prior). Red flags: "the ELBO
is the likelihood". Remediation: U05-C02.
O5.2. Strong: 0.5*(0.25 + 0.25 - 1 - ln 0.25)
= 0.44315 nats. Red flags: bits without
conversion. Remediation: U05-C05.
O5.3. Strong: log p(x) = E_q[log p(x,z)/q]
+ KL(q || p(z|x)). the first term is the
ELBO, the second the gap. Red flags:
dropped gap. Remediation: U05-C02, C10.
O5.4. Strong: z = mu + sigma*eps moves the
sampling outside the gradient path, so one
sample gives a low-variance gradient. the
score-function estimator keeps sampling
inside and pays high variance. Red flags:
"reparameterization removes randomness".
Remediation: U05-C04.
O5.5. Strong: analytic KL is exact but needs
a conjugate pair (Gaussian-Gaussian). MC KL
works for any q but adds variance. Red
flags: "analytic is always better".
Remediation: U05-C05.
O5.6. Strong: posterior collapse: the
decoder ignores z, q matches the prior, the
KL price goes to zero. Check latent usage
(mutual information), weaken the decoder or
anneal the KL. Red flags: "the KL is
converged, training is done". Remediation:
U05-C07.
O5.7. Strong: a diagonal Gaussian cannot
capture correlations or multimodality in the
true posterior. the gap term absorbs the
mismatch silently. Red flags: "more
dimensions fix it". Remediation: U05-C03.
O5.8. Strong: sweep beta over {0.5, 1, 2,
4}, measure a disentanglement score and
reconstruction per beta with seeds. predict
an inverted-U on the combined metric. Red
flags: "higher beta is always better".
Remediation: U05-C08.

## O-U06, vector quantization

O6.1. Strong: the codebook is a table of K
vectors. quantization replaces z with its
nearest code. Red flags: "the codebook is a
layer". Remediation: U06-C01, C02.
O6.2. Strong: distances [0.40, 0.80, 3.60],
winner e1, commitment loss 0.40, codebook
loss 0.40. Red flags: winner e2. Remediation:
U06-C02, C03.
O6.3. Strong: reconstruction trains the
encoder/decoder. commitment pulls z toward
the code. codebook loss pulls the code
toward z (or the EMA does). Each term has
one job. Red flags: "one loss would do".
Remediation: U06-C03.
O6.4. Strong: forward argmin, backward
identity: grad wrt z = grad wrt e_k. Lookup
is O(K*d) per vector. Red flags: "the argmin
differentiates". Remediation: U06-C04.
O6.5. Strong: gradient updates move codes by
the loss gradient (coupled to the encoder
step). EMA moves codes toward assigned-encoder
means with a decay, more stable but with a
stale-code risk. Red flags: "EMA has no
hyperparameters". Remediation: U06-C05.
O6.6. Strong: dead codes: no assignments, no
gradient, stale vectors. Fix: reset dead
codes to random encoder outputs or use a
usage-threshold reset. Red flags: "train
longer". Remediation: U06-C06.
O6.7. Strong: the STE pretends the backward
pass sees the identity. the pretense fails
when z sits far from its code, because the
copied gradient then points from the wrong
location. Red flags: "STE is exact".
Remediation: U06-C04.
O6.8. Strong: sweep K over powers of 2,
measure distortion (reconstruction) and rate
(log K per token) with seeds. predict
diminishing returns and rising dead-code
fraction. Red flags: "bigger K always wins".
Remediation: U06-C09, C12.

## O-U07, the DDPM loss

O7.1. Strong: q(x_t | x_{t-1}) adds a little
Gaussian noise each step for T steps. Red
flags: "the chain learns". Remediation:
U07-C01.
O7.2. Strong: at t = 50, 77.718 percent of
the signal variance survives. the rest is
noise. Red flags: "77 percent of the image".
Remediation: U07-C03.
O7.3. Strong: the KL between the forward
posterior and the reverse Gaussian has a
closed form. with the variance fixed, only
the means differ, and matching means is
matching epsilons up to a known scale.
Red flags: skipping why the variance can be
fixed. Remediation: U07-C07, C10.
O7.4. Strong: sample x_0, t, eps. form x_t.
one net call. MSE to eps. O(batch) net
cost. The 0-index bug: rng labels 0.T-1
while the net expects 1.T. fix by
sampling t in 1.T with ab[t-1]. Red flags:
"indexing does not matter". Remediation:
U07-C09 and E-018.
O7.5. Strong: eps multiplies 1 (full
strength). x0 multiplies 0.03956209 at t =
50 (a whisper). v mixes both. Red flags:
"parameterization is cosmetic".
Remediation: U07-C08.
O7.6. Strong: the net predicts a constant
(the loss trains, the output does not vary),
or train/serve mismatch (wrong schedule or
labeling at sampling). Check output
variance across t first. Red flags: "train
longer". Remediation: U07-C12.
O7.7. Strong: diagonal reverse steps cannot
model correlated denoising. data with strong
spatial or temporal correlation pays an
approximation error at every step. Red
flags: "Gaussian is universal".
Remediation: U07-C06.
O7.8. Strong: train two nets, ELBO-weighted
vs simplified, same budget and seeds.
compare sample quality and the t = 50
emphasis. predict the simplified net wins
perceptually but loses likelihood. Red
flags: "weighting does not matter".
Remediation: U07-C10.

## O-U08, DDIM sampling

O8.1. Strong: keep the DDPM marginals
q(x_t | x_0) for every t, drop the Markov
requirement, so the trained epsilon net
transfers unchanged. Red flags: "DDIM
retrains the net". Remediation: U08-C01.
O8.2. Strong: x_100 is the noisy start,
x0_hat = 2.13230847 is the Tweedie clean
estimate, x_90 = 1.71491474 is the jump
landing at eta = 0. Red flags: mixing up
x0_hat and x_90. Remediation: U08-C01.
O8.3. Strong: sigma = eta * sqrt((1-ab_p)/
(1-ab_t)) * sqrt(1 - ab_t/ab_p). at eta = 1
sigma^2 = 0.15405473, the 100->90 jump
posterior variance, not the single-step
0.01976684. Red flags: quoting the
single-step variance. Remediation: U08-C01
and E-019.
O8.4. Strong: 156672 FLOPs per forward:
DDPM-100 = 15.6672 MFLOP, DDIM-10 = 1.56672
MFLOP, a 10x serving cut. Red flags:
"DDIM is 10x better quality". Remediation:
U08-C11.
O8.5. Strong: proved: same training
objective and per-jump marginals. Not
proved: equal joint samples or equal quality
at 10 vs 100 steps. Red flags: "same
marginals means same samples". Remediation:
U08-C01 and interview R1.
O8.6. Strong: the ab index in x0_hat is off
by one (ab[t] instead of ab[t-1]). asserts
1-3 do not touch that index. Fix the index.
Red flags: "the net is wrong".
Remediation: U08-C12 and E-018.
O8.7. Strong: with a constant net and eta =
0 the Tweedie estimate telescopes through
every jump, so the endpoint is
schedule-invariant (measured 0.5049503 on
all schedules). It fails for a learned net
because x0_hat then varies with its input
and the telescoping breaks. Red flags:
"schedules never matter". Remediation: the
research capstone.
O8.8. Strong: freeze a trained net, sweep
S in {10, 20, 50} and schedules
{uniform, late-dense, early-dense} at matched
net-eval budgets, MMD vs DDPM-100 samples,
8+ seeds, bootstrap intervals,
pre-registered rejection rule. Red flags:
claiming the toy result transfers.
Remediation: the research capstone proposal.

## O-U09, score matching

O9.1. Strong: the score is the gradient of
the log density, pointing uphill. Red flags:
"the score is a probability". Remediation:
U09-C01.
O9.2. Strong: positive means x = 1.0 sits
left of the right mode, so the score points
right, uphill toward 1.5. Red flags: "the
sign is arbitrary". Remediation: U09-C01.
O9.3. Strong: perturb x to x_tilde, regress
s_theta(x_tilde) onto (x - x_tilde)/sigma^2.
the single-sample target is exact for the
conditional score, and averaging over the
posterior gives the marginal score. Red
flags: "the target is heuristic".
Remediation: U09-C02.
O9.4. Strong: x <- x + (a/2) s(x) +
sqrt(a) z, O(1) per step on the toy. Lesson
measured: a = 0.1 gives 56 crossings, mean
0.161. a = 0.5 gives 414 crossings, mean
0.0428: bigger steps mix faster but bias
more. Red flags: "bigger a is always
better". Remediation: U09-C03.
O9.5. Strong: DSM needs Gaussian
perturbation and has low variance where the
noise covers. sliced score matching needs no
perturbation but pays higher variance from
random projections. Red flags: "sliced is a
kind of DSM". Remediation: U09-C02.
O9.6. Strong: the step is too small for the
valley depth (p(0) = 0.008864): 56
crossings cannot balance two modes, so the
mean sticks at 0.161. Fix: anneal the noise
or run longer, and check crossings, not just
the mean. Red flags: "the score is wrong".
Remediation: U09-C03 and E-025.
O9.7. Strong: the score is undefined where
p = 0, so plain score matching never sees
those regions. the noise ladder fills low-
density regions with samples, giving the
regression targets everywhere. Red flags:
"more data fixes it". Remediation: U09-C02.
O9.8. Strong: sweep annealing schedules
(high-to-low vs constant sigma), measure
valley crossings and mode-balance bias with
seeds. predict crossings rise and bias falls
with stronger annealing up to a compute
limit. Red flags: no bias metric.
Remediation: U09-C03.

## O-U10, preference alignment

O10.1. Strong: P(prefer a over b) =
sigmoid(r_a - r_b): the win probability is a
logistic function of the reward gap. Red
flags: "BT gives rewards". Remediation:
U10-C06.
O10.2. Strong: sigmoid(0.7) = 0.66818. The
lesson toy: P = 0.7109, loss = -ln P =
0.3412, the BT negative log-likelihood.
Red flags: "the loss is accuracy".
Remediation: U10-C06.
O10.3. Strong: start from the KL-
constrained RL optimum, solve for the
reward in terms of the optimal policy, plug
into the BT likelihood: the reward model
cancels and leaves a loss on policy log-
ratios. Red flags: dropping the KL term.
Rubric: 2/2 with the cancellation named.
1/2 otherwise. Remediation: U10-C08.
O10.4. Strong: the gradient pushes up the
log-ratio on winners and down on losers,
O(1) per pair with two forward passes. No
reward model because the BT likelihood is
written directly on policy ratios. no
sampling loop because it is offline.
Red flags: "DPO needs rollouts".
Remediation: U10-C08.
O10.5. Strong: PPO assumes the reward model
generalizes to on-policy samples and needs
stable online optimization. DPO assumes the
BT model holds on the fixed pair set and
needs no environment. Red flags: "DPO is
PPO without the code". Remediation: U10-C07,
C08.
O10.6. Strong: reward hacking: the policy
exploits the preference model's blind spots
(the lesson toy: proxy 1.20 wins while truth
0.70 loses). Diagnose with held-out human
judgments, never the proxy score. Red flags:
"the metric went up, ship it".
Remediation: U10-C10.
O10.7. Strong: BT assumes preferences are
transitive and context-free. real
preferences cycle (rock-paper-scissors
tastes) and depend on the prompt framing.
Red flags: "more pairs fix it".
Remediation: U10-C06.
O10.8. Strong: sweep the KL coefficient,
measure held-out win-rate and KL drift with
seeds. predict win-rate rises then falls as
drift breaks the reference anchor. Gate on
the drift, not just the win-rate. Red flags:
"KL is just regularization". Remediation:
U10-C09, C11.
