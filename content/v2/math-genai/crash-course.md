# Crash course: math-genai in about fifty minutes

Date: 2026-10-07. Baseline: October 6, 2026.

Ten units, about five minutes each. Every worked number below
comes from the named lesson. The crash course points, the
lessons prove. This is a provisional independent bridge built
from the course lessons. It does not claim to reproduce the
instructor's lectures. Source attribution for leaf concepts is
PENDING (see source_manifest.md, source_gaps.md).

## U01, Probabilistic generative modelling (lessons/u01)

Three objects, never one. The data law p_data is the unknown
truth. The empirical law p_hat is counts over n: the dataset
as a distribution. The model law q is the formula you fit.
Score q against data with log-likelihood, never with vibes.

The running toy: eight tiny files, 4-bit patterns.
0000 appears 3 times, 1111 appears 2 times, 0101 appears
2 times, 1010 appears 1 time. So p_hat gives 0000
probability 0.375, 1111 and 0101 probability 0.25 each,
1010 probability 0.125.

Worked number (U01): entropy H(p_hat) = 1.9056 bits.
Cross-entropy against the uniform law = 2.0 bits. So
KL(p_hat || uniform) = 2.0 - 1.9056 = 0.0944 bits. The
uniform law wastes 0.0944 bits per file on this toy.

Memory aid: mass sums to 1, density integrates to 1.
Likelihood is a function of the parameters, not the data.
Zero mass on real data is fatal: one impossible sample
kills the whole likelihood.

Self-test: Q: Why is held-out likelihood the honest score?
A: Training likelihood rewards memorization. Held-out data
punishes it.

## U02, Variational divergence minimization (lessons/u02)

Jensen's inequality is the engine of every bound in this
course. For concave f, f(E[X]) >= E[f(X)]. Log is concave,
so log of an expectation bounds the expectation of the log.
That one move creates the ELBO and every variational bound
after it.

Worked number (U02): E[X] = 0.55, f(E[X]) = 0.3025,
E[f(X)] = 0.325. Jensen gap = 0.0225. The bound sits
0.0225 below the truth, and the gap is exactly the
price of the approximation.

Divergences have dual faces. The variational dual writes
KL as a max over test functions T: KL = max_T E_p[T] -
log E_q[e^T]. Worked number (U02): dual value 0.1838
nats = 0.2651 bits, matching the direct KL to all shown
digits. The dual turns a ratio into an optimization.

Worked number (U02): Monte Carlo with N = 10000 gives
estimate 1.001125 with predicted standard error 0.006124.
Error shrinks like 1/sqrt(N): 100x samples buys 10x
precision. Gradients check out too: analytic
-0.3873127314 versus finite-difference -0.3873127312.

Memory aid: every bound has a gap, and the gap has a
name. Name it or you cannot close it.

Self-test: Q: When is a variational bound tight? A: When
the approximating family contains the truth. Then the
gap is zero.

## U03, GAN foundations (lessons/u03)

Two networks, one game. The generator fakes. The
discriminator scores real versus fake. The minimax
objective: min_G max_D E[log D(x)] + E[log(1 - D(G(z)))].
At the optimum, the discriminator reports the density
ratio D* = p_data / (p_data + p_g), and the game value
equals -log 4 + 2 * JSD(p_data || p_g). Train the
generator and you minimize the Jensen-Shannon divergence
without ever writing a density.

Worked number (U03): at the toy setting the max_D value
reads -1.3537, and 2*JS - 2 = -1.3537, so JS = 0.3231
bits. The identity holds at every generator position
in the sweep, error 4.4e-16. The math is exact, the
training is not.

The original loss saturates. When D is perfect, the
generator gradient dies. The non-saturating fix flips
the generator to maximize log D(G(z)) instead. Same
fixed point, live gradients early. Mode collapse is
the failure the loss cannot see: the generator covers
one mode, the discriminator cannot punish missing
modes, recall dies silently.

Memory aid: the discriminator is a ratio meter. Perfect
accuracy means saturation, not convergence.

Self-test: Q: The discriminator hits 100 percent on
real versus fake. Good news? A: No. The generator gets
zero gradient and training stalls.

## U04, Wasserstein and improved adversarial training (lessons/u04)

When two laws share no support, JS reads 1 bit and KL
reads infinity, and neither moves when the generator
shifts. The toy: P0 = delta_0, P_theta = delta_theta,
theta = 1. JS = 1 bit, KL = infinity, both frozen.
You need a ruler that sees the gap of 1.

The Wasserstein-1 distance is the cheapest transport
cost: W1 = min over couplings of E[|x - y|]. On the
toy, W1 = |theta| = 1.0 with gradient dW1/dtheta = 1.0.
Finite, nonzero, pointing the right way. The
Kantorovich dual turns it into a max over 1-Lipschitz
functions: W1 = max_{||f||_L <= 1} E_p[f] - E_q[f].
The critic replaces the discriminator, and the
Lipschitz cap replaces the sigmoid.

Enforcement is the whole game. Weight clipping
(C05) gives 0.01 on the critic toy: an underestimate
that starves capacity. The gradient penalty enforces
the constraint on interpolations instead: a heuristic
with live gradients, not a proof. Training stabilizes
because the critic signal stays linear in the gap.

Memory aid: JS asks "same or different". W1 asks "how
far". Only the second trains on disjoint laws.

Self-test: Q: Why does the WGAN critic drop the
sigmoid? A: It estimates a distance, not a
probability. The output must stay unbounded.

## U05, Variational autoencoders (lessons/u05)

The marginal p(x) = sum_z p(x, z) is intractable, so
Jensen (U02) builds the ELBO: log p(x) >=
E_q[log p(x|z)] - KL(q(z|x) || p(z)). Reconstruction
minus rate. The encoder proposes q, the decoder
scores, the KL prices the code.

Worked number (U05): on the toy with drawn z =
0.824264, reconstruction = -0.7342 nats (-1.0593
bits), KL = 0.1766 nats (0.2547 bits), one-sample
ELBO = -0.9108 nats (-1.3140 bits). Both terms are
costs. The sum is the bound, labeled as a bound.

The reparameterization trick moves the noise outside
the parameters: z = mu + sigma * eps. Gradients flow
through mu and sigma while eps stays frozen. The
Gaussian KL has a closed form, so no sampling is
needed for the rate term.

Posterior collapse is the silent killer: a powerful
decoder ignores z, the KL reads ~0, and the model is
an autoregressive memorizer wearing a VAE costume.
Beta scales the rate price along the rate-distortion
frontier. Diagnose collapse by the KL per dim, not
by the reconstructions.

Memory aid: rate buys distortion. Dead latents cost
nothing and explain nothing.

Self-test: Q: KL reads 0.0003 per dim but samples
look sharp. What happened? A: Collapse. The decoder
memorized and ignores z.
## U06, Discrete latent modelling and VQ-VAE (lessons/u06)

Continuous latents blur. Quantize them: the encoder
emits z_e, the nearest codebook vector e_1 wins, the
decoder sees only z_q. Three losses train the three
parts. Reconstruction pulls the decoder. The codebook
loss pulls the winner toward the encoder. The
commitment loss pulls the encoder toward the winner.

Worked number (U06): z_e = [0.9, 0.2], winner e_1.
Reconstruction ||x - decoder(z_q)||^2 = 0.05.
Codebook ||sg[z_e] - e_1||^2 = 0.05. Commitment
0.25 * ||z_e - sg[e_1]||^2 = 0.0125. The
stop-gradient sg[.] routes each loss to its owner:
no owner, no learning.

Argmin has no gradient, so the straight-through
estimator copies the decoder gradient from z_q to
z_e as if quantization were the identity. Worked
number (U06): copied STE gradient [0, -1] versus
true gradient [0, 0]. The copy is biased by
construction. It works when z_e sits close to z_q,
which the commitment loss enforces.

Dead codes never win and never learn. EMA updates
track usage counts N_k and sums m_k instead of
backprop through the winners. The prior over codes
is a separate autoregressive model: the VQ-VAE
compresses, the prior generates.

Memory aid: three losses, three owners. The
stop-gradient is the fence between them.

Self-test: Q: Half the codebook has zero usage
after training. What happened? A: Dead codes.
Winners took all. Restart them near live codes or
raise the commitment weight.

## U07, DDPM derivation and parameterizations (lessons/u07)

Destroy, then learn to un-destroy. The forward chain
adds Gaussian noise on a schedule: beta_1 = 0.0001
to beta_100 = 0.02, T = 100 steps. Direct noising
jumps to any step: x_t = sqrt(alpha_bar_t) * x_0 +
sqrt(1 - alpha_bar_t) * eps. The reverse step
removes the predicted noise.

Worked number (U07): x_0 = 2.0, drawn noise eps =
0.5. At t = 50, x_50 = 1.99917538. The network
predicts eps_hat = 0.4 (true 0.5, error 0.1).
Estimated clean x0_hat = 2.05354466, reverse mean
mu_rev = 2.00072225. The 0.1 noise error moved the
mean by 0.0007. The posterior coefficients shrink
errors: small mistakes in noise space stay small
in data space.

The ELBO collapses to the epsilon loss: predict
the noise, nothing else. The parameterization
choice (noise, clean x_0, or score) changes what
the network outputs but not the sampling rule.
Boundary variance needs care: at t near 0 the
posterior variance has two defensible choices
and the wrong one shows in samples.

Memory aid: the schedule is the curriculum. Too
fast and the signal dies early. Too slow and you
waste steps on pure noise.

Self-test: Q: Why predict noise instead of x_0
directly? A: The noise target has unit scale at
every t. x_0 prediction mixes scales across the
chain and trains worse.

## U08, Diffusion variants and implementation (lessons/u08)

DDPM needs all 100 steps. DDIM keeps the same
marginals with a non-Markovian construction and
strides the chain. Deterministic when eta = 0,
stochastic when eta = 1. Fewer steps, same
endpoints.

Worked number (U08): alpha_bar_50 = 0.77718008.
One DDIM jump from 100 to 90: x_100 =
1.60480905, predicted clean x0_hat =
2.13230847, x_90 = 1.71491474. Ten such jumps
reach x_0. Ten network calls instead of one
hundred, same toy start.

The network must know what time it is. The U-Net
takes the timestep embedding alongside the noisy
input: same weights, time-conditioned behavior.
Conditioning (class labels, text) steers the
reverse process. Loss weighting rebalances the
t values: uniform weighting wastes capacity on
near-pure noise.

Memory aid: DDIM changes the path, not the
destination. The marginals match by construction.

Self-test: Q: DDIM with 10 steps looks worse
than DDPM with 100. What do you check first?
A: The step schedule. Even spacing is rarely
optimal. Spend steps where alpha_bar moves fast.

## U09, Score-based modelling and autoregressive models (lessons/u09)

The score is grad_x log p(x): it points uphill
with no normalizer. Denoising score matching
fits it by regression: perturb x with Gaussian
noise, train the network to predict
(x - x_tilde)/sigma^2. Low-density regions get
signal because the noise puts them there.

Worked number (U09): two-mode mixture. s(1.0) =
1.999926, s(1.5) = 0, s(-1.0) = -1.999926.
Zero at the modes and the valley, positive left
of the right mode (pointing right, uphill).
DSM target at x_tilde = 1.7 from clean x = 1.5
with noise var 0.04: target = -5.0. Langevin
dynamics then walks uphill with calibrated
noise: 5 steps from 0 read 0.0, 0.039759,
0.060272, 0.354732, 0.608576, 0.617063,
drifting toward the right mode.

The SDE/ODE bridge unifies it: the forward SDE
destroys, the reverse SDE (or probability-flow
ODE) rebuilds, and the score is the only learned
object. DDPM is one discretization of this
picture.

Then the other family: autoregressive models.
The chain rule factors the joint exactly:
p(x) = prod p(x_i | x_<i). Teacher forcing
trains on true prefixes. Sampling feeds the
model its own outputs (exposure bias). The
causal mask is the arrow of time. The
transformer implements it with attention at
O(L^2 d) per layer and a KV cache at sampling.

Memory aid: the score needs no partition
function. The chain rule needs no
approximation. Pick your poison honestly.

Self-test: Q: Why does the score blow up at
distribution boundaries? A: log p dives to
negative infinity there, so its gradient
explodes. Likelihoods need a named noise
floor.

## U10, LLM inference, quantization, and alignment (lessons/u10)

Training is parallel, generation is serial.
Temperature reshapes the draw without
retraining: softmax(z/T). Worked number (U10):
T = 0.5/1.0/2.0 gives pmax 0.8282/0.5745/0.4056
and entropy 0.8754/1.6174/1.9017 bits. Cold
sharpens, hot flattens. Top-p truncates the
tail and renormalizes.

The KV cache is the whole economics. Attention
at step n+1 needs K and V for positions 1..n:
cache them. Worked number (U10): L = 12, h =
8, d_h = 64, n = 512, fp16: cache =
12,582,912 bytes = 12.00 MiB, 24,576 bytes
per new token. Prefill costs 3,221,225,472
FLOPs versus 6,291,456 per decode token: ratio
512. Decode is memory-bound, so shrinking the
model speeds it up linearly.

Worked number (U10): INT8 affine on a 4x2 toy:
scale = 3.5/255 = 0.013725, zero-point 87,
max abs error 0.005882, mean abs error
0.004167. Bytes 32 -> 8, a 4x win. Roofline:
7B fp16 at 14 GB and 2 TB/s gives 7.00
ms/token, 142.9 tok/s ceiling. INT4 at 3.5 GB
gives 1.75 ms, 571.4 tok/s. All C04 numbers
are authored arithmetic, not benchmarks.

Alignment, three stages. SFT teaches the format
on demonstrations. Reward modelling scores
pairs: r_chosen = 1.2 versus r_rejected = 0.3
on the toy. PPO climbs the reward with a KL
leash to the reference. DPO skips the reward
model and pushes the log-ratio margin directly,
beta = 0.1 on the toy, with the same KL
regularizer keeping the policy near the
reference. Reward hacking is the failure mode:
the policy games the proxy, not the goal.

Memory aid: decode reuses the past, quantization
shrinks the present, alignment steers the
future. Three budgets, three knobs.

Self-test: Q: Doubling the context length
doubles decode latency per token? A: Roughly
yes for attention over the cache (O(n) per
step), but the cache bytes double too. Both
budgets move together.

## The whole course in ten lines

U01: three laws (data, empirical, model). Score
with logs. Zero mass kills.
U02: Jensen builds every bound. The gap always
has a name. Monte Carlo error falls as
1/sqrt(N).
U03: the game minimizes JS at an optimum that
never holds. Saturation kills gradients, not
convergence.
U04: JS and KL freeze on disjoint laws. W1 =
|theta| moves with gradient 1.
U05: ELBO = reconstruction minus KL. The
reparameterization trick routes gradients.
Watch for collapse.
U06: three losses, three owners. The
straight-through copy is biased by design.
Dead codes never learn.
U07: destroy on a schedule, predict the noise.
0.1 noise error moves the mean 0.0007.
U08: DDIM strides the chain, ten steps for
one hundred. The network must know the time.
U09: learn the uphill direction, walk it with
noise. Or factor the joint exactly and pay
O(L^2).
U10: cache the past (12 MiB per 512 tokens).
Shrink the weights (4x for INT8). Steer with
preferences (KL leash on).
