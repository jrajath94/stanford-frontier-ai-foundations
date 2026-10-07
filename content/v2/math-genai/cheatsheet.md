# Cheatsheet: Mathematical Foundations of Generative AI

Date: 2026-10-07. Baseline: October 6, 2026. One-stop dense
reference for the 10 units. Numbers below are the lesson
toys, not universal constants.

## U01, Generative modelling basics

| Object | Symbol | Meaning |
|---|---|---|
| Data law | p_data | Unknown truth |
| Empirical law | p_hat | Counts / n |
| Model law | q or p_theta | The fitted formula |
| Log-likelihood | l = sum log q(x_i) | Score in nats or bits |
| Entropy | H = -E[log p] | Expected surprise |
| Cross-entropy | H(p,q) = -E_p[log q] | Score of q under p |
| KL | KL(p\|\|q) = H(p,q) - H(p) | Directed price of mismatch |

Toy numbers (8 files): H = 1.9056 bits. H(p_hat,
uniform) = 2.0 bits. KL = 0.0944 bits.

Rules: zero mass on real data kills the likelihood.
Held-out likelihood is the honest score. Mass sums,
density integrates.

## U02, Variational bounds

| Item | Formula |
|---|---|
| Jensen (concave f) | f(E[X]) >= E[f(X)] |
| ELBO | log p(x) >= E_q[log p(x,z) - log q(z)] |
| Gap | KL(q(z\|x) \|\| p(z\|x)) |
| KL dual | max_T E_p[T] - log E_q[e^T] |
| MC error | se = sigma / sqrt(N) |

Toy numbers: Jensen gap 0.0225. Dual 0.1838 nats =
0.2651 bits. MC N=10000: est 1.001125, se 0.006124.
Gradient check: analytic -0.3873127314, finite-diff
-0.3873127312.

Rules: every bound has a named gap. 100x samples
buys 10x precision. Check gradients against
finite differences before trusting them.

## U03, GANs

| Item | Formula |
|---|---|
| Minimax | min_G max_D E[log D(x)] + E[log(1-D(G(z)))] |
| Optimal D | D* = p_data / (p_data + p_g) |
| At optimum | max_D V = -log 4 + 2 JSD |
| Non-saturating G | max_G E[log D(G(z))] |

Toy numbers: max_D V = -1.3537, JS = 0.3231 bits,
identity exact to 4.4e-16 across the sweep.

Rules: D at 100 percent means saturation, not
convergence. Mode collapse is a recall failure
the loss cannot see. Compare generators with
recall, not with D's accuracy.

## U04, Wasserstein GANs

| Item | Formula |
|---|---|
| W1 | min_coupling E[\|x - y\|] |
| Dual | max_{1-Lipschitz f} E_p[f] - E_q[f] |
| Clipping | crude box on weights, biased |
| Gradient penalty | on interpolations, heuristic |

Toy numbers (theta = 1): JS = 1 bit, KL = inf,
both frozen. W1 = 1.0, dW1/dtheta = 1.0. Clipping
critic toy: 0.01, an underestimate.

Rules: on disjoint laws only W1 trains. The
Lipschitz cap is the whole game. Clipping
starves capacity, the penalty keeps gradients
alive.

## U05, VAEs

| Item | Formula |
|---|---|
| ELBO | E_q[log p(x\|z)] - KL(q(z\|x) \|\| p(z)) |
| Reparam | z = mu + sigma * eps, eps ~ N(0,1) |
| Gaussian KL | 0.5 (mu^2 + s2 - 1 - log s2) |
| Beta-VAE | recon - beta * KL |

Toy numbers: recon -0.7342 nats, KL 0.1766 nats,
ELBO -0.9108 nats (-1.3140 bits).

Rules: KL near 0 per dim with sharp samples =
collapse. Beta moves along rate-distortion.
The KL term is load-bearing: drop it and q
collapses to a point.

## U06, VQ-VAE

| Loss | Formula | Toy value |
|---|---|---|
| Reconstruction | \|\|x - dec(z_q)\|\|^2 | 0.05 |
| Codebook | \|\|sg[z_e] - e\|\|^2 | 0.05 |
| Commitment | 0.25 \|\|z_e - sg[e]\|\|^2 | 0.0125 |

STE: backward copies decoder grad from z_q to
z_e. Toy: copied [0,-1], true [0,0]. Biased by
design.

Rules: three losses, three owners, sg[.] is the
fence. Dead codes (zero usage) never learn:
restart near live codes. The prior over codes
is a separate AR model.

## U07, DDPM

| Item | Formula |
|---|---|
| Forward | x_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps |
| Schedule | beta 1e-4 -> 0.02, T = 100 |
| Reverse mean | c0 * x0_hat + ct * x_t |
| x0_hat | (x_t - sqrt(1-ab_t) eps_hat) / sqrt(ab_t) |
| Loss | E\|\|eps - eps_hat(x_t, t)\|\|^2 |

Toy numbers: x_50 = 1.99917538, eps_hat = 0.4
(true 0.5). x0_hat = 2.05354466, mu_rev =
2.00072225. Error 0.1 in noise moves the mean
0.0007.

Rules: predict noise (unit scale every t), not
x_0. The schedule is the curriculum. Posterior
coefficients shrink errors.

## U08, Diffusion variants

| Item | Notes |
|---|---|
| DDIM | non-Markovian, same marginals, strides |
| eta = 0 | deterministic. eta = 1 stochastic |
| Timestep embed | network must know t |
| Conditioning | class/text steers reverse |
| Loss weighting | rebalance t, do not waste on pure noise |

Toy numbers: ab_50 = 0.77718008. Jump
100 -> 90: 1.60480905 -> 1.71491474. Ten
jumps reach x_0.

Rules: DDIM changes the path, not the
destination. Spend steps where alpha_bar
moves fast.

## U09, Score models and AR models

| Item | Formula |
|---|---|
| Score | s(x) = grad_x log p(x) |
| DSM loss | E\|\|s(x_tilde) - (x-x_tilde)/s2\|\|^2 |
| Langevin | x <- x + (a/2) s(x) + sqrt(a) z |
| AR | p(x) = prod p(x_i \| x_<i) |
| Attention | O(L^2 d) per layer |

Toy numbers: s(1.0) = 1.999926, s(1.5) = 0.
DSM target -5.0. Langevin: 0, 0.0398,
0.0603, 0.3547, 0.6086, 0.6171.

Rules: the score needs no normalizer. It
explodes at boundaries (name a noise floor).
AR likelihood is exact. Sampling is serial
and suffers exposure bias.

## U10, LLM inference and alignment

| Knob | Effect (toy V=4) |
|---|---|
| T = 0.5 / 1.0 / 2.0 | pmax 0.8282 / 0.5745 / 0.4056 |
| Entropy | 0.8754 / 1.6174 / 1.9017 bits |
| Top-p | truncate tail, renormalize |

KV cache (L=12, h=8, d_h=64, n=512, fp16):
12,582,912 bytes = 12.00 MiB. 24,576
bytes per new token. Prefill 3,221,225,472
FLOPs vs decode 6,291,456 per token: ratio
512. Decode is memory-bound.

INT8 affine: scale 3.5/255 = 0.013725,
zero-point 87, max abs err 0.005882, mean
0.004167. Bytes 32 -> 8 (4x). 7B fp16:
7.00 ms/token, 142.9 tok/s ceiling. INT4:
1.75 ms, 571.4 tok/s. (Authored arithmetic,
not benchmarks.)

Alignment: SFT (format) -> reward model
(r_chosen 1.2 vs r_rejected 0.3) -> PPO
with KL leash, or DPO direct (beta 0.1,
log-ratios 0.4 / -0.3). Watch reward
hacking: the proxy is not the goal.

## Decision rules across the course

| Question | Answer |
|---|---|
| Bound or exact? | AR exact. VAE bound. Flows exact. GAN game |
| Disjoint laws? | W1, never JS/KL |
| KL ~ 0, samples sharp? | Posterior collapse |
| D at 100%? | Saturation, training stalled |
| Fewer diffusion steps? | DDIM, check schedule |
| Decode too slow? | KV cache, then quantize (memory-bound) |
| Model misbehaves after RL? | Reward hacking. Check KL to reference |
| Dead codes? | Restart near live codes |
