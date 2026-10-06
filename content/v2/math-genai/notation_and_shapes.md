# notation_and_shapes.md, math-genai symbol registry

Date: 2026-10-06. Follows the shared symbol contract in
shared/prerequisites/notation_and_shapes.md. Symbols are introduced
before first use in the U01 lesson.

## U01 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| x | one file as a vector in R^d (a 4-bit pattern in the toy) | R1 |
| d | dimension of x. Toy: d = 4 | R1 |
| x_i | i-th item of x (index). x_2 = 1 for x = [0, 1, 0, 1] | R1 |
| D | dataset, a list of n items | C01 |
| n | dataset size. Toy: n = 8 | C01 |
| p_data | true (unknown) distribution that produced D | C01 |
| p_hat | empirical distribution from D (counts / n) | C01 |
| q, q_theta | model distribution, with parameters theta | C01 |
| theta | model parameter (a number the model can change) | C01 |
| p(x), q(x) | probability the distribution assigns to item x | R5 |
| sum_x | sum over every possible x | R5 |
| E_p[f] | expectation of f under p. Sum_x p(x) f(x) | R6 |
| L(theta) | likelihood: product of q_theta over the n items of D | C03 |
| l(theta) | log-likelihood: sum of log q_theta over the n items of D | C03 |
| log | log base 2 everywhere in this course, unless stated | R9 |
| bits | unit of information. One bit is one binary choice | R9 |
| support | set of x with p(x) > 0 | C04 |
| z | latent variable (hidden, never observed) | C05 |
| p(x, z) | joint probability of x and z | C06 |
| p(x | z) | conditional probability of x given z | C06 |
| p(z | x) | posterior probability of z given observed x | C06 |
| H(p) | entropy of p: E_p[-log p(x)], in bits | C07 |
| H(p, q) | cross-entropy: E_p[-log q(x)], in bits | C08 |
| D_KL(p \|\| q) | KL divergence: H(p, q) - H(p), in bits | C09 |
| I(X, Y) | mutual information of X and Y (comma form, no semicolon) | glossary |
| p(z) | prior distribution over the latent variable | C06 |

## U02 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| lambda | convex weight in [0, 1] | C01 |
| f (generator) | convex function with f(1) = 0, selects the divergence | C05 |
| D_f(p \|\| q) | f-divergence: E_q[f(p/q)], in bits | C05 |
| r(x) | density ratio p(x)/q(x), unit-free | C04 |
| d(x) | classifier P(x came from p). r = d/(1-d) | C04 |
| T(x) | witness function in the dual | C06 |
| f*(s) | convex conjugate: sup_t (s t - f(t)) | C06 |
| sup | supremum: least upper bound | R18 |
| phi | variational family parameter vector | C02 |
| q_phi | family member at phi | C02 |
| epsilon | smoothing mass for support repair | C07 |
| N | Monte Carlo sample count | C08 |
| sigma | true std of the integrand under the sampling distribution | C08 |
| SE | standard error: sigma / sqrt(N) | C08 |
| J(a, b) | dual objective over witness parameters, in nats | C09 |
| Bernoulli(p) | coin with P(heads) = p | lesson toy |

Log base 2 in bits for divergences. Natural log (nats) inside
conjugate computations, converted at the end.

## U03 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| G, theta_G | generator network and its weights | C01 |
| D, theta_D | discriminator network and its weights | C01 |
| z | noise input to G, usually N(0, I) | C01 |
| V(D, G) | game value: E_pd[log2 D] + E_z[log2(1-D(G(z)))], bits | C01 |
| D*_G(x) | optimal discriminator: p_data(x)/(p_data(x)+p_g(x)) | C03 |
| p_g | generator's distribution (pushforward of noise) | C01 |
| m | midpoint density: (p_data + p_g)/2 | C04 |
| JS(p \|\| q) | Jensen-Shannon divergence, bits, max 1 | C04 |
| t | discriminator logit: D = sigmoid(t) | C05 |
| k | discriminator steps per generator step | C02 |
| y | conditioning label (conditional GAN) | C08 |
| E | encoder network: x -> z (BiGAN) | C10 |
| eps | clip floor keeping D in [eps, 1-eps] | C07 |

## Shape conventions for this course

- One file x is a column vector in R^d. Toy: d = 4, shape (4,).
- A dataset of n files is the list D, not one big matrix, at U01.
- Probabilities are scalars in [0, 1]. Log-probabilities are
  negative numbers or zero. Information is in bits.
- A distribution over K outcomes is a list of K numbers that sum
  to 1 (a probability vector).
## U04 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| W1(p, q) | Wasserstein-1 distance: min coupling cost, data units | C01 |
| gamma | coupling: joint with marginals p and q | C01 |
| c(x, y) | ground cost, here |x - y| | C02 |
| f | critic: real-valued witness in the dual | C03 |
| L(f) | Lipschitz constant of f | C03 |
| E_p[f] - E_q[f] | dual gap, in data units | C03 |
| sv(W) | largest singular value of W (spectral norm) | C04 |
| c | weight clip bound. Toy: c = 0.01 | C05 |
| lambda | gradient-penalty weight. Toy: lambda = 10 | C06 |
| xhat | interpolated point: eps x + (1-eps) y | C06 |
| n_critic | critic steps per generator step. Authored: 5 | C08 |
| z* | inverted latent code: argmin_z ||G(z) - x||^2 | C09 |
| E | encoder in latent regression / BiGAN (U03 reuse) | C09 |
| d | domain label bit (domain adversarial) | C09 |

## U05 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| q(z | x) | encoder: approximate posterior over the latent code | C01 |
| p(x | z) | decoder: likelihood of x given the code | C01 |
| mu(x), sigma^2(x) | encoder outputs: posterior mean and variance | C01 |
| logvar | log sigma^2, the encoder's unconstrained output | C03 |
| ELBO | E_q[log p(x | z)] - KL(q || p), lower bound on log p(x) | C02 |
| eps | frozen noise, N(0, 1), in reparameterization | C04 |
| beta | KL weight in the beta-VAE objective | C08 |
| gamma_ik | GMM responsibility of component k for point i | SB |
| KL(q || p) | 0.5 (mu^2 + sigma^2 - 1 - log sigma^2), nats | C05 |

## U06 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| e_k | codebook entry k, in R^D | C01 |
| K | codebook size. Toy: K = 4 | C01 |
| z_e | encoder continuous output | C02 |
| z_q | quantized code: e_{k*} | C02 |
| k* | argmin_k ||z_e - e_k||^2 | C02 |
| sg[.] | stop-gradient operator | C03 |
| N_k, M_k | EMA count and embedding sum for code k | C05 |
| gamma | EMA decay. Toy: 0.99 | C05 |
| R | rate: log2 K bits per code | C09 |
| D | distortion: mean ||z_e - z_q||^2 (rate/distortion only) | C09 |

## U07 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| beta_t | per-step noise variance. Toy: 1e-4 to 0.02 | C02 |
| alpha_t | 1 - beta_t | C01 |
| alpha_bar_t | cumulative product of alpha_s, s = 1..t | C03 |
| SNR_t | alpha_bar_t / (1 - alpha_bar_t) | C03 |
| x_t | noisy state at step t | C01 |
| beta_tilde_t | posterior variance: (1-ab_{t-1})/(1-ab_t) beta_t | C05 |
| c0, ct | posterior mean coefficients on x_0 and x_t | C05 |
| eps_hat | network noise prediction | C06 |
| x0_hat | predicted x_0 from eps_hat | C06 |
| mu_rev | reverse mean from x0_hat | C06 |
| sigma_t^2 | reverse variance choice (boundary) | C11 |
| L_T, L_{t-1}, L_0 | ELBO terms: prior, middle KLs, decoder | C07 |

## U08 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| tau | DDIM subsequence of timesteps, length S | C01 |
| eta | DDIM stochasticity knob in [0, 1] | C01 |
| sigma_t(eta) | per-jump noise scale | C01 |
| x0_hat | implied clean prediction from eps_hat | C01 |
| PE(t) | sinusoidal timestep embedding vector | C05 |
| w(t) | per-timestep ELBO loss weight | C06 |
| s_hat | score estimate: -eps_hat/sqrt(1-ab_t) | C07 |
| eps_u, eps_c | unconditional / conditional noise prediction | C09 |
| g | classifier-free guidance scale | C09 |
| S | number of DDIM evals | C03 |

## U09 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| s(x) | score: d/dx log p(x) | C01 |
| x_tilde | noise-perturbed sample | C02 |
| a | Langevin step size | C03 |
| P (bigram) | 4x4 next-token matrix, rows sum to 1 | C05 |
| ppl | perplexity: 2^{NLL/n} | C05 |
| A (attention) | masked softmax weights, (n, n) | C07 |
| Q, K, V | query, key, value projections | C08 |
| T (temp) | sampling temperature (U10 reuse) | C10 |
| h, L | attention heads, layers (budget) | C11 |

## U10 symbols

| Symbol | Meaning | First defined |
|---|---|---|
| T | decoding temperature | C01 |
| p (top-p) | nucleus truncation threshold | C01 |
| bpe | bytes per element (fp16: 2) | C02 |
| s, zp | quantization scale and zero-point | C03 |
| r_c, r_r | Bradley-Terry rewards, chosen/rejected | C06 |
| rho | PPO policy ratio pi/pi_old | C07 |
| e | PPO clip range. Toy: 0.2 | C07 |
| beta | DPO/KL strength. Toy: 0.1 | C08 |
| pi_ref | reference (SFT) policy | C08 |
