# glossary.md, math-genai U01 terms

Date: 2026-10-06. One-line definitions, module pointers for depth.

- Data distribution (p_data): the unknown law that produced the
  dataset. U01-C01.
- Model distribution (q): the formula we fit to stand in for
  p_data. U01-C01.
- Empirical distribution (p_hat): counts divided by n. The dataset
  as a distribution. U01-C01.
- Density: probability per point for continuous x. U01-C02.
- Sample: one observed draw from a distribution. U01-C02.
- Likelihood (L): product of model probabilities over the dataset.
  U01-C03.
- Log-likelihood (l): sum of log model probabilities. U01-C03.
- Support: outcomes with positive probability. U01-C04.
- Latent variable (z): an unobserved variable that explains
  structure in x. U01-C05.
- Marginalization: sum (or integral) over z to get p(x) from
  p(x, z). U01-C06.
- Joint (p(x, z)): probability of x and z together. U01-C06.
- Conditional (p(x | z)): probability of x given z. U01-C06.
- Posterior (p(z | x)): probability of z after seeing x. U01-C06.
- Entropy (H): expected surprise of p, in bits. U01-C07.
- Surprisal: -log p(x). Bits needed to encode one draw. U01-C07.
- Cross-entropy (H(p, q)): expected surprise of p measured with
  q's probabilities, in bits. U01-C08.
- KL divergence (D_KL): cross-entropy minus entropy. Extra bits
  paid for using q instead of p. U01-C09.
- Divergence: any score of difference between two distributions.
  U01-C10.
- Estimator: a rule that turns data into a parameter value.
  U01-C11.
- Maximum likelihood (MLE): the estimator that maximizes
  log-likelihood. U01-C11.
- Bias: systematic error of an estimator. U01-C11.
- Held-out data: data the model did not train on. U01-C12.
- Evaluation: scoring a model on held-out data or on samples.
  U01-C12.
- Overfit: fit the training data at the price of held-out
  performance. U01-C12.
- Mutual information (I(X, Y)): reduction in uncertainty about X
  from knowing Y, in bits. RUN 2.
- Parameter (theta): a knob inside the model family. U01-C01.

## U02 terms

Date: 2026-10-06.

- Convex function: every chord lies above the graph. U02-C01.
- Jensen inequality: f(E[X]) <= E[f(X)] for convex f. U02-C01.
- Jensen gap: E[f(X)] - f(E[X]), non-negative for convex f.
  U02-C01.
- Variational family (Q): restricted parameterized set of
  candidate distributions. U02-C02.
- Lower bound: tractable E_q[log p(x,z) - log q(z)] below log
  p(x). U02-C03.
- Density ratio (r): p(x)/q(x), unit-free. U02-C04.
- f-divergence (D_f): E_q[f(p/q)] for convex f, f(1) = 0.
  U02-C05.
- Generator (f): the convex knob selecting the divergence.
  U02-C05.
- Convex conjugate (f*): sup_t (s t - f(t)). U02-C06.
- Witness (T): the function maximized in the dual. U02-C06.
- Primal: minimize the divergence over distributions. U02-C06.
- Dual: maximize over functions, needs samples only. U02-C06.
- Support mismatch: p > 0 where q = 0. U02-C07.
- Smoothing: epsilon mass on every outcome. U02-C07.
- Monte Carlo: sample-mean estimate of an expectation. U02-C08.
- Standard error: sigma/sqrt(N). U02-C08.
- Approximation gap: restricted-class sup below the full sup.
  U02-C10.
- Estimation gap: empirical dual versus population dual.
  U02-C10.
- Optimization gap: found maximizer versus best in class.
  U02-C10.
- Consistency: convergence to the truth as N and the class
  grow under four assumptions. U02-C11.

## U03 terms

Date: 2026-10-06.

- Generator (G): network mapping noise z to samples. U03-C01.
- Discriminator (D): network mapping x to (0, 1), its bet
  that x is real. U03-C01.
- Game value (V): E_pd[log2 D] + E_z[log2(1-D(G(z)))],
  in bits. U03-C01.
- Minimax: min over G of max over D of V. U03-C02.
- Nash equilibrium: p_g = p_data, D = 1/2, V = -2 bits.
  U03-C02.
- Optimal discriminator (D*): p_data/(p_data+p_g), the
  Bayes ratio. U03-C03.
- JS divergence: (KL(pd||m)+KL(pg||m))/2, m the midpoint.
  Max 1 bit. U03-C04.
- Saturation: minimax G gradient -D dying where D is
  confident. U03-C05.
- Non-saturating loss: E_z[-log D(G(z))], gradient
  -(1-D). U03-C05.
- Mode collapse: generator assigns ~0 mass to a data
  mode. U03-C06.
- Log-boundary explosion: 1/(D ln 2) gradient spikes as
  D -> 0. U03-C07.
- Conditional GAN: G(z,y), D(x,y), one game per class.
  U03-C08.
- DCGAN: strided convs, batchnorm, ReLU/LeakyReLU, tanh
  output. U03-C09.
- BiGAN: pair discriminator D(x,z) matching the G-joint
  and the E-joint. U03-C10.
- Update balance: k D-steps per G-step. U03-C11.
- Coverage: generator mass near a data mode. U03-C12.
- Coupling (gamma): a joint distribution with marginals p and
  q. A transport plan. U04-C01.
- Transport cost: sum gamma(i,j) |x_i - y_j|, the work of a
  plan. U04-C01.
- Wasserstein-1 (W1): the cheapest transport cost over all
  couplings. In data units. U04-C02.
- Ground cost: c(x, y), the per-unit move price. Here |x - y|.
  U04-C02.
- Kantorovich-Rubinstein dual: W1 = max over 1-Lipschitz f of
  E_p[f] - E_q[f]. U04-C03.
- Critic (f): the real-valued witness in the W dual. Not a
  classifier. U04-C04.
- Lipschitz constant (L): the maximum slope of f. U04-C04.
- Spectral norm: the largest singular value of a weight matrix.
  The affine map's Lipschitz constant. U04-C04.
- Weight clipping: forcing every weight into [-c, c] after each
  critic step. U04-C05.
- Gradient penalty: lambda E[(||grad f(xhat)|| - 1)^2] on
  real-fake interpolation points. U04-C06.
- Interpolated point (xhat): eps x + (1-eps) y, eps in [0,1].
  U04-C06.
- Calibration ratio: critic gap divided by exact W1 on a
  reference pair. U04-C07.
- Critic frequency (n_critic): critic steps per generator step.
  U04-C08.
- GAN inversion: z* = argmin ||G(z) - x||^2, the latent code
  for a target. U04-C09.
- Latent regression: a learned encoder E trained on (G(z), z)
  pairs. U04-C09.
- Domain adversarial training: task loss minus lambda times
  domain loss, with the domain gradient reversed. U04-C09.
- Gradient reversal: flipping the sign of the domain-classifier
  gradient into the feature extractor. U04-C09.
- Notebook evidence: opened, executed, hash-recorded artifact
  content. File names are not evidence. U04-C10.
- Fair comparison: same architectures, same seeds, fixed flop
  budget, reported calibration and coverage. U04-C12.

## U05 terms (2026-10-06)

- Prior p(z): the model's belief about the latent code before
  seeing x. U05-C01.
- Encoder q(z | x): the approximate posterior. Outputs mu and
  logvar per x. U05-C01.
- Decoder p(x | z): the likelihood of x given a code. U05-C01.
- ELBO: E_q[log p(x | z)] - KL(q || p). A lower bound on log
  p(x). U05-C02.
- Reconstruction term: E_q[log p(x | z)]. Rewards codes that
  decode well. U05-C02.
- Reparameterization: z = mu + sigma eps with eps ~ N(0, 1)
  fixed. Makes sampling differentiable. U05-C04.
- Score-function gradient: E[f(z) grad log q(z)]. Valid for any
  q, high variance. U05-C04.
- Posterior collapse: q(z | x) = p(z) for all x. The model
  ignores the latent code. U05-C07.
- Beta-VAE: E_q[log p(x | z)] - beta KL. One knob trades
  reconstruction against latent use. U05-C08.
- Latent traversal: sweep z, decode each point. The cheapest
  collapse diagnostic. U05-C09.
- Amortized inference: one trained encoder answers every new x
  in one forward pass. U05-C11.
- Responsibility: gamma_ik, the soft component assignment in a
  GMM E-step. U05 source block.
- EM: alternate soft assignments (E-step) and weighted updates
  (M-step). Monotone likelihood. U05 source block.

## U06 terms (2026-10-06)

- Codebook: the table of K learned vectors in R^D. The discrete
  alphabet. U06-C01.
- Vector quantization: replace z_e with the nearest codebook
  entry. U06-C02.
- Stop-gradient (sg[. ]): freezes its argument in backprop.
  U06-C03.
- Codebook loss: ||sg[z_e] - e_k||^2. Moves the entry toward
  the encoding. U06-C03.
- Commitment loss: beta ||z_e - sg[e_k]||^2. Moves the encoding
  toward the entry. U06-C03.
- Straight-through estimator: forward z_q, backward copy the
  gradient to z_e. Biased but useful. U06-C04.
- EMA update: the codebook tracks running means of assigned
  encodings with horizon 1/(1 - gamma). U06-C05.
- Dead code: an entry no z_e ever selects. Usage 0. U06-C06.
- Effective K: the number of codes with nonzero usage. U06-C06.
- Index prior: p(k) over code indices. Stage 2 of the VQ-VAE
  generative story. U06-C07.
- Rate: log2 K bits per code. U06-C09.
- Distortion: mean ||z_e - z_q||^2. U06-C09.
- Controlled ablation: vary exactly one factor, hold seed/data/
  architecture fixed. U06-C12.

## U07 terms (2026-10-06)

- Forward process: the fixed Markov chain q(x_t | x_{t-1}) =
  N(sqrt(1 - beta_t) x_{t-1}, beta_t). U07-C01.
- Noise schedule: the list beta_1..beta_T. U07-C02.
- Alpha_bar_t: the cumulative product of (1 - beta_s). The
  signal fraction surviving at t. U07-C03.
- SNR_t: alpha_bar_t / (1 - alpha_bar_t). U07-C03.
- Direct noising: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 -
  alpha_bar_t) eps. One jump, no chain. U07-C04.
- Posterior coefficient: the blend weights on x_0 and x_t in
  q(x_{t-1} | x_t, x_0). U07-C05.
- Beta_tilde_t: (1 - alpha_bar_{t-1})/(1 - alpha_bar_t) *
  beta_t. The posterior variance. U07-C05.
- Reverse Gaussian: p(x_{t-1} | x_t) = N(mu_rev, sigma_t^2)
  with x_0 replaced by x0_hat. U07-C06.
- Epsilon prediction: the network outputs eps_hat. x0_hat
  follows by algebra. U07-C08.
- Simplified loss: ||eps - eps_hat(x_t, t)||^2 over uniform t.
  U07-C10.
- Boundary variance: the ELBO-underdetermined choice of
  sigma_t^2 (beta_t, beta_tilde_t, or 0). U07-C11.
- Sampling loop: from x_T ~ N(0, 1), iterate the reverse
  Gaussian T times. U07-C12.

## U08 terms (2026-10-06)

- DDIM: a non-Markovian diffusion construction with the same
  marginals as DDPM, allowing big jumps. U08-C01.
- Marginal contract: q(x_t | x_0) fixed, process free. The
  condition that lets one net serve many samplers. U08-C01.
- Eta: the stochasticity knob: 0 is deterministic (ODE), 1 is
  stochastic (SDE-like). U08-C02.
- Step schedule (tau): the subsequence of timesteps the sampler
  visits. U08-C03.
- Timestep embedding: sinusoidal vector form of t. U08-C05.
- Loss weighting: per-timestep multiplier, ELBO vs simplified.
  U08-C06.
- Tweedie estimate: x0_hat from the score estimate. U08-C07.
- Classifier-free guidance: eps_u + g (eps_c - eps_u). U08-C09.
- Hit-rate: toy quality proxy P(|x0_hat - target| < 0.5). U08-C10.
- Check battery: the six-assert sampler contract test. U08-C12.

## U09 terms (2026-10-06)

- Score: grad log p(x), the uphill vector field. U09-C01.
- Denoising score matching: regression of the score on
  noise-perturbed samples. U09-C02.
- Langevin dynamics: score drift plus noise for MCMC sampling.
  U09-C03.
- Probability-flow ODE: the deterministic twin of the diffusion
  SDE with the same marginals. U09-C04.
- Chain rule: p(x_1..x_n) = prod p(x_i | x_<i). U09-C05.
- Teacher forcing: training each factor on the true prefix.
  U09-C06.
- Exposure bias: the train/test prefix mismatch and its cost.
  U09-C06.
- Causal mask: lower-triangular attention mask. U09-C07.
- Perplexity: 2^{NLL/n}, the effective branching factor. U09-C09.
- Context budget: the O(n^2) attention memory price of n.
  U09-C11.

## U10 terms (2026-10-06)

- Temperature: the softmax sharpening/flattening knob. U10-C01.
- Top-p (nucleus): truncation to the smallest set with mass >= p.
  U10-C01.
- KV cache: stored keys and values making decode O(n) per token.
  U10-C02.
- Prefill: the one O(n^2) pass over the prompt. U10-C02.
- Quantization: fewer bits per weight via scale and zero-point.
  U10-C03.
- Roofline: tok/s <= bandwidth / model bytes (decode). U10-C04.
- SFT: cross-entropy on demonstration responses. U10-C05.
- Bradley-Terry model: P(chosen > rejected) = sigmoid(r_c - r_r).
  U10-C06.
- PPO: clipped online policy optimization. U10-C07.
- DPO: offline preference loss on policy log-ratios. U10-C08.
- KL regularization: the price of drifting from pi_ref. U10-C09.
- Reward hacking: the policy exploiting the proxy. U10-C10.
- Win-rate: mean BT probability over held-out pairs. U10-C11.
