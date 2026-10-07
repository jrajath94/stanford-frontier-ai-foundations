# Cheatsheet: math-genmodels

Date: 2026-10-06. Baseline: October 6, 2026.

One dense page per unit family. Formulas use the notation fixed
in notation_and_shapes.md. Section pointers name the lesson that
proves each line. Honesty banner applies throughout.

## U01, probability and density estimation

| Formula / shape | Trap | Decision rule |
| --- | --- | --- |
| Density integrates to 1, mass sums to 1 | Reading p(x) as a probability | Integrate before you believe |
| Likelihood L(theta) = product p(x_i, theta) | Calling it a function of the data | It moves with theta |
| Empirical law: mass 1/n per sample | Trusting n = 12 | Watch it converge with n |
| Entropy H = -E[log p] | Units: nats vs bits | State the base |
| KL(p\|q) directed | Symmetrizing it | Direction is the penalty |

## U02, latent models and variational inference

| Formula / shape | Trap | Decision rule |
| --- | --- | --- |
| ELBO = E_q[log p(x,z) - log q(z)] | Calling it the likelihood | It is a floor |
| Gap = KL(q\|\|posterior) | Ignoring the gap | Tighten q or label it |
| Forward KL covers, reverse seeks | Wrong direction for the job | Cover for recall, seek for sharpness |
| z = mu + sigma x eps | Detaching eps wrongly | eps carries no gradient anyway |
| Amortized q: one net for all x | The amortization gap | Compare against per-point q |

## U03, autoencoders and representation

| Formula / shape | Trap | Decision rule |
| --- | --- | --- |
| VAE loss = recon + KL | KL = 0 looks efficient | Zero KL is collapse |
| Gaussian KL closed form | Sampling it instead | Use the formula |
| beta-VAE: recon + beta x KL | beta too large kills latents | Ride the frontier, pick beta by it |
| VQ: nearest codebook vector | Straight-through bias 0.6011 | Measure the bias on a toy |
| Rate-distortion frontier | More rate always helps | Only up to the frontier |

## U04, flows

| Formula / shape | Trap | Decision rule |
| --- | --- | --- |
| p_x = p_z(f^-1(x)) x \|det J\| | Dropping \|det J\| | The result is not a density |
| log p(y) = log p_z - log\|det J\| | The sign | Minus. Always minus |
| Triangular det = diagonal product | A stray nonzero above diagonal | Audit the structure |
| MAF: density 1 pass, sample D steps | Training a MAF to generate | Match the cheap direction to the job |
| Accumulate log-dets | Linear-space products | 100 layers of 1e6 overflow to inf |

## U05, adversarial and ratio estimation

| Formula / shape | Trap | Decision rule |
| --- | --- | --- |
| D*(x) = p/(p+q) = r/(1+r) | D* = p/q | The +q in the denominator |
| max_D V = -log 4 + 2 JSD | Assuming D is optimal | It never is. The identity is theory |
| JSD <= log 2, saturates disjoint | 100% accuracy as success | Saturation means zero gradient |
| W1 dual needs 1-Lipschitz f | Unconstrained critic | The value explodes |
| Gradient penalty, lambda = 10 | Penalty as proof | It constrains interpolations only |
| Mode collapse: recall 0 | Precision alone | The pair or nothing |

## U06, scores and diffusion

| Formula / shape | Trap | Decision rule |
| --- | --- | --- |
| s(x) = grad log p(x) | Gradient of the density | Log first, then gradient |
| x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps | Simulating the chain | The closed form is O(1) |
| DDPM loss = E\|eps - eps_hat\|^2 | Calling it a likelihood | The reweighting breaks the bound |
| Tweedie: x + sigma^2 score | Using it as a sampler | It returns the mean |
| DDIM eta = 0: deterministic | Striding for free | Validate quality vs steps |
| Guidance: s + w(s_c - s_u) | w too large | Watch for overshoot artifacts |
| Boundary score blows up | ODE likelihood without a floor | Name sigma_min or void the number |

## U07, autoregressive models

| Formula / shape | Trap | Decision rule |
| --- | --- | --- |
| p(x) = product p(x_t\|x_<t) | Forgetting the conditioning | Each factor sees the past |
| Perplexity = exp(NLL/L) | Across tokenizations | Bits per character first |
| Causal mask: lower triangular | One unmasked layer | The future leaks everywhere |
| Attention: softmax(QK^T/sqrt(d))V | Dropping the scale | Softmax saturates at init |
| Cost O(L^2 d), cache 2Ld bytes | Generating without cache | O(L^3): throughput dies |
| Temperature T on logits | High T without truncation | Truncate, then temper |
| Flow matching: regress x_1 - x_0 | Crossing paths | OT couplings straighten them |

## U08, evaluation

| Formula / shape | Trap | Decision rule |
| --- | --- | --- |
| Held-out loglik | Tuning on it | Lock it away |
| Bound vs exact | Ranking across the gap | Exact vs exact only |
| Precision / recall | One without the other | The pair is the minimum |
| FID: moments only | FID 0 as verdict | Build the bimodal counterexample |
| NN memorization test | Eyeballing samples | Histogram the distances |
| Compute-matched budgets | The idea vs the money | Fix params, flops, inference cost |
| Uncertainty over seeds | Single-seed leaderboard | 5 seeds minimum, report std |
| Pre-registered failure rule | HARKing | Write it before the run |

## Cross-unit decision rules

| Choice | Rule |
| --- | --- |
| Need exact likelihood | Flow or AR. Never a GAN, never a bare VAE number |
| Need fast sampling | GAN, DDIM, or IAF. Not MAF, not DDPM-1000 |
| Laws start far apart | Wasserstein-flavored. Not JSD |
| Discrete data | AR or discrete diffusion. Not Gaussian diffusion raw |
| Compare two models | Matched budgets, frozen metrics, seed error bars |
| Report a number | Name the assumption it needs |
