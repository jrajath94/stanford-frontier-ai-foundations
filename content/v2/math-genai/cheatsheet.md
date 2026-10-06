---
page_id: math-genai-cheatsheet
course_slug: math-genai
course_name: "Mathematical Foundations of Generative AI"
course_order: 11
order: 98
nav: "Cheatsheet"
title: "MATH-GENAI Cheatsheet"
summary: "Every formula, decision rule, and key number from the course on one page."
instructor: "Prof. Prathosh A P"
offering: "2025"
---

## The recipe

`θ* = argmin_θ D(P_X, P_θ)`: family, divergence, optimization.
Three gaps: wrong family, finite samples, local minima.

## Divergences: one-glance

| Name | Formula | Coin toy {0.5,0.5} vs {0.9,0.1} | Personality | Use when |
|---|---|---|---|---|
| KL (forward) | Σ P_X log(P_X/P_θ) | 0.511 nats | Asymmetric, infinite on zeros, mode-covering | Default; you have data and a density |
| Reverse KL | Σ P_θ log(P_θ/P_X) | ∞ on sharp toy | Mode-seeking, drops modes | You want a compact approximation |
| Jensen-Shannon | ½KL(P_X‖M) + ½KL(P_θ‖M), M = average | 0.102 | Symmetric, calm, saturates | GAN theory; not far-apart training |
| Total variation | ½Σ\|P_X − P_θ\| | 0.4 | Symmetric, simple, kinked gradient | Proofs, not gradient descent |
| Wasserstein-1 | min cost of moving mass | \|θ\| on point masses | Never saturates, gradient 1 | Distributions may not overlap |

f-divergence: `D_f = ∫ p_θ(x) f(p_X(x)/p_θ(x)) dx`, f convex, f(1) = 0.
Non-negative. Zero iff distributions match. KL = u log u member.

## MLE identity

`argmin_θ KL(P_X‖P_θ) = argmax_θ (1/n) Σ log P_θ(x_i)`
Cross-entropy = entropy + KL. Entropy is the floor: no model beats it.
On H,H,T: 0.7-model wins (−1.917 vs −2.079).

## Variational bound (→ GAN)

`D_f = max_T E[T(x)] − E[f*(T(x̂))]`, samples only, no densities.
Bound, not value: a weak critic underestimates. Train the critic first.
GAN's f: `f(u) = u log u − (u+1) log(u+1)`.
`J = E[log D(x)] + E[log(1 − D(G(z)))]`: D maximizes, G minimizes.
Non-saturating G loss: `−log D(G(z))` (0.693 vs 0.001 signal at d = 0.001→0.002).
Mode collapse: truth {0:.5, 10:.5}, model {0:1} → JS = 0.216, G stuck at 0.693.

## WGAN

`J = E[f(x)] − E[f(x̂)]`, f 1-Lipschitz (speed limit ≤ 1). No sigmoid, no logs.
Enforce by weight clipping (crude) or gradient penalty `(‖∇f‖−1)²`.

## FID

`FID = ‖μ−μ̂‖² + Tr(Σ + Σ̂ − 2(ΣΣ̂)^{1/2})`: Inception features, Gaussian fit.
Toy: identical clouds shifted by 2 → FID = 4. Lower is better. 0 = match.
Blind spots: memorization scores ~0. Pair with nearest-neighbor + diversity checks.

## ELBO

`log p_θ(x) ≥ E_q[log p_θ(x,z) − log q(z|x)]` (Jensen: log E ≥ E log).
Gap = `KL(q(z|x) ‖ p_θ(z|x))`.
Split: `E_q[log p_θ(x|z)] − KL(q(z|x) ‖ p(z))`: reconstruction minus rent.
Toy: ELBO −1.031, truth −0.968, gap 0.063 = KL exactly.
q choices: exact posterior (EM, gap 0) · learned network (VAE) · fixed process (diffusion).

## VAE

Encoder `q(z|x) = N(μ(x), σ(x))`, prior N(0,1), decoder `p(x|z)`.
Reparameterize: `z = μ + σ·ε`, ε ~ N(0,1). `∂z/∂μ = 1`, `∂z/∂σ = ε`.
KL per dim: `−½(1 + log σ² − μ² − σ²)`. Toy μ=0.5, σ²=0.25 → 0.443.
Loss = reconstruction + KL. Posterior collapse: KL → 0 with good reconstructions = dead latents.
β-VAE: β·KL (disentanglement vs fidelity). VQ-VAE: discrete codebook, no Gaussian KL.

## DDPM forward

`x_t = √α_t x_{t−1} + √(1−α_t) ε_t`: fixed Markov chain, nothing learned.
Jump: `x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε`, `ᾱ_t = α_1···α_t`.
Toy: 4.0 → 4.111 → 3.900 (α = 0.9). Stationary: x_T ≈ N(0,1).
Schedule: tuned by experiment. Too fast kills the signal early.

## DDPM loss

`L_simple = E‖ε − ε_θ(x_t, t)‖²`: predict the noise. Toy: (0.688−0.5)² = 0.0354.
Equivalent faces: predict ε, predict x_0, predict score `−ε/√(1−ᾱ_t)` (−1.578).
Sample: `x_{t−1} = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t))ε_θ) + √β̃_t·z`.
Toy: x_2 = 3.9 → x_1 = 4.059 (z = 0.3). Reweighted bound: samples over likelihood.

## Score / DDIM / guidance / latent

Langevin: `x ← x + (δ/2)·score + √δ·z`. Toy: 3.9 → 3.947.
DDIM stride: `x̂_0 = (x_t − √(1−ᾱ_t)ε_θ)/√ᾱ_t`, `x_s = √ᾱ_s x̂_0 + √(1−ᾱ_s)ε_θ`.
Toy: x_100 = 1.5 → x_50 = 2.844 via x̂_0 = 3.221. 20× fewer steps, less diversity.
Guidance: `score + s·∇log p(y|x_t)`. Toy: −1.578 + 2·0.5 = −0.578. Too much s caricatures.
Latent: 512×512×3 → 64×64×4 = 48× cheaper per step. Price: VAE artifacts.

## Autoregressive

`p(x_1..x_n) = Π p(x_i|x_<i)`. Toy: p(a,b,a) = 0.6·0.3·0.2 = 0.036.
Exact likelihood, sequential sampling. Next-token prediction = AR + MLE.

## Four families, four prices

GAN: saddle point (saturation, mode collapse). VAE: blur (mode-covering + Gaussian decoder).
Diffusion: slow sampling (T steps). AR: sequential generation.

## Failure → diagnostic → fix

| Symptom | Likely cause | Check | Fix |
|---|---|---|---|
| Generator loss flat at start | Saturation (original minimax loss) | D(G(z)) ≈ 0, loss barely moves | Non-saturating loss −log D |
| All samples look alike, loss fine | Mode collapse | Cluster samples vs cluster data | Rebalance players; change divergence family |
| VAE samples fine but KL ≈ 0 | Posterior collapse | KL per dim glued to 0 | Weaker decoder; anneal KL weight 0→1 |
| DDPM noisy at low t | Schedule too fast | Signal fraction vs t curve | Slow the schedule |
| FID good but samples are copies | Memorization | Nearest-neighbor distance to train set | It is not generating; fix the model, not the metric |
| Guided samples are caricatures | Guidance scale too high | Lower s, compare | Reduce s until steering bends without breaking |
| Loss per t varies wildly | Some noise levels starved | Plot loss broken down by t | Rebalance schedule or capacity per t |

## Numbers you must know cold

0.511 (KL coin toy) · 0.102 (JS) · 0.4 (TV) · 0.693 (forward sharp toy. JS max = log 2) ·
−1.917 vs −2.079 (MLE picks 0.7) · 0.001 vs 0.693 (saturation fix) · 0.216 (mode-collapse JS) ·
4 (FID toy) · −1.031 vs −0.968, gap 0.063 (ELBO toy) · 0.443 (VAE KL rent) ·
3.900 (forward toy) · 3.944 (posterior mean) · 0.0354 (L_simple toy) · 4.059 (one reverse step) ·
−1.578 (score toy) · 2.844 (DDIM stride) · 48× (latent saving) · 0.036 (AR toy)
