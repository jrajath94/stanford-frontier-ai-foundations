---
page_id: math-genai-cheatsheet
course_slug: math-genai
course_name: "Mathematical Foundations of Generative AI"
course_order: 11
order: 98
nav: "Cheatsheet"
title: "MATH-GENAI Cheatsheet"
summary: "Every formula and key number from the course on one page."
instructor: "Prof. Prathosh A P"
offering: "2025"
---

## The recipe

`θ* = argmin_θ D(P_X, P_θ)`: family, divergence, optimization.

## Divergences

| Name | Formula | Coin toy {0.5,0.5} vs {0.9,0.1} |
|---|---|---|
| KL (forward) | Σ P_X log(P_X/P_θ) | 0.511 nats |
| Jensen-Shannon | ½KL(P_X‖M) + ½KL(P_θ‖M), M = average | 0.102 |
| Total variation | ½Σ\|P_X − P_θ\| | 0.4 |
| Wasserstein-1 | min cost of moving mass | \|θ\| on point masses (10 vs 100, not 0.693 both) |

f-divergence: `D_f = ∫ p_θ(x) f(p_X(x)/p_θ(x)) dx`, f convex, f(1) = 0.
Non-negative. Zero iff distributions match.

## MLE identity

`argmin_θ KL(P_X‖P_θ) = argmax_θ (1/n) Σ log P_θ(x_i)`
Cross-entropy = entropy + KL. On H,H,T: 0.7-model wins (−1.918 vs −2.079).

## Variational bound (→ GAN)

`D_f = max_T E[T(x)] − E[f*(T(x̂))]`, samples only, no densities.
GAN's f: `f(u) = u log u − (u+1) log(u+1)`.
`J = E[log D(x)] + E[log(1 − D(G(z)))]`: D maximizes, G minimizes.
Non-saturating G loss: `−log D(G(z))` (0.693 vs 0.001 signal at d = 0.001→0.002).
Mode collapse: truth {0:.5, 10:.5}, model {0:1} → JS = 0.216, G stuck at 0.693.

## WGAN

`J = E[f(x)] − E[f(x̂)]`, f 1-Lipschitz (speed limit ≤ 1). No sigmoid, no logs.

## FID

`FID = ‖μ−μ̂‖² + Tr(Σ + Σ̂ − 2(ΣΣ̂)^{1/2})`: Inception features, Gaussian fit.
Toy: identical clouds shifted by 2 → FID = 4. Lower is better.

## ELBO

`log p_θ(x) ≥ E_q[log p_θ(x,z) − log q(z|x)]` (Jensen: log E ≥ E log).
Gap = `KL(q(z|x) ‖ p_θ(z|x))`.
Split: `E_q[log p_θ(x|z)] − KL(q(z|x) ‖ p(z))`: reconstruction minus rent.
Toy: ELBO −1.031, truth −0.968, gap 0.063 = KL exactly.

## VAE

Encoder `q(z|x) = N(μ(x), σ(x))`, prior N(0,1), decoder `p(x|z)`.
Reparameterize: `z = μ + σ·ε`, ε ~ N(0,1). `∂z/∂μ = 1`, `∂z/∂σ = ε`.
KL per dim: `−½(1 + log σ² − μ² − σ²)`. Toy μ=0.5, σ²=0.25 → 0.443.
Loss = pixel error + KL. Posterior collapse: KL → 0 with good reconstructions = dead latents.

## DDPM forward

`x_t = √α_t x_{t−1} + √(1−α_t) ε_t`: fixed Markov chain, nothing learned.
Jump: `x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε`, `ᾱ_t = α_1···α_t`.
Toy: 4.0 → 4.111 → 3.900 (α = 0.9). Stationary: x_T ≈ N(0,1).

## DDPM loss

`L_simple = E‖ε − ε_θ(x_t, t)‖²`: predict the noise. Toy: (0.688−0.5)² = 0.0354.
Equivalent faces: predict ε, predict x_0, predict score `−ε/√(1−ᾱ_t)` (−1.578).
Sample: `x_{t−1} = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t))ε_θ) + √β̃_t·z`.
Toy: x_2 = 3.9 → x_1 = 4.059 (z = 0.3).

## Score / DDIM / guidance / latent

Langevin: `x ← x + (δ/2)·score + √δ·z`. Toy: 3.9 → 3.947.
DDIM stride: `x̂_0 = (x_t − √(1−ᾱ_t)ε_θ)/√ᾱ_t`, `x_s = √ᾱ_s x̂_0 + √(1−ᾱ_s)ε_θ`.
Toy: x_100 = 1.5 → x_50 = 2.844 via x̂_0 = 3.221.
Guidance: `score + s·∇log p(y|x_t)`. Toy: −1.578 + 2·0.5 = −0.578.
Latent: 512×512×3 → 64×64×4 = 48× cheaper per step.

## Autoregressive

`p(x_1..x_n) = Π p(x_i|x_<i)`. Toy: p(a,b,a) = 0.6·0.3·0.2 = 0.036.
Exact likelihood, sequential sampling.

## Four families, four prices

GAN: saddle point (saturation, mode collapse). VAE: blur (mode-covering + Gaussian decoder).
Diffusion: slow sampling (T steps). AR: sequential generation.
