# course_map.md, math-genai unit map

Date: 2026-10-06. Playlist and repository references are title-level
and path-level only (SRC-04, SRC-05). Transcript inspection remains
open. The BSDA5002 week structure (SRC-02) is the binding unit map.

## Unit to syllabus and playlist mapping (week level)

| Unit | Parent topic | BSDA5002 weeks | Playlist weeks (titles) | Prerequisite modules |
|---|---|---|---|---|
| math-genai-U01 | Probabilistic generative modelling | Week 1 | W1_L1 course outline. W1_L2 introduction and problem setting. Local bridge content | P03, P06, P07, P08 |
| math-genai-U02 | Variational divergence minimization | Week 2 | W1L3 f-Divergence. W1L4 variational divergence minimization. W5T10 proof of Jensen's inequality | P05, P08, P09, P18 |
| math-genai-U03 | GAN foundations and applications | Week 3 | W2_L6, W2_L7 GANs intro and formulation. W2_T5 GAN implementation. W3L8 GANs as classifier-guided generative sampler. W3L9 DCGAN and conditional GANs. W3T6 DC-GAN implementation. W4L10 saturation of GAN training | P08, P09, P11 |
| math-genai-U04 | Wasserstein and improved adversarial training | Week 4 | W4L11 Wasserstein GANs. W4L12 inversion with GANs. W4L13 bi-directional GANs. W4L14 GAN inversion via latent regression. W4L15 domain adversarial networks. W4L16 evaluation of generative models. W4T7-T9 Bi-GAN, UDA, WGAN implementations | P04, P08, P09 |
| math-genai-U05 | Variational autoencoders | Week 5 | W5L17 intro to latent variable models. W5L18 ELBO. W5L19 GMM and EM. W5L20 VAE. W5T11 GMM. W6L21-L23 training VAE, reparameterization, inference. W6T12 VAE implementation | P08, P11, P18 |
| math-genai-U06 | Discrete latent modelling and VQ-VAE | Week 6 | W6L24 beta-VAE. W6L25 VQ-VAE. W6T13 VQ-VAE implementation | P04, P09, P11, P18 |
| math-genai-U07 | DDPM derivation and parameterizations | Week 7 | W7L26 DDPMs. W7L27 DDPM formulation. W7T14 U-Net. W8L28-L33 ELBO parts 1-2, optimization, equivalence, training, inference. W8T15 DDPM implementation. W8T16 proofs | P05, P06, P08, P18 |
| math-genai-U08 | Diffusion variants and implementation | Week 8-9 | W9L34 alternate interpretations of DDPMs. W9L35 DDPMs as score-predictors. W9L36 guided diffusion models. W9L37 latent diffusion models. W9L38-L39 DDIMs and inference. W9T17-T19 DDPM noise, DDIM, guided DDPM implementations | P09, P11, P12, P18 |
| math-genai-U09 | Score-based models and autoregressive LMs | Week 9-10 | W9L34-L35 score reading of DDPM (bridge). W10L40 autoregressive models. W10L41-L46 attention, transformers, position embeddings, training and inference | P05, P08, P13, P14, P18 |
| math-genai-U10 | LLM inference, quantization, and alignment | Week 11-12 | W11L47 overview of RL. W11L48 policy gradient theorem. W11L49 AR-LM as RL policy. W11L50 PPO. W11L51 TRPO. W12L52 reward-modelling. W12L53 DPO. W12L54 state-space-models | P12, P14, P15, P17 |

## Syllabus week structure (SRC-02, official)

Week 1 Introduction to Probabilistic Deep Generative Modelling.
Week 2 Generative Modelling via variational Divergence Minimization.
Week 3 Generative Adversarial Networks: Part 1 (Introduction and
Formulation).
Week 4 Generative Adversarial Networks: Part 2 (WGANs and
Applications).
Week 5 Generative Modelling via Variational Auto Encoding.
Week 6 Variational Auto Encoders: Improvisations and VQVAE.
Week 7 Denoising Diffusion Probabilistic Models (DDPMs) - Formulation.
Week 8 Diffusion Models: Multiple forms and Implementation.
Week 9 Conditional Diffusion Models and Score-based models.
Week 10 Auto-Regressive Models and Large Language Models Introduction.
Week 11 LLMs: Models, Sampling, Inference and Quantization Methods.
Week 12 LLMs Reinforcement Learning based Alignment Methods (PPO,
DPO).

## Prerequisite dependency order for RUN 2 onward

Build P03, P06, P07, P08 before U02 deepens (P05, P09, P18 for U02
come next).
U03 needs P09, P11. U04 needs P04 added.
U05 needs P18. U06 needs P04. U07 needs P05, P18.
U08 needs P12. U09 needs P13, P14. U10 needs P15, P17.
Shared modules live in shared/prerequisites/ and are already built.
This course adds only the local remediation it needs.

## Figure and assessment status

- U01: figure f01_data_vs_model.png in visual_audit.md. Diagnostic
  diagnostic-01. Lesson exercises E01-E20 + ladders L01-L05, keys
  separate.
- U02-U10: figures and assessments planned. No artifacts yet.
