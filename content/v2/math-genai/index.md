---
page_id: math-genai-index
course_slug: math-genai
course_name: "Mathematical Foundations of Generative AI"
course_order: 11
order: 0
nav: "Mathematical Foundations of Generative AI · Overview"
title: "Mathematical Foundations of Generative AI"
summary: "How do you teach a machine to create? From density estimation to generation: divergences, GANs, VAEs, diffusion: each mechanism built from zero with hand-worked numbers."
instructor: "Prof. Prathosh A P"
offering: "2025"
---

How do you teach a machine to create? That is the one question
behind this course. We have samples from an unknown distribution
P_X. We want new samples from it: faces nobody photographed,
sentences nobody wrote.

The course answers in four families, each a different point in
the same design space. The recipe is always: pick a parametric
family P_θ, choose a divergence to the truth, optimize.

## The lessons

1. **[L01 · The Job](l01-the-job.md)**: The task (make what does not exist), the three-step recipe, the memory machine that fails, and the push-forward trick.
2. **[L02 · KL Divergence and MLE](l02-kl-and-mle.md)**: The divergence, worked on a coin toy (KL 0.511), forward vs reverse, and the identity: min KL = max likelihood.
3. **[L03 · f-Divergences](l03-f-divergences.md)**: KL is one member of a family (JS 0.102, TV 0.4 on the toy), and the variational lower bound that measures distance from samples alone.
4. **[L04 · GANs](l04-gans.md)**: The minimax game traced by hand, saturation (0.001 vs 0.693), and mode collapse (JS 0.216 with half the world missing).
5. **[L05 · WGAN and Evaluation](l05-wgan-evaluation.md)**: The Wasserstein distance (gradient 1 everywhere) and FID, the industry judge, computed by hand (FID = 4).
6. **[L06 · Latent Variables and ELBO](l06-latent-variables-elbo.md)**: Hidden causes, the intractable log-integral, Jensen's inequality, and the ELBO worked on a two-coin toy (gap exactly 0.063).
7. **[L07 · VAE](l07-vae.md)**: The encoder network, the reparameterization trick (z = μ + σε), the KL rent (0.443), and posterior collapse.
8. **[L08 · DDPM Forward](l08-ddpm-forward.md)**: A fixed Markov chain turns data into noise (4.0 → 4.111 → 3.900), the closed-form jump, the stationary distribution.
9. **[L09 · DDPM Loss](l09-ddpm-loss.md)**: The ELBO collapses to noise prediction (loss 0.0354 in the toy). The three equivalent faces. Walking the chain backward.
10. **[L10 · Score, DDIM, AR](l10-score-ddim-ar.md)**: The score view, DDIM strides (100 → 50 in one jump), guided and latent diffusion, and the autoregressive family.

**[Cheatsheet](cheatsheet.md)**: every formula, every key number, one page.
**[Crash course](crash-course.md)**: the whole arc in fifteen minutes.

## How to read this course

Each lesson is self-contained: it builds its ideas from zero
with hand-worked toys and real numbers. Read in order the
first time. The arc is deliberate. After that, use the
cheatsheet as the map and dive into single lessons.

The playlist this course follows is Prof. Prathosh A P's
"Mathematical Foundations of Generative AI" (73 videos).
PyTorch tutorial videos are not covered here. The W10-W12
transformer and RL material belongs to the CS336/CS229S
courses and appears here only as the autoregressive frame.
