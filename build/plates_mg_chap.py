#!/usr/bin/env python3
"""math-genai chapter plates (fix round 1): one dense plate per lesson l01-l09.

Spec: left = cost without the rule, center = the stored object/rule,
right = cost with the rule, bottom = tradeoff in one line.
Filenames: assets/plate-lNN-chap-<name>.webp.
All numbers are hand-verified against the lesson text (see greps in the fix log).
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import (Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW,
                              ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW,
                              GREEN, _font, FR, FB)
from PIL import Image, ImageDraw

MEAS = ImageDraw.Draw(Image.new("RGB", (8, 8)))

def wrap(s, size, maxw):
    f = _font(FR, size)
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if MEAS.textlength(t, font=f) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines

def block(p, x, y, lines, size=13, maxw=258, lh=22, bold_first=False, color=INK):
    yy = y
    for i, s in enumerate(lines):
        for ln in wrap(s, size, maxw):
            p.text(x, yy, ln, size=size, color=color,
                   bold=(bold_first and i == 0))
            yy += lh
    return yy

def chap(fname, title, claim, left, center, right, bottom, footer):
    p = Plate(title, claim, footer,
              source="original synthesis of the lesson",
              width=960, inner_h=500)
    y = p.top
    p.panel(32, y, 290, 340, label="without the rule", fill=PANEL)
    p.panel(338, y, 284, 340, label="the rule", fill=YELLOW)
    p.panel(638, y, 290, 340, label="with the rule", fill=NEW)
    block(p, 48, y + 52, left)
    block(p, 354, y + 52, center, bold_first=True)
    block(p, 654, y + 52, right)
    by = y + 368
    p.text(32, by, "tradeoff:", size=14, bold=True, color=ORANGE)
    for ln in wrap(bottom, 14, 860):
        p.text(130, by, ln, size=14, bold=True)
        by += 24
    p.save(fname)

# L01: the recipe
chap("plate-l01-chap-recipe.webp",
     "The recipe: family, divergence, optimization",
     "Three slots. Fill them and you get a training procedure.",
     ["no recipe: fit by eyeball",
      "Gaussian, mean 5, spread 2",
      "memory machine {2,4,6,8}",
      "replays data, never emits 5",
      "P(5) = 0.20 vs 0: no comparison"],
     ["theta* = argmin D(P_X, P_theta)",
      "slot 1: family P_theta",
      "slot 2: divergence D",
      "slot 3: optimizer",
      "one equation, every lesson"],
     ["L02: D = KL gives MLE",
      "L03-4: samples only gives critic",
      "L06-7: latent z gives ELBO",
      "L08-10: fixed chain gives noise MSE",
      "each lesson fills the slots"],
     "the recipe never picks the family. three gaps stay: wrong family, finite samples, local minima.",
     "Chapter plate. Every lesson in this course fills the same three slots.")

# L02: KL -> MLE
chap("plate-l02-chap-kl-mle.webp",
     "KL splits, and MLE falls out",
     "The truth's entropy has no theta. Drop it.",
     ["min KL(P_X || P_theta)",
      "needs the truth's formula P_X.",
      "unknowable: the wall.",
      "no likelihood without it"],
     ["KL = H(P_X) - E[log P_theta]",
      "H has no theta: drop it",
      "min KL = max E[log P_theta]",
      "the identity that starts MLE"],
     ["(1/n) sum log P_theta(x_i)",
      "H,H,T: -1.917 beats -2.079",
      "Gaussian: mu_hat = 5,",
      "sigma_hat = 2.236"],
     "price: needs log p_theta(x) per point. explicit models only: the fork.",
     "Chapter plate. KL minimization is maximum likelihood once the constant is dropped.")

# L03: f-divergences
chap("plate-l03-chap-fdiv.webp",
     "One menu of divergences",
     "Every convex f gives a divergence. The critic makes it computable.",
     ["KL-only world: 0.511 nats",
      "on the coin toy.",
      "density formulas die",
      "in high dimensions"],
     ["D_f = E[f(p_X / p_theta)]",
      "KL 0.511, JS 0.102, TV 0.4",
      "variational: max over critics",
      "E[T(x)] - E[f*(T(x_hat))]"],
     ["the GAN objective falls out:",
      "E[log D] + E[log(1-D)]",
      "two-parameter critic: 0.194,",
      "an honest bound under 0.203"],
     "the bound is only as honest as the critic. weak critic, weak bound.",
     "Chapter plate. The f menu turns density ratios into a game critics can play.")

# L04: GANs
chap("plate-l04-chap-gan.webp",
     "The minimax game",
     "No density, no MLE. Learn the boundary instead.",
     ["saturating loss at D = 0.001:",
      "double the score to 0.002,",
      "loss moves 0.001: a whisper.",
      "the generator learns nothing"],
     ["min_G max_D J",
      "D maximizes: real vs fake",
      "G minimizes: fool D",
      "one objective, opposite goals"],
     ["non-saturating fix: 0.693,",
      "the same move screams.",
      "mode collapse: JS = 0.216,",
      "generator stuck at 0.693"],
     "saddle point, no convergence proof. toward 10: D ~ 0, loss explodes, so it stays.",
     "Chapter plate. The game replaces the likelihood when densities are out of reach.")

# L05: WGAN + evaluation
chap("plate-l05-chap-wgan.webp",
     "Wasserstein sees the distance",
     "JS is flat at 0.693. The moving cost is not.",
     ["theta = 10: JS = 0.693,",
      "theta = 100: JS = 0.693.",
      "same score, zero gradient.",
      "the generator is blind"],
     ["W = |theta|, the moving cost",
      "dual: max over 1-Lipschitz f",
      "J = E[f(x)] - E[f(x_hat)]",
      "raw scores, no sigmoid"],
     ["gradient 1 at every step:",
      "theta walks 100 -> 0.",
      "FID toy: mean 4 + spread 0 = 4.",
      "judge samples with samples"],
     "the Lipschitz limit must be enforced. FID blind spot: replayed data scores ~0.",
     "Chapter plate. A distance with gradients beats a divergence without them.")

# L06: ELBO
chap("plate-l06-chap-elbo.webp",
     "The ELBO floor",
     "The log of an integral is a wall. Jensen builds a floor under it.",
     ["log integral p(x,z) dz:",
      "intractable. the wall.",
      "log E != E log.",
      "MLE cannot start"],
     ["ELBO = E_q[log p(x,z)/q]",
      "gap = KL(q || posterior)",
      "E-step: q = posterior, gap 0",
      "M-step: push the floor up"],
     ["two coins: ELBO -1.031,",
      "truth -0.968, gap 0.063.",
      "0.063 = KL to the digit.",
      "likelihood never drops"],
     "the bound is only as tight as q. collapse loosens it while the loss looks fine.",
     "Chapter plate. A computable floor with a measured gap to the truth.")

# L07: VAE
chap("plate-l07-chap-vae.webp",
     "Amortize the posterior",
     "One encoder for every x. Move the randomness out of the gradient path.",
     ["sample z ~ q(z|x) directly:",
      "the draw has no gradient.",
      "backprop stops here.",
      "the encoder never learns"],
     ["z = mu + sigma * eps",
      "mu = 0.5, sigma = 0.5,",
      "eps = 1.2 -> z = 1.1",
      "KL rent: 0.443 nats"],
     ["amortized: one network,",
      "not one optimization per x.",
      "diagnostic: watch the KL term.",
      "KL -> 0 means dead latents"],
     "price: blurry samples. KL = 0 with good reconstructions is the corpse.",
     "Chapter plate. The reparameterization trick makes the ELBO trainable.")

# L08: DDPM forward
chap("plate-l08-chap-forward.webp",
     "Destroy slowly, reverse easily",
     "One giant leap is hopeless. A thousand small steps are easy.",
     ["x_0 -> x_1 ~ pure noise:",
      "the reverse must rebuild",
      "the photo from nothing.",
      "as hard as generation itself"],
     ["x_t = sqrt(a_t) x_{t-1}",
      "+ sqrt(1-a_t) eps_t",
      "closed form: jump to any t,",
      "no chaining, ever"],
     ["4.0 -> 4.111 -> 3.900,",
      "= 0.9*4.0 + 0.4359*0.688.",
      "a_bar_100 = 0.00003:",
      "pure noise N(0,1)"],
     "price: T network evals to sample. the schedule is tuned by experiment, not derived.",
     "Chapter plate. The fixed encoder buys a tractable reverse, one small step at a time.")

# L09: DDPM loss
chap("plate-l09-chap-ddpm-loss.webp",
     "Three moves to one MSE",
     "The chain ELBO is a monster. Three moves tame it.",
     ["chain ELBO: T KL terms,",
      "twitchy variances, slow.",
      "direct optimization",
      "never converges"],
     ["move 1: condition on x_0,",
      "posterior mean 3.944, var 0.0526",
      "move 2: KL becomes squared error",
      "move 3: predict the noise"],
     ["L_simple = E||eps-eps_t||^2",
      "= 0.0354 on the toy.",
      "three faces: 0.688, 4.0, -1.578.",
      "one reverse step: 3.9 -> 4.059"],
     "price: per-step weights dropped. reweighted bound: worse likelihood, better samples.",
     "Chapter plate. Three exact moves and one deliberate cheat: the field chose samples.")

print("chapter plates done")
