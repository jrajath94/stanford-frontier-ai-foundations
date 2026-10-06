#!/usr/bin/env python3
"""L07 plates: reparameterization, KL term, posterior collapse."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. Reparameterization trick
p = Plate("Move the randomness out: z = mu + sigma * eps",
          "Same distribution N(mu, sigma^2). Gradients flow through mu and sigma.",
          "Shell 3. mu = 0.5, sigma = 0.5, eps = 1.2 gives z = 1.1. Source: original toy.",
          source="original toy", inner_h=440)
p.panel(32, p.top, 420, 360, label="before: sample directly", fill=PINK)
p.text(56, p.top + 64, "z ~ N(mu, sigma^2)", size=16, bold=True)
p.text(56, p.top + 110, "the random draw has", size=14)
p.text(56, p.top + 138, "no gradient", size=14, bold=True, color=ORANGE)
p.text(56, p.top + 184, "backprop stops here", size=14, color=MUTED)
p.text(56, p.top + 230, "encoder never learns", size=14, bold=True, color=ORANGE)
p.arrow(468, p.top + 180, 548, p.top + 180, label="reparam")
p.panel(556, p.top, 372, 360, label="after: move randomness out", fill=NEW)
p.text(580, p.top + 64, "eps ~ N(0, 1), fixed", size=16, bold=True)
p.text(580, p.top + 110, "z = mu + sigma * eps", size=15)
p.text(580, p.top + 156, "dz/dmu = 1", size=15, bold=True, color=TEAL)
p.text(580, p.top + 188, "dz/dsigma = eps = 1.2", size=15, bold=True, color=TEAL)
p.text(580, p.top + 234, "z = 0.5 + 0.5*1.2 = 1.1", size=14)
p.text(580, p.top + 270, "gradients flow: encoder learns", size=14, bold=True, color=TEAL)
p.save("l07-reparam.webp")

# 2. KL term incentives
p = Plate("The KL rent: 0.443 nats, and what each term punishes",
          "KL = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2). Cheapest q is the prior itself.",
          "Shell 2. mu = 0.5, sigma^2 = 0.25: -0.5*(1 - 1.386 - 0.5) = 0.443. Source: original toy.",
          source="original toy", inner_h=440)
y = p.top + 24
p.text(32, y, "KL = -0.5 * ( 1 + log(sigma^2) - mu^2 - sigma^2 )", size=16, bold=True)
terms = [
    ("log(sigma^2)", "punishes tiny sigma", "overconfidence costs", PINK),
    ("-mu^2", "punishes drift from 0", "stay near the prior", YELLOW),
    ("-sigma^2", "punishes excess spread", "do not spray", ACTIVE),
]
yy = y + 56
for term, what, why, fill in terms:
    p.rect(32, yy, 220, 64, fill, label=term, size=14, rx=12)
    p.text(276, yy + 14, what, size=14)
    p.text(276, yy + 38, why, size=13, color=MUTED)
    yy += 88
p.text(32, yy + 16, "cheapest q: mu = 0, sigma = 1 -> KL = 0 (but latents carry nothing)", size=14, bold=True, color=FOCUS)
p.text(32, yy + 48, "training balances rent against reconstruction, automatically", size=13, color=MUTED)
p.save("l07-kl-term.webp")

# 3. Posterior collapse diagnostic
p = Plate("Posterior collapse: KL = 0 with good reconstructions is the corpse",
          "Healthy training keeps KL clearly above zero. Zero KL means dead latents.",
          "Shell 3. Watch the KL term, not the total loss. Source: original.",
          source="original", inner_h=400)
p.panel(32, p.top, 430, 320, label="healthy", fill=NEW)
p.bars(56, p.top + 200, [("KL", 0.443, TEAL)], 0.6, bar_w=100, gap=40, height=120, size=13)
p.text(56, p.top + 240, "reconstructions good", size=14)
p.text(56, p.top + 268, "KL = 0.443: encoder says", size=14, bold=True, color=TEAL)
p.text(56, p.top + 292, "something. Latents alive.", size=14)
p.panel(498, p.top, 430, 320, label="collapsed", fill=PINK)
p.bars(522, p.top + 200, [("KL", 0.001, PINK)], 0.6, bar_w=100, gap=40, height=120, size=13)
p.text(522, p.top + 240, "reconstructions good", size=14)
p.text(522, p.top + 268, "KL ~ 0: q = prior for every x.", size=14, bold=True, color=ORANGE)
p.text(522, p.top + 292, "Latents dead. Decoder alone.", size=14)
p.text(32, p.top + 344, "fixes: weaker decoder, anneal KL weight 0 -> 1, beta-VAE, VQ-VAE", size=14)
p.save("l07-collapse.webp")
