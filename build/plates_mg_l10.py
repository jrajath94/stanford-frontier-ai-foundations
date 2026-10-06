#!/usr/bin/env python3
"""L10 plates: score field, DDIM stride, latent compression, four families."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. Score field on 1-D toy
p = Plate("The score is an arrow field pointing uphill on probability",
          "At x = 3.9 the arrow points left with strength 1.578: toward the clean 4.0.",
          "Shell 2. score = -eps/sqrt(1-a_bar) = -0.688/0.4359 = -1.578. Source: original toy.",
          source="original toy", inner_h=440)
x0, y0, x1, y1 = 80, 400, 880, 140
p.axes(x0, y0, x1, y1)
p.text(x1 - 30, y0 + 24, "x", size=13, color=MUTED)
def X(v): return x0 + (v - 2.5) / 3.0 * (x1 - x0)
# probability hump centered at 4.0
pts = [(X(2.5 + i * 0.03), y0 - math.exp(-((2.5 + i * 0.03 - 4.0) ** 2) / 0.5) * (y0 - y1)) for i in range(101)]
p.curve(pts, color=TEAL, width=3)
p.text(X(4.0) - 20, y1 + 16, "clean 4.0", size=13, color=TEAL, bold=True)
# arrows: score points toward 4.0
for xv, strength in [(3.0, 0.6), (3.5, 1.0), (3.9, 1.578), (4.5, 0.8), (5.0, 0.4)]:
    xx = X(xv)
    yy2 = y0 - math.exp(-((xv - 4.0) ** 2) / 0.5) * (y0 - y1) - 40
    direction = 1 if xv < 4.0 else -1
    L = 30 + strength * 40
    p.arrow(xx, yy2, xx + direction * L, yy2, color=FOCUS, width=3)
p.text(X(3.9) - 60, 200, "at 3.9: arrow left, strength 1.578", size=14, bold=True, color=FOCUS)
p.text(80, y0 + 60, "the denoiser learned this whole arrow field: one score per point", size=14)
p.text(80, y0 + 88, "Langevin walks the arrows: 3.9 -> 3.947 in one step (delta = 0.1, z = 0.4)", size=14)
p.save("l10-score-field.webp")

# 2. DDIM stride
p = Plate("DDIM: predict clean, re-noise to the target step",
          "One stride from t = 100 to s = 50: 1.5 -> 2.844 via x_hat_0 = 3.221.",
          "Shell 3. Same marginals, non-Markovian reverse. No retraining. Source: original toy.",
          source="original toy", inner_h=440)
y = p.top + 40
p.rect(32, y, 200, 96, COUNT, label="x_100 = 1.5", size=15, rx=12)
p.arrow(248, y + 48, 330, y + 48, label="predict")
p.rect(340, y, 240, 96, YELLOW, label="x_hat_0 = 3.221", size=15, rx=12)
p.text(400, y + 64, "(1.5-0.7798)/0.2236", size=12, color=MUTED)
p.arrow(596, y + 48, 678, y + 48, label="re-noise")
p.rect(688, y, 200, 96, NEW, label="x_50 = 2.844", size=15, rx=12)
p.text(728, y + 64, "2.278+0.566", size=12, color=MUTED)
p.text(32, y + 150, "50 strides replace 1000 DDPM steps: 20x fewer network evals", size=15, bold=True, color=TEAL)
p.text(32, y + 190, "price: deterministic stride explores less -> slightly lower diversity", size=14)
p.text(32, y + 230, "fix: re-add controlled noise per stride", size=14, color=MUTED)
p.text(32, y + 280, "why it works: training used only the marginals q(x_t|x_0)", size=14, bold=True)
p.text(32, y + 308, "the non-Markovian process shares them, so eps_theta still applies", size=13, color=MUTED)
p.save("l10-ddim-stride.webp")

# 3. Latent diffusion compression
p = Plate("Latent diffusion: shrink the space, keep the process",
          "512x512x3 = 786,432 numbers become 64x64x4 = 16,384: 48x smaller.",
          "Shell 2. Every diffusion step costs 48x less. The VAE bottleneck is the price. Source: original computation.",
          source="original computation", inner_h=400)
p.panel(32, p.top, 380, 280, label="pixel space", fill=PINK)
p.text(56, p.top + 80, "512 x 512 x 3", size=18, bold=True)
p.text(56, p.top + 120, "= 786,432 numbers", size=15)
p.text(56, p.top + 170, "per diffusion step", size=14, color=MUTED)
p.arrow(428, p.top + 140, 500, p.top + 140, label="VAE encode")
p.panel(516, p.top, 412, 280, label="latent space", fill=NEW)
p.text(540, p.top + 80, "64 x 64 x 4", size=18, bold=True)
p.text(540, p.top + 120, "= 16,384 numbers", size=15)
p.text(540, p.top + 170, "48x cheaper per step", size=14, bold=True, color=TEAL)
p.text(32, p.top + 320, "diffuse here, decode once at the end. Price: VAE compression artifacts.", size=14)
p.save("l10-latent-diffusion.webp")

# 4. Four families chapter plate
p = Plate("Four families, one recipe, four prices",
          "GAN, VAE, diffusion, autoregressive: same three ingredients, different bills.",
          "Chapter plate. The course arc: from counting what exists to making what does not. Source: original.",
          source="original", inner_h=520)
headers = ["family", "latent?", "objective", "price"]
rows = [
    ("GAN (L3-5)", "none", "minimax game", "saddle point", PINK),
    ("VAE (L6-7)", "learned z", "ELBO", "blurry samples", YELLOW),
    ("diffusion (L8-10)", "fixed chain", "noise MSE", "slow sampling", COUNT),
    ("AR (L10)", "none", "exact MLE", "sequential", NEW),
]
yy = p.top + 16
xx = 32
widths = [200, 180, 260, 288]
for i, htxt in enumerate(headers):
    p.text(xx + sum(widths[:i]), yy, htxt, size=14, bold=True, color=MUTED)
yy += 40
for name, lat, obj, price, fill in rows:
    p.rect(32, yy, widths[0], 64, PANEL, label=name, size=14, rx=8)
    p.rect(32 + widths[0], yy, widths[1], 64, PANEL, label=lat, size=14, rx=8)
    p.rect(32 + widths[0] + widths[1], yy, widths[2], 64, PANEL, label=obj, size=14, rx=8)
    p.rect(32 + widths[0] + widths[1] + widths[2], yy, widths[3], 64, fill, label=price, size=14, rx=8)
    yy += 80
p.text(32, yy + 24, "the arc: GANs learn the real/fake boundary. VAEs learn a compressed imagination.", size=14)
p.text(32, yy + 52, "Diffusion learns to clean noise. AR learns what comes next.", size=14)
p.text(32, yy + 88, "one recipe (family, divergence, optimization). Four answers to 'how do you teach a machine to create?'", size=14, bold=True, color=FOCUS)
p.save("l10-four-families.webp")
