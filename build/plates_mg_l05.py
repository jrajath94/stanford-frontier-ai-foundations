#!/usr/bin/env python3
"""L05 plates: JS vs Wasserstein, WGAN game, FID pipeline."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. JS flat vs W linear
p = Plate("JS is flat at 0.693; Wasserstein sees the distance",
          "Point masses at 0 and theta. JS cannot tell 10 from 100. W can.",
          "Shell 2. d(JS)/d(theta) = 0. dW/d(theta) = 1. Source: original toy.",
          source="original toy", inner_h=460)
x0, y0, x1, y1 = 80, 400, 460, 120
p.axes(x0, y0, x1, y1)
p.text(24, y1 - 8, "distance", size=13, color=MUTED)
p.text(x1 - 60, y0 + 24, "theta", size=13, color=MUTED)
def X(t): return x0 + t / 120 * (x1 - x0)
# JS: 0.693 for all theta>0 (plot at fixed height)
p.curve([(X(2), y0 - 0.693 / 110 * (y0 - y1)), (X(118), y0 - 0.693 / 110 * (y0 - y1))], color=PINK, width=3)
pts = [(X(t), y0 - t / 110 * (y0 - y1)) for t in range(2, 112, 4)]
p.curve(pts, color=TEAL, width=3)
p.text(X(60), y0 - 0.693 / 110 * (y0 - y1) - 28, "JS = 0.693, flat", size=14, color=ORANGE, bold=True)
p.text(X(78), y0 - 78 / 110 * (y0 - y1) - 28, "W = |theta|, slope 1", size=14, color=TEAL, bold=True)
p.circle(X(10), y0 - 0.693 / 110 * (y0 - y1), 6, ORANGE)
p.circle(X(100), y0 - 0.693 / 110 * (y0 - y1), 6, ORANGE)
p.text(80, y0 + 60, "theta = 10:  JS = 0.693,  W = 10", size=14)
p.text(80, y0 + 88, "theta = 100: JS = 0.693,  W = 100", size=14)
p.text(80, y0 + 124, "generator gradient: 0 under JS, 1 under W", size=14, bold=True, color=FOCUS)
p.save("l05-js-vs-w.webp")

# 2. WGAN game
p = Plate("WGAN: the same game, raw scores, a speed limit",
          "No logs, no sigmoid. The critic's slope never exceeds 1.",
          "Shell 3. J = E[f(x)] - E[f(xhat)] over 1-Lipschitz f. Source: WGAN paper.",
          source="WGAN paper", inner_h=420)
p.panel(32, p.top, 420, 340, label="the critic f", fill=COUNT)
p.text(56, p.top + 64, "outputs a raw score", size=15, bold=True)
p.text(56, p.top + 96, "not a probability", size=14)
p.text(56, p.top + 140, "1-Lipschitz: slope <= 1", size=15, bold=True, color=FOCUS)
p.text(56, p.top + 172, "everywhere, always", size=14)
p.text(56, p.top + 216, "without the limit the max", size=14, color=MUTED)
p.text(56, p.top + 244, "is infinite: cheat by going", size=14, color=MUTED)
p.text(56, p.top + 272, "huge on real, -huge on fake", size=14, color=MUTED)
p.arrow(468, p.top + 170, 548, p.top + 170, label="dual")
p.panel(556, p.top, 372, 340, label="the walk", fill=NEW)
p.text(580, p.top + 64, "f(x) = -x, slope -1", size=15, bold=True)
p.text(580, p.top + 100, "J = E[-x] - E[-theta] = theta", size=15)
p.text(580, p.top + 144, "theta: 100 -> 0", size=14)
p.text(580, p.top + 172, "gradient 1 at every step", size=14, bold=True, color=TEAL)
p.text(580, p.top + 216, "never stalls, never flat", size=14)
p.text(580, p.top + 260, "enforcement: weight clipping", size=13, color=MUTED)
p.text(580, p.top + 284, "(crude) or gradient penalty", size=13, color=MUTED)
p.save("l05-wgan-game.webp")

# 3. FID pipeline
p = Plate("FID: judge samples with samples",
          "Real and generated images become Gaussians in Inception feature space.",
          "Shell 3. FID = 4 on the toy: identical clouds, centers 2 apart. Source: original toy.",
          source="original toy", inner_h=460)
y = p.top + 24
p.rect(32, y, 180, 88, COUNT, label="real images", size=14, rx=12)
p.rect(32, y + 112, 180, 88, PINK, label="fake images", size=14, rx=12)
p.arrow(228, y + 44, 292, y + 44, label="Inception")
p.arrow(228, y + 156, 292, y + 156, label="Inception")
p.rect(300, y, 200, 200, PANEL, label="feature layer L", size=14, rx=12)
p.text(340, y + 120, "deep features", size=13, color=MUTED)
p.arrow(516, y + 100, 580, y + 100, label="fit")
p.rect(588, y, 340, 200, YELLOW, label="two Gaussians", size=14, rx=12)
p.text(612, y + 64, "(mu, Sigma) real", size=14)
p.text(612, y + 96, "(mu_hat, Sigma_hat) fake", size=14)
p.text(612, y + 140, "assumption: features", size=13, color=MUTED)
p.text(612, y + 164, "are Gaussian (false, useful)", size=13, color=MUTED)
p.text(32, y + 248, "FID = ||mu - mu_hat||^2 + Tr(Sigma + Sigma_hat - 2(Sigma Sigma_hat)^{1/2})", size=14, bold=True)
p.text(32, y + 292, "toy: mu = (1, 1/3), mu_hat = (1, 7/3). Same Sigma.", size=14)
p.text(32, y + 320, "mean term: 0 + 2^2 = 4.   spread term: 0.   FID = 4.", size=14, bold=True, color=FOCUS)
p.text(32, y + 360, "blind spot: replayed training images score ~0 without creating anything", size=13, color=ORANGE)
p.save("l05-fid.webp")
