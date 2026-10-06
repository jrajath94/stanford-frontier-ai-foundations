#!/usr/bin/env python3
"""L03 plates: three members, variational bound, GAN derivation chain."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathgenai import Plate, BG, INK, MUTED, LINE, PANEL, COUNT, NEW, ACTIVE, CHIP, TEAL, ORANGE, FOCUS, PINK, YELLOW, GREEN

# 1. Three members, one toy
p = Plate("Three f's, one coin toy: 0.511, 0.102, 0.400",
          "Same truth {0.5,0.5}, same model {0.9,0.1}. The f chooses the score.",
          "Shell 2. Count the toy: KL punishes hardest, JS stays calmest. Source: original toy.",
          source="original toy", inner_h=420)
p.bars(64, p.top + 260, [("KL", 0.511, ORANGE), ("Jensen-Shannon", 0.102, TEAL), ("total variation", 0.400, FOCUS)],
       0.6, bar_w=130, gap=80, height=180, size=13)
p.text(64, p.top + 300, "KL: f = u log u. Asymmetric. Infinite on zeros.", size=14)
p.text(64, p.top + 328, "JS: symmetric. Calm. Saturates when far apart.", size=14)
p.text(64, p.top + 356, "TV: f = 0.5|u-1|. Simple. Kinked gradient.", size=14)
p.text(64, p.top + 392, "decision rule: f sets the training dynamics, not just the number", size=13, bold=True, color=FOCUS)
p.save("l03-three-members.webp")

# 2. The variational bound
p = Plate("A critic lower-bounds the divergence from samples alone",
          "Best critic = true divergence. Weak critic = honest underestimate.",
          "Shell 3. Max over T of E[T(x)] - E[f*(T(xhat))]. No densities. Source: original toy.",
          source="original toy", inner_h=420)
p.panel(32, p.top, 420, 320, label="true JS divergence", fill=COUNT)
p.text(56, p.top + 64, "D = 0.102", size=18, bold=True)
p.text(56, p.top + 100, "needs both densities", size=14, color=MUTED)
p.text(56, p.top + 128, "p_X and p_theta", size=14, color=MUTED)
p.text(56, p.top + 180, "unavailable:", size=14, bold=True, color=ORANGE)
p.text(56, p.top + 208, "we have samples only", size=14)
p.arrow(468, p.top + 160, 548, p.top + 160, label="replace")
p.panel(556, p.top, 372, 320, label="variational bound", fill=NEW)
p.text(580, p.top + 64, "max over critic T", size=18, bold=True)
p.text(580, p.top + 100, "E[T(x)] - E[f*(T(xhat))]", size=15)
p.text(580, p.top + 140, "samples only, no densities", size=14, color=TEAL)
p.bars(580, p.top + 280, [("weak critic", 0.06, PINK), ("true D", 0.102, TEAL)],
       0.12, bar_w=80, gap=40, height=120, size=12)
p.text(32, p.top + 344, "the bound never overclaims: weak critic scores 0.06 < 0.102", size=14, bold=True)
p.save("l03-variational-bound.webp")

# 3. The GAN falls out: derivation chain
p = Plate("Choose the GAN's f, and the GAN objective falls out",
          "One substitution chain: f -> conjugate -> critic reparameterization -> the game.",
          "Shell 3. f(u) = u log u - (u+1) log(u+1) becomes E[log D] + E[log(1-D)]. Source: f-GAN paper.",
          source="f-GAN paper", inner_h=420)
y = p.top + 24
p.rect(32, y, 280, 88, COUNT, label="f(u) = u log u", size=14, rx=12)
p.text(120, y + 56, "- (u+1) log(u+1)", size=14, color=MUTED)
p.arrow(328, y + 44, 392, y + 44, label="conjugate")
p.rect(400, y, 280, 88, PANEL, label="f*(t) = -log(1 - e^t)", size=14, rx=12)
p.text(480, y + 56, "t < 0", size=13, color=MUTED)
p.arrow(696, y + 44, 760, y + 44, label="reparam")
p.rect(768, y, 160, 88, YELLOW, label="T = sigma_f(V)", size=14, rx=12)
p.arrow(480, y + 120, 480, y + 168, label="substitute into bound")
p.rect(240, y + 176, 560, 96, FOCUS, label="J = E[log D(x)] + E[log(1 - D(xhat))]", color=BG, size=15, rx=12)
p.text(240, y + 288, "critic maximizes J (D -> 1 on real, -> 0 on fake)", size=14)
p.text(240, y + 316, "generator minimizes J (D -> 1 on its fakes)", size=14)
p.text(240, y + 352, "change f, change the game: the GAN is one point in a design space", size=13, bold=True, color=FOCUS)
p.save("l03-gan-falls-out.webp")
