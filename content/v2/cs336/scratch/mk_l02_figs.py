import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- f: memory movement cartoon (HBM -> accelerator -> HBM) ----
p = Plate(960, 500); p.defs_arrow()
y = p.title("Memory moves set the clock", "Every tensor travels from HBM to the chip and back.")
px, py, pw, ph = 60, 150, 220, 200
p.panel(px, py, pw, ph, "BEFORE: HBM")
p.chip(px+30, py+60, 160, 60, "tensor x", "1M bf16 = 2 MB", fill=BLUE)
p.chip(px+30, py+135, 160, 50, "weights w", "sits in HBM", fill=CHIP)
p.arrow(px+pw+10, px+pw+130, py+100, "read 2n bytes")
cx, cy, cw, chh = 420, 150, 220, 200
p.panel(cx, cy, cw, chh, "RULE: compute on chip")
p.chip(cx+55, cy+70, 110, 60, "ReLU", "n FLOPs", fill=ACTIVE)
p.text(cx+20, cy+170, "compute time = FLOPs / FLOP/s", 13, MUT)
p.text(cx+20, cy+192, "move time = bytes / 3.3 TB/s", 13, MUT)
p.arrow(cx+cw+10, cx+cw+130, cy+100, "write 2n bytes")
rx, ry, rw, rh = 780, 150, 120, 200
p.panel(rx, ry, rw, rh, "AFTER")
p.chip(rx+15, ry+60, 90, 60, "tensor y", "2 MB back", fill=NEWTOK)
p.text(60, 420, "Total time = max(move time, compute time). ReLU: 1us vs 0.001us.", 15, INK)
p.text(60, 446, "Move time wins by 1000x. ReLU is memory bound.", 15, INK, 700)
p.footer("Source: lecture board, original plate. Shell 2: count bytes and FLOPs, name the bottleneck.")
p.save("l02-mem-move.svg")

# ---- f: roofline ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("The roofline: intensity decides realized speed", "Low intensity ops never reach the peak FLOP/s.")
ox, oy, ow, oh = 110, 140, 700, 320
p.panel(ox-30, oy-10, ow+120, oh+90)
# axes
p.parts.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="{INK}" stroke-width="2"/>')
p.parts.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox}" y2="{oy}" stroke="{INK}" stroke-width="2"/>')
p.text(ox+ow/2, oy+oh+40, "arithmetic intensity (FLOPs per byte), log scale", 14, MUT, anchor="middle")
p.text(ox-70, oy+oh/2, "realized FLOP/s", 14, MUT, anchor="middle")
# log x positions: 0.1 -> 1000
import math
def X(v): return ox + (math.log10(v) - math.log10(0.1)) / (math.log10(2000) - math.log10(0.1)) * ow
def Y(v):  # v in TFLOP/s 0..1100
    return oy + oh - v / 1100 * oh
knee = 295
p.parts.append(f'<line x1="{X(0.1)}" y1="{Y(0.1*knee/10*10)}" x2="{X(knee)}" y2="{Y(989)}" stroke="{TEAL}" stroke-width="3"/>')
# memory-bound ramp: y = intensity * 3.35 TB/s -> TFLOP/s
p.parts.append(f'<line x1="{X(0.1)}" y1="{Y(0.1*3.35)}" x2="{X(knee)}" y2="{Y(989)}" stroke="{TEAL}" stroke-width="3"/>')
p.parts.append(f'<line x1="{X(knee)}" y1="{Y(989)}" x2="{X(2000)}" y2="{Y(989)}" stroke="{ORANGE}" stroke-width="3"/>')
p.text(X(2), Y(120), "memory bound", 14, TEAL, 700)
p.text(X(700), Y(880), "compute bound", 14, ORANGE, 700)
p.text(X(knee), Y(1030), "peak 989 TFLOP/s", 13, MUT)
for v, lab, c in [(0.25, "ReLU 0.25", INK), (5, "GELU 5", INK), (340, "matmul ~340", INK)]:
    p.parts.append(f'<circle cx="{X(v)}" cy="{Y(min(v*3.35,989))}" r="7" fill="{c}"/>')
    p.text(X(v), Y(min(v*3.35,989)) - 14, lab, 13, INK, 700, anchor="middle")
p.parts.append(f'<line x1="{X(knee)}" y1="{Y(0)}" x2="{X(knee)}" y2="{Y(989)}" stroke="{MUT}" stroke-width="1.5" stroke-dasharray="6,5"/>')
p.text(X(knee), Y(40), "knee = 295", 13, MUT, 700, anchor="middle")
p.footer("Source: lecture roofline slide, original plate. Shell 3: the knee is the accelerator intensity.")
p.save("l02-roofline.svg")

# ---- f: activation checkpointing ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Checkpointing trades memory for recompute", "Store every sqrt(L) layer, recompute the rest.")
# top: layer stack
lx = 80
for i in range(8):
    fill = FOCUS if i % 3 == 0 else PANEL
    tc = "#FFFFFF" if i % 3 == 0 else INK
    p.chip(lx + i*105, 150, 95, 70, f"L{i+1}", "checkpoint" if i % 3 == 0 else "dropped", fill=fill, tcolor=tc)
p.text(80, 140, "FORWARD: keep 3 of 8 layer activations", 14, MUT, 700)
p.text(80, 250, "BACKWARD: recompute dropped activations from the last checkpoint", 14, MUT, 700)
for i in range(8):
    if i % 3 == 0:
        p.chip(lx + i*105, 280, 95, 70, f"L{i+1}", "stored", fill=FOCUS, tcolor="#FFFFFF")
    else:
        p.chip(lx + i*105, 280, 95, 70, f"L{i+1}", "recomputed", fill=CHIP, hatch=True)
p.arrow(80, 900, 385, "trade")
p.text(80, 420, "Extreme: store 0 layers -> memory O(1), recompute O(L^2).", 15, INK)
p.text(80, 446, "Sweet spot: store sqrt(L) -> memory O(sqrt(L)), recompute O(sqrt(L)).", 15, INK, 700)
p.footer("Source: lecture board, original plate. Shell 4: the checkpoint symbol returns in parallelism.")
p.save("l02-checkpoint.svg")
