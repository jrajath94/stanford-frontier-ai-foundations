import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- FLOPs vs FLOP/s ----
p = Plate(960, 420); p.defs_arrow()
y = p.title("FLOPs is work. FLOP/s is speed.", "Lowercase s counts operations done. /s is hardware speed.")
px, py, pw, ph = 80, 140, 360, 180
p.panel(px, py, pw, ph, "FLOPs (lowercase s)")
p.text(px+24, py+70, "number of floating-point", 17, INK)
p.text(px+24, py+98, "operations performed", 17, INK)
p.text(px+24, py+140, "GPT-3: ~3.14e23 FLOPs", 15, MUT)
p.text(px+24, py+164, "6ND for one training run", 15, MUT)
rx, ry, rw, rh = 520, 140, 360, 180
p.panel(rx, ry, rw, rh, "FLOP/s (per second)")
p.text(rx+24, ry+70, "operations the hardware", 17, INK)
p.text(rx+24, ry+98, "can do per second", 17, INK)
p.text(rx+24, ry+140, "H100 bf16: 989 TFLOP/s", 15, MUT)
p.text(rx+24, ry+164, "divide the 1979 spec by 2", 15, MUT)
p.text(80, 380, "Time = FLOPs / FLOP/s. Never confuse the two in an interview.", 16, INK, 700)
p.footer("Source: lecture board, original plate. Shell 1: name the question, work vs speed.")
p.save("l02-flops-vs-flops.svg")

# ---- precision ladder ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Precision: fewer bits, same range, less resolution", "bf16 keeps fp32's exponent. That is the whole trick.")
rows = [
    ("fp32", "32", "1+8+23", "safe default", "4 B", CHIP),
    ("fp16", "16", "1+5+10", "underflows: 1e-8 -> 0", "2 B", PINK),
    ("bf16", "16", "1+8+7", "sweet spot for training", "2 B", NEWTOK),
    ("fp8", "8", "E4M3 / E5M2", "two variants, range vs precision", "1 B", ACTIVE),
    ("nvfp4", "4", "block-scaled", "Nemotron 3 Super trained in it", "0.5 B", CHIP),
]
x0, y0, cw = 70, 140, [110, 70, 130, 300, 70]
headers = ["format", "bits", "sign+exp+mant", "verdict", "bytes"]
for j, h in enumerate(headers):
    p.text(x0 + sum(cw[:j]) + 8, y0 - 12, h, 13, MUT, 700)
for i, (f, b, s, v, by, col) in enumerate(rows):
    yy = y0 + i * 72
    p.chip(x0, yy, 830, 62, "", fill=PANEL)
    vals = [f, b, s, v, by]
    for j, val in enumerate(vals):
        wgt = 700 if j == 0 else 400
        fill = INK if j == 0 else (TEAL if (j == 3 and f == "bf16") else INK)
        p.text(x0 + sum(cw[:j]) + 8, yy + 38, val, 15, fill, wgt)
    p.parts.append(f'<rect x="{x0}" y="{yy}" width="10" height="62" rx="5" fill="{col}"/>')
p.footer("Source: lecture precision slides, original plate. Shell 3: the exponent bits decide the range.")
p.save("l02-precision.svg")

# ---- 6ND ----
p = Plate(960, 500); p.defs_arrow()
y = p.title("Training costs 6ND FLOPs", "Forward is 2ND. Backward is 4ND. Add them.")
p.panel(70, 140, 250, 150, "FORWARD")
p.text(94, 200, "one multiply +", 17, INK)
p.text(94, 228, "one add per triple", 17, INK)
p.text(94, 262, "= 2 x N x P", 20, FOCUS, 700)
p.arrow(330, 420, 215, "2x the work")
p.panel(430, 140, 250, 150, "BACKWARD")
p.text(454, 200, "grad w.r.t. input", 17, INK)
p.text(454, 228, "grad w.r.t. weights", 17, INK)
p.text(454, 262, "= 4 x N x P", 20, FOCUS, 700)
p.arrow(690, 780, 215, "sum")
p.panel(790, 140, 120, 150, "TOTAL")
p.text(810, 215, "6NP", 26, FOCUS, 700)
p.text(70, 350, "N = tokens, P = parameters. Two worked checks from the lecture:", 15, MUT)
p.text(70, 380, "70B params x 15T tokens x 6 = 6.3e24 FLOPs -> 143 days on 1024 H100s.", 16, INK)
p.text(70, 408, "Holds for transformers while attention stays cheaper than the MLPs.", 16, INK)
p.footer("Source: lecture board, original plate. Shell 4: 6ND becomes the scaling-law currency.")
p.save("l02-6nd.svg")

# ---- bytes per param ----
p = Plate(960, 460); p.defs_arrow()
y = p.title("AdamW needs 12 bytes per parameter", "Two for the model, two for the gradient, eight for Adam.")
items = [("params", "2 B", "bf16", BLUE), ("gradients", "2 B", "bf16", BLUE),
         ("Adam m", "4 B", "fp32 1st moment", ORANGE), ("Adam v", "4 B", "fp32 2nd moment", ORANGE)]
x = 70
for lab, by, sub, col in items:
    p.chip(x, 150, 190, 110, lab, by + "  " + sub, fill=col)
    x += 215
p.text(70, 310, "8 H100s hold 640 GB. 640e9 / 12 bytes = ~53B parameters.", 17, INK, 700)
p.text(70, 340, "Optimizer states use fp32 for stability: squares and running averages", 15, MUT)
p.text(70, 364, "of small gradients underflow in bf16.", 15, MUT)
p.text(70, 394, "Activations (2*B*D*L) excluded: they depend on batch size, not the model.", 15, MUT)
p.footer("Source: lecture board, original plate. Shell 2: count the bytes that decide fit.")
p.save("l02-bytes-per-param.svg")

# ---- backward = 2x forward ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("The backward pass runs two matmuls", "One for the input gradient, one for the weight gradient.")
p.panel(60, 140, 840, 120, "FORWARD (einsum)")
p.text(84, 190, "h2[batch, out] = sum_in  h1[batch, in] x w2[in, out]", 17, INK, 600)
p.text(84, 222, "cost: 2 x batch x in x out", 15, MUT)
p.panel(60, 300, 410, 130, "BACKWARD message")
p.text(84, 346, "dh1[batch, in] = sum_out", 16, INK, 600)
p.text(84, 372, "dh2[batch, out] x w2[in, out]", 16, INK, 600)
p.text(84, 406, "same cost: 2 x batch x in x out", 15, TEAL, 700)
p.panel(490, 300, 410, 130, "WEIGHT gradient")
p.text(514, 346, "dw2[in, out] = sum_batch", 16, INK, 600)
p.text(514, 372, "dh2[batch, out] x h1[batch, in]", 16, INK, 600)
p.text(514, 406, "same cost: 2 x batch x in x out", 15, TEAL, 700)
p.footer("Source: lecture einsum board, original plate. Shell 3: einsum names kill transpose bugs.")
p.save("l02-backward.svg")
