import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- prenorm vs postnorm ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Keep the residual stream clean", "Move the norm out of the residual path. Everyone agrees on this.")
px, py, pw, ph = 60, 140, 390, 300
p.panel(px, py, pw, ph, "POSTNORM (original, 2017)")
p.chip(px+140, py+30, 110, 44, "x", "residual stream", fill=BLUE)
p.text(px+30, py+110, "attn(x) added, THEN norm", 14, INK)
p.chip(px+140, py+120, 110, 44, "LayerNorm", "in the stream", fill=PINK)
p.text(px+30, py+200, "every block rescales x", 14, INK)
p.text(px+30, py+224, "gradient norms drift", 14, ORANGE, 700)
p.text(px+30, py+248, "needs warmup to converge", 14, MUT)
rx, ry, rw, rh = 510, 140, 390, 300
p.panel(rx, ry, rw, rh, "PRENORM (modern)")
p.chip(rx+140, ry+30, 110, 44, "x", "untouched stream", fill=BLUE)
p.text(rx+30, ry+110, "norm BEFORE attn / FFN", 14, INK)
p.chip(rx+140, ry+120, 110, 44, "RMSNorm", "outside stream", fill=NEWTOK)
p.text(rx+30, ry+200, "x flows straight through", 14, INK)
p.text(rx+30, ry+224, "gradient size constant", 14, TEAL, 700)
p.text(rx+30, ry+248, "converges without warmup", 14, MUT)
p.arrow(px+pw+8, rx-8, 290, "move")
p.text(60, 490, "Rule of thumb from practitioners: keep your residual stream clean.", 15, INK, 700)
p.footer("Source: lecture norm slides, original plate. Shell 4: prenorm is the default in every later lesson.")
p.save("l03-prenorm.svg")

# ---- RMSNorm ----
p = Plate(960, 500); p.defs_arrow()
y = p.title("RMSNorm drops the mean subtraction", "Same modeling power. Far less memory traffic.")
p.panel(60, 140, 400, 200, "LayerNorm")
p.text(84, 190, "subtract mean, divide by std,", 16, INK)
p.text(84, 216, "then scale and shift", 16, INK)
p.text(84, 262, "0.17% of FLOPs ...", 15, MUT)
p.text(84, 288, "up to 25% of runtime", 15, ORANGE, 700)
p.panel(500, 140, 400, 200, "RMSNorm")
p.text(524, 190, "divide by RMS, then scale.", 16, INK)
p.text(524, 216, "No mean. No bias.", 16, INK)
p.text(524, 262, "no expressiveness loss", 15, TEAL, 700)
p.text(524, 288, "free systems win", 15, TEAL, 700)
p.arrow(460, 500, 240, "drop")
p.text(60, 400, "LayerNorm is memory-movement heavy: the workload is moving activations,", 15, INK)
p.text(60, 426, "not computing. Same logic kills bias terms in linear layers.", 15, INK)
p.footer("Source: lecture norm slides (Narang et al. 2020), original plate. Shell 3: intensity decides.")
p.save("l03-rmsnorm.svg")

# ---- GLU ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("A gated MLP multiplies two projections", "SwiGLU = Swish(xW) times (xV), then down-project.")
p.panel(60, 140, 840, 260, "SwiGLU block")
p.chip(110, 220, 90, 60, "x", "d_model", fill=BLUE)
p.arrow(200, 260, 250, "")
p.chip(260, 200, 110, 50, "W", "up-project", fill=CHIP)
p.chip(260, 260, 110, 50, "V", "gate", fill=CHIP)
p.text(390, 225, "Swish", 15, TEAL, 700)
p.text(390, 285, "x", 22, FOCUS, 700)
p.text(340, 340, "elementwise multiply", 14, MUT)
p.arrow(430, 540, 250, "")
p.chip(540, 220, 110, 60, "W2", "down-project", fill=CHIP)
p.arrow(650, 710, 250, "")
p.chip(710, 220, 130, 60, "output", "d_model", fill=NEWTOK)
p.text(110, 440, "Three matrices instead of two: shrink ff dim by 2/3 to match parameters.", 15, INK, 700)
p.text(110, 466, "Google uses GeGLU (Gemma, T5). Llama descendants use SwiGLU.", 15, MUT)
p.footer("Source: lecture activation slides (Shazeer 2020), original plate. Shell 4: the gate symbol.")
p.save("l03-glu.svg")

# ---- RoPE ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("RoPE rotates queries and keys by position", "Inner products are rotation-invariant, so only relative distance matters.")
p.panel(60, 140, 400, 240, "BEFORE: absolute positions")
p.chip(110, 210, 90, 60, "we", "pos 0", fill=BLUE)
p.chip(230, 210, 90, 60, "know", "pos 1", fill=BLUE)
p.text(110, 320, "shift the sentence,", 14, INK)
p.text(110, 344, "scores change", 14, ORANGE, 700)
p.panel(500, 140, 400, 240, "AFTER: rotate by position")
p.chip(550, 210, 90, 60, "we", "rot 2t", fill=NEWTOK)
p.chip(670, 210, 90, 60, "know", "rot 3t", fill=NEWTOK)
p.text(550, 320, "shift the sentence,", 14, INK)
p.text(550, 344, "relative angle still 1t", 14, TEAL, 700)
p.arrow(460, 500, 260, "rotate")
p.text(60, 440, "Split d dims into pairs. Rotate each pair at its own frequency.", 15, INK)
p.text(60, 466, "Slow pairs catch long range. Fast pairs catch neighbors.", 15, INK)
p.text(60, 492, "Apply to Q and K at every attention layer. No cross terms, purely relative.", 15, INK, 700)
p.footer("Source: lecture RoPE slides, original plate. Shell 4: rotated Q/K reused in inference.")
p.save("l03-rope.svg")

# ---- hyperparams ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Three forgiving hyperparameters", "Basins are wide. Defaults win. Only FLOPs truly matter.")
rows = [
    ("ff dim / d_model", "4x  (2.67x GLU, 3.5x Llama)", "Kaplan basin: 1-10 flat", TEAL),
    ("head dim x heads", "= d_model  (ratio ~1)", "wide basin, not critical", TEAL),
    ("d_model / n_layers", "~100", "wide better for systems", TEAL),
    ("vocab size", "30k mono / 100-200k multi", "bigger model, bigger vocab", CHIP),
]
x0, y0 = 70, 150
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 88
    p.chip(x0, yy, 830, 76, "", fill=PANEL)
    p.parts.append(f'<rect x="{x0}" y="{yy}" width="10" height="76" rx="5" fill="{col}"/>')
    p.text(x0 + 26, yy + 32, a, 16, INK, 700)
    p.text(x0 + 26, yy + 58, b, 15, FOCUS, 600)
    p.text(x0 + 480, yy + 44, c, 14, MUT)
p.footer("Source: lecture survey table (Kaplan 2020), original plate. Shell 2: count the ratios.")
p.save("l03-hparams.svg")

# ---- z-loss ----
p = Plate(960, 480); p.defs_arrow()
y = p.title("z-loss pins the softmax normalizer", "Add (log z)^2 to the loss. Penalize drift from zero.")
p.panel(60, 140, 840, 130, "log prob = u - log z")
p.text(84, 190, "u: model output, well-behaved", 16, INK)
p.text(84, 220, "log z: exp() can blow up either way", 16, ORANGE, 700)
p.panel(60, 300, 840, 80, "fix: loss += c * (log z)^2")
p.text(84, 350, "z near 1 keeps the output softmax numerically stable", 16, TEAL, 700)
p.text(60, 430, "Devlin 2014. Revived by Baichuan, DCLM, OLMo.", 14, MUT)
p.footer("Source: lecture stability slides, original plate. Shell 3: one added term, one danger removed.")
p.save("l03-zloss.svg")

# ---- QK norm ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("QK norm: normalize before the QK multiply", "Q and K enter the softmax at scale ~1. Attention stays stable.")
p.panel(60, 140, 840, 200, "attention with QK norm")
p.chip(100, 200, 100, 60, "Q", "RMSNorm", fill=NEWTOK)
p.chip(230, 200, 100, 60, "K", "RMSNorm", fill=NEWTOK)
p.arrow(330, 420, 230, "matmul")
p.chip(430, 200, 120, 60, "softmax", "stable inputs", fill=ACTIVE)
p.arrow(550, 640, 230, "")
p.chip(650, 200, 100, 60, "x V", "weighted avg", fill=BLUE)
p.text(100, 320, "Without the norms, QK magnitudes drift and the softmax saturates or blows up.", 15, INK)
p.text(60, 400, "From multimodal models (Idefics, Chameleon). Now standard in large LMs.", 15, MUT)
p.text(60, 426, "No measured performance cost. Prevents attention degeneracies.", 15, TEAL, 700)
p.text(60, 472, "Stronger variant: logit soft-capping (Gemma 2/3/4) bounds logits with tanh.", 15, INK)
p.text(60, 498, "Soft-capping is safer but costs expressiveness: very confident signals get clipped.", 15, MUT)
p.footer("Source: lecture stability slides, original plate. Shell 4: norms inside attention.")
p.save("l03-qknorm.svg")

# ---- GQA ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("GQA shares key-value heads across query heads", "Less KV cache to move. Nearly the same quality.")
cols = [("MHA", 8, 8, CHIP), ("GQA", 8, 2, NEWTOK), ("MQA", 8, 1, PINK)]
x = 70
for name, nq, nkv, col in cols:
    p.panel(x, 150, 260, 330, name)
    for i in range(min(nq, 6)):
        p.chip(x + 20 + (i % 3) * 80, 200 + (i // 3) * 70, 70, 56, f"Q{i+1}", "", fill=BLUE)
    if nq > 6:
        p.text(x + 20, 350, f"... {nq} query heads", 13, MUT)
    yy = 390
    for i in range(nkv):
        p.chip(x + 20 + i * 80, yy, 70, 56, "KV" if nkv < 8 else f"KV{i+1}", "shared" if nkv < 8 else "", fill=col)
    x += 290
p.text(70, 530, "Decode is memory bound on KV cache reads. Fewer KV heads = less movement.", 15, INK, 700)
p.text(70, 556, "GQA is the sweet spot: inference cost near MQA, quality near MHA.", 15, INK)
p.footer("Source: lecture attention slides, original plate. Shell 5: KV cache blocks reuse this layout.")
p.save("l03-gqa.svg")
