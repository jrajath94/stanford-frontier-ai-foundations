#!/usr/bin/env python3
"""Chapter plates for cs229 l14 (transformers). All numbers code-computed with asserts."""
import math, os

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/cs229/assets")
SANS = "Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------- number computations (asserted) ----------------
# attention toy: scores 1, 1, 2
e1, e2 = math.exp(1), math.exp(2)
tot = 2 * e1 + e2
w = [e1 / tot, e1 / tot, e2 / tot]
assert abs(tot - 12.82462) < 1e-3, tot
assert abs(w[0] - 0.21194) < 1e-4 and abs(w[2] - 0.57612) < 1e-4, w
assert abs(sum(w) - 1.0) < 1e-9  # spec: softmax sums to 1
frame = [w[0]*1 + w[1]*0 + w[2]*1, w[0]*0 + w[1]*1 + w[2]*1]
assert abs(frame[0] - 0.78806) < 1e-4, frame
# query [1,0]: scores 1, 0, 1
tot2 = 2 * e1 + math.exp(0)
w2 = [e1 / tot2, 1 / tot2, e1 / tot2]
assert abs(tot2 - 6.43656) < 1e-3
assert abs(w2[0] - 0.42232) < 1e-4 and abs(w2[1] - 0.15536) < 1e-4, w2
# 1/sqrt(d) audit: d=64, std 8
ratio_raw = math.exp(32)
assert abs(ratio_raw - 7.896e13) / 7.896e13 < 1e-3, ratio_raw  # ~8e13
ratio_scaled = math.exp(4)
assert abs(ratio_scaled - 54.598) < 1e-2
# quadratic price
scores_4096 = 4096 ** 2
assert scores_4096 == 16_777_216  # 16.7M
scores_8192 = 8192 ** 2
floats_per_head = scores_8192  # 67M floats in HBM
assert floats_per_head == 67_108_864
# block count: d=4096
d = 4096
attn_params = 4 * d * d
mlp_params = 2 * 4 * d * d
block = attn_params + mlp_params
assert attn_params == 67_108_864, attn_params  # 67.1M
assert mlp_params == 134_217_728, mlp_params   # 134.2M
assert block == 201_326_592, block             # 201.3M
blocks32 = 32 * block
assert blocks32 == 6_442_450_944, blocks32     # 6.44B
emb = 50_000 * 4096
assert emb == 204_800_000
assert abs((blocks32 + emb) / 1e9 - 6.65) < 0.01
assert block == 12 * d * d  # the sizing rule
# decoding toy: logits [3,2,1]
def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; s = sum(ex)
    return [x / s for x in ex]
p1 = softmax([3, 2, 1])
assert abs(p1[0] - 0.66524) < 1e-4 and abs(p1[1] - 0.24473) < 1e-4 and abs(p1[2] - 0.09003) < 1e-4, p1
p05 = softmax([6, 4, 2])
assert abs(p05[0] - 0.86681) < 1e-4 and abs(p05[2] - 0.01588) < 1e-4, p05
p2 = softmax([1.5, 1.0, 0.5])
assert abs(p2[0] - 0.50648) < 1e-4 and abs(p2[1] - 0.30720) < 1e-4 and abs(p2[2] - 0.18632) < 1e-4, p2
# NOTE: the lesson text prints T=2 as [0.468, 0.284, 0.248]; the code shows that is wrong.
# Code-computed: [0.506, 0.307, 0.186]. Flagged in the audit report.
# top-p 0.9 on [0.5,0.3,0.15,0.05]
kept = [0.5, 0.3, 0.15]
s = sum(kept)
renorm = [x / s for x in kept]
assert abs(s - 0.95) < 1e-9
assert abs(renorm[0] - 0.52632) < 1e-4 and abs(renorm[2] - 0.15789) < 1e-4, renorm
# exposure bias: 1% per token, 100 tokens
p_err = 1 - 0.99 ** 100
assert abs(p_err - 0.63397) < 1e-4, p_err  # 63%
# perplexity toy
assert abs(math.exp(3.0) - 20.086) < 1e-2
assert abs(math.exp(2.3) - 9.975) < 1e-2
# embedding matrix: 128256 x 4096
assert 128_256 * 4096 == 525_336_576  # 525M

# ---------------- svg builder ----------------
def plate(path, title, subtitle, left_label, center_label, right_label,
          left_lines, center_lines, right_lines, bottom_lines, footer,
          panel_h=300, bottom_h=72):
    W = 960
    y_panels = 140
    y_bottom = y_panels + panel_h + 16
    H = y_bottom + bottom_h + 56
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{SANS}">')
    parts.append(f'<rect width="{W}" height="{H}" fill="#F7F4EE"/>')
    parts.append(f'<text x="48" y="56" font-size="30" font-weight="600" fill="#1B2838">{esc(title)}</text>')
    parts.append(f'<text x="48" y="86" font-size="17" fill="#5C6B7A">{esc(subtitle)}</text>')
    parts.append('<g font-size="18" font-weight="600" fill="#1B2838">')
    parts.append(f'<text x="48" y="124">{esc(left_label)}</text>')
    parts.append(f'<text x="328" y="124">{esc(center_label)}</text>')
    parts.append(f'<text x="648" y="124">{esc(right_label)}</text>')
    parts.append('</g>')
    def region(x, w, fill, stroke, sw, lines, cx):
        parts.append(f'<rect x="{x}" y="{y_panels}" width="{w}" height="{panel_h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        y = y_panels + 44
        for (t, size, color, weight) in lines:
            parts.append(f'<text x="{cx}" y="{y}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{color}">{esc(t)}</text>')
            y += 30 if size >= 16 else 26
    region(48, 264, "#FFFDF8", "#1B2838", 1.5, left_lines, 180)
    region(328, 304, "#E7F1F8", "#1E4D8C", 1.5, center_lines, 480)
    region(648, 264, "#E7F4EF", "#1F7A72", 2, right_lines, 780)
    parts.append(f'<rect x="48" y="{y_bottom}" width="864" height="{bottom_h}" rx="12" fill="#FFFDF8" stroke="#1B2838" stroke-width="1.5"/>')
    y = y_bottom + 30
    for (t, color) in bottom_lines:
        parts.append(f'<text x="480" y="{y}" text-anchor="middle" font-size="15" fill="{color}">{esc(t)}</text>')
        y += 26
    parts.append(f'<text x="48" y="{H - 22}" font-size="16" font-weight="500" fill="#1B2838">{esc(footer)}</text>')
    parts.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", path)

DK, MU, TE, BL = "#1B2838", "#5C6B7A", "#1F7A72", "#1E4D8C"

plate(os.path.join(ASSETS, "plate-l14-chap-attention.svg"),
    "Chapter plate: attention, the QKV lookup",
    "Chapter plate. The query decides the mix. Source: original synthesis of the session.",
    "WITHOUT the rule: fixed windows", "the stored object: the toy", "WITH the rule: quadratic",
    [("mix with 5 neighbors", 15, DK, 600), ("fixed weights, fixed reach", 15, MU, 500),
     ("key word sits 20 back", 15, MU, 500), ("window cannot reach it", 15, MU, 500),
     ("cannot choose which matters", 15, MU, 500), ("context by accident", 15, MU, 500)],
    [("scores: 1, 1, 2", 15, DK, 600), ("weights: 0.212, 0.212, 0.576", 16, DK, 600),
     ("new frame: [0.788, 0.788]", 16, DK, 600), ("softmax sums to 1", 15, MU, 500),
     ("query [1,0]: 0.422, 0.155, 0.422", 15, TE, 600), ("same keys, new question", 15, MU, 500)],
    [("N x N scores", 16, DK, 600), ("N = 4096: 16.7M", 15, MU, 500),
     ("per layer per head", 15, MU, 500), ("divide by sqrt(d)", 15, TE, 600),
     ("d = 64: ratio 8e13 -> 54.6", 15, MU, 500), ("unscaled: winner-take-all", 15, MU, 500)],
    [("Tradeoff: every token sees every token. That is the power, and it is exactly the quadratic bill.", DK),
     ("FlashAttention retiled it to O(N) traffic; the math stayed exact.", MU)],
    "One connection: attention is a data-dependent lookup, and the lookup table is the whole sentence.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l14-chap-positions.svg"),
    "Chapter plate: positions, from tables to rotation",
    "Chapter plate. Attention has no order. Source: original synthesis of the session.",
    "WITHOUT the rule: no order", "the stored object: the encoding", "WITH the rule: relative",
    [("'dog bites man'", 16, DK, 600), ("equals 'man bites dog'", 15, MU, 500),
     ("permutation-invariant", 15, MU, 500), ("shuffle in, shuffle out", 15, MU, 500),
     ("order is invisible", 15, MU, 500)],
    [("sinusoidal: fixed waves", 15, DK, 600), ("learned: the table ends", 15, MU, 500),
     ("RoPE: rotate by position", 15, DK, 600), ("ALiBi: penalize distance", 15, MU, 500),
     ("NoPE: the mask leaks it", 15, MU, 500)],
    [("2026 default: RoPE", 16, TE, 600), ("theta = 500,000", 15, MU, 500),
     ("dot keeps cos((m-n)theta)", 15, MU, 500), ("NoPE: 2.1 vs 250 perplexity", 15, MU, 500),
     ("ALiBi: 2.3, stable to 1.8k", 15, MU, 500), ("[uncertain: secondary analysis]", 13, MU, 500)],
    [("Tradeoff: RoPE is the default; ALiBi extrapolates best; NoPE is free only with a causal mask.", DK),
     ("Absolute methods are history, except learned tables in BERT-style encoders.", MU)],
    "One connection: RoPE turns absolute labels into relative rotation, and the dot product keeps only the distance.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l14-chap-block.svg"),
    "Chapter plate: the block, counted",
    "Chapter plate. 12d^2 sizes any decoder. Source: original synthesis of the session.",
    "WITHOUT the rule: post-norm", "the stored object: one block", "WITH the rule: 32 blocks",
    [("normalize after the add", 15, DK, 600), ("stream rescaled per block", 15, MU, 500),
     ("96 blocks: destabilize", 15, MU, 500), ("deep stacks untrainable", 15, MU, 500),
     ("depth capped", 15, MU, 500)],
    [("attention: 4d^2 = 67.1M", 16, DK, 600), ("MLP: 8d^2 = 134.2M", 16, DK, 600),
     ("block: 201.3M", 18, DK, 600), ("pre-norm + residuals", 15, TE, 600),
     ("the stream never rescales", 15, MU, 500), ("SwiGLU gates the MLP", 15, MU, 500)],
    [("32 x 201.3M = 6.44B", 16, DK, 600), ("+ embeddings 204.8M", 15, MU, 500),
     ("= ~6.6B: a 7B model", 16, TE, 600), ("the rule: 12d^2 per block", 15, MU, 500),
     ("RMSNorm, no centering", 15, MU, 500)],
    [("Tradeoff: depth buys composition; pre-norm buys depth. The sizing rule is 12d^2 per block, nothing else.", DK),
     ("The block is a shared whiteboard: attention and MLP only add to the stream.", MU)],
    "One connection: the residual stream is the model's working memory, and pre-norm is what lets it reach 96 blocks.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l14-chap-decoding.svg"),
    "Chapter plate: decoding, from distribution to text",
    "Chapter plate. The distribution is not text. Source: original synthesis of the session.",
    "WITHOUT the rule: greedy", "the stored object: the distribution", "WITH the rule: the nucleus",
    [("argmax every step", 15, DK, 600), ("repetition loops", 15, MU, 500),
     ("'the the the'", 15, MU, 500), ("no backtrack", 15, MU, 500),
     ("locally optimal, globally dull", 15, MU, 500)],
    [("logits [3, 2, 1]", 15, DK, 600), ("T = 1: 0.665, 0.245, 0.090", 16, DK, 600),
     ("T = 0.5: 0.867, 0.117, 0.016", 15, MU, 500), ("T = 2: 0.506, 0.307, 0.186", 15, MU, 500),
     ("T -> 0 is greedy, T -> inf is uniform", 14, MU, 500)],
    [("top-p 0.9: cut the tail", 16, TE, 600), ("keep 0.526, 0.316, 0.158", 15, DK, 600),
     ("drop the 0.05 token", 15, MU, 500), ("code: low T, prose: high T", 15, MU, 500),
     ("penalty 1.0-1.2 for repeats", 15, MU, 500)],
    [("Tradeoff: temperature sets creativity; top-p sets safety. Decoding tunes the draw from the distribution.", DK),
     ("It cannot fix the distribution. Exposure bias: 1% per token means 63% of 100-token answers err.", MU)],
    "One connection: decoding turns one distribution into many possible texts, and the dials pick which one.",
    panel_h=300, bottom_h=72)
print("l14 plates done")
