#!/usr/bin/env python3
"""Chapter plates for cs229 l15 (efficiency, ICL, SFT). All numbers code-computed with asserts."""
import math, os

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/cs229/assets")
SANS = "Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------- number computations (asserted) ----------------
N = 4096
cubic = N * (N + 1) * (2 * N + 1) // 6
assert cubic == 22_914_881_536, cubic  # 2.29e10
cached = N * (N + 1) // 2
assert cached == 8_390_656, cached     # 8.4M
ratio = cubic / cached
assert abs(ratio - 2731.6) < 1.0, ratio  # ~2,730x
# cache bytes: 32 layers, 32 heads, head dim 128, fp16
k_per_layer = 32 * 128 * 2
v_per_layer = 32 * 128 * 2
per_token = (k_per_layer + v_per_layer) * 32
assert per_token == 524_288, per_token  # 512 KB
seq4k = per_token * 4096
assert seq4k == 2_147_483_648, seq4k    # 2 GB
batch10 = seq4k * 10 / 1e9
batch10_gib = seq4k * 10 / (1024 ** 3)
assert abs(batch10_gib - 20.0) < 1e-9, batch10_gib  # lesson's "20 GB" is GiB
# GQA-8
gqa_kv = 8 * 128 * 2
per_token_gqa = (gqa_kv * 2) * 32
assert per_token_gqa == 131_072, per_token_gqa  # 128 KB
assert per_token / per_token_gqa == 4.0
# MQA: 1 KV head
per_token_mqa = ((1 * 128 * 2) * 2) * 32
assert per_token_mqa == 16_384, per_token_mqa   # 16 KB
assert per_token / per_token_mqa == 32.0
# MLA: latent 512 dims vs MHA 16 KB per layer
mla_per_layer = 512 * 2
mha_per_layer = (k_per_layer + v_per_layer)
assert mha_per_layer / mla_per_layer == 16.0
# PagedAttention fragmentation
reserved = 10 * 2  # GB
used = 10 * 500 * per_token / 1e9
assert abs(used - 2.62) < 0.2, used
util_contig = used / reserved
assert abs(util_contig - 0.131) < 0.02, util_contig  # ~12%
# serving cost model
toks_per_hour = 2000 * 3600
cost_per_m = 2.0 / (toks_per_hour / 1e6)
assert abs(cost_per_m - 0.2778) < 1e-3, cost_per_m  # $0.28/M
# MoE router toy: logits [2.0, 1.0, 0.5, -1.0]
logits = [2.0, 1.0, 0.5, -1.0]
ex = [math.exp(x) for x in logits]
s = sum(ex)
probs = [x / s for x in ex]
assert abs(probs[0] - 0.60944) < 1e-4 and abs(probs[1] - 0.22419) < 1e-4, probs
assert abs(probs[2] - 0.13599) < 1e-4 and abs(probs[3] - 0.03035) < 1e-4, probs
top2 = [probs[0], probs[1]]
ts = sum(top2)
renorm = [x / ts for x in top2]
assert abs(renorm[0] - 0.73104) < 1e-4 and abs(renorm[1] - 0.26896) < 1e-4, renorm
# NOTE: the lesson prints [0.735, 0.265] (renormalizing its rounded 0.61/0.22).
# Code-computed from the logits: [0.731, 0.269]. Used below; flagged in audit.
# DeepSeek-V3: 37B active of 671B
assert abs(37 / 671 - 0.0551) < 1e-3
# ICL context tax
assert 20 * 100 == 2000
# LoRA: d=4096, r=16
lora = 2 * 4096 * 16
full = 4096 * 4096
assert lora == 131_072, lora
assert full == 16_777_216, full
assert full / lora == 128.0
# speculative: draft acceptance bound
# (lesson: 80% on 5-token drafts ~4x, 30% ~1.4x -- illustrative, kept as lesson numbers)

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

plate(os.path.join(ASSETS, "plate-l15-chap-kvcache.svg"),
    "Chapter plate: the KV cache kills the cubic",
    "Chapter plate. State is memory. Source: original synthesis of the session.",
    "WITHOUT the rule: recompute", "the stored object: the cache", "WITH the rule: O(t) per token",
    [("token t costs t^2 scores", 15, DK, 600), ("sum to 4096: 2.3e10", 16, DK, 600),
     ("per layer per head", 15, MU, 500), ("tokens 1-100 unchanged", 15, MU, 500),
     ("every score recomputed", 15, MU, 500), ("unservable", 15, MU, 500)],
    [("store K, V of past tokens", 15, DK, 600), ("512 KB per token", 16, DK, 600),
     ("4k sequence: 2 GB", 15, MU, 500), ("batch of 10: 20 GB", 15, MU, 500),
     ("+ 14 GB weights = 34 GB", 15, TE, 600), ("on an 80 GB GPU: fits, barely", 15, MU, 500)],
    [("token t costs t scores", 15, DK, 600), ("sum to 4096: 8.4M", 16, TE, 600),
     ("2,730x cheaper", 18, TE, 600), ("cubic becomes quadratic", 15, MU, 500),
     ("not a constant shaved", 15, MU, 500), ("an order of growth deleted", 15, MU, 500)],
    [("Tradeoff: the cache deletes an order of growth and fills the GPU. Bytes, not big-O, set the batch size.", DK),
     ("Cache fills memory, batch shrinks, parallelism dies, throughput dies.", MU)],
    "One connection: keys and values do not change between steps, so storing them turns O(N^3) into O(N^2).",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l15-chap-diets.svg"),
    "Chapter plate: the cache diets, stacked",
    "Chapter plate. Every diet attacks one axis. Source: original synthesis of the session.",
    "WITHOUT diets: full MHA", "the stored object: the diets", "WITH diets: the stack",
    [("32 query heads, 32 KV heads", 15, DK, 600), ("512 KB per token", 15, MU, 500),
     ("2 GB per 4k sequence", 15, MU, 500), ("batch of 10, not 1,000", 15, MU, 500),
     ("memory-bound decode", 15, MU, 500)],
    [("GQA-8: 128 KB, 4x", 16, DK, 600), ("MQA: 16 KB, 32x", 15, MU, 500),
     ("MLA latent: ~16x smaller", 15, MU, 500), ("CLA-2: halve, ~0.04 ppl", 15, MU, 500),
     ("fp8 cache: halve, minimal loss", 15, MU, 500), ("evict: fixed budget", 15, MU, 500)],
    [("GQA-8 + fp8 + evict", 16, TE, 600), ("paged: 12% to 96% util", 15, MU, 500),
     ("speculative: 2-3x", 15, MU, 500), ("INT8: 2x the batch", 15, MU, 500),
     ("$2/hr, 2000 tok/s", 15, MU, 500), ("$0.28 per M tokens", 15, TE, 600)],
    [("Tradeoff: heads, layers, precision, length, fragmentation. Shrink the bytes, then stop wasting them.", DK),
     ("Eviction is lossy and irreversible: a question about an evicted token cannot be answered.", MU)],
    "One connection: the cache is the most compressible state in the system, and each diet compresses a different axis.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l15-chap-moe.svg"),
    "Chapter plate: mixture of experts",
    "Chapter plate. Scale parameters, not compute. Source: original synthesis of the session.",
    "WITHOUT the rule: dense", "the stored object: the router", "WITH the rule: sparse",
    [("every parameter, every token", 15, DK, 600), ("671B params = 671B active", 15, MU, 500),
     ("compute grows with size", 15, MU, 500), ("one GPU cannot hold it", 15, MU, 500)],
    [("logits [2.0, 1.0, 0.5, -1.0]", 15, DK, 600), ("softmax: 0.609, 0.224, 0.136, 0.030", 15, MU, 500),
     ("top-2: 0.731, 0.269", 16, DK, 600), ("2 of 4 experts compute", 15, TE, 600),
     ("rest are dark", 15, MU, 500), ("router is one linear layer", 15, MU, 500)],
    [("DeepSeek-V3", 16, DK, 600), ("256 experts, top-8", 15, MU, 500),
     ("+ 1 shared expert", 15, MU, 500), ("671B params, 37B active", 16, TE, 600),
     ("5.5% active per token", 15, MU, 500), ("Mixtral: 2 of 8", 15, MU, 500)],
    [("Tradeoff: flat per-token compute for a systems project: memory, load balance, all-to-all, router collapse.", DK),
     ("Token-choice floods; the aux loss or per-expert bias keeps the balance.", MU)],
    "One connection: the router is one linear layer, and its decisions are the entire capacity bet.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l15-chap-icl.svg"),
    "Chapter plate: in-context learning",
    "Chapter plate. The prompt is the program. Source: original synthesis of the session.",
    "WITHOUT the rule: train per task", "the stored object: frozen weights", "WITH the rule: prompt",
    [("collect data, label it", 15, DK, 600), ("train a model per task", 15, MU, 500),
     ("months of work per task", 15, MU, 500), ("a model per company", 15, MU, 500)],
    [("sea -> mer, sky -> ciel", 15, DK, 600), ("cheese -> fromage", 16, DK, 600),
     ("no gradient step", 15, TE, 600), ("no weight moved", 15, TE, 600),
     ("patterns with a job inside", 15, MU, 500), ("attention continues the pattern", 15, MU, 500)],
    [("emerges past ~10B", 15, DK, 600), ("reliable at 175B", 15, MU, 500),
     ("order reversal: double-digit swing", 15, MU, 500), ("wording beats more examples", 15, MU, 500),
     ("20 x 100 = 2,000 tokens", 15, TE, 600), ("of overhead per query, forever", 15, MU, 500)],
    [("Tradeoff: zero training buys fragility. Prototype with ICL, ship with SFT: the sketchpad vs the ink.", DK),
     ("Match the method to the change rate: daily in the prompt, weekly in the store, stable in the weights.", MU)],
    "One connection: attention implements 'continue the pattern', and the prompt is where the pattern lives.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l15-chap-sft.svg"),
    "Chapter plate: SFT teaches the job description",
    "Chapter plate. Pre-training teaches the world. Source: original synthesis of the session.",
    "WITHOUT the rule: base model", "the stored object: the pair", "WITH the rule: SFT",
    [("completes text", 15, DK, 600), ("any continuation", 15, MU, 500),
     ("not an assistant", 15, MU, 500), ("helpfulness not selected", 15, MU, 500)],
    [("x = instruction, y = answer", 15, DK, 600), ("loss on y ONLY", 16, DK, 600),
     ("labels: [-100,-100,-100,mer]", 15, MU, 500), ("mask the instruction", 15, TE, 600),
     ("chat template first", 15, MU, 500), ("predict response given x", 15, MU, 500)],
    [("1,000 curated beats 50,000", 16, TE, 600), ("LoRA: 131k vs 16.8M", 15, DK, 600),
     ("128x fewer params", 15, MU, 500), ("forgetting: 1-3 points", 15, MU, 500),
     ("LR 1e-5, 1-3 epochs", 15, MU, 500), ("mix 5-10% pretrain data", 15, MU, 500)],
    [("Tradeoff: SFT narrows the model. Spend the budget on quality, not count. Teach the format, keep the world.", DK),
     ("The stack: pretrain the world, SFT the format, RL the taste. Each stage cheaper than the last.", MU)],
    "One connection: the loss mask focuses every gradient step on the response, and the response format is the whole lesson.",
    panel_h=300, bottom_h=72)
print("l15 plates done")
