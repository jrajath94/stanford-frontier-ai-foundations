import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- metrics ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("Three speeds, three questions", "TTFT: wait for the first token. Latency: streaming for one user. Throughput: tokens per second for all.")
rows = [
    ("TTFT", "time to first token", "interactive wait; prefill time", TEAL),
    ("latency", "seconds per token, one query", "streaming speed per user", BLUE),
    ("throughput", "tokens per second, many queries", "batch jobs; total output", ACTIVE),
]
y0 = 140
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 36, b, 15, INK, 600)
    p.text(84, yy + 62, c, 14, MUT)
p.text(60, 460, "Batch size trades them against each other. No single number is fast.", 15, MUT)
p.footer("Source: lecture metrics slides, original plate. Shell 1: name the metric first.")
p.save("l10-metrics.svg")

# ---- why different ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("Inference cannot parallelize the sequence", "Training sees all tokens at once. Inference generates one at a time. Intensity collapses.")
p.panel(60, 140, 400, 200, "training")
p.text(84, 190, "all tokens visible", 15, INK)
p.text(84, 216, "sequence = a dimension", 15, INK)
p.text(84, 250, "big fat matmuls", 15, TEAL, 700)
p.panel(500, 140, 400, 200, "inference")
p.text(524, 190, "one token at a time", 15, INK)
p.text(524, 216, "autoregressive: no future", 15, INK)
p.text(524, 250, "thin matvecs, intensity ~1", 15, ORANGE, 700)
p.text(60, 390, "Agents removed the human reading limit: tokens are now pure compute spend.", 15, MUT)
p.footer("Source: lecture comparison slides, original plate. Shell 2: the fundamental asymmetry.")
p.save("l10-why-different.svg")

# ---- kv cache ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("The KV cache kills the cube", "Naive: T^3. Causal reuse: prefill once in parallel, decode one token at a time.")
p.panel(60, 140, 400, 180, "naive")
p.text(84, 190, "each token recomputes", 15, INK)
p.text(84, 216, "all previous keys/values", 15, INK)
p.text(84, 250, "T tokens -> O(T^3)", 15, ORANGE, 700)
p.panel(500, 140, 400, 180, "KV cache")
p.text(524, 190, "causal: past never changes", 15, INK)
p.text(524, 216, "store K,V per layer/head", 15, INK)
p.text(524, 250, "prefill + O(T) decode", 15, TEAL, 700)
p.panel(60, 350, 840, 140, "size = B x S x layers x KV-heads x head-dim x 2 (K,V) x 2 bytes")
p.text(84, 400, "prefill: encode prompt in parallel, compute-bound. decode: one token, memory-bound.", 15, INK, 600)
p.footer("Source: lecture KV-cache slides, original plate. Shell 4: cache the past.")
p.save("l10-kv-cache.svg")

# ---- intensity table ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Attention in generation: the bottleneck", "MLP intensity scales with batch. Attention intensity is ~1 forever. Nothing fixes it in a transformer.")
p.panel(60, 140, 840, 260, "arithmetic intensity")
p.text(84, 190, "MLP prefill:    B x S      (good)", 16, INK, 600)
p.text(84, 222, "MLP generate:   B          (ok if many concurrent users)", 16, INK, 600)
p.text(84, 254, "attn prefill:   S / 2      (workable for long prompts)", 16, INK, 600)
p.text(84, 286, "attn generate:  S/(S+1) ~ 1  (THE BOTTLENECK)", 16, ORANGE, 700)
p.panel(60, 430, 840, 90, "why batching cannot help")
p.text(84, 480, "MLP weights are shared: load once, serve B. Each sequence has its own KV cache: B more matvecs.", 15, INK, 600)
p.footer("Source: lecture intensity slides, original plate. Shell 2: intensity decides everything.")
p.save("l10-intensity.svg")

# ---- latency vs throughput ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Batch size: the bus tradeoff", "Small batches: good latency, bad throughput. Big batches: the reverse. Memory caps the bus.")
p.panel(60, 140, 400, 200, "batch = 1 (car)")
p.text(84, 190, "latency 0.008 s/token", 16, INK, 700)
p.text(84, 220, "throughput 124 tok/s", 15, MUT)
p.text(84, 250, "Llama2-13B on H100", 14, MUT)
p.panel(500, 140, 400, 200, "batch = 256 (bus)")
p.text(524, 190, "latency worse", 16, ORANGE, 700)
p.text(524, 220, "throughput much better", 15, INK)
p.text(524, 250, "until KV cache fills HBM", 14, MUT)
p.text(60, 390, "Latency ~ linear in B (KV cache grows). Throughput ~ B/(B + const): asymptotes.", 15, INK, 700)
p.text(60, 420, "Shrinking memory helps both. Only the batch dimension pits them against each other.", 15, TEAL, 700)
p.footer("Source: lecture Llama2 example, original plate. Shell 4: pick the metric, then the batch.")
p.save("l10-latency-throughput.svg")

# ---- kv shrink ----
p = Plate(960, 620); p.defs_arrow()
y = p.title("Four ways to shrink the KV cache", "GQA shares heads. MLA compresses. CLA shares layers. Sliding window bounds the past.")
rows = [
    ("GQA", "fewer KV heads: KV / (N/K)", "K=8 keeps accuracy, much faster", TEAL),
    ("MLA", "project K,V to C dims (16000->512)", "DeepSeek: ~MHA accuracy, tiny cache", ACTIVE),
    ("CLA", "share KV across layers", "another axis of sharing", BLUE),
    ("sliding window", "attend to last K tokens only", "cache independent of length; hybrid with global", ORANGE),
]
y0 = 140
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 108
    p.panel(60, yy, 840, 92, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="92" rx="5" fill="{col}"/>')
    p.text(84, yy + 38, b, 15, INK, 600)
    p.text(84, yy + 64, c, 14, MUT)
p.text(60, 586, "Rule: shrink the cache, keep accuracy. Linear attention/Mamba compress history into a state.", 14, MUT)
p.footer("Source: lecture KV-cache slides, original plate. Shell 4: find the squeeze.")
p.save("l10-kv-shrink.svg")

# ---- speculative decoding ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Speculative decoding: check beats generate", "Draft K tokens with a cheap model. Verify in parallel with the big one. Accept with min(1, q/p).")
p.panel(60, 140, 840, 180, "the loop")
p.chip(100, 180, 220, 80, "draft model p", "cheap, sequential", fill=BLUE)
p.arrow(320, 220, 60, "K tokens")
p.chip(380, 180, 220, 80, "target model q", "big, parallel check", fill=ACTIVE)
p.arrow(600, 220, 60, "accept?")
p.chip(660, 180, 200, 80, "exact sample", "from q", fill=TEAL)
p.panel(60, 350, 840, 120, "why it is exact")
p.text(84, 400, "rejection sampling with a guaranteed sample: accept w.p. min(1, q/p), else sample the residual.", 15, INK, 600)
p.text(84, 426, "sweet spot K = 3-4. Distill the draft toward the target.", 14, MUT)
p.footer("Source: lecture speculative-decoding slides, original plate. Shell 4: verify in parallel.")
p.save("l10-spec-decode.svg")

# ---- serving ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Serving: batch the chaos", "Continuous batching fills gaps. Selective batching flattens MLPs. PagedAttention defrags the cache.")
rows = [
    ("continuous batching (Orca)", "decode one token per seq per step; evict finished, admit new", TEAL),
    ("selective batching", "MLP: concatenate jagged seqs into one mega-seq; attention stays per-seq", BLUE),
    ("PagedAttention (vLLM)", "non-contiguous KV blocks; kills fragmentation; prefix sharing; copy-on-write", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 108
    p.panel(60, yy, 840, 92, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="92" rx="5" fill="{col}"/>')
    p.text(84, yy + 62, b, 14, MUT)
p.text(60, 486, "OS ideas, reused: paging for memory, batching for the queue.", 14, MUT)
p.footer("Source: lecture serving slides, original plate. Shell 5: systems meet the OS.")
p.save("l10-serving.svg")
