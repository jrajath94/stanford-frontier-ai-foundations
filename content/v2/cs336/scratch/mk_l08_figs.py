import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- zero ladder ----
p = Plate(960, 620); p.defs_arrow()
y = p.title("ZeRO: shard more, pay nothing", "Stage 1 and 2 reuse the all-reduce budget. Stage 3 adds one all-gather, hidden by overlap.")
rows = [
    ("DDP", "replicate all", "2P (all-reduce)", MUT),
    ("ZeRO-1", "shard optimizer state", "2P (RS + AG)", TEAL),
    ("ZeRO-2", "shard grads too", "2P (incremental)", TEAL),
    ("ZeRO-3 / FSDP", "shard params too", "3P (2 AG + 1 RS)", ORANGE),
]
y0 = 140
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 108
    p.panel(60, yy, 840, 92, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="92" rx="5" fill="{col}"/>')
    p.text(84, yy + 38, b, 15, INK, 600)
    p.text(84, yy + 64, "comm per step: " + c, 14, MUT)
p.text(60, 586, "Stage 2 trick: reduce-scatter grads during the backward sweep, never materialize. Stage 3: all-gather params per layer, free after use.", 14, MUT)
p.footer("Source: lecture ZeRO slides, original plate. Shell 4: free memory is the best memory.")
p.save("l08-zero-ladder.svg")

# ---- topology philosophy ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Mesh vs tree, and convergence", "TPU torus: neighbors only, scales forever. GPU fat tree: all-to-all, flexible. MoE pushed TPUs toward trees.")
p.panel(60, 140, 400, 240, "TPU: toroidal mesh")
p.text(84, 190, "each chip talks to neighbors", 15, INK)
p.text(84, 216, "neighbors wrap around", 15, INK)
p.text(84, 242, "same degree at any scale", 15, TEAL, 700)
p.text(84, 280, "great for dense, regular", 14, MUT)
p.panel(500, 140, 400, 240, "GPU: fat tree")
p.text(524, 190, "fast links at bottom", 15, INK)
p.text(524, 216, "pods + spine switches", 15, INK)
p.text(524, 242, "any-to-any routing", 15, TEAL, 700)
p.text(524, 280, "great for MoE, irregular", 14, MUT)
p.panel(60, 410, 840, 90, "convergent evolution: TPU8i moved to a tree. Workloads define the network.")
p.text(84, 456, "MoE token routing needs all-to-all. Huawei Ascend: weaker chips, fiber everywhere, 4x power.", 14, MUT)
p.footer("Source: lecture hardware slides, original plate. Shell 3: workloads shape the wire.")
p.save("l08-topology-philosophy.svg")

# ---- pipeline bubble ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Bubbles are idle tax", "Naive pipeline: one GPU works at a time. Micro-batches fill the pipe. Zero-bubble splits B and W.")
p.panel(60, 140, 400, 200, "naive: 4 stages, 1 batch")
for r in range(4):
    x = 90 + r * 80
    p.parts.append(f'<rect x="{x}" y="{190+r*30}" width="60" height="22" rx="4" fill="{ACTIVE if r==0 else CHIP}" stroke="{INK}" stroke-width="1"/>')
p.text(90, 330, "utilization: terrible", 14, ORANGE, 700)
p.panel(500, 140, 400, 200, "micro-batches: 4 stages, 8 micros")
for r in range(4):
    for m in range(3):
        x = 530 + m * 90 + r * 12
        p.parts.append(f'<rect x="{x}" y="{190+r*30}" width="70" height="22" rx="4" fill="{ACTIVE}" stroke="{INK}" stroke-width="1"/>')
p.text(530, 330, "bubble -> 0 as 1 / microbatches", 14, TEAL, 700)
p.panel(60, 370, 840, 120, "zero-bubble: backward = B (propagate partials) + W (weight grads)")
p.text(84, 420, "B must come first: the next stage waits. W can wait. Do B now, W in the gaps.", 15, INK, 600)
p.footer("Source: lecture pipeline slides, original plate. Shell 4: schedule around the critical path.")
p.save("l08-bubble.svg")

# ---- tp cuts ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Tensor parallel cuts the matmuls", "Column cuts on the way up, row cuts on the way down. Small layers replicate.")
p.panel(60, 140, 840, 220, "one transformer block")
p.chip(90, 190, 200, 60, "x -> A", "column cut", fill=ACTIVE)
p.chip(330, 190, 140, 60, "GeLU", "replicated", fill=CHIP)
p.chip(510, 190, 200, 60, "B -> z", "row cut", fill=ACTIVE)
p.arrow(290, 220, 40, "")
p.arrow(470, 220, 40, "")
p.text(90, 300, "MLP up-projection and attention QKV: column-wise.", 14, INK)
p.text(90, 326, "MLP down-projection and attention out: row-wise.", 14, INK)
p.panel(60, 390, 840, 100, "the f/g duality")
p.text(84, 434, "forward: f = identity, g = all-reduce. Backward: flipped. Know this to write TP.", 15, INK, 600)
p.footer("Source: lecture tensor-parallel slides, original plate. Shell 4: cut wide, reduce narrow.")
p.save("l08-tp-cuts.svg")

# ---- activation memory ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Activations dwarf parameters", "34sbh + 5as/h. TP divides the big terms. Sequence parallel divides the rest.")
p.panel(60, 140, 840, 120, "no parallelism")
p.text(84, 190, "34 x s x b x h   +   5 x a x s / h  (quadratic attention + dropout)", 16, INK, 600)
p.panel(60, 290, 840, 120, "with tensor parallel t")
p.text(84, 340, "(24 sbh + 5 as/h) / t   +   10 sbh  (layernorms, residuals: not split)", 16, INK, 600)
p.panel(60, 440, 840, 60, "with sequence parallel + recompute: 34 sbh / t")
p.footer("Source: lecture activation-memory slides, original plate. Shell 2: count activations first.")
p.save("l08-activation-memory.svg")

# ---- expert parallel ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Expert parallel beats tensor for MoE", "Route sparse tokens with all-to-all instead of slicing dense matmuls thinner.")
p.panel(60, 140, 840, 200, "why EP over TP for MoE")
p.text(84, 190, "TP cuts matmuls finer: small tiles, GPU utilization suffers.", 15, INK)
p.text(84, 216, "EP routes tokens to whole experts: matmuls stay big.", 15, TEAL, 700)
p.text(84, 242, "all-to-all dispatch; load balance keeps it near a transpose.", 15, INK)
p.text(84, 280, "DeepSeek DeepEP, NVIDIA Hybrid EP: latency is everything.", 14, MUT)
p.panel(60, 370, 840, 110, "the wrinkle")
p.text(84, 414, "EP parallelizes MLPs, not attention. Decouple: high TP for attention, low TP for MoE layers.", 15, ORANGE, 700)
p.footer("Source: lecture expert-parallel slides, original plate. Shell 5: MoE changes the prescription.")
p.save("l08-ep-vs-tp.svg")

# ---- 4d prescription ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("The 4D prescription", "Fit the model first, then spend the rest on data parallel. Simple rules win.")
steps = [
    ("1. fit the model", "TP or EP of 8 inside the fast node", TEAL),
    ("2. fit across nodes", "pipeline parallel or ZeRO-3 on the slow links", BLUE),
    ("3. spend the rest", "data parallel: maximize it, minimize model parallel", ACTIVE),
    ("4. long context", "context parallel (ring attention)", NEWTOK),
]
y0 = 140
for i, (a, b, col) in enumerate(steps):
    yy = y0 + i * 84
    p.panel(60, yy, 840, 72, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="72" rx="5" fill="{col}"/>')
    p.text(84, yy + 46, b, 14, MUT)
p.text(60, 500, "Batch too small: gradient accumulation. Comm hiding: keep compute longer than comm.", 14, MUT)
p.footer("Source: lecture strategy + Megatron guidelines, original plate. Shell 5: rules of thumb.")
p.save("l08-4d-prescription.svg")

# ---- training runs ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("The prescription in the wild", "Every frontier run is the same recipe with different numbers.")
runs = [
    ("OLMo 7B", "FSDP only", "small dense: FSDP scales fine", TEAL),
    ("DeepSeek V3", "EP64 + PP", "MoE: expert parallel replaces tensor", ACTIVE),
    ("Llama3 405B", "TP8 PP16 DP128", "dense giant; 148 GPU failures", BLUE),
    ("Gemma 2", "FSDP + TP + seq", "TPU mesh: tensor parallel at scale", NEWTOK),
    ("Mixtral 8x22B", "EP8 PP4 TP4", "TP4 for the attention layers", ORANGE),
    ("Qwen 3", "EP32 PP8 TP2", "DeepSeek-style recipe", PINK),
]
y0 = 140
for i, (a, b, c, col) in enumerate(runs):
    yy = y0 + i * 68
    p.panel(60, yy, 840, 58, "")
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="58" rx="5" fill="{col}"/>')
    p.text(84, yy + 24, a, 15, INK, 700)
    p.text(84, yy + 46, b + "  -  " + c, 13, MUT)
p.footer("Source: lecture survey of published runs, original plate. Shell 5: same recipe, new numbers.")
p.save("l08-training-runs.svg")
