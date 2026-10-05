import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- lifetime ----
p = Plate(960, 620); p.defs_arrow()
y = p.title("The lifetime of a token", "Request to intelligence, one scheduling loop.")
stages = [
    ("request", "tokenize the prompt", MUT),
    ("schedule", "assign to GPUs (prefill vs decode pools)", MUT),
    ("KV cache", "have we seen these tokens before?", TEAL),
    ("execute", "prefill once, decode one token at a time", ACTIVE),
    ("sample", "stop tokens, safety checks, return", MUT),
]
y0 = 140
for i, (a, b, col) in enumerate(stages):
    yy = y0 + i * 78
    p.panel(60, yy, 840, 62, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="62" rx="5" fill="{col}"/>')
    p.text(84, yy + 40, b, 13.5, MUT)
p.text(60, 580, "The engine repeats: schedule, execute, sample. Inference turns electricity into intelligence.", 14, INK, 700)
p.footer("Source: guest lecture overview slides, original plate. Shell 1: the lifetime.")
p.save("l18-lifetime.svg")

# ---- workloads ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Workloads define the system", "Coding, chat, and agents look nothing like training traffic.")
rows = [
    ("coding agent", "tens of thousands of input tokens, short outputs", TEAL),
    ("chat", "short in, short out, wants first token in under a second", MUT),
    ("agentic loop", "many turns, tool calls, idle gaps between turns", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 56, b, 14, MUT)
p.text(60, 480, "SLAs differ: time to first token for chat, time to completion for batch.", 14, MUT)
p.footer("Source: guest lecture workload slides, original plate. Shell 1: workloads.")
p.save("l18-workloads.svg")

# ---- prefill vs decode ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Prefill vs decode", "Two operations, two bottlenecks, two machine fleets.")
p.panel(60, 140, 840, 140, "prefill: 10,000 tokens in, 1 token out")
p.text(84, 188, "compute bound: like training without the backward pass", 15, INK, 600)
p.text(84, 214, "uses the GPU well. Runs once per request.", 14, MUT)
p.panel(60, 306, 840, 140, "decode: 1 token at a time")
p.text(84, 354, "memory bandwidth bound: load the whole model for one token", 15, ORANGE, 700)
p.text(84, 380, "runs once per generated token. Prefill takes longer; decode runs far more steps.", 14, MUT)
p.text(60, 486, "Split them onto different machines: NVIDIA GPUs for prefill, LPU chips for decode.", 14, TEAL, 700)
p.footer("Source: guest lecture prefill/decode slides, original plate. Shell 2: the split.")
p.save("l18-prefill-decode.svg")

# ---- continuous batching ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Continuous batching", "Time flows down. Requests join and leave mid-step.")
p.text(60, 150, "step 1: long request generating", 14, INK)
p.text(60, 200, "step 2: short request joins, shares compute and KV cache", 14, MUT)
p.text(60, 250, "step 3: short request finishes, new one arrives", 14, MUT)
p.text(60, 300, "step 4: GPU memory full: KV cache cannot grow, queue forms", 14, ORANGE, 700)
p.text(60, 350, "step 5: long request ends, queued request starts", 14, MUT)
p.text(60, 420, "Many requests live in one system. Batching is per-step, not per-request.", 15, INK, 700)
p.footer("Source: guest lecture batching slides, original plate. Shell 2: batching.")
p.save("l18-continuous-batching.svg")

# ---- KV cache ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("The KV cache", "Do not recompute what you already computed.")
p.panel(60, 140, 840, 130, "the idea")
p.text(84, 188, "everyone says hi ChatGPT: share one prefix across all users", 15, INK, 600)
p.text(84, 214, "radix tree lookup: seen tokens hit the cache, new tokens compute", 14, MUT)
p.panel(60, 296, 840, 140, "the hierarchy")
p.text(84, 344, "GPU memory -> CPU DRAM -> SSD: each tier cheaper, each slower", 15, INK, 600)
p.text(84, 370, "LRU eviction. Prefetch when a user reopens an old conversation.", 14, MUT)
p.footer("Source: guest lecture KV cache slides, original plate. Shell 3: caching.")
p.save("l18-kv-cache.svg")

# ---- disaggregation ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Cache-aware disaggregation", "Route by cache hit rate. Two lines of code, 40% faster.")
p.panel(60, 140, 840, 130, "the routing")
p.text(84, 188, "fresh request (book paste, low cache hits) -> one GPU pool", 15, INK, 600)
p.text(84, 214, "warm conversation (mid-chat) -> another pool. Never mix them.", 14, MUT)
p.panel(60, 296, 840, 130, "the logic")
p.text(84, 344, "10-turn conversations mean ~10% of requests are fresh and expensive", 15, INK, 600)
p.text(84, 370, "up to 40% faster serving from this routing alone", 14, TEAL, 700)
p.footer("Source: guest lecture disaggregation slides, original plate. Shell 3: routing.")
p.save("l18-disaggregation.svg")

# ---- megakernel ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Megakernels", "One kernel per op leaves the GPU idle. Fuse the whole layer.")
p.panel(60, 140, 840, 140, "the problem")
p.text(84, 188, "132 SMs on H100. Kernel launches, tail effects, gaps between ops.", 15, INK, 600)
p.text(84, 214, "decode loads weights for one token: the GPU is a glorified memory loader", 14, MUT)
p.panel(60, 306, 840, 140, "the megakernel")
p.text(84, 354, "one kernel, many ops: overlap KV load with QKV, weight load with attention", 15, INK, 600)
p.text(84, 380, "30 to 70% attention speedup. 72% bandwidth: near the speed of light.", 14, TEAL, 700)
p.footer("Source: guest lecture Megakernel slides, original plate. Shell 4: kernels.")
p.save("l18-megakernel.svg")

# ---- Parcae ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Parcae: loops that do not blow up", "Loop transformers scaled by constraining the spectral radius.")
p.panel(60, 140, 840, 140, "the instability")
p.text(84, 188, "naive looping blows up: powering the A matrix amplifies activations", 15, INK, 600)
p.text(84, 214, "norms fight expansion: loss spikes, NaNs, 9 of 10 learning rates fail", 14, MUT)
p.panel(60, 306, 840, 140, "the fix")
p.text(84, 354, "A becomes negative diagonal: spectral radius under 1, cannot explode", 15, INK, 600)
p.text(84, 380, "scaling laws: as data grows, scale recurrence alongside parameters", 14, TEAL, 700)
p.footer("Source: guest lecture Parcae slides, original plate. Shell 5: architectures.")
p.save("l18-parcae.svg")
