import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- CPU vs GPU ----
p = Plate(960, 500); p.defs_arrow()
y = p.title("CPU optimizes latency. GPU optimizes throughput.", "Few complex cores vs hundreds of simple cores.")
p.panel(60, 140, 400, 240, "CPU")
p.chip(110, 200, 90, 90, "core", "big control", fill=CHIP)
p.chip(220, 200, 90, 90, "core", "big control", fill=CHIP)
p.chip(330, 200, 90, 90, "core", "big control", fill=CHIP)
p.text(110, 340, "serial, branchy, low latency", 15, INK)
p.panel(500, 140, 400, 240, "GPU")
for i in range(8):
    p.chip(520 + (i % 4) * 95, 190 + (i // 4) * 100, 85, 85, "SM", "", fill=BLUE)
p.text(520, 340, "108 SMs, SIMT, high throughput", 15, INK)
p.text(60, 430, "Dennard scaling ended: clocks stopped rising. Progress now comes from", 15, INK)
p.text(60, 456, "parallelism. V100 (2017) added tensor cores: matmul became the privileged op.", 15, INK, 700)
p.footer("Source: lecture hardware slides, original plate. Shell 2: count the cores.")
p.save("l05-cpu-vs-gpu.svg")

# ---- memory hierarchy ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("The memory hierarchy is the whole game", "Registers are free. Global memory is 10x L1. Respect the ladder.")
levels = [
    ("registers", "~1 cycle", "per thread", NEWTOK, 300),
    ("L1 / shared", "20-30 cycles", "per SM, programmable", ACTIVE, 260),
    ("L2 cache", "~200 cycles", "on chip", CHIP, 220),
    ("HBM global", "~10x L1", "off chip, 80-144 GB", PINK, 180),
]
x0, y0 = 70, 150
for i, (name, lat, note, col, wdt) in enumerate(levels):
    yy = y0 + i * 88
    p.parts.append(f'<rect x="{x0}" y="{yy}" width="{wdt}" height="64" rx="8" fill="{col}" stroke="{INK}" stroke-width="1.5"/>')
    p.text(x0 + wdt + 20, yy + 28, name, 17, INK, 700)
    p.text(x0 + wdt + 20, yy + 52, lat + "  -  " + note, 14, MUT)
p.text(70, 500, "SRAM is hundreds of times more expensive than DRAM. Hence the hierarchy.", 14, MUT)
p.footer("Source: lecture memory slides (A100 latencies), original plate. Shell 2: name the latencies.")
p.save("l05-mem-hierarchy.svg")

# ---- SIMT divergence ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("A warp executes both branches", "32 threads, one instruction. if/else runs twice with masking.")
p.panel(60, 140, 840, 200, "warp of 32 threads hits:  if (x > 0)")
p.chip(110, 210, 200, 70, "16 threads", "take IF", fill=NEWTOK)
p.chip(350, 210, 200, 70, "16 threads", "take ELSE", fill=PINK)
p.text(600, 225, "both run,", 15, INK)
p.text(600, 249, "losers idle", 15, ORANGE, 700)
p.panel(60, 370, 840, 90, "fix: multiply by a mask, not a branch")
p.text(84, 424, "ReLU as x * (x > 0)  instead of  if (x > 0) x else 0", 16, INK, 600)
p.footer("Source: lecture SIMT slides, original plate. Shell 3: the divergence rule.")
p.save("l05-simt-divergence.svg")

# ---- fusion ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("Fusion: one factory, not five", "Read once, compute everything in the SM, write once.")
p.panel(60, 140, 380, 220, "BEFORE: 5 kernels")
ops = ["sin", "cos", "square", "square", "add"]
for i, op in enumerate(ops):
    p.chip(90 + i * 68, 210, 62, 60, op, "", fill=CHIP)
    if i < 4:
        p.arrow(152 + i * 68, 158 + i * 68, 240, "")
p.text(90, 320, "read + write global", 14, ORANGE, 700)
p.text(90, 342, "memory 5 times", 14, ORANGE, 700)
p.panel(520, 140, 380, 220, "AFTER: 1 kernel")
p.chip(560, 200, 300, 80, "sin^2 + cos^2", "fused", fill=NEWTOK)
p.text(560, 320, "read once, write once", 14, TEAL, 700)
p.arrow(440, 520, 250, "fuse")
p.text(60, 420, "Easy fusions: torch.compile / JAX do them automatically.", 15, INK)
p.text(60, 446, "Hard fusions (FlashAttention): write the kernel by hand.", 15, INK, 700)
p.footer("Source: lecture fusion slides, original plate. Shell 4: the fused kernel.")
p.save("l05-fusion.svg")

# ---- tiling ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Tiling: load once, reuse T times", "Cut matrices into tiles. Compute in shared memory.")
p.panel(60, 140, 400, 280, "naive N x N matmul")
p.text(84, 200, "each element read", 16, INK)
p.text(84, 230, "N times from HBM", 16, ORANGE, 700)
p.text(84, 280, "global reads: O(N^3)", 15, MUT)
p.panel(500, 140, 400, 280, "tiled, tile size T")
p.parts.append(f'<rect x="540" y="190" width="120" height="120" rx="6" fill="{ACTIVE}" stroke="{INK}" stroke-width="1.5"/>')
p.parts.append(f'<rect x="680" y="190" width="120" height="120" rx="6" fill="{PANEL}" stroke="{LINE}" stroke-width="1.5"/>')
p.parts.append(f'<rect x="540" y="320" width="120" height="60" rx="6" fill="{PANEL}" stroke="{LINE}" stroke-width="1.5"/>')
p.text(540, 410, "each element: N/T HBM reads", 15, TEAL, 700)
p.arrow(460, 500, 280, "tile")
p.text(60, 470, "T = N: read once from HBM, reuse N times in SRAM. T-limited by shared memory size.", 15, INK)
p.text(60, 496, "Tile sizes are tuned: torch.compile max-autotune benchmarks them for ~15 minutes.", 15, MUT)
p.footer("Source: lecture tiling slides, original plate. Shell 3: the reuse rule.")
p.save("l05-tiling.svg")

# ---- coalescing ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Coalesce: one burst per warp", "DRAM serves 128-byte bursts. Threads in a burst read for free.")
p.panel(60, 140, 400, 200, "NOT coalesced")
for i in range(4):
    p.parts.append(f'<rect x="{100+i*90}" y="200" width="60" height="60" rx="6" fill="{PINK}" stroke="{INK}" stroke-width="1.5"/>')
    p.text(100 + i * 90 + 30, 235, "t" + str(i), 14, INK, 700, anchor="middle")
p.text(100, 300, "4 threads, 4 bursts", 14, ORANGE, 700)
p.text(100, 322, "column reads in row-major", 14, MUT)
p.panel(500, 140, 400, 200, "coalesced")
p.parts.append(f'<rect x="540" y="200" width="320" height="60" rx="6" fill="{NEWTOK}" stroke="{INK}" stroke-width="1.5"/>')
for i in range(4):
    p.text(580 + i * 80, 235, "t" + str(i), 14, INK, 700, anchor="middle")
p.text(540, 300, "4 threads, 1 burst", 14, TEAL, 700)
p.text(540, 322, "row reads in row-major", 14, MUT)
p.arrow(460, 500, 240, "reorder")
p.text(60, 400, "Mnemonic: threads moving along the major axis are NOT coalesced.", 15, INK, 700)
p.text(60, 426, "Karpathy: padding vocab 50257 -> 50304 gave 25% speedup via alignment.", 15, INK)
p.text(60, 472, "Sizes divisible by 16/32 fill whole burst windows. Powers of 2 are not magic.", 15, MUT)
p.footer("Source: lecture coalescing slides, original plate. Shell 3: the burst rule.")
p.save("l05-coalesce.svg")

# ---- wave quantization ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Wave quantization: 98 tiles fly, 120 crawl", "A100 has 108 SMs. One extra tile row costs a whole wave.")
p.panel(60, 140, 400, 220, "1792: 98 tiles")
for r in range(4):
    for c in range(6):
        p.parts.append(f'<rect x="{100+c*52}" y="{180+r*40}" width="46" height="34" rx="4" fill="{TEAL}" opacity="0.8"/>')
p.text(100, 380, "98 < 108: one wave", 14, TEAL, 700)
p.panel(500, 140, 400, 220, "1793: 120 tiles")
for r in range(4):
    for c in range(6):
        p.parts.append(f'<rect x="{540+c*52}" y="{180+r*40}" width="46" height="34" rx="4" fill="{TEAL}" opacity="0.8"/>')
for c in range(6):
    p.parts.append(f'<rect x="{540+c*52}" y="{340}" width="46" height="14" rx="4" fill="{ORANGE}"/>')
p.text(540, 380, "120 > 108: two waves, 12 stragglers", 14, ORANGE, 700)
p.text(60, 430, "Adding ONE dimension (1792 -> 1793) with 256x128 tiles: 98 -> 120 tiles.", 15, INK)
p.text(60, 456, "The second wave runs 12 tiles while 96 SMs sit idle. Throughput craters.", 15, INK, 700)
p.footer("Source: lecture mystery-plot slides, original plate. Shell 4: count tiles vs SMs.")
p.save("l05-wave-quant.svg")

# ---- flash attention ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("FlashAttention = tiling + online softmax + recompute", "Never materialize the n x n matrix. Three tricks, one kernel.")
p.panel(60, 140, 840, 130, "1. TILE the QK^T and PV matmuls")
p.text(84, 190, "dashed blocks live in SRAM. K, Q, V stream from HBM in tiles.", 15, INK)
p.panel(60, 300, 840, 130, "2. ONLINE softmax per tile")
p.text(84, 350, "track running max m and sum l. New max? Rescale and continue.", 15, INK, 600)
p.text(84, 380, "m_new = max(m, rowmax).  l_new = l*e^(m-m_new) + rowsum*e^(rowmax-m_new).", 14, MUT)
p.panel(60, 460, 840, 80, "3. FUSE + RECOMPUTE backward")
p.text(84, 510, "one kernel end to end. Backward recomputes tiles: no n^2 activation stored.", 15, INK, 700)
p.footer("Source: FlashAttention paper, lecture slides. Shell 5: the fused kernel returns in Triton.")
p.save("l05-flashattn.svg")
