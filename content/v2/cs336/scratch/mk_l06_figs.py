import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- grid/blocks/threads ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Think in blocks, not threads", "Grid of blocks. Block on one SM. Threads share its memory.")
p.panel(60, 140, 840, 200, "GRID: launch a grid of thread blocks")
for b in range(4):
    p.chip(110 + b * 200, 200, 170, 100, f"block {b}", "one SM", fill=BLUE)
p.panel(60, 370, 840, 130, "inside one block")
p.chip(110, 410, 130, 60, "threads", "32 = warp", fill=CHIP)
p.chip(280, 410, 200, 60, "shared memory", "block-local", fill=ACTIVE)
p.chip(520, 410, 130, 60, "registers", "per thread", fill=NEWTOK)
p.text(680, 445, "HBM: global to all", 14, MUT)
p.text(60, 530, "Why blocks: softmax and matmul need threads to talk. Blocks share memory. Threads alone cannot.", 15, INK, 700)
p.footer("Source: lecture programming-model slides, original plate. Shell 4: the block symbol.")
p.save("l06-grid-blocks.svg")

# ---- occupancy ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Registers cap your occupancy", "128 threads x 160 regs = 20K regs/block. 3 blocks/SM. 12 of 64 warps = 18%.")
p.panel(60, 140, 840, 180, "the math")
p.text(84, 190, "registers per block = 128 threads x 160 regs = 20,480", 17, INK, 600)
p.text(84, 222, "B200: 65,536 regs/SM  ->  3 blocks fit  (3 x 20,480)", 17, INK, 600)
p.text(84, 254, "3 blocks x 4 warps = 12 warps. Max 64. Occupancy = 12/64 = 18%.", 17, ORANGE, 700)
p.panel(60, 350, 410, 130, "high occupancy?")
p.text(84, 396, "more warps hide latency", 15, INK)
p.text(84, 422, "not always better", 15, MUT)
p.panel(490, 350, 410, 130, "thread coarsening")
p.text(514, 396, "fewer, fatter threads", 15, INK)
p.text(514, 422, "8 elements per thread", 15, TEAL, 700)
p.footer("Source: lecture occupancy slides, original plate. Shell 2: count registers.")
p.save("l06-occupancy.svg")

# ---- bank conflicts ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("32 banks, one thread each", "32 threads hitting one column serialize: a 32-way bank conflict.")
p.panel(60, 140, 840, 200, "shared memory: 32 banks x 4 bytes")
for i in range(8):
    col = PINK if i == 0 else CHIP
    p.parts.append(f'<rect x="{100+i*95}" y="200" width="80" height="80" rx="6" fill="{col}" stroke="{INK}" stroke-width="1.5"/>')
    p.text(100 + i * 95 + 40, 245, f"b{i}", 14, INK, 700, anchor="middle")
p.text(100, 320, "32 threads read column 0 -> all hit bank 0 -> 32x serialized", 15, ORANGE, 700)
p.panel(60, 370, 840, 110, "fix")
p.text(84, 414, "rows are fine (spread across banks). Matmul needs rows AND columns:", 15, INK)
p.text(84, 440, "swizzling rearranges shared memory to dodge the conflict.", 15, TEAL, 700)
p.footer("Source: lecture bank-conflict slides, original plate. Shell 3: the bank rule.")
p.save("l06-bank-conflict.svg")

# ---- benchmark rules ----
p = Plate(960, 500); p.defs_arrow()
y = p.title("Benchmark like you mean it", "Warm up. CUDA events. Synchronize. Repeat. Then profile.")
steps = [
    ("1. warm up", "lazy compilation must not pollute timing", TEAL),
    ("2. CUDA events", "record start/end on the device, not the wall", TEAL),
    ("3. synchronize", "GPU is async: sync before reading the clock", ORANGE),
    ("4. repeat + average", "variance is real; P95 if you are picky", TEAL),
    ("5. profile", "profiler names the kernels: time tells you where", FOCUS),
]
x0, y0 = 70, 150
for i, (a, b, col) in enumerate(steps):
    yy = y0 + i * 62
    p.parts.append(f'<rect x="{x0}" y="{yy}" width="10" height="52" rx="5" fill="{col}"/>')
    p.text(x0 + 26, yy + 22, a, 16, INK, 700)
    p.text(x0 + 26, yy + 44, b, 14, MUT)
p.footer("Source: lecture benchmarking slides, original plate. Shell 1: the measurement loop.")
p.save("l06-bench-rules.svg")

# ---- gelu race ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("Three GeLUs, one winner", "Naive: many kernels, HBM round trips. Fused: one kernel, one trip each way.")
p.panel(60, 140, 260, 220, "naive PyTorch")
for i, op in enumerate(["tanh", "mul", "add", "mul"]):
    p.chip(90, 180 + i * 44, 200, 36, op, "", fill=CHIP)
p.text(90, 380, "3.75 (slowest)", 15, ORANGE, 700)
p.panel(350, 140, 260, 220, "torch.compile")
p.chip(380, 240, 200, 80, "1 Triton kernel", "auto-fused", fill=ACTIVE)
p.text(380, 380, "fast", 15, TEAL, 700)
p.panel(640, 140, 260, 220, "built-in")
p.chip(670, 240, 200, 80, "1 CUDA kernel", "hand-written", fill=NEWTOK)
p.text(670, 380, "fastest here", 15, TEAL, 700)
p.text(60, 430, "Between kernels, data returns to HBM. Fusion deletes the round trips.", 15, INK, 700)
p.text(60, 456, "Profiler proof: naive shows many kernels; compiled shows one Triton kernel.", 15, MUT)
p.footer("Source: lecture GeLU slides, original plate. Shell 4: fusion measured.")
p.save("l06-gelu-race.svg")

# ---- triton kernel shape ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Every Triton kernel has the same shape", "Who am I. What do I read. Compute. Write back.")
steps = [
    ("pid = tl.program_id(0)", "who am I: which block", BLUE),
    ("offsets = pid*B + arange(B)", "what I read: my slice (+mask the tail)", BLUE),
    ("x = tl.load(X + offsets)", "read: HBM -> registers/shared", ACTIVE),
    ("y = gelu(x)", "compute: looks like PyTorch", NEWTOK),
    ("tl.store(Y + offsets, y)", "write: back to HBM", ORANGE),
]
x0, y0 = 70, 150
for i, (a, b, col) in enumerate(steps):
    yy = y0 + i * 72
    p.chip(x0, yy, 830, 62, "", fill=PANEL)
    p.parts.append(f'<rect x="{x0}" y="{yy}" width="10" height="62" rx="5" fill="{col}"/>')
    p.text(x0 + 26, yy + 26, a, 15, INK, 700)
    p.text(x0 + 26, yy + 48, b, 13, MUT)
p.footer("Source: lecture Triton slides (lecture_06.py), original plate. Shell 4: the kernel shape.")
p.save("l06-triton-gelu.svg")

# ---- softmax block ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Softmax: one row per block", "Row fits: direct. Row too long: loop over tiles with an accumulator.")
p.panel(60, 140, 400, 220, "row fits in block")
p.text(84, 190, "blocksize = next pow2(cols)", 15, INK, 600)
p.text(84, 216, "max, sub, exp, sum, div", 15, INK)
p.text(84, 242, "masked -inf past the tail", 15, MUT)
p.text(84, 280, "looks like PyTorch", 15, TEAL, 700)
p.panel(500, 140, 400, 220, "row too long: tiles")
p.text(524, 190, "tile 0: cols 0-3", 15, INK, 600)
p.text(524, 216, "tile 1: cols 4-7 ...", 15, INK, 600)
p.text(524, 242, "accumulator += tile", 15, INK, 600)
p.text(524, 280, "for loop inside the block", 15, ORANGE, 700)
p.arrow(460, 500, 250, "loop")
p.text(60, 420, "Naive PyTorch softmax: 5MN reads + 3MN writes. Triton: 1 read + 1 write per row.", 15, INK, 700)
p.text(60, 446, "Blocks, not threads: the row is the unit. Tiles handle the overflow.", 15, MUT)
p.footer("Source: lecture softmax slides, original plate. Shell 4: the reduction pattern.")
p.save("l06-softmax-block.svg")

# ---- matmul tiling ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Matmul: one C tile per block", "Sweep A row-tiles and B column-tiles. Accumulate in shared memory.")
p.panel(60, 140, 840, 260, "block (m, n) computes C tile")
p.chip(110, 200, 120, 80, "A tile", "row m", fill=BLUE)
p.chip(270, 200, 120, 80, "B tile", "col n", fill=BLUE)
p.arrow(390, 470, 240, "tl.dot")
p.chip(480, 200, 150, 80, "acc", "shared mem", fill=ACTIVE)
p.text(660, 225, "+= A_tile @ B_tile", 15, INK, 600)
p.text(660, 251, "sweep k tiles", 15, MUT)
p.text(110, 330, "naive per-element: M*K*N HBM reads, intensity O(1)", 14, ORANGE, 700)
p.text(110, 356, "tiled: intensity O(tile size). Ideal needs all of A,B in shared: too big.", 14, INK)
p.panel(60, 430, 840, 80, "free fusion")
p.text(84, 480, "apply ReLU to acc before the write. The epilogue costs nothing extra.", 15, TEAL, 700)
p.footer("Source: lecture matmul slides, original plate. Shell 5: tiling returns in FlashAttention.")
p.save("l06-matmul-tiling.svg")
