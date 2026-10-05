import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- memory hierarchy ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Same game, bigger hierarchy", "What was slow last week is fast this week. Orchestrate to avoid data bottlenecks.")
tiers = [
    ("shared / L1", "fastest", TEAL, "registers, SM-local"),
    ("HBM", "8 TB/s", BLUE, "was slow, now fast"),
    ("NVLink / NVSwitch", "1.8 TB/s", ACTIVE, "~4x slower than HBM"),
    ("InfiniBand", "much slower", ORANGE, "GPU -> PCIe -> IB"),
    ("Ethernet", "slowest", PINK, "via CPU socket buffer"),
]
y0 = 150
for i, (a, b, col, note) in enumerate(tiers):
    yy = y0 + i * 70
    p.chip(70, yy, 830, 60, "", fill=PANEL)
    p.parts.append(f'<rect x="70" y="{yy}" width="10" height="60" rx="5" fill="{col}"/>')
    p.text(96, yy + 24, a, 16, INK, 700)
    p.text(96, yy + 46, note, 13, MUT)
    p.text(860, yy + 38, b, 15, col, 700, anchor="end")
p.footer("Source: lecture hierarchy slide, original plate. Shell 2: distance = cost.")
p.save("l07-memory-hierarchy.svg")

# ---- collectives ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Three collectives run training", "All-gather, reduce-scatter, all-reduce. Rank = device. World size = count.")
defs = [
    ("all-gather", "each rank's piece -> everyone holds everything", TEAL, "forward: assemble full params"),
    ("reduce-scatter", "reduce per shard, scatter results", ACTIVE, "backward: sum grads, split storage"),
    ("all-reduce", "reduce + replicate to all", BLUE, "= reduce-scatter + all-gather"),
]
y0 = 140
for i, (a, b, col, note) in enumerate(defs):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 96, a)
    p.text(84, yy + 44, b, 15, INK, 600)
    p.text(84, yy + 70, note, 13, MUT)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="96" rx="5" fill="{col}"/>')
p.text(60, 540, "Broadcast/scatter/gather/reduce are warmups. All-to-all routes tokens to experts in MoE.", 15, MUT)
p.footer("Source: lecture collectives slides, original plate. Shell 3: the three you must know.")
p.save("l07-collectives.svg")

# ---- topology ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Topology: 8 GPUs per node", "NVLink to NVSwitch inside a node. InfiniBand between pods. NVL72 for the rich.")
p.panel(60, 140, 540, 200, "one node: 8 GPUs")
for r in range(2):
    for c in range(4):
        p.parts.append(f'<rect x="{90+c*120}" y="{200+r*70}" width="100" height="50" rx="8" fill="{BLUE}" stroke="{INK}" stroke-width="1.5"/>')
        p.text(90 + c * 120 + 50, 232 + r * 70, f"GPU{c*2+r+1 if r==0 else c*2+r+4}", 13, INK, 700, anchor="middle")
p.text(90, 330, "NVSwitch: any GPU -> any GPU", 13, ACTIVE, 700)
p.panel(640, 140, 260, 200, "between nodes")
p.text(664, 190, "InfiniBand", 16, INK, 700)
p.text(664, 216, "slower than NVLink", 14, ORANGE, 700)
p.text(664, 242, "then Ethernet", 14, MUT)
p.panel(60, 370, 840, 100, "NVL72: 9 trays of 8 = 72 GPUs in one NVLink domain")
p.text(84, 424, "fast interconnect up to 72 GPUs if you can pay", 14, MUT)
p.footer("Source: lecture topology slides, original plate. Shell 3: fast inside, slow across.")
p.save("l07-topology.svg")

# ---- rdma ----
p = Plate(960, 500); p.defs_arrow()
y = p.title("RDMA: cut out the CPU", "Ethernet copies via the CPU kernel. RDMA lets GPUs write each other's memory directly.")
p.panel(60, 140, 380, 220, "standard Ethernet")
p.text(84, 190, "GPU -> CPU socket buffer", 15, INK)
p.arrow(200, 260, 220, "copy")
p.text(84, 290, "CPU builds packets -> NIC", 15, INK)
p.text(84, 330, "high latency", 15, ORANGE, 700)
p.panel(480, 140, 420, 220, "RDMA: NVLink / InfiniBand / RoCE")
p.text(504, 190, "GPU writes GPU memory", 15, INK, 700)
p.text(504, 216, "no CPU in the path", 15, TEAL, 700)
p.text(504, 242, "RoCE = RDMA over Ethernet", 14, MUT)
p.footer("Source: lecture networking slides, original plate. Shell 3: bypass the middleman.")
p.save("l07-rdma.svg")

# ---- data parallel ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Data parallel: split the rows", "Each GPU sees 1/4 of the batch. One all-reduce of gradients. Parameters stay identical.")
p.panel(60, 140, 840, 180, "batch 128 x 1024, world size 4")
for r in range(4):
    p.chip(100 + r * 200, 190, 170, 70, f"GPU {r}", "rows 32*(r)", fill=BLUE)
p.panel(60, 350, 840, 140, "the one line that matters")
p.text(84, 400, "all_reduce(param.grad)  ->  divide by world size", 17, INK, 600)
p.text(84, 430, "forward + backward stay local. Only gradients sync. Losses differ, params do not.", 14, MUT)
p.footer("Source: lecture data-parallel code (lecture_07.py), original plate. Shell 4: sync only gradients.")
p.save("l07-data-parallel.svg")

# ---- tensor parallel ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Tensor parallel: split the columns", "Each rank holds 1/4 of each layer. All-gather activations forward, reduce-scatter backward.")
p.panel(60, 140, 840, 180, "layer weight: dim x dim")
p.text(84, 190, "column split: rank r holds dim x (dim/4)", 16, INK, 600)
for r in range(4):
    p.chip(100 + r * 200, 230, 170, 60, f"rank {r}", "cols r/4", fill=ACTIVE)
p.panel(60, 350, 840, 140, "forward / backward")
p.text(84, 400, "forward: each rank computes local activations, all-gather to full", 15, INK, 600)
p.text(84, 430, "backward: reduce-scatter the gradients (dual of all-gather)", 15, INK, 600)
p.footer("Source: lecture tensor-parallel code, original plate. Shell 4: communication per layer.")
p.save("l07-tensor-parallel.svg")

# ---- pipeline parallel ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Pipeline parallel: split the layers", "Each rank owns some layers. Micro-batches shrink the idle bubbles.")
p.panel(60, 140, 840, 140, "4 ranks, 4 layer groups")
for r in range(4):
    p.chip(100 + r * 200, 170, 170, 70, f"rank {r}", f"layers {r}", fill=NEWTOK)
p.arrow(270, 205, 30, "")
p.arrow(470, 205, 30, "")
p.arrow(670, 205, 30, "")
p.panel(60, 310, 840, 180, "micro-batches")
p.text(84, 360, "batch -> micro-batches: send activations forward, grads backward (send/recv)", 15, INK, 600)
p.text(84, 386, "bubble: rank idle while waiting on neighbors. More micro-batches = smaller bubbles.", 15, ORANGE, 700)
p.text(84, 412, "overlap comm + compute: receive while computing", 14, TEAL, 700)
p.footer("Source: lecture pipeline slides, original plate. Shell 4: bubbles are the tax.")
p.save("l07-pipeline-parallel.svg")

# ---- which where ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("Which parallelism where", "Match the communication pattern to the interconnect. Hardware decides.")
rows = [
    ("tensor parallel", "every layer, big activations", "NVLink domain only", ACTIVE),
    ("data parallel", "one all-reduce per step", "scales far, needs batch > world size", BLUE),
    ("pipeline parallel", "point-to-point, small tensors", "tolerates slow links, even cross-world", TEAL),
]
y0 = 150
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 36, b, 14, INK)
    p.text(84, yy + 62, c, 14, MUT)
p.text(60, 470, "Critical batch size: past it, data parallel wastes compute. Switch to tensor.", 15, MUT)
p.footer("Source: lecture strategy discussion, original plate. Shell 5: hardware picks the strategy.")
p.save("l07-where-which.svg")
