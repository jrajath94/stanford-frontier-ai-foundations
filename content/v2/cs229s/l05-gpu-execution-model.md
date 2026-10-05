---
page_id: cs229s-l05
course_slug: cs229s
course_name: "CS229S: Systems for Machine Learning"
course_order: 3
order: 5
nav: "L05 · GPU Execution Model"
title: "Lecture 5: CUDA and GPU Programming for AI"
summary: "How attention actually executes on GPUs: kernels, warps, tiling, tensor cores. Defines the device block reused by MS&E435."
date: "2024-10-07"
instructor: "Azalia Mirhoseini"
offering: "Fall 2024"
concepts: [device-block, cuda, kernel, warp, simt, tiling, tensor-cores, coalesced-access, nvlink, infiniband]
sources:
  - tag: slides
    label: "Hardware Aware Algorithm Design slide deck, GPU execution model sections (Fall 2023 headers)"
  - tag: supplement
    label: "NVIDIA CUDA C++ Programming Guide"
    url: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - tag: supplement
    label: "Dan Fu, Chris Re, NeurIPS MLSys Keynote 2023 (tensor cores)"
---

## How to read this lesson

**Level 1 (Core)** defines the **device block**: one GPU, one node,
one cluster, and the links between them. MS&E435 reuses this symbol;
it is defined here and nowhere else. **Level 2 (Deep)** covers the
execution model: kernels, warps, coalesced access, tiling, and
tensor cores. Kernel code lives in [CS336 L06](../cs336/l06-triton-kernels.html);
interconnect depth in [CS336 L07](../cs336/l07-parallelism.html).

## Level 1: The device block

![Device block](assets/plate-device-block.svg "Shell 4. One GPU, one node of 8 GPUs on NVLink, many nodes on InfiniBand. Source: original plate; defined here for MS&E435.")

The device block is the machine as a memory hierarchy with three
shells.

**One GPU (device).** Streaming multiprocessors do the math: 108
SMs on an A100. On-chip memory sits next to them: registers
(256 KB per SM, private to threads), shared memory (192 KB per
SM, visible to a thread block), L2 cache (40 MB). HBM device
memory (40 GB at about 2 TB/s) holds the model. A **kernel** is one
GPU operation: threads load inputs from HBM to SRAM and registers,
compute, and write results back to HBM.

**One node.** Eight GPUs connected by NVLink, a very high bandwidth
link inside the server. Communication within the node is fast.

**One cluster.** Many nodes connected by InfiniBand, which is much
slower than NVLink. Communication across nodes is the expensive
move.

The placement rule follows from Lecture 1: intra-node links are
orders of magnitude faster than inter-node networking. So
communication-heavy patterns (tensor parallelism) live inside the
node, and rarer, larger patterns (data parallelism) span the
cluster. [L10](l10-parallelism.html) derives the mapping.

> [!QA]
> Q: What is the device block?
> A: The three-shell symbol for the machine: one GPU (108 SMs, registers, shared memory, L2, 40 GB HBM), one node (8 GPUs on NVLink), and one cluster (nodes on InfiniBand). A kernel is one GPU operation that loads from HBM, computes on-chip, and writes back. The symbol is defined in this lecture and reused by MS&E435.
> Follow-up: Why does the hierarchy decide the parallelism strategy?
> A: Because communication cost follows the hierarchy. Tensor parallelism all-reduces after every attention and MLP block, so it needs NVLink speed and stays inside one node. Data parallelism synchronizes once per step, so it can afford InfiniBand across nodes.

## Level 1: How a kernel runs

A kernel launches a **grid** of **blocks**. The hardware distributes
blocks to SMs with free capacity. Multiple blocks can run
concurrently on one SM. Inside a block, threads also run
concurrently on that SM.

Inside the SM, threads execute in groups of 32 called **warps**
under SIMT: single instruction, multiple threads. One instruction
unit drives many processors in lockstep on different data.

Maximum active threads = (number of SMs) x (max blocks per SM) x
(max threads per block). Expose at least this much work or SMs sit
idle: this is the parallelization principle from Lecture 3, stated
in hardware terms.

```ascii
grid    : [block 0][block 1][block 2]...
             |         |         |
SMs     :  SM A      SM B      SM C      <- blocks land where free
threads :  warp = 32 threads, one instruction, lockstep
```

## Level 1: Three memories a thread sees

**Global memory.** Visible to all threads. Slow: it is HBM.

**Shared memory.** Visible to threads of the same block. One thread
can load data that many threads need, avoiding duplicate reads.

**Registers.** Private to each thread. Fastest.

| Memory | Visible to | Speed | Use for |
|---|---|---|---|
| Global (HBM) | All threads | Slow | Model weights, large tensors |
| Shared | One block | Fast | Tiles shared by block threads |
| Registers | One thread | Fastest | Per-thread accumulators |

Efficient loading has two rules. **Coalesced access:** threads in
the same warp must touch adjacent memory elements, or bandwidth
collapses. **Tiling:** partition data into subsets that fit in
shared memory; compute on tiles independently.

The tiling example from the slides: four threads multiply
matrices. Their accessed elements overlap. If the threads
collaborate through shared memory instead of loading
independently, half the memory accesses disappear. Tiling is the
same idea as Lecture 3's 2D matmul tiling, now stated as a CUDA
programming pattern.

> [!QA]
> Q: What is tiling and why does it cut memory traffic?
> A: Tiling partitions data into subsets that fit in shared memory, then computes on each tile independently. Threads in a block collaborate: one thread's load serves many threads' compute, instead of every thread loading its own copy. The slides' matmul example saves half the memory accesses this way.
> Follow-up: What breaks if threads in a warp access scattered addresses?
> A: Coalescing breaks. The hardware fetches memory in contiguous chunks, so scattered accesses waste most of each fetched chunk. Adjacent threads must touch adjacent elements or effective bandwidth collapses.

## Level 2: Tensor cores want big tiles

Modern GPUs carry tensor cores: specialized units for generalized
matrix multiplication, D = AB + C. They natively multiply 16 by 16
tiles. The speed gap with versus without tensor cores reaches 16x
on H100 (Fu and Re, MLSys keynote 2023).

![Tensor cores](assets/slide-l05-tensor-cores.png "Tensor cores multiply 16x16 tiles; keep them busy with large tiles. Source: Stanford slides, credit Dan Fu and Chris Re.")

Two consequences. First, algorithms should be expressed as large
matmuls so tensor cores can take them. Second, tile sizes should
keep tensor cores busy: launching them costs time, and the latency
of a 64 by 64 multiply is roughly the same as a 16 by 16 one. Small
tiles waste the launch cost; large tiles amortize it.

This is the hardware-specific optimization principle from Lecture
3, and it explains a FlashAttention design choice in
[L06](l06-flash-attention.html): tile sizes are chosen for the
hardware's fast path, not for the algorithm's elegance.

## Level 2: Caching versus recompute in the backward pass

Training's backward pass needs intermediate values from the forward
pass, which adds reads and writes. If the workload is memory bound
and the recomputation is cheap relative to memory, skip storing the
intermediates and recompute them during the backward pass.

| Regime | Forward intermediates | Why |
|---|---|---|
| Compute bound | Store them | Math is the limit; traffic is affordable |
| Memory bound | Recompute them | Traffic is the limit; math is free |

This is the same rule as Lecture 3, applied to training.
FlashAttention's backward pass (next lecture) recomputes the
attention matrix from the stored softmax normalization vectors
instead of keeping the full N by N matrix: less HBM traffic at the
price of more FLOPs, a winning trade when memory bound.

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/plate-device-block.svg" alt="Device block">
<div class="rc-body">
<strong>1. The device block has three shells</strong>
<p>One GPU (SMs, registers, shared, L2, HBM), one node (8 GPUs,
NVLink), one cluster (nodes, InfiniBand). Defined here, reused by
MS&E435.</p>
<p class="rc-num">Key: the machine is a hierarchy</p>
</div>
</div>
<div class="recap-card">
<img src="assets/plate-device-block.svg" alt="Kernel">
<div class="rc-body">
<strong>2. A kernel loads, computes, writes</strong>
<p>Threads load inputs from HBM to SRAM and registers, compute,
write results back. Grids of blocks map to SMs with free capacity.</p>
<p class="rc-num">Key: kernel = one GPU operation</p>
</div>
</div>
<div class="recap-card">
<img src="assets/plate-device-block.svg" alt="Warps">
<div class="rc-body">
<strong>3. Warps execute in lockstep</strong>
<p>32 threads, one instruction, many data: SIMT. Expose enough
threads or SMs idle. Max threads = SMs x blocks x threads.</p>
<p class="rc-num">Key: 32 threads per warp</p>
</div>
</div>
<div class="recap-card">
<img src="assets/plate-device-block.svg" alt="Thread memories">
<div class="rc-body">
<strong>4. Threads see three memories</strong>
<p>Global (all threads, slow), shared (one block, collaborative),
registers (one thread, fastest). Place data by sharing pattern.</p>
<p class="rc-num">Key: share through shared memory</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l05-tiling.png" alt="Tiling">
<div class="rc-body">
<strong>5. Tiling halves redundant loads</strong>
<p>Partition data into shared-memory subsets. Threads collaborate
on overlapping inputs instead of loading independently.</p>
<p class="rc-num">Key: coalesce, then tile</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l05-tensor-cores.png" alt="Tensor cores">
<div class="rc-body">
<strong>6. Tensor cores want big tiles</strong>
<p>16 by 16 native tiles, up to 16x faster on H100. Launch cost is
flat, so large tiles amortize it. Express work as big matmuls.</p>
<p class="rc-num">Key: 16x16 tiles, big batches</p>
</div>
</div>
<div class="recap-card">
<img src="assets/plate-device-block.svg" alt="Interconnects">
<div class="rc-body">
<strong>7. NVLink inside, InfiniBand between</strong>
<p>Intra-node links are orders of magnitude faster. Frequent
collectives stay in the node; rare ones span the cluster.</p>
<p class="rc-num">Key: topology decides strategy</p>
</div>
</div>
<div class="recap-card">
<img src="assets/plate-device-block.svg" alt="Recompute">
<div class="rc-body">
<strong>8. Recompute when memory bound</strong>
<p>Skip storing forward intermediates; recompute them backward.
More FLOPs, less HBM traffic: the winning trade under a memory
roof.</p>
<p class="rc-num">Key: memory bound means math is free</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Hardware Aware Algorithm Design slide deck, GPU execution model sections (Fall 2023 headers).

**Further reading:**
- NVIDIA CUDA C++ Programming Guide: the authoritative execution model.
- [CS336 L06](../cs336/l06-triton-kernels.html): writing tiled, fused kernels in Triton.
- [CS336 L07](../cs336/l07-parallelism.html): NVLink, InfiniBand, and collectives in depth.
- Dan Fu and Chris Re, NeurIPS MLSys keynote 2023: tensor core utilization.

**Caveats from these sources.** SM counts, memory sizes, and the
16x tensor-core figure are A100/H100 generation numbers from the
slides; newer GPUs differ. The 8-GPUs-per-node layout is the
standard DGX/HGX configuration the course assumes. Coalescing and
tiling rules are NVIDIA CUDA specifics; other accelerators differ
in detail but not in spirit.

## Connections to the other courses

- **MS&E435:** reuses the device block defined here. Do not redefine it; link here.
- **CS336 L06:** kernel programming: this lecture's patterns in code.
- **CS336 L07:** interconnects and collectives in full.
- **CS229S L03:** the six principles; this lecture is their hardware substrate.
- **CS229S L06:** FlashAttention uses every pattern on this page.

> [!CHEAT]
> **GPU execution cheatsheet.** Device block: GPU (108 SMs, 256 KB registers/SM, 192 KB shared/SM, 40 MB L2, 40 GB HBM) / node (8 GPUs, NVLink) / cluster (nodes, InfiniBand). Kernel: load HBM to SRAM/registers, compute, write back. Grid of blocks to SMs; warps of 32 threads, SIMT. Max threads = SMs x blocks/SM x threads/block. Thread memories: global (all, slow), shared (block), registers (thread). Coalesced access: adjacent threads touch adjacent elements. Tiling: shared-memory subsets, collaborate to halve loads. Tensor cores: 16x16 GEMM tiles, up to 16x on H100; big tiles amortize launch. Recompute backward intermediates when memory bound. Topology rule: frequent collectives inside node, rare ones across.

> [!MEMORY]
> **The device in one line.** Fast small memories near the math, slow big memory far away, fast links inside the box and slow links between boxes.
