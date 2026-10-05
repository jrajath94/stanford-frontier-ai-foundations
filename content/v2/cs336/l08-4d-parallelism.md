---
page_id: cs336-l08
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 8
nav: "L08 · 4D Parallelism"
title: "Lecture 8: 4D Parallelism at Scale"
summary: "The full parallelism toolbox: ZeRO stages, pipeline bubbles and zero-bubble scheduling, tensor-parallel cuts, activation memory, expert parallel, and the 4D prescription behind real training runs."
date: "2026-04-22"
instructor: "Tatsunori Hashimoto"
offering: "Spring 2026"
duration: "1:20:01"
video_id: 6-cXp-aOmdg
video_title: "Stanford CS336 Spring 2026 Lecture 8: 4D Parallelism"
video_caption: "Original lecture. Tatsunori Hashimoto goes deep: ZeRO, bubbles, tensor and expert parallel, and the real training runs that combine them."
concepts: [zero, FSDP, pipeline-parallelism, zero-bubble, tensor-parallelism, expert-parallelism, sequence-parallelism, context-parallelism, 4D-parallelism]
sources:
  - tag: video
    label: "Lecture 8 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=6-cXp-aOmdg
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: slides
    label: "lecture_8.pdf, official course slides"
---

## How to read this lesson

Lecture 7 cut the work three ways. This lecture goes deep on the
algorithms, the memory math, and the real runs. **Level 1 (Core):**
ZeRO stages and the memory crisis. **Level 2 (Deep):** pipeline
bubbles, tensor-parallel cuts, activation accounting, expert parallel,
and the 4D prescription.

Two bottlenecks drive everything: compute (more than one chip holds)
and memory (models do not fit) [01:27](ts:01:27). The new unit of
compute is the data center [10:34](ts:10:34).

## Level 1: The memory crisis

Naive data parallel replicates everything and saves zero memory. The
real cost: with Adam you hold roughly 16 bytes per parameter, about
five copies of the weights [13:55](ts:13:55). Parameters (blue) and
gradients (orange) match. Optimizer state (green), the Adam first and
second moments in high precision, dominates.

## Level 1: The ZeRO ladder

Shard the state instead of replicating it. Three stages, each one
sharding more.

![ZeRO ladder](assets/l08-zero-ladder.svg "Stage 1: shard optimizer state. Stage 2: shard gradients. Stage 3/FSDP: shard parameters. Cost stays 2P until stage 3.")

**Stage 1:** shard optimizer state. Each GPU updates its slice:
reduce-scatter the gradients to the right owners, update, all-gather
the new parameters. Communication: reduce-scatter plus all-gather,
which equals one all-reduce. Same cost as DDP, memory divided by N.
Free [18:52](ts:18:52).

**Stage 2:** shard gradients too. Trick: during the backward sweep,
reduce-scatter each layer's gradients the moment they are computed and
free them. Never materialize the full gradient. Same communication,
more memory saved [19:16](ts:19:16).

**Stage 3 (FSDP):** shard parameters too. Each GPU holds a slice of
everything. All-gather the layer's parameters on demand, forward,
free. All-gather again for backward, reduce-scatter the gradients,
free. Cost: two all-gathers plus one reduce-scatter. One all-gather
more than DDP [20:25](ts:20:25).

Two ideas make stage 3 near-free: sweep-and-free (communicate, use,
immediately release) and overlap (all-gather layer n+1 while computing
layer n). The FSDP timeline shows compute and communication streams
interleaved. If compute outlasts communication, the extra traffic
hides completely [24:02](ts:24:02). On an A100, stage 3 fits 50B
parameters where the baseline could not fit 7B [28:04](ts:28:04).

> [!QA]
> Q: ZeRO-1 is literally free. Why does anyone run plain DDP?
> A: DDP is the simple mental model and the debugging baseline: every rank holds the full model, so any rank's state is inspectable and checkpointable directly. ZeRO-1 adds the reduce-scatter/all-gather choreography and sharded optimizer bookkeeping. For small models that fit comfortably, the memory savings buy nothing and the complexity buys risk. Use DDP until memory forces your hand, then climb the ladder.
> Follow-up: Why does overlap make FSDP "essentially free"?
> A: Communication and computation use different hardware: the network moves bytes while the SMs multiply matrices. If the all-gather for the next layer finishes before the current matmul does, its cost never appears on the critical path. The requirement is that compute dominates per layer, which holds for big layers on fast interconnects. Small layers or slow networks break the illusion.

## Level 1: Data parallel consumes batch size

DDP is capped by the batch: 8 examples can feed at most 8 GPUs
[29:15](ts:29:15). Past the critical batch size, extra examples add
less than another SGD step would: diminishing returns, wasted compute.
And ZeRO-1/2 never touch activation memory, which dwarfs parameters at
scale. Hence model parallelism: communicate activations, not just
parameters [31:02](ts:31:02).

```ascii
DDP cap      : 8 examples feed at most 8 GPUs
past B_crit  : extra examples add less than another SGD step
way out      : model parallelism, communicate activations
```

## Level 2: Pipeline bubbles

Cut by layers: GPU 0 owns layers 0-3, GPU 1 owns 4-7, and so on.
Naive execution: one GPU works at a time, forward then backward.
Utilization is terrible [32:15](ts:32:15).

![Pipeline bubbles](assets/l08-bubble.svg "Naive: one active GPU at a time. Micro-batches: the pipe fills. Zero-bubble: split backward into B and W.")

Micro-batches chop the batch so work flows continuously: the bubble
shrinks as 1/microbatches [33:52](ts:33:52). Pipeline's virtue is
communication: point-to-point, only b x s x h activations, far less
than whole parameter matrices. It lives on the slowest links: across
pods, across data centers [34:58](ts:34:58).

Zero-bubble scheduling splits the backward pass in two. Propagating
partials (B) is on the critical path: the next stage waits for it.
Weight gradients (W) are leaf nodes: they can happen anytime. Do B
first, defer W into the gaps, and the pipeline nearly fills
[37:28](ts:37:28).

## Level 2: Tensor-parallel cuts

Cut by width: matmuls split into smaller matmuls, partial sums added
back. The transformer rule: column cuts on the way up (MLP
up-projection, attention QKV), row cuts on the way down (MLP
down-projection, attention output). Small layers (norms, nonlinears,
MoE routers) replicate: not worth the overhead
[41:41](ts:41:41).

![Tensor parallel cuts](assets/l08-tp-cuts.svg "Column-wise on the way up, row-wise on the way down. f/g duality flips between forward and backward.")

The f/g duality: forward uses f=identity then g=all-reduce. Backward
flips them [40:48](ts:40:48). Tensor parallel is communication-hungry:
an all-reduce per matmul at activation size, every layer. Keep it
inside the fast node: TP of 8 on GPUs, then stop [42:50](ts:42:50).
TPUs, with their uniform mesh, can push tensor parallel much further
than GPUs can [43:28](ts:43:28).

## Level 2: Activation memory, exactly

Storing everything costs 34 x s x b x h plus 5 x a x s / h, where the
second term is the quadratic attention (softmax, dropout)
[46:51](ts:46:51). FlashAttention recomputation drops that term.

![Activation memory](assets/l08-activation-memory.svg "34sbh baseline. Tensor parallel divides the MLP and attention terms. Sequence parallel divides the rest. Lower bound: 34sbh/t.")

Tensor parallel divides the MLP (24 of the 34) and attention terms by
t, but not the layernorms, dropouts, and residuals: a stubborn 10sbh
remains [48:53](ts:48:53). Sequence parallel splits those leftovers
along the sequence axis, FSDP-style: shard now, materialize on demand
[49:25](ts:49:25). With recomputation, the floor is 34sbh/t: the number
to memorize for fitting models by hand [51:44](ts:51:44).

## Level 2: Expert parallel

For MoEs, prefer expert parallel over tensor parallel. Tensor slicing
makes matmuls small and utilization suffers. Routing whole tokens to
whole experts keeps matmuls big [54:45](ts:54:45). The dispatch is
all-to-all, latency-sensitive: computation waits for tokens to arrive.
DeepSeek's DeepEP and NVIDIA's Hybrid EP push this to undocumented-PTX
extremes [56:16](ts:56:16).

![Expert parallel](assets/l08-ep-vs-tp.svg "Route tokens to whole experts instead of slicing matmuls. But EP only covers MLPs: decouple TP for attention.")

The wrinkle: EP parallelizes the MLPs, not the attention. Want high TP
for attention but low TP for MLPs, or the slices get tiny. Modern
setups decouple them: one TP degree for attention, another for MoE
layers [60:27](ts:60:27).

Context parallel (ring attention) splits long sequences across
devices, passing activations around the ring: the standard move for
long-context extension and serving [61:05](ts:61:05).

> [!QA]
> Q: When do you pick tensor parallel over expert parallel for an MoE?
> A: For the attention layers, always tensor: experts do not touch attention, so EP cannot help there. For the MoE layers, prefer expert parallel: it keeps expert matmuls whole and routes sparse tokens instead. The modern answer is both at once, with decoupled degrees: high TP on attention, high EP on the experts. The Megatron guideline says exactly this: EP over TP for MoE layers, TP for attention.
> Follow-up: Why does sequence parallel have a misleading name?
> A: It does not parallelize the sequence computation the way context parallel does. It is an activation-memory trick that rides along with tensor parallel: shard the layernorm and residual activations along the sequence axis, materialize on demand. Conceptually it is FSDP for activations, not a way to compute attention faster.

## Level 2: The 4D prescription

No strategy dominates. The table of tradeoffs has red in every row
[62:07](ts:62:07). But the combination rule is simple:

![4D prescription](assets/l08-4d-prescription.svg "Fit the model first with TP/EP in the fast node, then pipeline or FSDP, then spend everything left on data parallel.")

1. Until the model fits: tensor or expert parallel of 8 inside the
   fast node.
2. Across nodes: pipeline parallel or ZeRO-3 on the slow links.
3. Then data parallel for everything left. Minimize model parallel,
   maximize data parallel.
4. Long sequences: add context parallel.

Batch too small: gradient accumulation restores utilization. The
compute-vs-communication roofline tells you when to add a dimension:
big batch, FSDP alone is compute-bound. Shrinking batch, add tensor
parallel to stay compute-bound [64:49](ts:64:49). The old
NVIDIA/Stanford scaling study shows the pattern exactly: DP maxed, TP
rises to 8 and stops, PP grows, DP falls at the extreme, utilization
flat throughout [70:13](ts:70:13).

Counterintuitive: do more computation via recomputation to get better
utilization. It saves memory, memory becomes batch size, batch size
becomes utilization [71:54](ts:71:54).

## Level 2: In the wild

![Training runs](assets/l08-training-runs.svg "OLMo to Qwen 3: the same recipe, different numbers. Tensor parallel stays at 8.")

OLMo 7B: FSDP only, small dense models do not need more
[72:46](ts:72:46). DeepSeek V1: ZeRO-1 plus tensor, sequence, pipeline.
DeepSeek V3: expert parallel of 64 across 8 machines, pipeline tricks
to keep utilization up. Yi: the classic ZeRO-1/TP/PP combo. Llama3
405B: TP8, CP1, PP16, DP128 for pretraining. More context parallel for
long-context extension. 148 GPU failures during the run: redundancy is
a distributed-systems problem too [76:02](ts:76:02). Gemma 2: FSDP plus
tensor plus sequence on the TPU mesh. Mixtral 8x22B: EP8, PP4, TP4 for
attention. Qwen 3: EP32, PP8, TP2, the DeepSeek recipe.

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l08-zero-ladder.svg" alt="ZeRO ladder">
<div class="rc-body">
<strong>1. ZeRO shards the state</strong>
<p>Stage 1: optimizer state. Stage 2: gradients. Stage 3: parameters.
Stages 1-2 cost exactly one all-reduce. Free memory.</p>
<p class="rc-num">Key: 2P comm until stage 3</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l08-topology-philosophy.svg" alt="Mesh vs tree">
<div class="rc-body">
<strong>2. Networks follow workloads</strong>
<p>TPU mesh: neighbor talk, scales forever. GPU tree: any-to-any,
flexible. MoE pushed TPUs toward trees.</p>
<p class="rc-num">Key: workloads define the wire</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l08-bubble.svg" alt="Pipeline bubbles">
<div class="rc-body">
<strong>3. Bubbles are idle tax</strong>
<p>Naive pipeline: one GPU at a time. Micro-batches fill the pipe.
Zero-bubble: B now, W in the gaps.</p>
<p class="rc-num">Key: bubble ~ 1/microbatches</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l08-tp-cuts.svg" alt="Tensor parallel cuts">
<div class="rc-body">
<strong>4. Cut wide, reduce narrow</strong>
<p>Columns up, rows down. Small layers replicate. f/g flips in
backward. Keep TP inside the fast node.</p>
<p class="rc-num">Key: TP of 8, then stop</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l08-activation-memory.svg" alt="Activation memory">
<div class="rc-body">
<strong>5. Count activations first</strong>
<p>34sbh plus attention quadratic. TP divides the big terms. Sequence
parallel divides the rest. Floor: 34sbh/t.</p>
<p class="rc-num">Key: 34sbh/t</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l08-ep-vs-tp.svg" alt="Expert parallel">
<div class="rc-body">
<strong>6. EP for MoE</strong>
<p>Route tokens to whole experts, keep matmuls big. All-to-all,
latency-critical. Decouple TP: high for attention, low for MoE.</p>
<p class="rc-num">Key: EP over TP for experts</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l08-4d-prescription.svg" alt="4D prescription">
<div class="rc-body">
<strong>7. Fit, then spend</strong>
<p>TP/EP in the node, PP/FSDP across, DP for the rest. Gradient
accumulation for small batches. Hide comm under compute.</p>
<p class="rc-num">Key: minimize MP, maximize DP</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l08-training-runs.svg" alt="Training runs">
<div class="rc-body">
<strong>8. Same recipe, new numbers</strong>
<p>OLMo to Qwen 3: every frontier run is the prescription with
different degrees. TP stays at 8 on GPUs.</p>
<p class="rc-num">Key: read the run table</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 8 video and slides (lecture_8.pdf).
- Megatron parallelism guidelines for MoEs: the practitioner version
  of the prescription.
- NVIDIA Megatron Bridge: recommended configs per model size.

**Further reading:**
- ZeRO paper (Rajbhandari et al.): the three stages.
- Zero-bubble pipeline paper. DeepSeek V3 report: EP at 64-way.
- The NVIDIA/Stanford large-scale parallelism study (Zaharia,
  Narayanan): how strategies shift with scale.

**Caveats from these sources.** Training-run configs are as reported
in public papers and repos. Internal runs may differ. The TPU8i
announcement was same-day news, treat topology claims as preliminary.
Undocumented-PTX claims about DeepEP are secondhand.

## Connections to the other courses

- **CS336 Lecture 7:** the three basic parallelisms this lecture
  extends.
- **CS336 Lecture 4:** MoE routing and load balancing feed the EP
  discussion.
- **CS229S:** the distributed-systems side: failures, redundancy,
  NCCL internals.

> [!CHEAT]
> **4D parallelism cheatsheet.** Bottlenecks: compute, memory. Unit: the data center. Memory: 16B/param Adam. ZeRO-1: shard opt state, 2P comm, free. ZeRO-2: shard grads, incremental reduce, free. ZeRO-3/FSDP: shard params, 2 AG + 1 RS, overlap hides it. DP cap: batch size, critical batch size. Pipeline: cut layers, bubbles, micro-batches, B/W split, bsh point-to-point, slow links. TP: columns up rows down, f/g duality, all-reduce per matmul, TP8 max on GPUs. Activations: 34sbh + 5as/h, floor 34sbh/t. Sequence parallel: shard layernorm leftovers on s. EP: route tokens, all-to-all, prefer over TP for MoE, decouple TP degrees. Context parallel: ring attention for long context. Prescription: fit with TP/EP in node, PP/FSDP across, DP the rest. Recompute to buy batch size.

> [!MEMORY]
> **Shard everything, hide the cost.** ZeRO, overlap, bubbles, cuts: every trick is memory divided by N with communication hidden under compute.
