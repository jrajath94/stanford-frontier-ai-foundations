---
page_id: cs229s-l10
course_slug: cs229s
course_name: "CS229S: Systems for Machine Learning"
course_order: 3
order: 10
nav: "L10 · Parallelism"
title: "Lecture 10: Parallelism Fundamentals at Scale"
summary: "The CS229S framing of distributed training: data, tensor, pipeline, context, and expert parallelism, ZeRO memory math, ring all-reduce, DualPipe, and the Megatron view from Deepak Narayanan's talk."
date: "2024-11-15"
instructor: "Azalia Mirhoseini"
offering: "Fall 2024"
duration: "0:56:00"
video_id: JA1l96tjrs4
video_title: "Deepak Narayanan: Training Large Language Models at Scale (MLIS seminar)"
video_caption: "Guest talk assigned with the parallelism lecture. Narayanan (NVIDIA Megatron-LM) covers data, tensor, and pipeline parallelism on real clusters. Timestamps link to the exact moments."
concepts: [data-parallelism, tensor-parallelism, pipeline-parallelism, context-parallelism, expert-parallelism, all-reduce, ring-allreduce, zero, fsdp, microbatch, ptd-parallelism, dualpipe, alpa]
sources:
  - tag: video
    label: "Deepak Narayanan parallelism talk (YouTube)"
    url: https://www.youtube.com/watch?v=JA1l96tjrs4
  - tag: slides
    label: "Parallelism Fundamentals slide deck (Fall 2023 headers)"
  - tag: paper
    label: "Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2019)"
    url: https://arxiv.org/abs/1910.02054
  - tag: paper
    label: "Shoeybi et al., Megatron-LM (2019)"
    url: https://arxiv.org/abs/1909.08053
  - tag: paper
    label: "Narayanan et al., PipeDream (SOSP 2019)"
    url: https://arxiv.org/abs/1811.06965
  - tag: paper
    label: "Zheng et al., Alpa: Automating Inter- and Intra-Operator Parallelism (OSDI 2022)"
    url: https://arxiv.org/abs/2201.12023
  - tag: paper
    label: "Jacobs et al., DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models (2023)"
    url: https://arxiv.org/abs/2309.14509
  - tag: paper
    label: "DeepSeek-AI, DeepSeek-V3 Technical Report (2024)"
    url: https://arxiv.org/abs/2412.19437
  - tag: paper
    label: "Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models (2022)"
    url: https://arxiv.org/abs/2205.05198
  - tag: supplement
    label: "Course text: narayanan-parallelism.txt (talk notes)"
---

### Coverage and sourcing

This lesson follows the CS229S "Parallelism Fundamentals"
slide deck (Fall 2024 offering, Fall 2023 headers),
taught by Azalia Mirhoseini, plus the assigned guest
talk by Deepak Narayanan (NVIDIA Megatron-LM, Stanford
MLSys #83, 56 minutes) and its course text notes. The
deck's arc runs: data parallelism and its memory wall,
ZeRO stages, ring all-reduce, tensor parallelism, the
pipeline bubble and schedules, the decision table, and
Alpa. The lesson adds the post-deck systems, each fact
dated: context/sequence parallelism (DeepSpeed-Ulysses,
Megatron CP), expert parallelism for MoE, FSDP2, and
DeepSeek-V3's DualPipe training stack (December 2024),
with October 2026 updates throughout. Talk timestamps
point to the auto-captioned YouTube video. Wording is
the speaker's paraphrase, verified against the cleaned
captions.

## The problem: one GPU is not enough

Two curves collide. GPU FLOPS double roughly every 2.5
years, but LLMs grow about 10x bigger every year (Epoch
AI, Megatron-Turing NLG 530B). A single device cannot
hold the weights, the activations, the gradients, and
the optimizer state, and even when it can, one device is
too slow. The only way forward is to split the work
across many GPUs. This lecture derives the three ways to
split it, the math that forces the choice, and the rules
for combining them.

Two challenges dominate every distributed design:
minimize **communication overhead** (time sending data
between GPUs) and minimize **synchronization overhead**
(dependencies that stop GPUs from working independently).
Everything below is a tradeoff between these two.

## First attempt: split the data

Computation partitions along two axes: the input **data**
or the model **parameters**. The simplest split is the
data.

**Data parallelism**: partition the input data, replicate
the model weights. Four GPUs and batch 64: each GPU takes
16 examples. Forward and backward run independently per
GPU. Each produces its own gradients for its own
sub-batch. At the end of the backward pass, the
gradients are aggregated across all GPUs and averaged.
The GPUs share their gradients with an **all-reduce**
operation: every GPU ends up with the sum of all GPUs'
gradients.

![Data parallel](assets/slide-l10-data-parallel.png "Shell 1. Partition the batch across GPUs. Replicate the weights. Source: Stanford slides. Project: Stanford Frontier AI.")

Pros: the most direct throughput lever, since the global
batch grows with each GPU, and easy to implement (PyTorch
native support). Narayanan walks the tradeoffs on real
clusters [06:26](ts:06:26).

### Subchapter: the global batch math

Four GPUs, batch 64: each GPU processes 16 examples, and
the gradient average covers all 64. Double the GPUs to 8
with batch 128: each GPU still processes 16, and
throughput doubles if communication keeps up. The global
batch is the product: data-parallel degree times
per-GPU batch. But bigger global batches change the
optimization: the learning rate and the number of steps
must be retuned, because each step now averages over
more examples. Throughput scales. The training recipe
shifts.

## Where it breaks: the memory wall

Data parallelism has a hard requirement: the model,
activations, gradients, and optimizer state must fit on
one GPU. GPT-3 cannot run on pure data parallelism on an
80 GB GPU. And every weight synchronizes every step, so
it is communication-intensive.

Work the memory math for P parameters in mixed-precision
training:

![16P memory](assets/slide-l10-16p-memory.png "Shell 2. Mixed-precision training needs 16P bytes: 2P params, 2P grads, 12P optimizer state. Source: Stanford slides. Project: Stanford Frontier AI.")

### Subchapter: the 16P, worked at GPT-3 scale

P = 175B. fp16 parameters: 350 GB. fp16 gradients: 350
GB. Adam's three fp32 states: 3 x 700 GB = 2.1 TB.
Total: 2.8 TB, before a single activation. The lecture's
10B worked example (160 GB) already exceeds an 80 GB
GPU. At 175B the bill is 35 GPUs of memory for one
replica.

![The 16P at GPT-3 scale](assets/plate-l10-16p-gpt3.webp "175B parameters need 2.8 TB in mixed-precision training: 350 plus 350 plus 2,100 GB. Shell 2. Source: original toy for the 16P sum. Project: Stanford Frontier AI.")

- fp16 parameters: 2P bytes
- fp16 gradients: 2P bytes
- Adam state: fp32 parameters 4P, momentum 4P, variance 4P
- Total: 16P bytes, before activations

A 10B-parameter model needs 160 GB for parameters and
optimizer state alone. No single GPU holds that. Data
parallelism replicates all of it on every GPU, which is
pure redundancy: each GPU only needs its share.

## The key question

What if each GPU held only its share of the 16P bytes?
**ZeRO** (Zero Redundancy Optimizer) removes the
replication in three stages.

![ZeRO stages](assets/slide-l10-zero-stages.png "Shell 3. Stage 1 partitions optimizer state, stage 2 adds gradients, stage 3 adds parameters. Source: Stanford slides. Project: Stanford Frontier AI.")

**Stage 1: optimizer state partitioning.** Group the
optimizer states into Nd equal partitions. Each data
parallel process updates only its partition, then
all-gathers (each GPU collects the full set of) the
updated parameters at the end of each step. Memory per
GPU: 4P + 12P/Nd.

**Stage 2: plus gradient partitioning.** Each process
only needs the reduced gradients for its own parameter
partition: 2P bytes of gradients become 2P/Nd. Memory
per GPU: 2P + 14P/Nd, about 2P.

**Stage 3: plus parameter partitioning.** Do not store
all parameters on all processes. Fetch (all-gather) the
ones needed for the forward and backward pass. Memory
per GPU: 16P/Nd. The price: about 1.5x more
communication volume.

| Stage | Partitions | Memory per GPU |
|---|---|---|
| Baseline | none | 16P |
| 1 | optimizer state | 4P + 12P/Nd |
| 2 | + gradients | 2P + 14P/Nd (about 2P) |
| 3 | + parameters | 16P/Nd |

PyTorch **FSDP** (Fully Sharded Data Parallel)
implements ZeRO-style sharding as a module wrapper: wrap
the model, and sharding, gathering, and reduction happen
underneath.

### Subchapter: the ZeRO stages, worked

10B parameters on Nd = 8 data-parallel GPUs. Baseline
16P: 160 GB per GPU, fits nowhere.

Stage 1: 4P + 12P/8 = 40 + 15 = 55 GB per GPU. Fits on
an 80 GB GPU with room for activations.

Stage 2: 2P + 14P/8 = 20 + 17.5 = 37.5 GB per GPU. More
headroom for longer sequences.

Stage 3: 16P/8 = 20 GB per GPU. The price: every step
all-gathers the full parameters, about 1.5x the
communication volume.

The ladder is the lesson: each stage halves the memory
roughly, and each stage costs bandwidth. Climb only as
far as the memory gap demands.

![The ZeRO stage math](assets/plate-l10-zero-stages-math.webp "10B on 8 GPUs: 160 GB baseline, 55, 37.5, 20 GB per stage. Shell 3. Source: original toy for the stage math. Project: Stanford Frontier AI.")

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
sharded_module = FSDP(module)   # ZeRO-style sharding under the hood
```

Match the stage to the memory gap, not the maximum: the
extra all-gathers cost bandwidth, so if the model fits
with stage 1 or 2, the lower stages communicate less.

### Subchapter: FSDP2 and the 2026 default

PyTorch's FSDP2 (the DTensor-based rewrite: DTensor is
PyTorch's distributed-tensor abstraction, which tracks
how each tensor is sharded across devices) is the 2026
default for ZeRO-3-style sharding: per-parameter
sharding instead of per-module, composable with tensor
parallelism, and lower memory spikes during the
all-gather. DeepSpeed's ZeRO-Infinity extends stage 3 to
CPU and NVMe (Non-Volatile Memory Express, the fast
solid-state-drive interface) offload for
trillion-parameter models.
The ladder keeps growing downward: shard more, offload
further, communicate only what the step needs.

## The collective everything stands on

Data parallelism averages gradients with an all-reduce:
every GPU ends up with the sum (then the mean) of all
GPUs' gradients. The **ring all-reduce** runs in 2(N-1)
iterations for N GPUs. Each GPU splits its data into N
fragments. The first N-1 iterations accumulate received
values, the next N-1 propagate the completed sums.

![Ring all-reduce](assets/slide-l10-ring-allreduce.png "Shell 4. 2(N-1) iterations. Each node sends 2(N-1)X/N bytes. Bandwidth-optimal. Source: Stanford slides, Patarasuk and Yuan. Project: Stanford Frontier AI.")

### Subchapter: the ring, worked on N = 4

Four GPUs, gradients X = 1 GB. Each GPU splits its 1 GB
into 4 fragments of 250 MB. Iterations: 2 x (4-1) = 6:
three scatter-reduce rounds accumulate the sums, three
all-gather rounds propagate them. Each node sends 2 x 3
x 1/4 = 1.5 GB total. At 100 GB/s NVLink that is 15 ms
of wire time, and it stays near 2X no matter how many
nodes join: per-node traffic approaches 2X as N grows.

The latency caveat lives in the iteration count: 2(N-1)
rounds of small messages pay latency, not bandwidth.
The ring is optimal for large transfers (gradient sync).
Small messages want fewer, fatter hops.

![The ring at N = 4](assets/plate-l10-ring-n4.webp "6 iterations, 1.5 GB per node, traffic near 2X at any scale. Shell 4. Source: original toy for the ring. Project: Stanford Frontier AI.")

Efficiency: each node sends 2(N-1) x X/N bytes for data
of size X, so communication time is 2(N-1)X / (N x B) at
bandwidth B. As N grows, per-node traffic approaches 2X:
independent of GPU count. That is why the ring is
bandwidth-optimal, and why data parallelism scales to
many nodes.

## The key question, again

ZeRO shrinks the per-GPU memory, but the full model
still passes through every GPU's hands. What if the
weights themselves must be split?

**Model parallelism** replicates the input data and
partitions the model weights. Two flavors. **Vertical
slicing** places subsets of layers on different GPUs.
Each GPU waits for the previous stage's activations, so
devices sit idle most of the time (PipeDream).
**Horizontal slicing** shards within the operation
instead.

For transformers, the horizontal style is Megatron
**tensor parallelism**: shard the self-attention and MLP
weights across GPUs, with an all-reduce after every
attention and MLP block to resynchronize. Pros: per-GPU
memory falls, and utilization stays high compared to
vertical slicing. Cons: the all-reduces are extremely
frequent, so throughput demands very fast interconnects,
and the synchronization ops must be added manually to the
attention module.

### Subchapter: Megatron's column-row trick, worked

The MLP is two matmuls: Y = GeLU(X W1) W2. GeLU
(Gaussian error linear unit) is a smooth activation
that multiplies each input by its cumulative-normal
probability. Split W1 by columns across 2 GPUs: each computes half the hidden
units, no communication needed (GeLU is elementwise).
Split W2 by rows: each GPU holds the rows matching its
hidden half and produces a partial output. One all-reduce
sums the partials. Two matmuls, one synchronization.

Attention follows the same pattern: Q, K, V projections
split by columns (heads divided across GPUs), the output
projection split by rows. One all-reduce per attention
block, one per MLP block. That is the price of tensor
parallelism, stated per layer: two all-reduces, every
layer, every step.

Narayanan's rule from the talk: tensor parallelism
becomes very unwieldy across nodes [14:48](ts:14:48).
Within a server, 8 GPUs connect over NVLink at very high
bandwidth [15:01](ts:15:01). Across servers, even
InfiniBand is much slower [15:09](ts:15:09). So tensor
parallelism lives inside the node, and its communication
never crosses the slow links. The device block from L05
is the map. This is the route.

![Device block: tensor parallelism stays in the node](assets/plate-device-block.svg "Shell 5. The device block, defined in L05. Tensor parallelism lives on NVLink. Data parallelism spans InfiniBand. Source: original plate. Project: Stanford Frontier AI.")

## The key question, once more

Tensor parallelism solved the memory problem but demands
fast networking. What if you have many servers, or one
server without fast interconnects? Revisit vertical
slicing, but fix the idleness with pipelining.

**Pipeline parallelism** splits each batch into
**microbatches** and injects several into the pipeline at
once. While stage 2 processes microbatch 1, stage 1
processes microbatch 2. Idle time shrinks.

![Pipeline microbatch](assets/slide-l10-pipeline-microbatch.png "Shell 6. Microbatches fill the pipeline: stages work concurrently instead of idling. Source: Stanford slides, PipeDream. Project: Stanford Frontier AI.")

### Subchapter: the bubble, worked

Four pipeline stages, eight microbatches. The pipeline
fills for 3 steps and drains for 3 steps: total steps 8
+ 4 - 1 = 11, idle steps 3. Bubble fraction: (p - 1) /
(m + p - 1) = 3/11 = 27 percent of the GPUs' time
wasted.

Double the microbatches to 16: bubble 3/19 = 16 percent.
The cost: smaller microbatches mean lower arithmetic
intensity. This is the knob the lecture names:
microbatch size trades bubble size against compute
efficiency, and no schedule removes the fill and drain
entirely.

![The pipeline bubble](assets/plate-l10-bubble.webp "4 stages, 8 microbatches: 27 percent idle. 16 microbatches cut it to 16 percent. Shell 6. Source: original toy for the bubble. Project: Stanford Frontier AI.")

Two tuning knobs. **Microbatch size**: larger means
higher arithmetic intensity, but smaller means smaller
pipeline bubbles. **Schedule**: which microbatch runs
where and when. GPipe runs all forwards then all
backwards. **1F1B** (one forward, one backward)
interleaves them and shrinks both the bubble and the
activation memory.

![Pipeline schedule](assets/plate-pipeline-schedule.svg "Shell 7. 1F1B interleaves forwards and backwards to shrink idle bubbles. Source: original plate. Narayanan et al. Project: Stanford Frontier AI.")

Narayanan's talk shows an **interleaved** variant: device
1 holds layers 1 and 5, device 2 holds layers 2 and 6,
and so on, so each device does two phases of forward
passes per microbatch [24:07](ts:24:07). More stages per
device means less idle time at the cost of more
point-to-point messages. TeraPipe extends the idea to
the sequence dimension: split the sequence itself into
subsequences processed incrementally per stage.

Pros: per-GPU memory falls. Communication is
point-to-point sends, not all-reduces. Cons: bubbles
never fully vanish. Scheduling microbatches concurrently
is genuinely hard to implement.

### Subchapter: DualPipe, the 2024 answer

DeepSeek-V3 (December 2024) trains with 16-way pipeline
parallelism and no tensor parallelism. **DualPipe** is
the schedule that makes it work: it duplicates the
pipeline so micro-batches flow from both ends, and the
fill of one direction overlaps the drain of the other.
Computation and communication overlap: the all-to-all
exchange for expert parallelism hides inside the
overlapped compute.

The paper's claim: near-zero pipeline bubble with most
communication hidden. The design logic: tensor
parallelism's per-block all-reduces were judged too
expensive at 671B scale, so the pipeline schedule had to
carry the parallelism alone. DualPipe is what a pipeline
schedule looks like when it cannot lean on tensor
parallelism. The V3 stack (16-way DualPipe PP, 64-way
expert parallelism, ZeRO-1 data parallelism) is derived
in full in L09.

## The fourth dimension: split the sequence

Data, tensor, and pipeline split the batch, the weights,
and the layers. Long context adds a fourth split: the
sequence itself. At 128K+ tokens, the activations for one
sequence can exceed the weights: splitting the sequence
across GPUs attacks the activation memory directly.

**DeepSpeed-Ulysses** (2023) is the clean design. Split
the sequence across GPUs, and for attention, all-to-all
the Q, K, V so each GPU holds the full sequence for a
subset of heads. Each GPU computes attention for its
heads over the whole sequence, then all-to-alls the
output back. Communication: a few all-to-alls per layer,
not per head. Measured: near-linear strong scaling from
64 to 256 GPUs at 131K sequence length, and it
outperforms Megatron's sequence parallelism and
Colossal-AI's on the sequences they can run.

Megatron-LM ships **context parallelism** as
`--context-parallel-size` with `--cp-comm-type p2p`:
shard activations along the sequence dimension,
exchange what attention needs. The decision rule: reach
for sequence parallelism when activations dominate the
memory bill, which is exactly the long-context regime.
As you learned in CS229S L06, context parallelism is the
last resort for attention bottlenecks because it
communicates every layer. For activation memory, it is
the first resort.

### Subchapter: when activations dominate, worked

One transformer layer, batch 1, N = 128K, d = 7168.
FlashAttention never materializes the N by N scores, so
the stored activations per layer follow the standard
estimate of about 34 x N x d bytes for the backward pass
(Korthikanti et al.): 34 x 131,072 x 7,168 = 31.9 GB
per layer. Times 61 layers: about 1.95 TB of
activations. That is why V3-scale training at long
context needs the sequence split: the activations dwarf
the 16P weight bill (671B params is 1.3 TB in FP16).
Split the sequence 8 ways and each GPU holds 244 GB.
Still large, which is why context parallelism composes
with the other four.

## The fifth dimension: split the experts

Mixture-of-experts adds a fifth split. **Expert
parallelism** places experts on different GPUs, derived
in full in L09: V3 runs 64-way over 8 nodes, tokens
travel by all-to-all dispatch, node-limited routing caps
the fan-out at 4 nodes, DeepEP kernels saturate the
links. Megatron-LM exposes it as
`--expert-model-parallel-size` with `--moe-grouped-gemm`
for the batched expert matmuls. One hard rule from the
Megatron docs: combining expert parallelism with tensor
parallelism requires sequence parallelism enabled. The
splits interact, and the framework enforces the
combination that keeps the communication patterns sound.

The full 2026 parallelism menu:

| Split | What it partitions | Communication | Lives on |
|---|---|---|---|
| Data | batch | all-reduce per step | anywhere |
| Tensor | weights within layers | all-reduce per block | NVLink, inside node |
| Pipeline | layers | point-to-point sends | across nodes |
| Context/sequence | sequence length | all-to-all or p2p per layer | across nodes |
| Expert | MoE experts | all-to-all dispatch per MoE layer | across nodes |

## What is used where: real training stacks

Distributed training is how every large model is built.
Facts as of October 2026.

| Stack | Parallelism | Source |
|---|---|---|
| Megatron-LM (NVIDIA) | tensor + pipeline + data + context + expert (5D) | public |
| DeepSpeed (Microsoft) | ZeRO stages 1-3, ZeRO-Infinity, Ulysses, DeepSpeed-MoE | public |
| PyTorch FSDP2 | ZeRO-3 style sharding, DTensor-based | public |
| DeepSeek-V3 | 16-way DualPipe + 64-way EP + ZeRO-1, no TP, FP8 | public in the V3 paper |
| vLLM / SGLang / TensorRT-LLM | tensor parallelism for serving | public |
| NVIDIA Dynamo | disaggregated prefill/decode serving | public |

The pattern: 5D parallelism at training scale, FSDP2 as
the default open fine-tuning path, DualPipe as the
newest public pipeline schedule, disaggregation as the
newest serving shape. Exact parallelism plans inside
frontier labs (GPT, Gemini) are not public.

## Mapping back: the decision table

All the techniques are now built. The lecture's summary
places each one where its tradeoff wins:

![Parallelism summary](assets/slide-l10-parallelism-summary.png "Shell 8. Data, tensor, pipeline: when each wins. Source: Stanford slides. Project: Stanford Frontier AI.")

| Situation | Choice | Why |
|---|---|---|
| Weights and activations fit on one GPU | Data parallelism | Simplest, most direct throughput lever |
| They do not fit, but one fast node is available | Tensor parallelism (NVLink) | Frequent all-reduces need the fast links |
| They do not fit, many nodes or slow interconnect | Pipeline parallelism | Point-to-point sends, no all-reduces |
| Activations dominate at long context | Context parallelism | Splits the sequence; attacks activation memory |
| MoE experts do not fit per GPU | Expert parallelism | All-to-all dispatch; needs EP-aware stack |

In practice, combine: data plus tensor plus pipeline
(**PTD** or 3D parallelism) scales to thousands of GPUs
(DeepSpeed, Megatron-LM), and the 2026 stacks add
context and expert parallelism to reach 5D. The
strategy space grows combinatorially and experiments at
scale are slow and expensive, so **Alpa** (Zheng et al.,
OSDI 2022) automates the search: it treats
inter-operator parallelism (pipeline) and intra-operator
parallelism (data, tensor) separately and picks the
optimal combination for a given model and cluster.

## The 2026 hardware map

The strategies route over physical links, and the links
kept getting faster. NVLink 5 (Blackwell B200): 1.8
TB/s bidirectional per GPU. NVLink 6 (Rubin, shipping
fall 2026): 3.6 TB/s. The NVL72 rack puts 72 GPUs in one
NVLink domain. Between racks, InfiniBand remains an
order of magnitude slower per GPU.

The map rules, updated: tensor parallelism lives inside
the NVLink domain (now up to 72 GPUs, not just 8).
Pipeline and expert parallelism cross racks on
InfiniBand, where DualPipe's overlap and DeepEP's
kernels hide the cost. Data parallelism spans
everything: the ring's per-node traffic approaches 2X no
matter how many racks join. As you learned in CS229S
L05, the device block is the map. This section is the
2026 edition of the routes drawn on it.

## The honest price

Every split pays in communication or synchronization.
Data parallelism synchronizes every weight every step.
Tensor parallelism all-reduces after every block and
dies on slow networks. Pipeline parallelism never fully
kills the bubble and is hard to implement. Context
parallelism communicates every attention layer. Expert
parallelism adds all-to-all dispatch to every MoE layer.
The 16P figure excludes activations, which vary with
sequence length and checkpointing, and at 128K they
exceed the weights (about 1.95 TB vs 1.3 TB in the
worked example). The strategy space is combinatorial:
Alpa exists because no human tunes thousand-GPU
parallelism by hand. And the activation math above
(about 1.95 TB at 128K) is an order-of-magnitude estimate: the
exact factor depends on the architecture and the
checkpointing schedule.

## Coverage map: every lecture claim and where it lives

| Lecture claim | Covered in | File line |
|---|---|---|
| Two curves: GPU FLOPS 2.5yr doubling vs 10x/yr model growth | The problem: one GPU is not enough | L68 |
| Two challenges: communication and synchronization overhead | The problem: one GPU is not enough | L68 |
| Data parallelism: partition data, replicate weights; 4 GPUs batch 64 | First attempt: split the data | L86 |
| Global batch math: degree times per-GPU batch; recipe shifts | the global batch math | L109 |
| Memory wall: 16P bytes; 10B is 160 GB; GPT-3 175B is 2.8 TB | Where it breaks: the memory wall | L122 |
| ZeRO stages: 4P+12P/Nd, 2P+14P/Nd, 16P/Nd; 1.5x comm at stage 3 | The key question | L156 |
| ZeRO worked: 10B on 8 GPUs, 160 to 55 to 37.5 to 20 GB | the ZeRO stages, worked | L194 |
| FSDP wrapper; FSDP2 DTensor default; ZeRO-Infinity offload (Oct 2026) | FSDP2 and the 2026 default | L224 |
| Ring all-reduce: 2(N-1) iterations; per-node 2(N-1)X/N, approaches 2X | The collective everything stands on | L239 |
| Ring worked N=4: 6 iterations, 1.5 GB per node | the ring, worked on N = 4 | L250 |
| Model parallelism: vertical vs horizontal; Megatron tensor parallelism | The key question, again | L274 |
| Column-row trick: split W1 by columns, W2 by rows; one all-reduce | Megatron's column-row trick, worked | L298 |
| Narayanan's rule: TP inside the node, never across (NVLink vs IB) | The key question, again | L274 |
| Pipeline: microbatches; bubble (p-1)/(m+p-1); 27% at 4/8, 16% at 4/16 | The key question, once more | L327 |
| 1F1B schedule; interleaved variant; TeraPipe sequence split | the bubble, worked | L341 |
| DualPipe: 16-way PP, no TP, near-zero bubble, overlapped comm (Dec 2024) | DualPipe, the 2024 answer | L382 |
| Context parallelism: Ulysses all-to-all, 64 to 256 GPUs at 131K | The fourth dimension: split the sequence | L403 |
| Activations dominate: 31.9 GB/layer, 1.95 TB at 128K; split 8 ways 244 GB | when activations dominate, worked | L433 |
| Expert parallelism: V3 64-way; EP+TP requires SP (Megatron rule) | The fifth dimension: split the experts | L448 |
| 5D parallelism menu table | The fifth dimension: split the experts | L448 |
| Training stacks table: Megatron, DeepSpeed, FSDP2, V3, vLLM, Dynamo (Oct 2026) | What is used where | L473 |
| Decision table: data, tensor, pipeline, context, expert rows | Mapping back: the decision table | L493 |
| Alpa automates inter/intra-operator search | Mapping back: the decision table | L493 |
| 2026 hardware map: NVLink 5 1.8 TB/s, NVLink 6 3.6 TB/s, NVL72 (Oct 2026) | The 2026 hardware map | L519 |
| Honest price: every split pays; activations exceed weights at 128K | The honest price | L538 |

> [!QA]
> Q: A 10B-parameter model trains in mixed precision with Adam. How much memory do the parameters and optimizer state need, and what does ZeRO stage 3 change?
> A: Baseline: 2P for fp16 params, 2P for fp16 grads, and 12P for fp32 params, momentum, and variance: 16P total, or 160 GB for 10B params. That fits on no single GPU. ZeRO stage 3 partitions parameters, gradients, and optimizer state across Nd data-parallel GPUs, cutting per-GPU memory to 16P/Nd at the cost of about 1.5x more communication for parameter gathering.
> Follow-up: Why not always use stage 3?
> A: Because the extra all-gathers cost bandwidth and can expose communication on the critical path. If the model fits with stage 1 or 2, the lower stages communicate less. Match the stage to the memory gap, not the maximum.

> [!QA]
> Q: Why is ring all-reduce bandwidth-optimal?
> A: Each of N nodes sends 2(N-1) x X/N bytes for data of size X, so communication time is 2(N-1)X / (N x B). As N grows, per-node traffic approaches 2X: independent of GPU count. Adding GPUs never increases any single node's traffic, which is why data parallelism scales to many nodes.
> Follow-up: What does bandwidth-optimal not guarantee?
> A: Low latency. The ring needs 2(N-1) iterations, so small messages still pay the per-iteration latency. Bandwidth-optimality is about large transfers, which is the gradient-sync regime.

> [!QA]
> Q: You have 64 GPUs across 8 nodes with NVLink inside nodes and InfiniBand between them, training a model that needs 16 GPUs worth of memory. Sketch the parallelism plan.
> A: Use all three. Tensor parallelism within each node (up to 8 ways) for the frequent attention and MLP all-reduces over NVLink. Pipeline parallelism across nodes for the layer stages, with microbatches sized to keep bubbles small. Data parallelism across the remaining replicas for throughput, with ZeRO stage 1 or 2 to fit optimizer state. This is PTD parallelism: each style lives where the network makes it cheap.
> Follow-up: What breaks if you run tensor parallelism across nodes instead?
> A: Throughput collapses. Tensor parallelism all-reduces after every attention and MLP block, and InfiniBand is much slower than NVLink. The GPUs would stall on communication constantly. Narayanan's talk makes this the cardinal rule: keep tensor parallelism inside the node.

> [!QA]
> Q: How do microbatches and the 1F1B schedule fight the pipeline bubble?
> A: Microbatches keep every stage fed: while stage 2 processes microbatch 1, stage 1 processes microbatch 2, so idle time shrinks to the fill and drain phases. 1F1B interleaves one forward and one backward per microbatch instead of running all forwards then all backwards, which shrinks the bubble further and cuts the activation memory that GPipe's schedule holds.
> Follow-up: Why not make microbatches tiny to kill the bubble entirely?
> A: Smaller microbatches mean lower arithmetic intensity: the math gets less efficient per byte moved. The microbatch size trades bubble size against compute efficiency, and the schedule trades implementation complexity against both.

> [!QA]
> Q: Walk me through ZeRO stages 1 to 3 on a 10B model across 8 GPUs.
> A: Baseline 16P = 160 GB per GPU: fits nowhere. Stage 1 partitions the 12P optimizer state: 4P + 12P/8 = 40 + 15 = 55 GB per GPU. Stage 2 also partitions the 2P gradients: 2P + 14P/8 = 20 + 17.5 = 37.5 GB. Stage 3 partitions the 2P parameters too: 16P/8 = 20 GB per GPU, with about 1.5x more communication for the parameter all-gathers. Each stage roughly halves the memory and costs bandwidth: stop at the stage that fits.
> Follow-up: Where does FSDP sit on this ladder?
> A: FSDP is ZeRO-3-style: it shards parameters, gradients, and optimizer state across the data-parallel group. FSDP2 is the 2026 default: per-parameter sharding on DTensor, composable with tensor parallelism. Use it when stage 3 is the answer. For stage 1 or 2, lighter sharding communicates less.

> [!QA]
> Q: Ring all-reduce, N = 16, X = 4 GB. How much does each node send, and how many iterations?
> A: Iterations: 2 x 15 = 30. Per node: 2 x 15 x 4/16 = 7.5 GB. Note the limit: as N grows the per-node traffic approaches 2X = 8 GB regardless of GPU count. That is the bandwidth-optimality claim in numbers: doubling the cluster never doubles anyone's traffic.
> Follow-up: The gradients are 4 MB instead of 4 GB. Does the ring still win?
> A: Not necessarily. Thirty iterations of small messages pay latency each round, and the latency term dominates at small sizes. Bandwidth-optimality is about large transfers. For tiny messages, tree or hierarchical all-reduce with fewer hops can win.

> [!QA]
> Q: When does context parallelism beat the other splits, and what does it cost?
> A: When activations dominate the memory bill: at 128K context with d = 7168, one layer stores about 31.9 GB of activations and 61 layers store about 1.95 TB, exceeding the 1.3 TB weight bill. Splitting the sequence 8 ways cuts each GPU's activation share to 244 GB. The cost: every attention layer now exchanges sequence shards (all-to-all in Ulysses, point-to-point in Megatron CP), converting a local compute problem into a network problem. Reach for it when the sequence is the memory, not when the weights are.
> Follow-up: Why did Ulysses outperform Megatron's sequence parallelism?
> A: The lesson's stated measurement is the whole answer: near-linear strong scaling from 64 to 256 GPUs at 131K sequence length, and it outperforms Megatron's sequence parallelism and Colossal-AI's on the sequences they can run.

> [!QA]
> Q: Applied design: train a 70B model on 64 H100s (8 nodes of 8). Plan the parallelism.
> A: First the memory: 16P = 1.12 TB, so pure data parallelism is out. Tensor parallelism: 8 ways within each node over NVLink, the frequent per-block all-reduces stay on the fast links. Pipeline parallelism: 8 stages across the 8 nodes, point-to-point sends over InfiniBand, microbatches sized so the bubble stays under 20 percent. Data parallelism: the remaining factor across replicas with ZeRO-1 for the optimizer state. If the context is long (128K+), add context parallelism to split the activation bill. The cardinal rules: tensor never crosses nodes, pipeline crosses cheaply, data scales throughput.
> Follow-up: The run is communication-bound on the tensor all-reduces. What do you change?
> A: Reduce the tensor-parallel degree (fewer all-reduce participants per block) and shift the split toward pipeline or data parallelism. Sequence parallelism does not answer this: the lesson teaches it as the fix for activation memory, not for tensor all-reduce traffic. Measure first: the all-reduce time per block against the compute time per block tells you which knob pays.

## Recap: the whole lesson on one screen

The story in ten steps. Each step answers the one before
it.

1. **One GPU is not enough.** Models grow 10x per year.
   GPU FLOPS double every 2.5 years. Split the work, and
   minimize communication and synchronization overhead.
2. **Split the data first.** Data parallelism: 4 GPUs,
   batch 64, 16 examples each, weights replicated,
   gradients averaged by all-reduce. The global batch is
   the product: degree times per-GPU batch.
3. **Training needs 16P bytes.** 2P params, 2P grads,
   12P Adam state in mixed precision. 10B params need
   160 GB: no single GPU. 175B needs 2.8 TB.
4. **ZeRO partitions the redundancy.** Stage 1:
   optimizer state (4P + 12P/Nd). Stage 2: plus
   gradients (2P + 14P/Nd). Stage 3: plus parameters
   (16P/Nd) for 1.5x communication. FSDP2 wraps it.
5. **Ring all-reduce is bandwidth-optimal.** 2(N-1)
   iterations. Per-node traffic approaches 2X regardless
   of GPU count. The collective data parallelism stands
   on.
6. **Split the model next.** Tensor parallelism shards
   attention and MLP weights Megatron-style:
   column-parallel then row-parallel, one all-reduce per
   block. It needs NVLink speed: never span nodes.
7. **Pipeline when the network is slow.** Stages hold
   layer groups. Microbatches flow through. 1F1B and
   interleaved schedules shrink the bubble. DualPipe
   overlaps both directions: near-zero bubble, no TP
   needed at V3 scale.
8. **Split the sequence when activations dominate.**
   Context parallelism (Ulysses all-to-all, Megatron CP):
   1.95 TB of activations at 128K, split 8 ways is 244
   GB per GPU.
9. **Split the experts for MoE.** Expert parallelism:
   64-way over 8 nodes in V3, all-to-all dispatch,
   node-limited routing. EP plus TP requires SP.
10. **Combine and automate.** Fits: data. One fast node:
    tensor. Many nodes: pipeline. Long context:
    context. MoE: expert. 5D at thousand-GPU scale.
    Alpa searches the combinatorial space.

## Go deeper

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/JA1l96tjrs4" title="Deepak Narayanan: Training Large Language Models at Scale" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

- Deepak Narayanan: Training Large Language Models at Scale (assigned talk): https://www.youtube.com/watch?v=JA1l96tjrs4
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models: https://arxiv.org/abs/1910.02054
- Megatron-LM: Training Multi-Billion Parameter Language Models: https://arxiv.org/abs/1909.08053
- DeepSpeed Ulysses: Training Extreme Long Sequence Transformers: https://arxiv.org/abs/2309.14509
- Alpa: Automating Inter- and Intra-Operator Parallelism: https://arxiv.org/abs/2201.12023
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSpeed ZeRO documentation: https://www.deepspeed.ai/tutorials/zero/

## Official sources and further reading

**Official:**
- Parallelism Fundamentals slide deck (Fall 2023
  headers).
- Deepak Narayanan talk (YouTube, JA1l96tjrs4): the
  practitioner view.
- Course text narayanan-parallelism.txt: talk notes.

**Further reading:**
- Rajbhandari et al., "ZeRO" (2019).
- Shoeybi et al., "Megatron-LM" (2019).
- Narayanan et al., "PipeDream" (SOSP 2019) and
  "Memory-Efficient Pipeline Parallelism" (2021).
- Zheng et al., "Alpa" (OSDI 2022).
- Jacobs et al., "DeepSpeed Ulysses" (2023).
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024):
  DualPipe and the no-TP stack.

**Caveats from these sources.** The "10x per year" model
growth and "2.5 year" GPU doubling are trend estimates
from the slides, not laws. The 16P figure excludes
activations, which vary with sequence length and
checkpointing. The 1.5x stage-3 communication factor is
from the ZeRO paper's analysis. Talk timestamps point to
the auto-captioned YouTube video. Wording is the
speaker's paraphrase, verified against the cleaned
captions. The activation estimate (34sbh bytes per
layer) follows Korthikanti et al.

## Connections to the other courses

- **CS336 L07/L08:** data, tensor, pipeline, and 4D/5D
  parallelism derived in full.
- **CS229S L05:** the device block: the map these
  strategies route over.
- **CS229S L03:** communication versus computation: the
  same bottleneck lens at cluster scale.
- **CS229S L08:** FSDP/ZeRO as the distributed answer to
  fine-tuning memory cost.
- **CS229S L09:** the V3 stack: DualPipe, expert
  parallelism, and why no tensor parallelism.
