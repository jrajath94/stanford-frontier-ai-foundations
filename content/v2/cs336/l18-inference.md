---
page_id: cs336-l18
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 18
nav: "L18 · Serving Inference"
title: "Lecture 18: Serving Inference"
summary: "The other side of the model: the lifetime of a token, prefill vs decode, KV caching, disaggregation, Megakernels, and Parcae loop transformers."
date: "2026-05-27"
instructor: "Dan Fu"
offering: "Spring 2026"
duration: "1:11:33"
video_id: 9EEm4iMAF5s
video_title: "Stanford CS336 Language Modeling from Scratch | Spring 2026 | Guest Lecture: Dan Fu"
video_caption: "Guest lecture by Dan Fu (UCSD / Together AI): serving models and the research behind it."
concepts: [inference, prefill, decode, kv-cache, continuous-batching, disaggregation, megakernels, parcae, loop-transformers]
sources:
  - tag: video
    label: "Lecture 18 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=9EEm4iMAF5s
  - tag: notes
    label: "Official subtitle transcript (en-US)"
---

## How to read this lesson

A guest lecture flips the course: from training models to serving
them. **Level 1 (Core):** the token's lifetime, prefill vs decode,
the KV cache. **Level 2 (Deep):** disaggregation, Megakernels,
Parcae.

## Level 1: The lifetime of a token

Inference is the engine that turns electricity into intelligence,
the way a car engine turns oil into motion
[05:09](ts:05:09). The whole inference stack in one loop:
schedule the request to GPUs, check the KV cache for seen tokens,
execute the model, sample a token, repeat
[07:34](ts:07:34).

![Lifetime](assets/l18-lifetime.svg "Request to intelligence, one scheduling loop.")

Know the kernels and you enable full-stack innovation: new routing,
new kernels, new architectures.

## Level 1: Workloads

Production traffic does not look like training traffic. Coding
agents send tens of thousands of input tokens and get short outputs
[10:24](ts:10:24). Chat wants the first token in under a second.
Agentic loops have many turns, tool calls, and idle gaps between
turns [11:21](ts:11:21). Batch jobs translate each document once.
The workload shape, and its SLA, decides every systems choice.

![Workloads](assets/l18-workloads.svg "Coding, chat, agents: different shapes, different SLAs.")

> [!QA]
> Q: Why would one inference stack not serve all workloads optimally?
> A: Because the workloads stress different bottlenecks. A coding agent needs huge KV caches kept hot across turns (memory capacity). A chat app needs low time-to-first-token (prefill latency). Batch translation needs throughput on first-token processing with almost no decode pressure. A stack tuned for one wastes the other's dominant resource. This is why the lecture ends on co-design: architectures like DeepSeek's MLA radically compress the KV cache precisely because agentic workflows are KV-cache bound, while BERT-era batch jobs could use cheap bidirectional encoders (the guest's example) since they barely decode at all.
> Follow-up: What is the single most important number for each workload?
> A: Interactive chat: time to first token (under a second keeps the user). Agents: KV cache capacity per session (hot cache across turns). Batch: tokens per second per dollar (throughput). The lecturer's rule: nobody ever asks to serve less traffic, subject to those SLAs.

## Level 1: Prefill vs decode

Two operations, two bottlenecks. Prefill: 10,000 never-seen tokens
in, compute the activations and one logit out. Compute bound, like
training without the backward pass [14:24](ts:14:24). Decode:
generate one token at a time, reloading the whole model for each
one. Memory bandwidth bound, light on flops [14:54](ts:14:54).

![Prefill vs decode](assets/l18-prefill-decode.svg "Compute-bound once, memory-bound forever.")

Prefill takes longer. Decode runs far more steps (once per
generated token). The standard move: disaggregate, run prefill on
one fleet, decode on another, specialize each stack
[20:00](ts:20:00). Hardware follows the split: NVIDIA plans GPUs for
prefill and Groq-style LPU chips for decode. OpenAI partners with
Cerebras for the same reason [21:07](ts:21:07).

## Level 1: The KV cache

Users repeat themselves. Every "hi ChatGPT" shares a prefix. Every
conversation turn repeats the book you pasted. Prefix sharing with
a radix tree: look up which tokens you have seen, compute only the
new ones [18:02](ts:18:02).

![KV cache](assets/l18-kv-cache.svg "Do not recompute what you already computed.")

When the GPU fills, offload: GPU to CPU DRAM to SSD, each tier
cheaper and slower [25:43](ts:25:43). Evict with LRU, prefetch when
a user reopens an old conversation. Classic OS paging, rediscovered
for activations [27:51](ts:27:51).

## Level 2: Cache-aware disaggregation

Two lines of routing code, 40% faster serving. Most requests are
mid-conversation (warm). About 10% are fresh (someone pasted a
book). Never run the book-paste on the same GPUs as the short
chat: route low cache-hit-rate requests to one prefill pool, warm
requests to another [31:31](ts:31:31).

![Disaggregation](assets/l18-disaggregation.svg "Route by cache hit rate. Two lines, 40% faster.")

Early days: in ten years this will look obvious. Scale bugs are the
tax: 0.001%-probability kernel errors that NaN logits into "hi hi
hi" loops, off-by-one reads that emit random Chinese characters,
tool-call handlers that loop for tens of thousands of tokens
[21:54](ts:21:54).

## Level 2: Megakernels

Decode loads the entire model to make one token, so the GPU is a
glorified memory loader [34:27](ts:34:27). Worse, one kernel per op
leaves streaming multiprocessors idle: launch gaps, tail effects,
gaps between kernels add up. The Megakernel fuses many ops into one
kernel and schedules them like a distributed system: start the KV
load before QKV-plus-RoPE finishes, load the O-projection weights
before attention ends [37:32](ts:37:32).

![Megakernels](assets/l18-megakernel.svg "Fuse the layer. Overlap everything. Near the speed of light.")

Built with ThunderKittens, a lower-level Triton-like kernel
library. Payoff: 30 to 70% on attention inference, one full Llama-1B
layer in a single kernel, 72% of peak memory bandwidth on H100
[40:56](ts:40:56). The cost: one kernel engineer writes
Megakernels for one hardware, two or three models, batch sizes 1
to 16, per year. Batch 17 means starting over [63:37](ts:63:37).

## Level 2: Parcae

A different scaling question: is more parameters the only way?
Parcae loops transformer blocks, reusing parameters for more flops.
Naive looping blows up (the A matrix powers amplify activations).
The fix from state-space theory: make A a negative diagonal matrix
so the spectral radius stays under 1, and the system cannot explode
[50:50](ts:50:50).

![Parcae](assets/l18-parcae.svg "Loops that do not blow up. Scale recurrence with data.")

The result: stable training where unconstrained loops NaN, and
better perplexity than the same-parameter transformer baseline
[53:31](ts:53:31). The scaling law finding: as data grows, scale
recurrence alongside parameters. Today's models have zero
recurrence and tons of data, sitting at the far left of the curve,
which suggests pre-training may be under-looping [58:15](ts:58:15).
Inference bonus: fewer parameters means more KV cache and less
cross-GPU communication [62:03](ts:62:03).

## Level 2: Co-design

Inference closes the loop back to architecture. Size your model to
the chip's memory (a Groq LPU holds ~250MB). Match quantization to
the hardware (NVFP4 on NVIDIA, MXFP4 on AMD) [64:47](ts:64:47).
Compress the KV cache if you serve agents (MLA). Skip the decoder
if you serve batch (BERT on search). The model's shape and the
fleet's shape are one decision.

| Serving target | Model choice |
|---|---|
| Groq LPU (~250MB) | size the model to fit with KV cache room |
| NVIDIA GPUs | train in NVFP4 |
| AMD | use MXFP4 |
| agentic workloads | compress the KV cache (MLA) |
| batch workloads | skip the decoder (BERT-style) |

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l18-lifetime.svg" alt="Lifetime">
<div class="rc-body">
<strong>1. Token lifetime</strong>
<p>Schedule, check KV cache, execute, sample, repeat. Electricity
to intelligence.</p>
<p class="rc-num">Key: the serving loop</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l18-workloads.svg" alt="Workloads">
<div class="rc-body">
<strong>2. Workloads</strong>
<p>Coding: long in. Chat: fast first token. Agents: turns and
tools. SLAs differ.</p>
<p class="rc-num">Key: shape decides system</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l18-prefill-decode.svg" alt="Prefill decode">
<div class="rc-body">
<strong>3. Prefill vs decode</strong>
<p>Prefill compute-bound, once. Decode memory-bound, per token.
Split the fleets.</p>
<p class="rc-num">Key: two bottlenecks</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l18-continuous-batching.svg" alt="Batching">
<div class="rc-body">
<strong>4. Batching</strong>
<p>Requests join and leave mid-step. KV memory is the hard
limit. Queue when full.</p>
<p class="rc-num">Key: per-step, not per-request</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l18-kv-cache.svg" alt="KV cache">
<div class="rc-body">
<strong>5. KV cache</strong>
<p>Radix-tree prefix sharing. GPU, CPU, SSD tiers. LRU
eviction, prefetch.</p>
<p class="rc-num">Key: never recompute</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l18-disaggregation.svg" alt="Disaggregation">
<div class="rc-body">
<strong>6. Routing</strong>
<p>Route by cache hit rate. Fresh and warm never share GPUs.
40% faster.</p>
<p class="rc-num">Key: two lines of code</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l18-megakernel.svg" alt="Megakernels">
<div class="rc-body">
<strong>7. Megakernels</strong>
<p>Fuse ops into one kernel. Overlap loads. 30-70% faster,
72% of peak bandwidth.</p>
<p class="rc-num">Key: kill the gaps</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l18-parcae.svg" alt="Parcae">
<div class="rc-body">
<strong>8. Parcae</strong>
<p>Loop blocks, constrain spectral radius below 1. Scale
recurrence with data.</p>
<p class="rc-num">Key: flops without parameters</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 18 video (guest lecture, Dan Fu, UCSD / Together AI).
- Papers named in lecture: Megakernels/ThunderKittens (Stanford and Together), Parcae (UCSD).

**Further reading:**
- DeepSeek V3/V4 inference papers (fused MoE kernels, MLA).
- vLLM and TensorRT-LLM docs (continuous batching, disaggregation).

**Caveats from these sources.** The lecturer's slides were AI
generated. He warned that fine details may be wrong, and this
lesson follows his spoken claims, not the slide pixels. Scale
figures (trillion-parameter open models, 5 to 10 trillion frontier)
are his 2026 estimates, not measurements. The Parcae Twitter story
was a fabrication the poster retracted.

## Connections to the other courses

- **CS336 Lecture 4:** flash attention and kernel writing meet
Megakernels.
- **CS336 Lecture 9:** scaling laws extended to recurrence.
- **CS336 Lecture 16:** RL rollouts are decode-heavy inference
workloads.

> [!CHEAT]
> **Inference cheatsheet.** Token lifetime: schedule, KV lookup, execute, sample, repeat. Workloads: coding (long in), chat (fast first token), agents (turns), batch (throughput). Prefill: compute bound, 10k in 1 out, once. Decode: memory bound, 1 token per full model load, per step. Disaggregate fleets. LPU/Cerebras for decode. Continuous batching: per-step joins, KV memory is the limit. KV cache: radix prefix sharing, GPU/CPU/SSD tiers, LRU, prefetch. Cache-aware routing: fresh vs warm pools, 40% faster. Megakernels: fuse ops, overlap loads, 30-70% speedup, 72% bandwidth, huge engineering cost. Parcae: loop blocks, spectral radius under 1, scale recurrence with data. Co-design: size to chip memory, match quantization, compress KV for agents.

> [!MEMORY]
> **Serving is a second training problem.** Every flop saved in training is re-spent at inference. The system that generates the tokens is half the model.
