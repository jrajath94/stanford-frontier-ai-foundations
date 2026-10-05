---
page_id: cs229s-crash
course_slug: cs229s
course_name: "CS229S: Systems for Machine Learning"
course_order: 3
order: 901
nav: "CS229S · Crash course"
title: "CS229S Crash Course"
summary: "Interview-speed review of CS229S: the full systems story in 30 minutes, with images and links into the deep lessons."
---

<span class="crash-timer">30 minutes · interview speed</span>

This page tells the whole systems story fast. Each section gives you
the working version: enough to answer interview questions with
confidence. Links at the end of each section take you into the full
lesson when you want the derivations and the follow-ups.

<div class="crash-section" markdown="1">

### 1. Three gaps drive the whole course

Deep learning compute demand grows 32x every 2 years; Moore's law
gives 2x. Model size outruns accelerator memory: GPT-4 scale towers
over the A100's 40 or 80 GB. And asymptotic complexity is not
wall-clock speed: a linear-time attention algorithm loses to
hardware-aware FlashAttention in measured runtime. Systems for ML
closes all three gaps.

Training is a large upfront bill: tens of millions of dollars for a
frontier model. Inference is cheap per call, under $0.0001, but it
compounds with every user. Design for the bill you pay.

<figure class="crash-fig"><img src="assets/media-generation-plate-chapter-l01-fullstack-0-ba96d864-a7fa-4e8c-b343-e346c5aa61cf.webp" alt="Full-stack efficiency"><figcaption>Data, model, software, hardware: every layer gets its own efficiency lever.</figcaption></figure>

<ul class="crash-links">
<li><a href="l01-introduction.html">Lecture 1: the economic case</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 2. The transformer block you are optimizing

A language model factors into next-token predictions via the chain
rule. The transformer mixes tokens with attention: queries look up
keys, values mix by similarity. Every token meets every other
token, which costs O(N squared). That single cost is what the rest
of the course attacks.

<figure class="crash-fig"><img src="assets/slide-l02-attention-math.png" alt="Attention math"><figcaption>O = QK^T: O(1) interaction distance, parallelizable, quadratic cost.</figcaption></figure>

<ul class="crash-links">
<li><a href="l02-sequence-models.html">Lecture 2: the systems framing</a> (deep mechanics in <a href="../cs336/l03-architecture.html">CS336 L03</a>)</li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 3. Diagnose first: memory bound or compute bound

Total time is max(Tmem, Tmath). Arithmetic intensity is FLOPs per
byte; the device ridge is peak FLOPs over peak bandwidth. On an
A100 the ridge is 161 FLOPs per byte. Below it you are memory
bound: move less data. Above it you are compute bound: do more
math. A matmul with N=128 sits at 124 (memory bound); with N=8192
it sits at 2731 (compute bound). Same kernel, different shape,
different fix.

<figure class="crash-fig"><img src="assets/plate-roofline.svg" alt="Roofline"><figcaption>Left of the ridge, buy bandwidth. Right of the ridge, buy compute.</figcaption></figure>

<ul class="crash-links">
<li><a href="l03-hardware-aware-design.html">Lecture 3: the decision procedure</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 4. The device block

One GPU: 108 SMs, registers, shared memory, L2, 40 GB of HBM.
One node: 8 GPUs on NVLink. One cluster: nodes on InfiniBand, much
slower. Kernels load from HBM, compute on-chip, write back.
Threads run in warps of 32 under SIMT. Tiling keeps data on-chip;
tensor cores multiply 16 by 16 tiles up to 16x faster. Frequent
collectives stay inside the node; rare ones span the cluster.

<figure class="crash-fig"><img src="assets/plate-device-block.svg" alt="Device block"><figcaption>Defined here; MS&E435 reuses this symbol.</figcaption></figure>

<ul class="crash-links">
<li><a href="l05-gpu-execution-model.html">Lecture 5: the device block</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 5. Training costs 3x a forward pass; decoding is memory bound

Forward is 2BHN per MLP layer; backward is 4BHN, so a training
step costs about 3x one forward pass, plus activation storage.
Decoding reads the whole model per token: over 3 GB per GPT-2-XL
token. Small batches are memory bound; batching amortizes the
weight reads. A 7B model on a 40 GB A100 caches 130k tokens:
batch 128 at length 1024. Memory sets the serving limit, not
FLOPs.

<figure class="crash-fig"><img src="assets/media-generation-plate-chapter-l04-inference-0-45c8f33f-3e8d-41ee-a2d5-173728fe9c9a.webp" alt="Inference chapter plate"><figcaption>Cache and guess when memory bound; batch when you can.</figcaption></figure>

<ul class="crash-links">
<li><a href="l04-transformer-performance.html">Lecture 4: FLOP counting and KV caching</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 6. Speculative decoding: guess, then verify

Decoding is memory bound, so extra compute is nearly free. A small
draft model (about 15x smaller) guesses the next tokens; the big
model checks them all in one batch; the agreed prefix is accepted
for free. Result: 2 to 3x speedup on 11B to 137B models. Wrong
guesses cost little because the batch was memory bound anyway.

<ul class="crash-links">
<li><a href="l04-transformer-performance.html">Lecture 4: the three draft strategies</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 7. FlashAttention: never write the N by N matrix

Standard attention is memory bound: the cost is reads and writes
around the score matrix, not the FLOPs. FlashAttention tiles Q, K,
V through SRAM, tracks a running softmax sum and max per tile, and
recomputes tiles in the backward pass. Exact output. 9x less HBM
traffic, 13% more FLOPs, 6x faster. Lower FLOPs do not imply
wall-clock speedup.

<figure class="crash-fig"><img src="assets/media-generation-plate-chapter-l06-flashattn-0-fef314f8-620b-419e-bef8-42397039204e.webp" alt="FlashAttention chapter plate"><figcaption>Fusion plus tiling plus recompute beats lower FLOPs in wall clock.</figcaption></figure>

<ul class="crash-links">
<li><a href="l06-flash-attention.html">Lecture 6: the full derivation</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 8. Shrink the model: prune, quantize, distill

Pruning removes weights; only structured patterns (like 2:4 N:M)
speed up hardware. Quantization spends fewer bits: k-means stores
indices plus centroids (3.2x on the toy), linear quantization runs
integer math via scale and zero-point. Distillation trains a small
student on the teacher's probability distributions. Each trades a
little accuracy for a lot of memory; pruning plus quantization
compose.

<figure class="crash-fig"><img src="assets/media-generation-plate-chapter-l07-memory-0-e99a1302-0e56-463a-917c-27764af8b66a.webp" alt="Memory chapter plate"><figcaption>Three cuts, one smaller model.</figcaption></figure>

<ul class="crash-links">
<li><a href="l07-memory-efficient-networks.html">Lecture 7: all numbers worked</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 9. Fine-tune the behavior, then fine-tune the cost

Base models complete text; instruction tuning teaches following
instructions. RLHF trains a reward model on human preference
rankings and optimizes it; Constitutional AI replaces the labels
with a short human-written constitution and AI feedback. Then the
systems problem: fine-tuning costs over 10x the weights in memory.
PEFT tunes almost nothing: prompt tuning trains input embeddings,
LoRA trains low-rank adapters that merge into the weights with
zero inference latency.

<ul class="crash-links">
<li><a href="l08-finetuning-and-peft.html">Lecture 8: RLHF, RLAIF, and PEFT</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 10. Attention-free architectures and the recall test

Convolutions via the FFT run in O(N log N); linear recurrences
train as convolutions and sample in O(1). S4 solved the long-range
benchmarks but lagged in language perplexity. The diagnosis:
recall needs input-dependent mixing, and convolutional mixing is
fixed. The direction is sub-quadratic plus selective: Mamba, gated
linear attention. Measure efficiency on the task, not on
asymptotics.

<ul class="crash-links">
<li><a href="l09-efficient-architectures.html">Lecture 9: FFT, S4, and recall</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 11. Parallelism: split what does not fit

Data parallelism splits the batch and all-reduces gradients; the
ring version is bandwidth-optimal. Mixed-precision Adam needs 16P
bytes; ZeRO partitions it down to 16P/Nd. Tensor parallelism
shards weights Megatron-style and must stay inside NVLink.
Pipeline parallelism stages layers and flows microbatches; 1F1B
shrinks the bubble. Combine all three (PTD) at thousand-GPU
scale; Alpa automates the search.

<figure class="crash-fig"><img src="assets/plate-pipeline-schedule.svg" alt="Pipeline schedule"><figcaption>Microbatches and 1F1B keep stages busy.</figcaption></figure>

<ul class="crash-links">
<li><a href="l10-parallelism.html">Lecture 10: the decision table</a> (deep mechanics in <a href="../cs336/l07-parallelism.html">CS336 L07</a> and <a href="../cs336/l08-4d-parallelism.html">L08</a>)</li>
</ul>

</div>

<div class="crash-section" markdown="1">

### Rapid-fire Q&A

**Q: Memory bound or compute bound, and how do you know?**
A: Compare arithmetic intensity (FLOPs/byte) to the ridge (161 on
A100). Below: memory bound, move less data. Above: compute bound,
do more math.

**Q: Why is decoding slow?**
A: One token costs a full model read. Small batches are memory
bound. Fix with batching, KV caching, and speculative decoding.

**Q: What is FlashAttention in one sentence?**
A: Exact attention that never materializes the N by N matrix:
tile through SRAM, rescale the softmax online, recompute
backward. 9x less traffic, 6x faster.

**Q: When does KV caching win?**
A: When compute bound: the extra memory traffic is the cheaper
side. When memory bound, recomputation can win instead.

**Q: LoRA vs full fine-tuning for 50 tasks?**
A: One 14 GB base plus 50 tiny adapters versus 50 copies of 14
GB. The adapters merge into the weights, so inference pays
nothing.

**Q: Why did sparsity need 2:4 structure?**
A: Unstructured pruning scatters memory access and rarely speeds
up. 2:4 matches Ampere sparse tensor cores: regular, fast, 50%
fewer weights.

**Q: Data, tensor, or pipeline parallelism?**
A: Model fits: data. One fast node: tensor (NVLink only). Many
nodes: pipeline. Combine at scale.

**Q: Why do convolutions lose to attention on recall?**
A: Recall needs input-dependent mixing. Attention adapts per
example; convolutional mixing is fixed. Flexibility beats
asymptotics.

</div>
