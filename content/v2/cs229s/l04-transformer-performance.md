---
page_id: cs229s-l04
course_slug: cs229s
course_name: "CS229S: Systems for Machine Learning"
course_order: 3
order: 4
nav: "L04 · Transformer Performance"
title: "Lecture 4: Analyzing the Performance of Transformers"
summary: "Count the FLOPs of training and inference by hand: backprop costs 2x the forward pass, KV caching trades memory traffic for compute, and speculative decoding exploits the memory-bound regime."
date: "2024-10-04"
instructor: "Azalia Mirhoseini"
offering: "Fall 2024"
concepts: [flops-counting, backpropagation, kv-caching, arithmetic-intensity, speculative-decoding, throughput, latency, batch-size, medusa]
sources:
  - tag: slides
    label: "Analyzing the Performance of Transformers slide deck (Fall 2023 headers)"
  - tag: paper
    label: "Leviathan, Kalman et al., Fast Inference from Transformers via Speculative Decoding (2023)"
    url: https://arxiv.org/abs/2211.17192
  - tag: paper
    label: "Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling (2023)"
    url: https://arxiv.org/abs/2302.01318
  - tag: supplement
    label: "Carol Chen, Transformer Inference Arithmetic (2022)"
    url: https://blog.eleuther.ai/transformer-math/
---

## How to read this lesson

**Level 1 (Core)** counts the cost of training and inference with
the lecture's own numbers: the 2x backprop rule, the KV cache
capacity of a 7B model, and why decoding is memory bound.
**Level 2 (Deep)** derives the arithmetic-intensity argument for KV
caching and the three speculative-decoding draft strategies.
Inference mechanics continue in [CS336 L10](../cs336/l10-inference.html);
the **KV block** symbol is defined there and reused here.

## Level 1: Backprop costs twice the forward pass

Training minimizes a loss by stepping parameters downhill. The
gradient measures sensitivity: how much the loss changes when a
parameter wiggles. Backpropagation computes it with the chain rule
on the computation graph.

Toy graph: L(w1, w2, w3) = 2 (w1 w2) w3 with u = w1 w2, v = 2u,
L = v w3. Forward: w1=2, w2=5, w3=3 gives u=10, v=20, L=60.
Backward multiplies local derivatives right to left: dL/dw1 =
(dL/dv)(dv/du)(du/dw1). Two facts matter for systems.

**Reuse.** Intermediate derivatives like dL/dv appear in several
parameter gradients. Efficient backprop reuses them instead of
recomputing.

**Storage.** To compute dL/dw3 you need v. Keeping activations for
the backward pass makes backpropagation memory-intensive. This
storage cost is why activation checkpointing exists.

Now count an MLP layer's FLOPs. A middle layer with batch B, hidden
H, output N needs 2BHN FLOPs forward: B times H times N
multiply-adds, times 2 for multiply plus add.

![FLOPs rule](assets/slide-l04-flops-rule.png "Forward: 2BHN. Rule of thumb: about 2 x batch x parameter count. Source: Stanford slides.")

The rule of thumb: forward FLOPs are about 2 x B x (number of
parameters). The backward pass needs two more matmuls per layer:
dL/dW2 = a1^T dL/dz2 (2HBN) and dL/da1 = dL/dz2 W2^T (2BHN), plus
the gradient through the nonlinearity. Total: 4BHN, about twice
the forward pass.

![Backprop FLOPs](assets/slide-l04-backprop-flops.png "Backward: 4BHN, about 2x the forward 2BHN. Activations must be cached. Source: Stanford slides.")

> [!QA]
> Q: How many FLOPs does training cost relative to one forward pass?
> A: About 3x per step: 1x for the forward pass plus 2x for the backward pass. The lecture's MLP accounting gives 2BHN forward and 4BHN backward per layer. The backward pass also forces you to store activations, which is the memory cost of training.
> Follow-up: Why does the backward pass need the activations at all?
> A: The chain rule multiplies upstream gradients by local derivatives evaluated at the forward values. For dL/dW2 = a1^T dL/dz2 you need a1, the layer input. Without storing it you must recompute it. Storage versus recomputation is the same tradeoff as KV caching below.

## Level 1: Autoregressive decoding repeats work

Generation predicts one token, appends it, and repeats. Without
caching, each step recomputes keys and values for the entire
prefix: computing K and V for the full sequence costs 2 x (2Nd^2)
per token, plus Q for the current token, QK^T (2Nd), and the
softmax-weighted sum (2Nd).

KV caching removes the repeated work. Keys and values of earlier
tokens never change, because each token only depends on tokens
before it. So compute K, Q, V for the current token only (3 x 2d^2),
score against the cached keys (2Nd), and mix the cached values
(2Nd). The cache grows by one entry per step.

The **KV block** (defined in [CS336 L10](../cs336/l10-inference.html),
reused here without redrawing) is the stored object: per token, per
layer, two vectors of dimension dmodel. In FP16 that is
2 x 2 x nlayers x dmodel bytes per token.

![KV block, defined in CS336](../cs336/assets/l10-kv-cache.svg "The KV block symbol, first defined in CS336 L10 and reused here. Source: CS336.")

## Level 1: How many tokens fit on one GPU

Worked example from the slides. A100 40 GB, 7B-parameter model with
24 layers and dmodel 2048, sequence length 1024.

1. Weights: 7e9 x 2 bytes = 14 GB.
2. Leftover for KV cache: 40 - 14 = 26 GB.
3. Per token: 2 x 2 x 24 x 2048 = 200k bytes = 0.0002 GB.
4. Capacity: 26 / 0.0002 = 130k tokens.
5. Batch size at 1024 tokens each: 130k / 1024 = 128.

![KV cache capacity](assets/slide-l04-kvcache-capacity.png "14 GB of weights leaves 26 GB; at 200k bytes per token that is 130k tokens, or batch 128 at length 1024. Source: Stanford slides.")

Memory, not compute, sets the serving batch size. This is the
calculation behind every serving-system capacity plan.

> [!QA]
> Q: A 7B model on a 40 GB A100 serves 1024-token sequences. What batch size fits?
> A: Weights take 14 GB, leaving 26 GB. Each token's KV cache costs 2 x 2 x 24 x 2048 = 200k bytes, so 26 GB holds about 130k tokens. At 1024 tokens per sequence that is batch size 128. The limit is HBM capacity, not FLOPs.
> Follow-up: What changes first if you double the sequence length?
> A: The KV cache per sequence doubles, so the batch size halves to 64. Longer contexts trade directly against concurrency on fixed memory.

## Level 1: Small batches are memory bound

Per-token decoding reads the whole model to produce one token: for
GPT-2-XL (1.5B params, 48 layers), each decode step loads about
24 MB for attention plus 41 MB for the MLP per layer, over 3 GB of
memory traffic per token. Arithmetic intensity is about 2 x batch:
tiny.

![Decode cost](assets/slide-l04-decode-cost.png "Over 3 GB read per token for GPT-2-XL. An RTX 4090 has compute for 30,000 tokens/s but decodes about 300. Source: Stanford slides.")

An RTX 4090 has the compute for roughly 30,000 tokens per second
but decodes only about 300: the memory bandwidth caps it. Small
batches spend all their time loading weights and process little
data. Large batches raise the FLOP count without re-reading
weights, moving toward the compute-bound regime. Batching is the
first serving optimization because it amortizes the weight reads.

Throughput, latency, and bandwidth are three different things.
Latency is time per item: what interactive users feel. Throughput is
items per second: what batch pipelines maximize. Bandwidth is the
hardware property: the ceiling neither can exceed. Do not trade one
vocabulary for another in an interview.

## Level 1: Speculative decoding

Decoding is memory bound, so extra compute is nearly free. The
idea: guess the next few tokens with a cheap draft, verify all
guesses in one big-model batch, keep the agreed prefix. Right
guesses are free tokens; wrong guesses cost almost nothing because
the batch was memory bound anyway. This is branch prediction for
language models.

![Speculative decoding](assets/plate-speculative-decoding.svg "Shell 4. Draft, verify in one batch, accept the agreed prefix. Source: original plate; Leviathan & Kalman et al. 2023.")

Procedure: run the prompt through the model once (prefill), saving
the KV cache. Then loop: draft K tokens, run the big model on
prompt plus drafts in parallel, check agreement token by token,
accept the longest agreeing prefix plus one fresh token.

Results: 2 to 3x speedup on Chinchilla 70B, T5 11B, and LaMDA 137B
(Leviathan and Kalman et al., 2023; Chen et al., 2023).

![Speculative results](assets/slide-l04-speculative-results.png "2-3x speedups on Chinchilla 70B, T5 11B, LaMDA 137B. Source: Stanford slides.")

> [!QA]
> Q: Explain speculative decoding and when it helps.
> A: A small draft model guesses the next few tokens. The big model verifies all guesses in one parallel forward pass, which costs about the same as one decode step because decoding is memory bound. Agreed tokens are accepted for free; the first disagreement is resampled. It helps exactly when decoding is memory bound with small batches: the verification batch soaks up idle compute. Reported speedups are 2 to 3x on 11B to 137B models.
> Follow-up: What limits the speedup?
> A: Draft agreement. If guesses are usually wrong, verification still costs a full batch and yields one token. Guessing also cannot be free: a bigger draft model agrees more but costs more per guess. The optimum the slides cite is a draft about 15x smaller than the main model.

## Level 2: The arithmetic-intensity argument for KV caching

KV caching trades memory traffic for saved compute. When does the
trade win? Compare per-token costs on an A100 (312e12 FLOPs/s
compute, 1.5e12 bytes/s memory in the slides' numbers).

Computing K and V for one token: (2 x nlayers x 2 x dmodel^2) /
312e12 seconds. Reading them from HBM instead: (2 x 2 x nlayers x
dmodel^2) / 1.5e12 seconds. The ratio Tmem / Tmath at dmodel 2048
is 312e12 / 1.5e12 = 208. Computing the KV for one token takes the
same time as the memory traffic of 208 tokens' worth. Below 208
tokens you are memory bound; above, compute bound.

```
Tmath(1 token) = (2 x L x 2 x d^2) / 312e12
Tmem(1 token)  = (2 x 2 x L x d^2) / 1.5e12
Tmem / Tmath   = 312e12 / 1.5e12 = 208   (d = 2048)
```

The lecture's punchline: KV caching makes sense when you are
compute bound, because then the extra memory traffic is the cheaper
side. Recall from Lecture 3: if you are not compute bound, extra
computation is free, so recomputation can beat caching. The same
rule, applied to the cache.

## Level 2: Three ways to draft

**Smaller draft model.** Train a small model on the same data. Pros:
transformers are well calibrated, so agreement is good; draft size
is tunable (about 15x smaller seems optimal); same code runs it.
Cons: another model to manage in infrastructure; agreement is
imperfect since it is a different model; it consumes extra memory.

**Medusa heads.** Train extra heads that predict the token after
next, and the one after that, directly. Pros: nearly free at
inference, easy to train and deploy. Cons: it approximates a joint
distribution with a mean-field factorization, so guesses degrade
fast. Good for about 2x, unlikely to reach 4x.

**Lossy optimization.** Make the model itself fast and reckless:
INT4 quantization, skipped layers, skipped heads, early exit. Pros:
no extra model; highly correlated with the full model; training
already optimizes for training cost more than inference cost, so
headroom exists. Cons: complicated code. Probably underexplored.

![Guessing options](assets/slide-l04-guessing-options.png "Draft model, Medusa heads, lossy optimization: the three guessing strategies and their tradeoffs. Source: Stanford slides.")

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/slide-l04-flops-rule.png" alt="FLOPs rule">
<div class="rc-body">
<strong>1. Forward is 2 x batch x params</strong>
<p>Each multiply-add counts twice. The rule of thumb estimates any
layer's forward FLOPs from its parameter count.</p>
<p class="rc-num">Key: 2BHN per MLP layer</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l04-backprop-flops.png" alt="Backprop FLOPs">
<div class="rc-body">
<strong>2. Backprop costs 2x the forward pass</strong>
<p>Two extra matmuls per layer: 4BHN. Training stores activations,
which is the memory cost of the backward pass.</p>
<p class="rc-num">Key: 1x forward + 2x backward</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l04-kvcache-capacity.png" alt="KV cache capacity">
<div class="rc-body">
<strong>3. KV caching removes repeated work</strong>
<p>Past keys and values never change, so cache them. Per token:
compute 3 x 2d^2, score 2Nd, mix 2Nd. The KV block is the unit.</p>
<p class="rc-num">Key: 200k bytes per token at 7B</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l04-kvcache-capacity.png" alt="Serving capacity">
<div class="rc-body">
<strong>4. Memory sets the batch size</strong>
<p>7B on 40 GB A100: 14 GB weights, 130k cacheable tokens, batch
128 at length 1024. HBM capacity is the serving limit.</p>
<p class="rc-num">Key: 130k tokens, batch 128</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l04-decode-cost.png" alt="Decode cost">
<div class="rc-body">
<strong>5. Decoding is memory bound</strong>
<p>Over 3 GB read per GPT-2-XL token. The RTX 4090 could compute
30,000 tokens/s but decodes 300. Batching amortizes weight reads.</p>
<p class="rc-num">Key: AI about 2 x batch</p>
</div>
</div>
<div class="recap-card">
<img src="assets/plate-speculative-decoding.svg" alt="Speculative decoding">
<div class="rc-body">
<strong>6. Speculative decoding guesses, then verifies</strong>
<p>Draft K tokens cheaply, check them in one big-model batch.
Right guesses are free; wrong ones cost little. 2 to 3x faster.</p>
<p class="rc-num">Key: branch prediction for LLMs</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l04-guessing-options.png" alt="Guessing options">
<div class="rc-body">
<strong>7. Three draft strategies</strong>
<p>Small draft model (15x smaller is sweet), Medusa heads (cheap,
caps near 2x), lossy tricks (no extra model, complex code).</p>
<p class="rc-num">Key: agreement limits speedup</p>
</div>
</div>
<div class="recap-card">
<img src="assets/media-generation-plate-chapter-l04-inference-0-45c8f33f-3e8d-41ee-a2d5-173728fe9c9a.webp" alt="Inference chapter plate">
<div class="rc-body">
<strong>8. Inference is a bandwidth game</strong>
<p>Latency serves users, throughput serves pipelines, bandwidth
caps both. Cache and guess when memory bound; batch when you can.</p>
<p class="rc-num">Key: cache, batch, speculate</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Analyzing the Performance of Transformers slide deck (Fall 2023 headers).
- Carol Chen, "Transformer Inference Arithmetic" (2022): the Tmem/Tmath 208 derivation.

**Further reading:**
- Leviathan, Kalman et al., "Fast Inference from Transformers via Speculative Decoding" (2023).
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).
- FasterDecoding/Medusa (GitHub): the Medusa heads implementation.
- [CS336 L10](../cs336/l10-inference.html): the KV block defined; inference systems in full.

**Caveats from these sources.** The A100 memory bandwidth in the
KV-caching derivation is 1.5e12 bytes/s in the slides (1.5 TB/s)
versus 1,935 GB/s in Lecture 3's spec slide; both orderings give
the same conclusion (memory bound at small batch). The 208 ratio
assumes dmodel 2048. The 130k-token capacity ignores attention
workspace and fragmentation. Medusa's 2x cap and the 15x draft size
are empirical rules from the slides, not theorems.

## Connections to the other courses

- **CS336 L10:** the KV block symbol, defined there and reused here.
- **CS336 L02:** FLOP counting and the backward-pass 2x rule, from scratch.
- **CS229S L03:** arithmetic intensity and the ridge; the 208 ratio is that machinery.
- **CS229S L06:** FlashAttention attacks the same memory-bound attention.
- **CS229S L07:** quantization shrinks the weights that decoding keeps re-reading.

> [!CHEAT]
> **Transformer performance cheatsheet.** Forward: 2BHN per MLP layer; rule of thumb 2 x B x params. Backward: 4BHN, 2x forward; must cache activations. KV per token: compute 2 x nlayers x 2 x dmodel^2 FLOPs; memory 2 x 2 x nlayers x dmodel bytes. A100: Tmem/Tmath = 208 at dmodel 2048. KV caching wins when compute bound. 7B/24L/2048d on 40 GB: 14 GB weights, 200k B/token, 130k tokens, batch 128 at 1024 length. Decode AI ~ 2 x batch; GPT-2-XL reads >3 GB/token; RTX 4090: 30k tokens/s compute, ~300 tokens/s actual. Latency: time per item. Throughput: items per second. Bandwidth: hardware ceiling. Speculative decoding: draft, verify in one batch, accept prefix; 2-3x on 11B-137B. Drafts: small model (~15x), Medusa heads (~2x cap), lossy tricks.

> [!MEMORY]
> **Decode math in one line.** One token costs a full model read; batching and guessing are the only ways to pay less per token.
