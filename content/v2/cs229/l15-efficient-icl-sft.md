---
page_id: cs229-l15
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 15
nav: "L15 · Efficient Attention, ICL, SFT"
title: "Lecture 15: Attention Variants, In-Context Learning, and SFT"
summary: "Making attention cheaper (KV cache, GQA, MoE), the shock of few-shot learning, and supervised fine-tuning as instruction tuning."
date: "2026-05-25"
instructor: "Tengyu Ma"
offering: "Spring 2026"
duration: "1:16:08"
video_id: hHC-SF3utxg
video_title: "Lecture 15: Efficient Attention and Adaptation"
video_caption: "Original lecture. Tengyu Ma attacks the quadratic price with KV caching and attention variants, then covers in-context learning and SFT."
concepts: [KV-cache, attention-variants, GQA, MoE, mixture-of-experts, in-context-learning, few-shot, zero-shot, SFT, instruction-tuning]
sources:
  - tag: video
    label: "Lecture 15 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=hHC-SF3utxg
  - tag: video
    label: "Explainer: efficient transformers and fine-tuning"
    url: https://www.youtube.com/watch?v=emZxqRScfc0
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: serve a million users without a million GPUs

Lecture 14 priced attention at N^2 per forward pass. Generation
makes it worse: each new token attends to all previous ones, so
naive generation recomputes the whole matrix per token. A 4,096-token
answer at 16.7M scores per layer per head per step is unservable.
The job: cut the per-token cost, then cut the memory, then ask what
the trained model can do without any weight updates at all.

## First attempt: recompute everything per token

The naive generator: to produce token 101, run the full transformer
on tokens 1-100. To produce token 102, run it on 1-101 from scratch.
Token t costs O(t^2). The full answer costs O(N^3). For N = 4,096,
that is ~2.3 x 10^10 score computations per layer per head. Nobody
serves this. The waste is visible: tokens 1-100 did not change
between steps 101 and 102, but every one of their scores was
recomputed.

### Subchapter: the cubic, audited

Verify the 2.3 x 10^10. Token t costs t^2 scores. Total:
sum_{t=1}^{4096} t^2 = N(N+1)(2N+1)/6 = 4096 * 4097 * 8193 / 6 =
22,914,881,536 ≈ 2.29 x 10^10. The lesson's number is exact. With
the KV cache, token t costs t scores: total 4096*4097/2 =
8,390,656 ≈ 8.4M. Ratio: 2.29e10 / 8.39e6 = 2,730x cheaper. The
cache does not shave a constant: it deletes an order of growth.
That 2,730x is the difference between unservable and servable.

![Cubic audit](assets/plate-l15-cubic-audit.webp "The cubic, audited. Naive: sum of t squared to 4096 = 2.3e10 scores. KV cache: sum of t = 8.4M. 2,730x cheaper. Source: original audit for the generation cost. Project: Stanford Frontier AI.")

### Subchapter: the cache in bytes

Big-O hides the bytes. Price them for a 7B model: d = 4096, 32
layers, 32 heads, head dim 128, fp16. Per token per layer: K is 32
heads x 128 x 2 bytes = 8,192. V the same: 16,384 bytes per
layer. Times 32 layers: 524,288 bytes = 512 KB per token. A 4,096-token
sequence: 2 GB. Batch of 10: 20 GB, plus 14 GB of weights = 34 GB
on an 80 GB GPU: fits, barely, before activations and overhead.
This is why the lecture's "batch of 10, not 1,000" is not
rhetoric: each sequence is a 2 GB object. Bytes, not big-O, set
the batch size.

![Cache bytes](assets/plate-l15-cache-bytes.webp "The cache in bytes. Per token: 512 KB. A 4k sequence: 2 GB. Batch of 10: 20 GB plus 14 GB weights. Source: original plate for the KV memory. Project: Stanford Frontier AI.")

## The KV cache

Keys and values for tokens 1-100 are identical whether you are
generating token 101 or 102. So save them. The **KV cache** stores
every layer's keys and values for all past tokens in GPU memory.
Generating token t+1 needs only its own query, key, and value:
attend the new query to the cached keys, mix the cached values,
append the new K/V to the cache. Per-token cost drops from O(t^2)
to O(t). The cubic becomes quadratic.

![KV cache](assets/svg/l15-kvcache.svg "The KV cache. Keys and values of past tokens are stored, not recomputed. Per-token cost falls from O(t^2) to O(t). Source: original plate for Stanford Frontier AI.")

The price is memory. The cache is linear in sequence length T, and
the lecture stresses how heavy it is: saving K and V for every head
of every layer for one long sequence can fill most of a GPU's
memory. Then batch size collapses: you can serve 10 sequences, not
1,000. Small batches starve the GPU's compute units (nothing to
parallelize across), so inference becomes **memory-bound**: the GPU
waits on memory bandwidth, not arithmetic. The lecture's chain:
KV cache fills memory -> batch shrinks -> parallelism dies ->
throughput dies. Reducing the cache is the highest-impact
systems work in LLM serving.

## Attention variants: shrink the cache

**Grouped-query attention** (GQA): share one key/value head across
a group of query heads. With 32 query heads and 8 KV heads, the
cache shrinks 4x. The queries keep their expressiveness (each still
attends differently). Only the stored K/V compress. Quality barely
moves. Memory moves a lot.

**Mixture of experts** (MoE): instead of one big MLP per block,
have E expert MLPs and a router that sends each token to the top-k
(usually 2). Parameters grow E-fold. Compute per token stays ~flat
because each token visits 2 experts. A 8-expert model has 8x the
MLP parameters for ~1.1x the compute. The price: the router must
balance load (experts starve or flood), and all experts must sit in
memory even though each token uses two.

![MoE](assets/svg/l15-moe.svg "Mixture of experts. A router sends each token to 2 of 8 experts. Parameters grow 8x, compute per token stays nearly flat. Source: original plate for Stanford Frontier AI.")

### Subchapter: GQA, counted

Count the diet. 32 query heads, 32 KV heads: 512 KB per token (the
audit above). GQA with 8 KV heads: K per layer is 8 heads x 128 x
2 bytes = 2,048. V the same: 4,096 bytes per layer. 32 layers:
131,072 bytes = 128 KB per token. Exactly 4x smaller. A 4,096-token
sequence: 512 MB instead of 2 GB. Batch of 10: 5 GB instead of 20.
The queries keep all 32 heads: each still attends with its own
pattern. Only the stored keys and values compress. The lecture's
claim "quality barely moves" is the empirical surprise: the KV
content is redundant across heads, the query patterns are not.

![GQA counted](assets/plate-l15-gqa-counted.webp "GQA, counted. 32 KV heads: 512 KB per token. 8 KV heads: 128 KB. 4x smaller cache, queries untouched. Source: original plate for the GQA arithmetic. Project: Stanford Frontier AI.")

## The shock: in-context learning

Now the pivot from systems to behavior. Take the trained model,
freeze every weight, and prompt it:

```ascii
Translate English to French:
  sea -> mer
  sky -> ciel
  cheese ->
```

The model answers "fromage". No gradient step. No weight moved.
Three examples in the prompt taught it the task. This is
**in-context learning** (ICL): few-shot learning with frozen
weights. The lecture presents it as the shock of the GPT-3 era:
nobody trained for this. It emerged from next-token prediction at
scale.

Why it works, roughly: the prompt's examples are processed by the
same attention machinery as everything else. Attention can
implement "find the pattern in the recent examples and continue
it": the query for the blank position matches the example slots,
and the values carry the mapping. The model learned, during
pre-training, to complete patterns. Few-shot prompts are patterns
with a job hidden inside. Scale matters: small models barely do
this. Large ones do it reliably. The lecture's honest note: the
mechanism is still partly mysterious, and ICL is brittle: change
the example order or wording and accuracy swings.

![In-context learning](assets/svg/l15-icl.svg "In-context learning. Examples live in the prompt, weights frozen. Attention implements 'continue the pattern'. Source: original plate for Stanford Frontier AI.")

### Subchapter: the brittleness, itemized

ICL is free and fragile. Three documented failure shapes. Order:
reversing the example order can swing few-shot accuracy by double
digits on some tasks. The model reads position as signal. Wording:
"Translate English to French" vs "English: sea, French: mer" can
move accuracy more than adding examples does. Recency: the model
over-weights the last example: a misleading final example corrupts
the pattern. The decision rule for production: prototype with ICL
(zero training, instant iteration), ship with SFT (the behavior is
in the weights, not in a prompt you hope nobody rewords). ICL is
the sketchpad. SFT is the ink.

![ICL brittle](assets/plate-l15-icl-brittle.webp "ICL is brittle. Reorder the examples: accuracy swings. Reword the prompt: bigger swing than adding examples. Prototype with ICL, ship with SFT. Source: original plate for the brittleness. Project: Stanford Frontier AI.")

## SFT: teach the format

Pre-trained models complete text. Users want assistants that
follow instructions. **Supervised fine-tuning** (SFT), also called
instruction tuning: collect (instruction, response) pairs written
by humans, and train the model to predict the response given the
instruction, with the usual cross-entropy loss. Ten thousand to a
million pairs. The model learns the *format* of helpfulness:
answer the question, be concise, refuse safely.

The lecture's framing: pre-training teaches the world. SFT teaches
the job description. The knowledge comes from pre-training. SFT
barely adds facts. What it adds is behavior: which of the many
possible completions is the one a user wants. The price: SFT
narrows the model. Overdo it and the model forgets things it knew
(catastrophic forgetting) or becomes sycophantic. And the pairs are
human-written: expensive, slow, and the quality ceiling of the
whole step.

![SFT](assets/svg/l15-sft.svg "Supervised fine-tuning. Train on (instruction, response) pairs. Pre-training teaches the world. SFT teaches the job description. Source: original plate for Stanford Frontier AI.")

## The honest price

Efficiency buys servability and pays in complexity: GQA's sharing,
MoE's routing and load balance, the KV cache's memory hunger that
no variant fully removes. ICL buys task flexibility with zero
training and pays in brittleness: prompt wording swings accuracy,
long example lists eat the context window, and nobody can fully
explain why it works. SFT buys helpful behavior and pays in human
hours per pair plus the narrowing tax. The through-line: every step
after pre-training is about spending the model's capability
wisely, because the capability itself was the expensive part.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| KV cache | Naive generation is O(N^3): recompute everything per token | Cache K/V: per-token O(t^2) -> O(t); memory linear in T |
| Memory-bound serving | Cache fills the GPU: batch of 10, not 1,000 | Diagnosis: memory, not compute, is the bottleneck; parallelism starves |
| GQA | KV cache too big | Share KV heads across query groups: 4x smaller cache, quality holds |
| MoE | Want 8x parameters at 1x compute | Router sends each token to 2 of 8 experts; price: load balance, memory |
| ICL | New task, no training budget | Frozen weights, examples in prompt: "cheese ->" gets "fromage"; brittle but free |
| SFT | Base models complete; users want assistants | (Instruction, response) pairs; teaches the job description, not the world |

> [!QA]
> Q: What is the KV cache and why does it make inference memory-bound?
> A: During generation, each token's keys and values never change once computed, so the cache stores them instead of recomputing: per-token cost drops from O(t^2) to O(t). The cache grows linearly with sequence length across every layer and head, and the lecture stresses it can fill most of GPU memory for a single long sequence. Small memory headroom means small batches (10 sequences, not 1,000), which means nothing to parallelize across, so the GPU's compute units idle waiting on memory bandwidth. Memory-bound, not compute-bound.
> Follow-up: How does GQA help?
> A: Grouped-query attention shares each key/value head across a group of query heads: 32 query heads with 8 KV heads cuts the cache 4x. Queries keep their distinct attention patterns. Only the stored keys and values compress. The lecture presents it as the highest-impact cache diet with minimal quality cost.

> [!QA]
> Q: How can a model learn a task with frozen weights in in-context learning?
> A: The task is encoded in the prompt as examples, and attention implements pattern continuation. For "sea -> mer, sky -> ciel, cheese ->", the blank position's query matches the example slots, and the values carry the English-to-French mapping the model must extend. Pre-training on billions of pattern-completion instances taught the machinery. The prompt supplies the pattern. No weights move. The lecture flags the mystery: this was not trained for explicitly, it emerged with scale, and it is brittle to wording and order.
> Follow-up: When does ICL fail?
> A: When the pattern needs more examples than fit in context, when the task contradicts pre-training priors too strongly, or when the wording buries the pattern. Small models barely do ICL at all: it is a scale-emergent capability. For reliability on a fixed task, SFT or fine-tuning beats prompting.

> [!QA]
> Q: What does SFT add that pre-training does not?
> A: Behavior, not knowledge. Pre-training teaches the world: facts, language, reasoning patterns. SFT on (instruction, response) pairs teaches the job description: answer the question directly, follow the format, refuse safely. The lecture's line: pre-training teaches the world, SFT teaches the format of helpfulness. SFT uses 10K-1M human-written pairs, a tiny fraction of pre-training data, because it steers rather than builds.
> Follow-up: What is catastrophic forgetting in SFT?
> A: Over-training on the instruction pairs erodes pre-training knowledge: the model gets helpful-shaped but dumber. The narrow pair distribution overwrites broad capabilities. Mitigations: mix pre-training data into SFT, keep SFT short, use low learning rates. It is lecture 6's bias-variance in new clothes: fit the instructions too hard and lose the world.


> [!QA]
> Q: Walk me through the mechanism: verify the 2.3e10 and the KV-cache savings.
> A: Naive: token t costs t^2 scores. Sum t^2 from 1 to 4096 = 4096*4097*8193/6 = 22,914,881,536 ≈ 2.29e10 per layer per head. KV cache: token t costs t scores (one query against t cached keys). Sum t = 4096*4097/2 = 8,390,656 ≈ 8.4M. Ratio: 2.29e10/8.39e6 = 2,730x. The cache deletes an order of growth, not a constant. That is why naive generation is unservable and cached generation ships.
> Follow-up: What does the cache not fix?
> A: The per-token O(t) and the memory: 8.4M scores per layer per head is still quadratic in total, and the cache itself is 2 GB per 4k sequence. GQA, eviction, and compression attack what the cache leaves behind.

> [!QA]
> Q: Applied design: serve 4k-context chat on one 80GB GPU with a 7B fp16 model. What is the max batch, and what binds?
> A: Weights: 14 GB. KV cache per 4k sequence: 512 KB/token * 4096 = 2 GB. Memory allows (80-14)/2 = 33 sequences, but activations, fragmentation, and overhead pull it to ~10-20 in practice: the lecture's "batch of 10". With GQA-8: 128 KB/token, 512 MB/sequence: ~100+ by memory. What binds at large batch is memory bandwidth: each generated token reads every sequence's full cache, so per-token latency grows with batch * length. Throughput (tokens/s) still rises with batch until bandwidth saturates. Decision rule: batch for throughput, watch per-token latency for the SLA.
> Follow-up: Why not just buy more GPUs?
> A: That is what production does (tensor/pipeline parallel serving), but the bytes-per-token math follows you: it determines how many GPUs per replica and the cost per million tokens. The audit is the pricing model.

> [!QA]
> Q: When does the MoE router fail, and what does it cost?
> A: Three failures. Load imbalance: the router floods one expert, which becomes the straggler while others idle. Fixed with an auxiliary load-balancing loss. Token dropping: over-capacity experts drop tokens, losing information. Fixed with capacity factors above 1. Expert collapse: the router converges to always picking the same experts, wasting the rest. Fixed with noise in routing during training. The standing cost: all 8 experts sit in memory (any token may visit any expert) while each token computes on 2. Memory pays for parameters. Compute pays for 2.
> Follow-up: Why top-2 and not top-1?
> A: Top-1 is cheaper but brittle: a routing mistake sends the token to exactly the wrong expert with no backup. Top-2 gives a weighted blend and a second opinion. The field settled on 2 as the quality/compute sweet spot. Top-1 variants exist for maximum efficiency.

> [!QA]
> Q: Fixed production task: ICL or SFT? Decide with numbers.
> A: ICL: 20 examples at 100 tokens each = 2,000 tokens of prompt overhead on every query, forever. At 1M queries/day, that is 2B prompt tokens/day of pure overhead. SFT: pay the human-pair cost once (10k-1M pairs), then every query is short. SFT wins on reliability (behavior in weights, not in wording) and on serving cost for fixed tasks. ICL wins for exploration: zero training, try 10 task framings in an afternoon. The hybrid the field uses: SFT the format and common cases, ICL the per-query variables.
> Follow-up: The task changes weekly. Does the answer change?
> A: Yes, toward ICL or retrieval: weekly SFT retraining is an ops burden and risks forgetting. If the change is in facts, use RAG (lecture 13): update the store, not the weights. If the change is in format, SFT on the new format with replay of the old. Match the adaptation method to the change rate: prompt for daily, RAG for weekly facts, SFT for stable behavior.

## Recap: the whole lesson on one screen

1. **The job.** Serve generation without O(N^3) per answer.
2. **Naive.** Recompute everything per token. 2.3 x 10^10 scores
   per layer per head at N = 4,096. Unservable.
3. **KV cache.** Store K/V: O(t^2) -> O(t) per token. Memory
   linear in T.
4. **Memory-bound.** Cache fills GPU. Batch collapses to ~10.
   Compute idles. Shrink the cache = serve more.
5. **GQA.** Share KV heads: 4x smaller cache. **MoE.** 8 experts,
   top-2 routing: 8x params, ~1x compute.
6. **The shock.** ICL: frozen weights, "cheese ->" gets
   "fromage". Pattern continuation via attention. Brittle, free.
7. **SFT.** (Instruction, response) pairs teach the job
   description. Knowledge from pre-training. Behavior from SFT.
8. **The honest price.** Systems complexity, prompt brittleness,
   human hours, the narrowing tax.
9. **The audit.** 2.29e10 naive, 8.4M cached: 2,730x. An order
   of growth, deleted.
10. **The bytes.** 512 KB/token, 2 GB per 4k sequence. Bytes set
    the batch.
11. **GQA.** 8 KV heads: 128 KB/token, 4x smaller. Queries
    untouched.
12. **The brittleness.** Order, wording, recency. Sketchpad vs
    ink: ICL prototypes, SFT ships.


## What is used where

**The KV cache runs all LLM serving.** vLLM, TensorRT-LLM, and
every production inference stack implement paged/cached KV
storage: the lesson's cache is the deployed primitive.
**GQA is the default** in Llama 3, Mistral, and most open models:
the 4x cache diet with minimal quality cost. **MoE is the
frontier scale-out:** Mixtral 8x7B and the large labs' models
route tokens to experts. **SFT is the post-training default:**
every instruction model ships it. ICL is the interface users
actually touch.

## Watch next

<div class="video-block"><div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/emZxqRScfc0" title="Explainer: efficient transformers and fine-tuning" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div><p class="video-cap">Explainer: efficient transformers and fine-tuning. KV cache, attention variants, and the SFT step in one visual pass. Watch after the MoE section.</p></div>

## Official sources and further reading

**Official:**
- Lecture 15 video, Stanford Online YouTube:
  - [Tengyu Ma derives](https://www.youtube.com/watch?v=hHC-SF3utxg)
  the KV cache and its memory-bound serving consequences, presents
  GQA and MoE, demonstrates in-context few-shot learning, and
  frames SFT as teaching the job description.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the formal
  treatment.

**Caveats from these sources.** The KV-cache memory arithmetic
(batch of ~10 vs ~1,000) is the lecture's illustration of the
memory-bound regime, not a benchmark. The ICL mechanism
("attention implements pattern continuation") is the lecture's
rough account of a still-partly-mysterious phenomenon. MoE load
balancing is presented as the price, with the details in the
literature.

## Connections to the other courses

- **CS229 L14:** the quadratic price this lesson attacks. The
  transformer block being served.
- **CS229 L12:** pre-training: the capability SFT steers and ICL
  exploits.
- **CS229 L17:** RL: the next step after SFT in the post-training
  stack.
- **CS336:** the systems half: parallelism, batching, and serving
  infrastructure for the KV cache era.
- **CS224N:** instruction tuning and few-shot prompting from the
  NLP side.
