---
page_id: cs229s-crash
course_slug: cs229s
course_name: "CS229S: Systems for Machine Learning"
course_order: 3
order: 901
nav: "CS229S · Crash course"
title: "CS229S Crash Course 1: The Three Gaps and the Stack of Levers"
summary: "The economic case for systems ML: 32x demand against 2x hardware, the memory wall, paper speed versus wall-clock, and the levers that close each gap (quantization, MoE, parallelism, hardware-aware design)."
concepts: [sysml, scaling-laws, compute-trends, memory-wall, quantization, llm-int8, mixture-of-experts, parallelism, flops-vs-runtime, co-design]
---

<span class="crash-timer">Crash course 1 of 10 · the economics</span>

[← Course index](index.html) · **Chapter 1** · [Next: Chapter 2 — the transformer, built from zero →](crash-course-02.html)

## The problem: the plan is scale, and scale is unaffordable

A new large language model drops. You rent A100s to fine-tune it. The job starts, the weights load, and the process dies with three words: out of memory. No bug in your code. This is the default experience of working with large models, and it is why this course exists.

The field's plan is fixed. **Scaling laws** (Kaplan et al., 2020) show performance improving smoothly with three inputs: compute used, dataset size, and parameter count. Size also buys **emergent behaviors**: capabilities that appear only past a size threshold. GPT-3 showed few-shot learning (Brown et al., 2020). Chain-of-thought reasoning emerged the same way (Wei et al., 2022). You cannot get these by training a small model cleverly. The size is the point.

Size needs a second step. A pretrained model predicts the next token on web text. Ask it to translate "cheese" to French and it keeps writing translation examples instead of answering. **Fine-tuning** updates the weights on instruction-following data so it answers instead of completing. Thoppilan et al. (LaMDA, 2022) is the example: the same base model, tuned for dialog, becomes an assistant. So the plan is: train big, then fine-tune. This chapter asks what that plan costs, and who pays.

## Gap 1: compute demand outruns hardware, 32x against 2x

The naive answer is hardware: models grow, buy better chips. That works only if chips improve as fast as models grow. They do not.

Sevilla et al. (2022) plotted training compute across three eras of machine learning. In the deep learning era, training compute grows **32x every 2 years**. Moore's law gives roughly **2x every 2 years**. The gap is **16x per 2-year window**, and it compounds.

![Compute trends](assets/slide-l01-compute-trends.png "Shell 1. Training compute rises 32x per 2 years against Moore's law 2x. Source: Stanford slides, Sevilla et al. 2022. Project: Stanford Frontier AI.")

Work the arithmetic. Two years pass. Hardware doubles. Model compute demand multiplies by 32. The missing factor of 16 must come from somewhere else: better algorithms, better parallelism, better utilization. That missing factor is this course.

## Gap 2: memory is the wall

Model size follows the same steep curve, but accelerator memory is nearly flat. The V100 holds 32 GB. The TPUv3 holds 32 GB. The A100 holds 40 or 80 GB. The largest models sit far above all of them.

![Memory wall](assets/slide-l01-memory-wall.png "Shell 2. Model size over time against fixed accelerator memories. Source: Stanford slides. Project: Stanford Frontier AI.")

The out-of-memory error from the opening scene is now explained. The weights do not fit. No setting in your training script fixes a 32 GB card facing a 100 GB model. The options are to shrink the model, split it across devices, or both. Every later chapter is one of those options.

## Gap 3: paper speed is not wall-clock speed

The third gap is the subtlest. An algorithm can be better on paper and slower on the machine. The lecture's example is attention, the core operation of the transformer. A new linear attention algorithm scales as O(N) against O(N squared) for standard attention. On paper it wins. But FlashAttention, a hardware-aware implementation of the exact O(N squared) algorithm, runs faster in measured wall-clock time.

![Algorithmic scaling vs wall-clock](assets/slide-l01-algorithmic-scaling-meme.png "Shell 3. Linear attention wins on asymptotics. FlashAttention wins on the GPU. Source: Stanford slides, meme credit Michael Zhang. Project: Stanford Frontier AI.")

Why? Big-O notation hides constants and ignores how an algorithm uses memory. A linear algorithm that ignores the memory hierarchy can lose to a quadratic one that respects it. The rule the course repeats: judge algorithms by measured runtime on the target hardware, never by asymptotics alone.

A second cost split runs through the whole course. **Training** is a large upfront bill: tens of millions of dollars for a frontier model. **Inference**, running the model for one input, is cheap per call, under $0.0001 for a large model, but it accumulates with every user. Design for training when the bill is upfront. Design for serving when the bill compounds.

![Training and inference costs](assets/slide-l01-train-infer-costs.png "Shell 4. Training is a large upfront cost. Inference compounds with users. Source: Stanford slides, OctoML. Project: Stanford Frontier AI.")

## The key question

If buying bigger chips cannot close the gap, what can? The lecture's answer is the whole stack. Efficiency is not one trick. It is a lever at every layer, and the levers compose.

![Full-stack efficiency](assets/media-generation-plate-chapter-l01-fullstack-0-ba96d864-a7fa-4e8c-b343-e346c5aa61cf.webp "Shell 5. Every layer of the stack gets its own efficiency lever. Source: original chapter plate. Project: Stanford Frontier AI.")

## Lever 1: quantization (shrink the numbers)

One billion weights in FP32 need 4 GB. The same weights in INT8 need 1 GB. That factor of 4 is the whole pitch: **quantization** stores each number in fewer bits. Naive quantization degrades accuracy unacceptably, and it scales poorly past about 6B parameters. The fix is model-aware.

The toy that shows the failure: eight weights, [0.5, -0.3, 0.8, 0.2, 127.0, -0.4, 0.6, 0.1]. INT8 spans -128 to 127. Naive quantization scales by the max: 127.0 maps to 127, and 0.5 maps to about 0. Every small weight rounds to zero. The matrix becomes the outlier plus zeros. Accuracy collapses.

Dettmers et al. (LLM.int8, 2022) diagnosed the failure in real models. Large transformers grow a few feature dimensions with huge magnitudes, the **outliers**, while the rest stay small. The fix is surgical: quantize 99.9 percent of the matrix to INT8 and keep the outlier columns in FP16. Memory still drops by nearly 4x. Accuracy holds. The paper ran OPT-175B this way with no accuracy loss.

![One outlier breaks naive INT8. LLM.int8 isolates it](assets/plate-l01-quant-outlier.webp "Outliers keep FP16, 99.9 percent of weights go INT8, accuracy holds. Shell 2. Source: original toy for the LLM.int8 fix. Project: Stanford Frontier AI.")

The quantization family, each answering the outlier problem differently:

| Scheme | Bits | What it protects | Calibration | Public use |
|---|---|---|---|---|
| LLM.int8 | 8 | outlier columns in FP16 | none, dynamic | HuggingFace transformers |
| GPTQ | 4 | rounding error spread via Hessian | yes, small set | most Llama 4-bit quants |
| AWQ | 4 | salient weights by activation scale | yes, small set | vLLM, AutoAWQ |
| FP8 | 8 | range via E4M3/E5M2 formats | none | DeepSeek-V3 training |

**FP8** is a floating format used during training itself. DeepSeek-V3 trained in FP8, public in its paper. DeepSeek-V4 moved to NVFP4, a 4-bit microscaling format. What quantization GPT-5 uses, if any, is not public. Unknown.

## Lever 2: mixture-of-experts (spend compute only where it matters)

Traditional networks are dense: each input is processed by the entire model. **Mixture-of-experts** (MoE) is dynamic: large weight matrices become a mixture of smaller matrices called **experts**, and a **gating function** routes each input to a small number of them. Shazeer, Mirhoseini et al. (2017) introduced the sparsely-gated MoE layer. Fedus et al. (2021) applied it to transformers. Total capacity grows with expert count while cost per token stays near flat.

The gate picks experts per token. Original MoE: top-2 of up to 2048 experts. Switch Transformers: top-1, simpler and faster. The toy: 4 tokens, 8 experts, top-2 routing. The gate assigns token 1 to {2, 5}, token 2 to {2, 7}, token 3 to {2, 3}, token 4 to {5, 2}. Expert 2 gets 4 assignments. Experts 1, 4, 6, 8 get zero. Left alone, the gate keeps picking the expert that already works, and the rest starve. That is **expert collapse**: you paid for 8 experts and use 1. The fix is an **auxiliary loss** that penalizes uneven assignment and forces the gate to spread tokens. Then experts specialize and the capacity pays off.

Then the systems bill: experts live on different devices, so each token's dispatch is an all-to-all communication across the network. That is **expert parallelism**, and it is why MoE is a systems topic as much as an architecture topic. The gate's decision is a network operation.

What is used where: Mixtral 8x7B routes top-2 of 8 experts, public. DeepSeek-V3 routes 8 of 256 fine-grained experts plus one shared expert, public. GPT-4 is rumored to be MoE. Not confirmed. Unknown.

![The gate routes. Imbalance starves experts](assets/plate-l01-moe-routing.webp "Uneven gates collapse MoE into one dense expert. The balance loss spreads the load. Shell 3. Source: original toy for MoE routing. Project: Stanford Frontier AI.")

## Lever 3: parallelism (split the work to fit)

Scaling across devices needs parallelization strategies, and the strategy must match the hardware. Three ways to split:

**Data parallelism.** Copy the whole model to each GPU. Split the batch. Each GPU trains on its slice, then all-reduce syncs the gradients. Memory need: the full model must fit on one GPU. Spans nodes happily.

**Tensor parallelism.** Split single operations across GPUs. One matrix multiply becomes two halves on two GPUs, results combined. Memory need: the model shards, so 70B in FP16 (140 GB) fits across two 80 GB cards. Communication: every layer, small and frequent. Stays inside one machine, on NVLink.

**Pipeline parallelism.** Split the layers. GPU 0 holds layers 1-8, GPU 1 holds 9-16. Feed microbatches through. Communication: only activations at the boundary, rare. Spans nodes. The cost is idle time: the pipeline drains and fills, the **bubble**.

The rule: frequent fine-grained communication stays intra-node. Rare coarse communication spans the cluster. Llama 3 trained on 16,384 H100s with data plus model plus pipeline parallelism, public in the Meta paper. Megatron-LM is the public tensor-parallel implementation. DeepSeek-V3's DualPipe overlaps communication with computation to hide the cost, public in its paper.

![Three ways to split the work](assets/plate-l01-parallel-family.webp "Frequent fine traffic stays in the node. Rare coarse traffic spans the cluster. Shell 3. Source: original toy for the parallelism family. Project: Stanford Frontier AI.")

## Lever 4: design for the device, not the FLOP count

EfficientNet introduced depthwise convolutions with fewer FLOPs (floating-point operations) than competing ResNets. But the operation took 5.00% of FLOPs and 65.30% of runtime. Fewer FLOPs, more time.

| Operation | Share of FLOPs | Share of runtime |
|---|---|---|
| DepthwiseConv2D | 5.00% | 65.30% |
| Conv2D | 94.67% | 34.20% |
| Other | 0.33% | 0.50% |

The operation did not suit the hardware, so compute utilization collapsed. FLOPs measure work, not speed. Speed is work divided by achieved throughput, and achieved throughput depends on memory access patterns and the device's strengths. The roofline model (Chapter 3) turns this into a number: every kernel is either compute-bound or memory-bound, and the achieved FLOPs per second tell you which. If your clever low-FLOP operation lands memory-bound, you lost. This is why the course judges everything by measured runtime.

![Fewer FLOPs, more runtime](assets/plate-l01-flops-lie.webp "Speed is work over achieved throughput, not FLOP count. Shell 3. Source: Stanford slides, EfficientNet table. Project: Stanford Frontier AI.")

The final lever is co-design. The FAST paper (Zhang et al., ASPLOS 2022) searched the hardware and software stack together: the datapath (how the chip moves data), the schedule (in what order operations run), and the fusion (which operations merge into one kernel). The best choices differ by workload: a datapath tuned for convolutions is not the one tuned for attention. Joint search delivered multi-fold gains over single-layer optimization. The price is portability: the answer is a chip for one workload family, not a general GPU.

## What is used where: the real models (Oct 2026)

Every lever in this chapter ships in a production model. Facts verified against public sources as of October 2026.

| Model | Public efficiency levers | Why these |
|---|---|---|
| DeepSeek-V4 (Apr 2026) | NVFP4 training, hybrid CSA+HCA attention (MLA dropped), MoE, 1M context | MIT-licensed; every lever cuts cost per token [uncertain: exact parameter counts vary by source] |
| DeepSeek-V3 (Dec 2024) | FP8 training, MLA attention, MoE (256 routed + 1 shared expert, 8 active), DualPipe parallelism | 671B parameters, 37B active per token; the full-stack cost playbook |
| Llama 4 Maverick (Apr 2025) | MoE (128 experts, 17B active of 400B), 1M context | the entire Llama 4 family went MoE; open weights |
| Mixtral 8x7B | MoE top-2 of 8, sliding window, GQA | open-weights MoE; sparse compute with long context |
| Gemini 3.1 Pro (Feb 2026) | sparse MoE transformer (per public model card), 1M context | long context plus reasoning as the product feature [uncertain: exact counts not public] |
| GPT-5 (Aug 2025) | not public | unknown; internals unconfirmed |

Read it as the course in miniature. DeepSeek is the full stack: train cheaper (FP8, then NVFP4), serve cheaper (MoE, latent attention), scale wider (DualPipe). Llama 4 shows even Meta moved its flagship line to MoE. GPT-5 reminds you that the table records only what makers announce.

## Mapping back: each gap gets its levers

| Gap | Levers that close it |
|---|---|
| Compute demand outruns hardware (32x vs 2x) | Parallelism across devices, hardware-aware kernels, full-stack co-search |
| Model size outruns memory (the OOM scene) | Quantization, sparsity, MoE, splitting weights across devices |
| Asymptotics outrun wall-clock (the meme) | Measure on target hardware; design for the memory hierarchy, not big-O |

## The honest price

None of these levers is free. Quantization risks accuracy and needs model-aware fixes past 6B parameters. MoE trades dense simplicity for routing and load-balancing complexity. Parallelism trades memory for communication overhead. Hardware-aware design trades portability for speed: what wins on an A100 may not win elsewhere. The course exists because the tradeoffs are real and the numbers decide them.

## Memory aids

> [!MEMORY]
> **Mnemonic — the three gaps:** "CMA": Compute (32x vs 2x), Memory (the wall), Asymptotics (paper speed lies). When an interviewer asks why systems ML exists, answer with CMA and one number each.

**Never-confuse pairs:**

- Training bill (tens of millions, upfront, once) vs inference bill (under $0.0001 per call, compounds with users). Design for the bill you pay.
- Model parallelism (partition the model into disjoint sub-models; less communication, more idle time) vs tensor parallelism (partition tensors within one operation; less idle time, more communication, needs NVLink).
- GPTQ (spreads rounding error via the Hessian) vs AWQ (scales up salient channels before quantizing). Both are 4-bit and need calibration. Pick GPTQ for a quick one-shot shrink. Pick AWQ for the best 4-bit accuracy.
- Expert collapse (the gate starves all but one expert) vs load balancing (the auxiliary loss that forces the spread). Collapse is the failure. The balance loss is the fix.

**If-this-then-that:**

- A100 fine-tune dies with out-of-memory → shrink the model first (quantize, Chapter 7). Splitting needs no new math but needs more GPUs.
- 70B model on limited GPUs → INT4 (GPTQ or AWQ): 140 GB becomes 35 GB. Fits one 80 GB card with cache headroom.
- Serving 50 fine-tuned variants → LoRA adapters, not 50 full copies (Chapter 8).
- Linear algorithm beats your kernel on paper → profile both on the target GPU before you believe the big-O (Chapter 3).

**Trap card — "interviewers love to ask":**

> [!QA]
> Q: Walk me through the LLM.int8 fix. Why does naive INT8 quantization break large transformers?
> A: Start with the toy: weights [0.5, -0.3, 0.8, 0.2, 127.0, -0.4, 0.6, 0.1]. INT8 covers -128 to 127, so naive quantization scales everything by the max. The outlier 127.0 becomes 127. The small weights become 0. The matrix is now the outlier plus zeros, and accuracy collapses. Dettmers et al. found real large transformers grow such outlier feature dimensions. The fix: keep the outlier columns (about 0.1 percent of weights) in FP16 and quantize the rest to INT8. Memory still drops nearly 4x. The paper ran OPT-175B with no accuracy loss.
> Follow-up: Why does the failure get worse past 6B parameters?
> A: Outliers emerge with scale. Small models have calm activation ranges, so naive INT8 works. Past about 6B, a few dimensions grow huge magnitudes, and one scale factor can no longer serve both the outliers and the rest. The diagnosis is scale-dependent, which is why the fix is model-aware rather than a blunt bit cut.

## Self-test

> [!QA]
> Q: Why does the field keep training larger models instead of better small ones?
> A: Two reasons. Scaling laws show smooth, predictable gains from more compute, data, and parameters, so size is a reliable lever. Emergent abilities such as few-shot learning and chain of thought appear only at large scale, so some capabilities have no small-model equivalent. Systems work exists because this appetite for scale collides with hardware limits.
> Follow-up: What breaks first as models grow?
> A: Money and memory. Training a frontier model costs tens of millions of dollars, and model size outruns accelerator memory: GPT-4-scale weights sit far above an A100's 40 or 80 GB.

> [!QA]
> Q: Quote the two growth rates and say why they matter together.
> A: Deep learning training compute grows 32x every 2 years. Moore's law gives roughly 2x every 2 years. The 16x gap per 2-year window compounds, so each generation of models demands far more than new hardware supplies. Every missing factor must come from better algorithms, better parallelism, or better utilization.

> [!QA]
> Q: An algorithm scales linearly while the baseline scales quadratically. Is it faster?
> A: Not necessarily. Big-O hides constants and ignores memory behavior. The lecture's example is linear attention versus FlashAttention: the linear algorithm has better asymptotics, but FlashAttention wins in wall-clock time because it is hardware-aware. Always profile on the target device.

> [!QA]
> Q: Walk me through MoE routing. What goes wrong without load balancing?
> A: Take 4 tokens and 8 experts with top-2 routing. The gate assigns token 1 to experts {2, 5}, token 2 to {2, 7}, token 3 to {2, 3}, token 4 to {5, 2}. Expert 2 gets 4 assignments. Experts 1, 4, 6, 8 get zero. Left alone, the gate keeps picking the one expert that already works, and the rest starve. That is expert collapse: you paid for 8 experts and use 1. The fix is an auxiliary loss that penalizes uneven assignment, forcing the gate to spread tokens. Then experts specialize and the capacity pays off.
> Follow-up: Why is MoE a systems topic, not just an architecture trick?
> A: Experts live on different devices. Every routing decision is an all-to-all dispatch across the network. The gate is a communication pattern, and its cost depends on interconnect bandwidth. That is expert parallelism.

> [!QA]
> Q: Applied design: you have 8 A100 40GB cards and must serve a 70B model at low latency. Stack your levers.
> A: Start with the memory wall. 70B weights in FP16 need 140 GB. No single 40 GB card holds them. First lever: quantize. INT4 (GPTQ or AWQ) brings weights to 35 GB. That fits one card, barely, before the KV cache. Second lever: split. Tensor parallel across 2 GPUs gives headroom for the cache and keeps the frequent communication on NVLink. Third lever: FlashAttention for the attention kernel, since decode is memory-bound (Chapters 5 and 6). Fourth: manage the KV cache per token, because at long context it becomes the new wall. The interview signal: name the binding constraint first (weights do not fit), then stack levers in the order that removes constraints.
> Follow-up: Why quantize before parallelizing?
> A: Quantization needs no extra hardware and cuts every downstream cost: smaller weights mean less memory per GPU, smaller activations to communicate, and fewer GPUs to buy. Parallelism spends communication to buy memory. Spend the free lever first.

## Go deeper

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/ccBMRryxGog" title="Switch Transformer (Mixture of Experts), Yannic Kilcher" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/1JWnEze9V5g" title="The Engineering Behind LLM Inference: Quantization" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

- Switch Transformer paper walkthrough (Yannic Kilcher): https://www.youtube.com/watch?v=ccBMRryxGog
- MoE from scratch: routing and load balancing: https://www.youtube.com/watch?v=YZKMtxccocA
- The Engineering Behind LLM Inference: Quantization (GPTQ, AWQ, FP8, outliers): https://www.youtube.com/watch?v=1JWnEze9V5g
- LLM.int8() (Dettmers et al., 2022): https://arxiv.org/abs/2208.07339
- Switch Transformers (Fedus et al., 2021): https://arxiv.org/abs/2101.03961
- FAST full-stack co-search (Zhang et al., ASPLOS 2022): https://dl.acm.org/doi/10.1145/3503222.3507767
- Full lesson: [Lecture 1](l01-introduction.html)

[← Course index](index.html) · **Chapter 1** · [Next: Chapter 2 — the transformer, built from zero →](crash-course-02.html)
