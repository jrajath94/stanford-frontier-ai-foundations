---
page_id: cs336-cheatsheet
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 900
nav: "CS336 · Cheatsheet"
title: "CS336 Cheatsheet"
summary: "Every key fact from CS336 on one dense page: definitions, formulas, numbers, decisions, mistakes, interview lines."
---

Every block below is the working version of one lecture: the numbers,
the decisions, and the one-liners that answer interview questions.
Each block links to its full lesson for the derivations.

<div class="cheat-cols" markdown="1">

<div class="cheat-block" markdown="1">

### Tokenization schemes

| Scheme | Unit | Vocab | Wins | Loses |
|---|---|---|---|---|
| Char-level | 1 character | ~256 | Never fails on new text | Sequences 4-5x longer |
| Word-level | 1 word | 170k+ | Clean word units | Breaks on new words (OOV) |
| Subword (BPE) | Frequent chunks | ~50k | Short sequences, no OOV | Splits are arbitrary |

BPE training: count adjacent pairs, merge the most frequent, repeat.
Inference: greedy longest match, left to right.

</div>

<div class="cheat-block" markdown="1">

### Memory aids: mnemonics, never-confuse pairs, if-this-then-that

**Mnemonics**

- **6ND**: "six end" = six times N (params) times D (tokens). Training FLOPs.
- **BREAD**: the serving stack order = **B**atch (continuous), **R**adix (prefix sharing), **E**vict (LRU tiers), **A**mplify (speculative), **D**isaggregate (fleets).
- **FROZEN BRIDGE**: LLaVA = freeze both ends, train the bridge (the projector).
- **MEDIUM**: Kimi's curriculum = train on the MEDIUM difficulty middle.
- **The reward is the ceiling**: RLVR. Everything else is how fast you reach it.

**Never-confuse pairs**

- **MFU vs throughput**: MFU = fraction of the GPU's promise you reach. Throughput = tokens per second. High MFU with a small model can still mean low throughput per dollar.
- **FLOPs vs FLOP/s**: FLOPs = work done. FLOP/s = speed. Time = FLOPs / FLOP/s.
- **Prefill vs decode**: prefill = compute-bound, once. Decode = memory-bound, per token. Different fleets, different chips.
- **PPO vs GRPO**: PPO = value network + clipping, general hammer. GRPO = group baseline, no critic, one page, bandit-shaped tasks only.
- **SFT vs RL**: SFT imitates demonstrations (teaches format). RL optimizes a reward (teaches behavior). Style comes from SFT, capability from RL.
- **CLIP vs SigLIP**: CLIP = softmax over the batch (batch is the objective). SigLIP = per-pair sigmoid (batch-size independent).
- **Dedup vs decontaminate**: dedup = remove repeats within training data. Decontaminate = remove test data from training data. Different jobs, different tools.
- **Data parallel vs tensor parallel**: DP = split the batch (needs big batches). TP = split the matrices (needs fast interconnect). DP caps at the critical batch size. TP caps at node boundaries.

**If-this-then-that rules**

- If decode is slow, buy bandwidth (not flops). If prefill is slow, buy compute.
- If the batch is the objective (CLIP), the batch size is not a hyperparameter.
- If all 8 rollouts agree, the advantage is 0: no update. (Dr. GRPO.)
- If the workload is bimodal (book-pastes + chats), disaggregate. If uniform, do not.
- If the context is long, compress the KV cache (MLA/GQA) before adding GPUs.
- If the reward is learnable, expect over-optimization. If it is verifiable, spend compute.
- If the problem mix is trivial or impossible, watch the per-group stds (the std-0 trap).
- If resolution matters, budget tokens first: 9,792 per document page at 1344px.
- If the mix is small and high-quality, cap repetition (UniMax): the 50-epoch trap.
- If the optimizer claim is new, check the compute-times-Chinchilla ratio and the baseline tuning.

</div>

<div class="cheat-block" markdown="1">

### Rapid self-tests (answers inline, say them first)

1. Training FLOPs for 7B params, 1T tokens? 6 x 7e9 x 1e12 = 4.2e22.
2. 70B bf16 decode: GB per token and ms at 3.3 TB/s? 140 GB, 42 ms.
3. KV cache bytes for B=8, S=4096, 32 layers, 8 KV heads, d=128, bf16? 8x4096x32x8x128x2x2 = 4.3 GB.
4. Chinchilla tokens for a 70B model? 20 x 70B = 1.4T tokens.
5. N=4 CLIP toy: how many softmax problems? 2N = 8.
6. GRPO group rewards [1,1,1,1,1,1,1,1]: advantage of rollout 1? 0. Std is 0: no signal.
7. Dr. GRPO drops which two normalizations? Std and length.
8. AnyRes 1344px document: image tokens? 17 x 576 = 9,792.
9. B=1 decode, intensity ~1: bound? Memory-bound. H100 knee is 295.
10. ZeRO-1 vs ZeRO-3: what is sharded? ZeRO-1: optimizer states. ZeRO-3: everything (params too).
11. 50-epoch trap in one line? Small quality sources get over-repeated in the mix.
12. MinHash LSH: P(collision)? Equals Jaccard similarity.
13. R1-Zero recipe? Base model + GRPO, accuracy + format rewards, outcome only, no SFT warm start.
14. Why did DeepSeek drop process supervision? Outcome was enough and scaled better.
15. Cache-aware routing win? 40% faster: fresh and warm never share GPUs.
16. Megakernel bandwidth? 72% of peak H100. Price: batch 17 means starting over.
17. Parcae stability condition? Spectral radius under 1 (negative diagonal A).
18. M-RoPE axes? Height, width, time concatenated.
19. muP promise? Optimal LR transfers across widths.
20. WSD schedule? Warmup, stable, decay in the last 10-20%. Rewind and re-decay.

</div>

<div class="cheat-block" markdown="1">

### Core formulas

**Language model:** p(x1..xn) = product over i of p(xi | x1..xi-1).
Each factor is one next-token prediction.

**Fertility:** tokens divided by words. English ~1.3. Some languages 3+.

**Embedding params:** vocab size x hidden dim.
50,000 x 12,288 = 614M params in the embedding matrix alone.

**Sequence cost:** attention costs O(n^2) in tokens. Longer token
sequences cost quadratically more compute.

</div>

<div class="cheat-block" markdown="1">

### Numbers to memorize

- GPT vocab: ~50,000 tokens (100k in newer models)
- English fertility: ~1.3 tokens per word
- Char fallback: 256 byte values cover all text losslessly
- Context is counted in tokens, not words or characters
- Pricing is per token: fertility sets the bill

</div>

<div class="cheat-block" markdown="1">

### Decisions

**Which scheme?** Subword always wins in practice. Char-level is a
teaching tool. Word-level is dead for open vocabulary.

**Bigger vocab?** Shorter sequences, faster attention, but a bigger
embedding matrix and rarer tokens that train poorly.

**Smaller vocab?** Smaller model, but higher fertility: longer sequences,
more compute, higher cost per word.

**Rule of thumb:** vocab size grows with model size. Small models use
32k. Frontier models use 100k+.

</div>

<div class="cheat-block" markdown="1">

### Common mistakes

- Counting context in words. The window counts tokens.
- Assuming one token = one word. Punctuation and spaces are tokens too.
- Swapping tokenizers after training. IDs index learned rows. New IDs
  point at wrong vectors. The model breaks.
- Trusting number splits. "123" and "124" can split differently, which
  hurts arithmetic. Known limitation, not a bug.
- Forgetting the corpus bias. The training data picks the vocabulary.
  English-heavy data punishes other languages with high fertility.

</div>

<div class="cheat-block" markdown="1">

### Interview one-liners

- "BPE is greedy longest-match: fast, simple, not optimal."
- "Fertility drives cost and context: you pay per token."
- "The tokenizer is fixed before training. Swapping it invalidates the model."
- "Numbers tokenize inconsistently, which hurts arithmetic."
- "Corpus bias in the vocabulary punishes low-resource languages."

</div>

<div class="cheat-block" markdown="1">

### Resource Accounting (L02)

Goal: best model for fixed compute and memory. Tensors: memory = elements x bytes. Precision: fp32 (1+8+23), fp16 underflows (5-bit exponent), bf16 sweet spot (1+8+7), fp8 two variants, nvfp4 block-scaled, 1-bit only post-training. Mixed precision: bf16 for params/acts/grads, fp32 for optimizer states (AMP). FLOPs = work. FLOP/s = speed. H100 dense bf16 = 989 TFLOP/s (halve the 1979 spec). Matmul = 2BDK FLOPs. MFU = actual/promised. 0.5 good, 0.1 broken. Intensity = FLOPs/byte. H100 knee = 295. ReLU 0.25, GELU 5, matmul n/3. time = max(move, compute). Backward = 2x forward. Total 6ND. AdamW = 12 B/param (2+2+4+4). Activations = 2BDL. Grad accum: microbatches, no zeroing, zero extra compute. Checkpointing: store sqrt(L), recompute rest.

<ul class="crash-links">
<li><a href="l02-resource-accounting.html">L02: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Architecture (L03)

Prenorm: norm outside the residual stream, gradients flow straight through. RMSNorm: drop mean and bias, free systems win (0.17% FLOPs, up to 25% runtime). Biases: drop everywhere. GLU: gate the MLP, shrink ff dim by 2/3, SwiGLU (Llama) vs GeGLU (Google). Blocks: serial beats parallel (parallel halves effective depth). RoPE: rotate Q/K by position, relative by construction, multiply not add. Hyperparams: ff 4x (2.67x GLU), heads multiply to d model, aspect ratio ~100, vocab 30k mono / 100-200k multi. Regularization: single-pass means no overfitting. Weight decay is an optimizer. Stability: z-loss for output softmax, QK norm for attention, soft-capping as last resort. GQA: share KV heads, fix decode economics. Sliding window: full every 4th layer for long context.

<ul class="crash-links">
<li><a href="l03-architecture.html">L03: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Attention Alternatives + MoE (L04)

Long context: attention O(n^2), FFN O(n). FlashAttention: 2x constant factor, no materialized matrix. Linear attention: drop softmax, (QK^T)V = Q(K^TV), n^2d to nd^2. Duality: dense trains, recurrent infers, equivalence exact, softmax-drop lossy. Mamba-2: gamma(t) forget gate, input-dependent. Gated delta net: beta(t) input gate + (I - beta k k^T) projector. Hybrids: 7:1 (Minimax M1), 3:1 (Qwen 3.5). Low ratios free, high ratios degrade. DSA: cheap indexer, top-k, full attention on subset, bolted on at extension. MoE: split FFN, route tokens, params up FLOPs flat. Routing: token-choice TopK, one-matmul router, hash/RL/assignment rejected. DeepSeek: fine-grained + shared experts. Training: sparsity is non-differentiable, heuristics win, noise removed. Balance: L = F x P, gradient pushes down popular experts. Removal is catastrophic. v3: per-expert bias, MLA (cache latent c), MTP (built-in speculative decode). Systems: expert parallel comms, downproject before all-to-all, no silent token drops anymore. Upcycling: copy dense MLPs, now unfashionable.

<ul class="crash-links">
<li><a href="l04-linear-attention-moe.html">L04: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### GPUs (L05)

CPU: latency, few complex cores. GPU: throughput, 108 SMs, SIMT. Hierarchy: registers ~1 cycle, L1/shared 20-30, L2 slower, HBM ~10x L1. SRAM 100x cost of DRAM. Thread/block/warp: block = one SM + shared memory, warp = 32 lockstep threads. TPU: convergent, few giant units, 64-dim minimum. Tensor cores (V100+): matmul 10x other ops. Curves: compute fast, bandwidth slow, interconnect slowest. Tricks: no ifs (mask instead), low precision (halve bytes. MXFP8 per-32 scales, two copies for transpose, 20-30% real), fusion (read once write once), recompute (8 to 5 accesses), coalesce (128B bursts, major axis), tile (N/T reads, tune sizes, pad). Mystery plot: divisibility 16/32 wins. Wave quantization at 1792/1793 over 108 SMs. FlashAttention: tiled matmuls + online softmax (running max) + fused kernel + recomputed backward.

<ul class="crash-links">
<li><a href="l05-gpus.html">L05: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Triton Kernels (L06)

Model: grid of blocks. Block on one SM. Shared per block. Registers per thread. HBM global. Warps: 32 lockstep. Zero-cost switch hides HBM latency. Occupancy: limited by registers and the 64-warp cap. 18% example. Fat threads are fine. Banks: 32 by 4B. One thread per cycle. Swizzle to dodge conflicts. Coalesce: 128B lines. Blocks: divide SM counts. Bench: warmup, events, sync, repeat, scale. Profile: kernel names encode lib, arch, dtype, tile. GeLU: naive 3.75, fused wins. Triton shape: pid, offsets plus mask, tl.load, compute, tl.store. Softmax: one row per block. -inf mask. Tile loop if long. Matmul: C tile per block. k-sweep. Acc in shared. Fuse the epilogue. PTX: ld.global and st.global. Compiler coarsens. Alternatives: ThunderKittens, CUTLASS.

<ul class="crash-links">
<li><a href="l06-triton-kernels.html">L06: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Parallelism (L07)

Why multi-GPU: model does not fit, or speed. Hierarchy: shared/L1, HBM 8TB/s, NVLink 1.8TB/s, IB, Ethernet. Rank = device. World size = count. Collectives: all-gather (assemble), reduce-scatter (sum and split), all-reduce (both). All-to-all: MoE routing. Topology: 8 GPUs per node, NVSwitch, IB across, NVL72 = 72. RDMA: GPU to GPU, no CPU. RoCE: cheap IB. NCCL: collectives to packets. torch.distributed: spawn, barrier, gloo/NCCL. Bandwidth: 2(N-1)/N x size/duration, ~400GB/s demo. DDP: split rows, all-reduce grads, one line. Tensor: split columns, all-gather fwd, reduce-scatter bwd, NVLink only. Pipeline: split layers, micro-batches kill bubbles, overlap comm. Strategy: tensor in node, data across, pipeline if needed. Critical batch size caps data parallel.

<ul class="crash-links">
<li><a href="l07-parallelism.html">L07: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### 4D Parallelism (L08)

Bottlenecks: compute, memory. Unit: the data center. Memory: 16B/param Adam. ZeRO-1: shard opt state, 2P comm, free. ZeRO-2: shard grads, incremental reduce, free. ZeRO-3/FSDP: shard params, 2 AG + 1 RS, overlap hides it. DP cap: batch size, critical batch size. Pipeline: cut layers, bubbles, micro-batches, B/W split, bsh point-to-point, slow links. TP: columns up rows down, f/g duality, all-reduce per matmul, TP8 max on GPUs. Activations: 34sbh + 5as/h, floor 34sbh/t. Sequence parallel: shard layernorm leftovers on s. EP: route tokens, all-to-all, prefer over TP for MoE, decouple TP degrees. Context parallel: ring attention for long context. Prescription: fit with TP/EP in node, PP/FSDP across, DP the rest. Recompute to buy batch size.

<ul class="crash-links">
<li><a href="l08-4d-parallelism.html">L08: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Scaling Laws (L09)

View: optimize small, extrapolate big. Log-log line = power law, slope = exponent. Stats: mean est -1, nets -0.1 (~nonparametric, 10 dims). History: 1993 Cortes/Vapnik, 2017 Hestness, 2020 Kaplan, 2022 Chinchilla. Data: mixtures move intercepts, slopes stay. Repetition: 4 epochs safe. Filtering loosens with scale. Arch: GLU good, Performer bad, Switch good. SGD vs Adam: same slopes. Aspect ratio ~100, scale-invariant. Critical batch: noise vs bias limited, B_crit = E_min/S_min, grows as loss drops. LR: 1/width or muP. Upstream vs downstream: fit ppl, verify tasks. Joint: Kaplan N^0.27 (giants), Chinchilla N^0.5 (20 tok/param). Fit: envelope, IsoFLOP (default), parametric. Kaplan lost on: unembedding count, warmup, batch size. Serve: overtrain, small and capable.

<ul class="crash-links">
<li><a href="l09-scaling-laws.html">L09: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Inference (L10)

Metrics: TTFT, latency, throughput. Training 1x cost, inference daily. Agents: tokens = spend, no ceiling. Asymmetry: autoregressive, no seq parallel. KV cache: naive T^3, cache K/V, prefill (parallel, compute) + decode (1 token, memory). Size: B*S*layers*KVheads*H*2*2B. Intensity: MLP B*T, attn S*T/(S+T). Prefill: BS and S/2. Generate: B and ~1. H100 needs ~295. Bottleneck: generation attention, batching cannot help. Llama2-13B/H100: B=1: 0.008s/tok, 124 tok/s. Latency ~B, throughput ~B/(B+c). Shrink KV: GQA (N/K), MLA (compress to C), CLA (layers), sliding window, linear/Mamba. Quant: QAT, PTQ, GPTQ, AWQ. Prune + heal + distill. Spec decode: draft K, verify parallel, min(1,q/p), K=3-4. Serving: continuous batching, selective batching, PagedAttention (blocks, prefix share, copy-on-write).

<ul class="crash-links">
<li><a href="l10-inference.html">L10: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Scaling Advanced (L11)

MiniCPM: muP, LR optimum 1e-2 at all sizes, 5x ladder. WSD: warmup, stable, decay 10-20%, rewind and re-decay. Philosophies: stabilize (muP) vs fit (DeepSeek). StepFun: batch ~ sqrt(D), LR down N up D, smooth contours. Optimizers: small wins lie, watch compute x Chinchilla ratio, tune baselines, Marin blowup. Muon: momentum + Newton-Schultz x5, SVD to UV', matrices only, Kimi K2. muP math: A1 activations O(1), A2 feature learning O(1), LR fan-out/fan-in (SGD) or 1/fan-in (Adam). Breaks: learned norm gain, Lion, big decay. MoE: Kimi sparsity 48, Hunyuan 96 tok/active, LLaMA3 sigmoid, MiniMax bake-off. No silver bullet: scaling is art plus vibes.

<ul class="crash-links">
<li><a href="l11-scaling-advanced.html">L11: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Evaluation (L12)

Good = benchmarks, cost, preference, usage. Eval = abstract construct to concrete metric. Perplexity: P(x) mass on test, best = entropy, GPT-2 zero-shot broke in-distribution. Catches: boring tokens, cloze in disguise, trust. Exams: MMLU (57, few-shot, 90s), MMLU-Pro (10-way, 88), GPQA (PhD, diamond, 94), HLE (private, 64.7). Chat: Arena (pairwise ELO, biases), AlpacaEval (LLM judge, debiased), WildBench (checklists). Agents: SWE-bench (93% verified), Terminal-Bench, CyBench (solved), MLE-bench. Scaffold = half the score. ARC: reasoning vs knowledge, o1/o3 solved, ARC-3 now. Safety: HarmBench, AIR-Bench, GCG, contextual, dual use. Validity: GDPVal, clinician tasks, detect/report/fresh/private contamination defenses, audit quality. Philosophy: purpose picks the bench, evaluate systems not methods.

<ul class="crash-links">
<li><a href="l12-evaluation.html">L12: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Training Data (L13)

Data is the secret sauce. Llama 3 hides it. Stages: pre (raw web), mid (quality, context), post (chat, RL). Big low-quality to small high-quality. Crawling: dynamic web, deep web, auth walls, robots.txt (~50% restricted by mid-2023), anti-bot, terms, licenses. Shadow libraries = piracy. Copyright: everything copyrighted, 75 years, license (CC) or fair use (4 factors). Training = fair use (narrow 2025 rulings), pirating = illegal (Anthropic $1.5B). Common Crawl: monthly since 2007, ~300B pages, WARC/WET, trafilatura better. Pockets: Wikipedia (dumps, poisonable), GitHub (permissive only), arXiv (LaTeX, CC). Datasets: BERT docs, GPT-2 Reddit links, CCNet classifier, C4 rules, GPT-3 classifier+dedup, Pile (Books3), Llama 1 (watershed), Refined/FineWeb, DCLM (1.4% kept), Nemotron (synthetic). Rules vs classifiers. The funnel is the lever. Stack: code process as data. Common Pile: license-only, reasonable.

<ul class="crash-links">
<li><a href="l13-training-data.html">L13: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Data Pipeline (L14)

Transform: HTML to text (boilerplate out, tables hard), PDFs rare and valuable (truncated, OCR). Filter: T small good, R huge raw, subset like T. fastText linear, KenLM generative. Quality = defined. Threshold depends on token budget (Ryan: quality wins early, loses after epoching). Dedupe: exact (C4 3-sentence spans), near (Jaccard 0.99). Gas mask 61k. Dedupe globally, decontaminate. MinHash: P(collision) = Jaccard. LSH: b bands of r, S-curve, threshold (1/b)^(1/r), 0.64 at center. Mixing: distribution over sources. 50-epoch trap. UniMax caps. Diversity. RegMix: small swarm, regression, optimize, scale. Two leaps of faith. Simulated epoching. Post-training: environments, tasks, teachers. OpenThoughts 1.2M, SWE-smith 50k, SWE-Zero 300k no-exec, 12M scaled.

<ul class="crash-links">
<li><a href="l14-data-pipeline.html">L14: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Post-training (L15)

GPT-3 to ChatGPT = SFT + RL. SFT data: FLAN (unnatural), self-instruct, Alpaca/Vicuna (distillation), Open Assistant (experts), WizardLM/Tulu3 (synthetic), agentic (tool calls). Shifts: chatty, expert, tools. Pitfalls: style vs capability, tail knowledge hallucinates, RL recalibrates. Safety: violation vs false refusal, 500 examples surgical, Llama 2 few thousand, OLMo 50k from WildChat. Mid-training: decay phase, base model is a lie. RLHF: maximize reward, may collapse. Raters differ from writers. Pipeline: sample, rank, reward model, PPO + KL. Annotators: experts $100+/hr, demographics transfer, owls transfer, formatting vs factuality. Model annotation: 10x cheaper, Zephyr gave up, length hacking. PPO: policy gradient, off-policy, clip. DPO: tilt reference by reward, up good down bad, surprise-scaled. Failures: over-optimization, mode collapse, miscalibration.

<ul class="crash-links">
<li><a href="l15-post-training.html">L15: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### RLVR (L16)

RLHF over-optimizes learned rewards. RLVR: verifiable rewards (math, code), compute keeps helping. PPO: REINFORCE + clipping, 37 details, value net memory, gamma=lambda=1 bandit trap. GRPO: group z-score advantage, no value net, one page. Std-0 trap: trivial/impossible groups explode under std normalization. Dr. GRPO: drop std and length normalization, advantages go to 0 on no-signal groups. Aha moment was in base. R1-Zero: base + GRPO, accuracy + format, outcome only, near o1. Production: long-CoT SFT, language consistency, RLHF finish. Distill R1 CoTs into Qwen/Llama. Kimi: best-of-8 curriculum, medium difficulty, length compression, RL beats expert iteration. Qwen 3: thinking fusion, early exit, RL on 4k. Coder-Next: mid-train agents, 4 experts distilled, 70.6% SWE-bench at 3B active. Hacking: git history, Lean strings, answer equivalence rabbit hole. Infra: rollouts stall, off-policy destabilizes. Moral: it is all about the reward.

<ul class="crash-links">
<li><a href="l16-rlvr.html">L16: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Multimodality (L17)

Omni: any in, any out. Tokenize everything. CLIP: 2N contrastive, 400M pairs, ViT-L/14, zero-shot beats ResNet. Text gives semantics. SigLIP: binary loss, batch decoupled, chunked rotation, 5 vs 10 days. LLaVA: CLIP + W + Vicuna, align then fine-tune, 158k synthetic. Projector = space alignment. Freeze both ends, train the bridge. AnyRes: crops not downsampling, overview + details. Token budget: 1344px doc = 16 crops + 1 overview = 9,792 image tokens. Resolution is a token purchase, the context window is the bottleneck. Qwen-VL: cross-attention, 1.4B stage 1. Qwen2: dynamic res, 2x2 compression, M-RoPE. Qwen3: SigLIP-2, interleaved frequencies, timestamps, sqrt loss, DeepStack, 256K context. Chameleon: VQ-VAE discrete, 1024 tokens per image, unstable (entropy), loses detail. State: continuous encode, diffusion generate, weight modalities.

<ul class="crash-links">
<li><a href="l17-multimodality.html">L17: full lesson</a></li>
</ul>

</div>

<div class="cheat-block" markdown="1">

### Serving Inference (L18)

Token lifetime: schedule, KV lookup, execute, sample, repeat. Workloads: coding (long in), chat (fast first token), agents (turns), batch (throughput). Prefill: compute bound, 10k in 1 out, once. Decode: memory bound, 1 token per full model load, per step. Decode tax: 70B bf16 = 140 GB per token, 42 ms at 3.3 TB/s. Decode scales with bandwidth, not flops. Disaggregate fleets. LPU/Cerebras for decode. Disaggregate when the mix is bimodal. Continuous batching: per-step joins, KV memory is the limit. KV cache: radix prefix sharing, GPU/CPU/SSD tiers, LRU, prefetch. Cache-aware routing: fresh vs warm pools, 40% faster. Megakernels: fuse ops, overlap loads, 30-70% speedup, 72% bandwidth, batch 17 means starting over. Parcae: loop blocks, spectral radius under 1, scale recurrence with data. Co-design: size to chip memory, match quantization, compress KV for agents.

<ul class="crash-links">
<li><a href="l18-inference.html">L18: full lesson</a></li>
</ul>

</div>

</div>
