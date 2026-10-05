---
page_id: cs229s-cheatsheet
course_slug: cs229s
course_name: "CS229S: Systems for Machine Learning"
course_order: 3
order: 900
nav: "CS229S · Cheatsheet"
title: "CS229S Cheatsheet"
summary: "Every key fact from CS229S on one dense page: definitions, formulas, numbers, decisions, mistakes, interview lines."
---

Grows with the course. Each finished lecture adds its blocks here.

<div class="cheat-cols" markdown="1">

<div class="cheat-block" markdown="1">

### The three gaps

- Compute demand 32x per 2 years vs Moore's law 2x (Sevilla 2022)
- Model size outruns accelerator memory: A100 40/80 GB, TPUv3 32 GB, V100 32 GB
- Asymptotics outrun wall-clock: linear attention loses to FlashAttention in runtime
- Training: tens of millions of dollars upfront. Inference: under $0.0001 per call, compounds with users

</div>

<div class="cheat-block" markdown="1">

### Bottleneck diagnosis

- Time = max(Tmem, Tmath). Tmath = ops / compute BW. Tmem = bytes / memory BW
- Compute bound: Tmath > Tmem. Memory bound: Tmem > Tmath
- Arithmetic intensity = FLOPs / bytes moved
- Ridge = peak FLOPs / peak BW. A100: 312 TFLOPS / 1935 GB/s = 161 FLOPs/byte
- Below ridge: memory bound (buy bandwidth, fuse, tile). Above: compute bound (buy FLOPs)
- Matmul FP16: FLOPs 2MNK, bytes 2(KM+NK+NM). N=128: AI 124, memory bound. N=8192: AI 2731, compute bound

</div>

<div class="cheat-block" markdown="1">

### Device block (defined here, reused by MS&E435)

- One GPU: 108 SMs, registers 256 KB/SM, shared 192 KB/SM, L2 40 MB, HBM 40 GB at ~2 TB/s
- Kernel: load HBM to SRAM/registers, compute, write back
- Warps: 32 threads, SIMT lockstep. Max threads = SMs x blocks/SM x threads/block
- Thread memories: global (all threads), shared (block), registers (thread)
- Coalesced access: adjacent threads touch adjacent elements
- Node: 8 GPUs on NVLink (fast). Cluster: nodes on InfiniBand (much slower)
- Tensor cores: 16x16 GEMM tiles, up to 16x on H100. Big tiles amortize launch

</div>

<div class="cheat-block" markdown="1">

### Six performance principles

1. Fusion: composite ops on resident data, fewer HBM trips
2. Parallelization: saturate all SMs (need 108+ outputs on A100)
3. Tiling: shared-memory subsets, threads collaborate, halve loads
4. Caching vs recompute: cache when compute bound, recompute when memory bound
5. Pipelining: overlap compute with memory transfers
6. Hardware-specific: tensor cores, intrinsics, new instructions

</div>

<div class="cheat-block" markdown="1">

### FLOP counting

- MLP layer forward: 2BHN. Rule of thumb: ~2 x batch x params
- Backward: 4BHN, ~2x forward. Must cache activations
- Training step: ~3x one forward pass (1x fwd + 2x bwd)
- Attention per token without cache: K,V full seq 2x(2Nd^2). Q 2d^2. QK^T 2Nd. AV 2Nd
- With KV cache: K,Q,V current token 3x(2d^2). QK^T 2Nd. AV 2Nd

</div>

<div class="cheat-block" markdown="1">

### KV cache numbers

- Per token: 2 x 2 x nlayers x dmodel bytes (FP16)
- 7B (24 layers, d=2048) on 40 GB A100: 14 GB weights, 200k B/token, 130k tokens, batch 128 at 1024 length
- Tmem/Tmath = 208 at d=2048: KV for 1 token costs the memory time of 208
- KV caching wins when compute bound
- Decode AI ~ 2 x batch. GPT-2-XL: >3 GB read per token. RTX 4090: 30k tokens/s compute, ~300 tokens/s actual
- Latency: time per item. Throughput: items per second. Bandwidth: hardware ceiling

</div>

<div class="cheat-block" markdown="1">

### Speculative decoding

- Draft K tokens cheaply, verify in one big-model batch, accept the agreed prefix
- 2-3x speedup on Chinchilla 70B, T5 11B, LaMDA 137B
- Draft options: small model (~15x smaller is sweet), Medusa heads (~2x cap, mean-field), lossy tricks (INT4, skip layers, no extra model)
- Limit: draft agreement rate

</div>

<div class="cheat-block" markdown="1">

### FlashAttention

- Standard attention: O(N^2) memory and compute, memory bound
- Never materialize N x N: tile QKV through SRAM
- Online softmax: running sum L and running max m per tile, rescale partial outputs. Exact
- Backward: cache L, M (size N), recompute tiles. +13% FLOPs
- HBM: 9x less traffic (40.3 to 4.4 GB). Runtime: 6x faster (41.7 to 7.3 ms)
- Results: MLPerf BERT +15%, 3x vs HF at 1k, LRA 2.4-2.8x, Path-X solved
- FA-2: CUTLASS 3, parallelize over seq/batch/heads
- Approximations: sparse (windows, BigBird), low-rank (Linformer, causality hard), kernel (linear transformer, needs custom kernel)

</div>

<div class="cheat-block" markdown="1">

### Pruning and sparsity

- Goal: min L(x;Wp) with ||Wp||0 < T
- Unstructured: accurate, scattered, rarely faster. Structured: regular, hardware-fast
- 2:4 N:M: 2 of every 4 zero. Ampere sparse tensor cores. 50% weights gone
- Iterative prune + fine-tune beats one-shot
- Heuristics: magnitude (fine/coarse), regression (min ||Z-Zhat||F^2), per-layer sensitivity analysis

</div>

<div class="cheat-block" markdown="1">

### Quantization

- Storage = params x bits. Add O(N), multiply O(N^2) in bit-width
- FP32: (-1)^s (1+f) 2^(e-127), bias 127. Subnormal: e=0. Inf/NaN: e=all-ones
- FP16: 5 exp + 10 frac. BF16: 8 + 7 (range for stable training)
- K-means (Deep Compression): log2(S)-bit index + S centroids. 4x4/S=4: 64B to 20B, 3.2x. Compute stays FP32. Huffman +20%
- Linear: r = S(q - Z). qY = (SwSx/Sy)(qW-Zw)(qX-Zx) + Zy, integer-only
- LLM.int8(): outlier-aware mixed precision to 175B

</div>

<div class="cheat-block" markdown="1">

### Distillation

- Student mimics teacher: logits, features, weights, gradients, sparsity
- Logits loss: cross-entropy E(-pt log ps) or L2 on probabilities
- Dark knowledge: the teacher's uncertainty, not just labels

</div>

<div class="cheat-block" markdown="1">

### Fine-tuning and PEFT

- Base model completes. Instruction tuning teaches following instructions
- RLHF: human preference ranks -> reward model -> RL optimize. <1% labels (Christiano 2017)
- Constitutional AI: ~10 principles replace ~10k labels. SL on self-critique, then RLAIF
- Fine-tune memory: >10x trainable params (weights + activations + grads + optimizer)
- PEFT: selective / additive-adaptive / hybrid
- Prompt tuning: m tunable tokens, m-by-e matrix, frozen base
- LoRA: h = Wx + BAx = (W+BA)x. Merge W_LoRA = W + BA. r << d. Wq, Wv best. Zero inference latency

</div>

<div class="cheat-block" markdown="1">

### Attention-free architectures

- RNN: slow train, fast O(1) sample, infinite context. Transformer: fast train, slow sample, finite
- Long conv: kernel = input size. Pad kernel_size-1 for causality. Naive O(N^2)
- FFT theorem: time conv = pointwise frequency multiply. Cooley-Tukey: O(n log n)
- Linear recurrence = convolution: train parallel, sample recurrent (S4)
- S4: solved LRA/Path-X. LM gap at 360M: attn 8.39 vs SSMs 9.79-13.13
- Recall needs input-dependent mixing. Conv mixing is diagonal-constant. Direction: Mamba, GLA, Based

</div>

<div class="cheat-block" markdown="1">

### Parallelism

- Data: split batch, replicate weights, all-reduce grads
- Tensor (Megatron): shard attn+MLP, all-reduce per block. Inside node only (NVLink)
- Pipeline: stages hold layers, microbatches flow. 1F1B and interleaved cut bubbles. Point-to-point only
- Memory: 16P bytes mixed-precision Adam (2P params + 2P grads + 12P state)
- ZeRO: S1 4P+12P/Nd, S2 2P+14P/Nd, S3 16P/Nd at 1.5x comm. FSDP = ZeRO as API
- Ring all-reduce: 2(N-1) iterations, 2(N-1)X/(NB) seconds, bandwidth-optimal
- Decision: fits -> data. One fast node -> tensor. Many nodes -> pipeline. PTD combines. Alpa automates

</div>

<div class="cheat-block" markdown="1">

### Interview one-liners

- "Below the ridge, move less data; above it, do more math."
- "FlashAttention never writes the N by N matrix: tile, rescale online, recompute backward."
- "Decoding is memory bound: one token costs a full model read."
- "LoRA merges into the weights, so inference pays nothing."
- "Sparsity only helps if the pattern matches the hardware."
- "Keep tensor parallelism inside NVLink; span nodes with data or pipeline."
- "KV caching wins when compute bound. Recompute wins when memory bound."
- "Recall needs input-dependent mixing: that is why attention survives."

</div>

<div class="cheat-block" markdown="1">

### Classic mistakes

- Optimizing math on a memory-bound kernel (the toy proves it buys nothing)
- Judging algorithms by big-O instead of wall-clock on target hardware
- Unstructured pruning with no speedup (access stays scattered)
- Naive quantization past 6B params (outliers break it; use LLM.int8-style)
- Tensor parallelism across nodes (InfiniBand stalls the all-reduces)
- One global sparsity ratio (layers differ; run sensitivity analysis)
- Confusing latency, throughput, and bandwidth
- Forgetting activations in training memory (the 16P is before them)

</div>

</div>
