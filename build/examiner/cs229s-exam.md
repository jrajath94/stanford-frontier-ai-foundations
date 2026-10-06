# CS229S Examiner Report — Gate E1
Course: CS229S (Systems for Machine Learning), all 10 lessons
Role: EXAMINER (Pipeline gate E1). 30 interview questions, answers derived from lesson text ONLY.
Rule applied: any question unanswerable from the lesson text is marked FAIL with what is missing.

---

## L01 — Introduction (Q1-Q2)

**Q1 (L01).** Quote the two growth rates that make hardware alone a losing bet, and work out the missing factor per 2-year window.
**A.** Training compute in the deep learning era grows 32x every 2 years; Moore's law gives roughly 2x every 2 years (l01-introduction.md:79). Two years pass, hardware doubles, demand multiplies by 32, so a factor of 16 must come from algorithms, parallelism, and utilization (l01:79-88). That 16x gap compounds each window.

**Q2 (L01).** DepthwiseConv2D is 5.00% of FLOPs but 65.30% of runtime, while Conv2D is 94.67% of FLOPs and 34.20% of runtime (l01-introduction.md:331-335). Explain the mechanism and the lesson's rule for judging speed.
**A.** The depthwise convolution does little work per memory access, so the chip waits on memory and achieved throughput collapses; it looks cheap in FLOP count but runs expensive on the clock (l01:342-352). The rule: speed is work divided by achieved throughput, not FLOP count — judge by measured runtime on target hardware (l01:346-359, 116-124).

## L02 — Sequence Models (Q3-Q4)

**Q3 (L02).** Work the attention toy end to end: tokens counselor=[1,0], helped=[0,1], frame=[1,1], query q=[1,1], identity projections. What are the scores, weights, and the new "frame"?
**A.** Scores are dot products: 1, 1, 2 (l02-sequence-models.md:351-355). Softmax: e^1=2.72, e^1=2.72, e^2=7.39, total 12.83, weights = [0.21, 0.21, 0.58] (l02:356-359). New "frame" = 0.21*[1,0] + 0.21*[0,1] + 0.58*[1,1] = [0.79, 0.79] (l02:360-363). The query-key match, not a fixed rule, decided the mix.

**Q4 (L02).** What does the causal mask do to the score matrix, and how does it make the KV cache possible?
**A.** Before softmax, every score (i,j) with j > i is set to negative infinity; softmax turns that into exactly 0, so token 3 sees only tokens 1, 2, 3 and the matrix becomes a triangle (l02:480-489). When token N+1 arrives, rows 1..N never change — past keys and values are final — so they can be cached instead of recomputed (l02:488-490).

## L03 — Hardware-Aware Design (Q5-Q7)

**Q5 (L03, depth: roofline model).** A kernel has AI 400 FLOPs/byte. Diagnose it on a B200, name the bottleneck, and prescribe the optimization. Then explain why FP8 raises the B200 ridge to 562.
**A.** B200 ridge = 2,250 TFLOPS / 8,000 GB/s = 281 FLOPs/byte (l03-hardware-aware-design.md:144-150). AI 400 is above 281, so the kernel is compute bound on a B200: optimize the math (tensor cores, FP8, better tiling), not the bandwidth; the same kernel is also compute bound on an H100 (ridge 295), but an AI-200 kernel flips from compute bound on the A100 (ridge 161) to memory bound on the H100 (l03:254-260). FP8 doubles peak FLOPs to 4,500 TFLOPS while bandwidth stays 8,000 GB/s, so the ridge doubles to 562: lower precision makes more kernels memory bound because each byte now feeds twice the math (l03:150-158, 261-263).

**Q6 (L03).** Work out whether C = AB with M=K=8192, N=128 (FP16) is memory or compute bound on an A100, and what changes when N=8192.
**A.** AI = 2MNK / 2(KM + NK + NM). At N=128, AI = 124.1, below the A100 ridge of 161: memory bound — the small-batch inference regime (l03:167-183). At N=8192, AI = 2730.6, above the ridge: compute bound — the large training regime (l03:181-183). Same operation, different shapes, different bottleneck.

**Q7 (L03).** Tiling raised matmul AI from 0.5 to 1.0 in the worked example. What physically changed, and why only 2x?
**A.** Data reuse changed, not the math: the naive thread loads 2K values per output; the 2D-tiled thread loads 4K values but computes 4 outputs, sharing rows and columns in fast on-chip memory, so bytes per operation fell (l03:229-245, QA ~390). Only 2x because the 2x2 tile is small — each loaded byte serves only 2 outputs; real A100-sized tiles (e.g. 128x128, reusing each value 128 times) push intensity orders of magnitude higher (l03:245-250, 420).

## L04 — Transformer Performance (Q8-Q11)

**Q8 (L04, depth: KV cache math).** Work the KV cache byte math for a 70B model (80 layers, dmodel 8192, FP16) and say at what point the cache rivals the weights.
**A.** Per token: 2 (K and V) x 2 bytes x 80 layers x 8192 = 2.6 MB (l04-transformer-performance.md:159-162). At 32K context: 2.6 MB x 32,768 = 86 GB of cache against 140 GB of weights (l04:161-163). With grouped-query attention at 8 KV heads instead of 64, the bill falls 8x to 0.3 MB per token (l04:163-165).

**Q9 (L04, depth: speculative decoding).** State the expected-tokens formula, work it for K=5 with alpha=0.8 and alpha=0.3, and name the two conditions under which the speedup collapses.
**A.** Expected tokens per target forward pass = (1 - alpha^(K+1)) / (1 - alpha) (l04:318-322). K=5, alpha=0.8: 0.8^6=0.262, (1-0.262)/0.2 = 3.69 tokens (l04:323-325). K=5, alpha=0.3: (1-0.3^6)/0.7 = 1.43 — barely better than 1 (l04:326-328). The speedup collapses when (1) the draft disagrees too often (low alpha: a disagreeing draft is a tax, not a trick), or (2) the batch is large, so verification leaves the memory-bound regime and the "free" compute was never free (l04:326-330).

**Q10 (L04).** Work the 6N rule for a 7B model trained on 1T tokens, and translate it to H100 time.
**A.** One training step costs forward 2N + backward 4N = 6N FLOPs per token (l04:92-96). For 7B: 6 x 7e9 = 4.2e10 FLOPs per token; on 1T tokens, 4.2e22 FLOPs total (l04:96-98). On one H100 (989 TFLOPS): 4.2e22 / 989e12 = 4.25e7 s = 492 days; on 1,000 H100s at 50% utilization, about 1 day (l04:98-100).

**Q11 (L04).** Why is prefill compute bound while decode is memory bound on the same chip?
**A.** Prefill feeds N prompt tokens through every layer at once; the N-by-d-by-d matmuls have high data reuse and AI far above the ridge (l04:254-258). Decode generates one token per step, streaming the full weights for one vector-matrix multiply per layer: about 2 FLOPs per byte, far below the ridge (l04:258-263). That is why batching helps decode (more tokens per weight read) but barely helps prefill (already compute bound) (l04:264-268).

## L05 — GPU Execution Model (Q12-Q13)

**Q12 (L05).** What is warp divergence, what does a 16/16 split cost, and why are attention masks fine but per-token early exits not?
**A.** Under SIMT one instruction drives 32 threads; a 16/16 branch split runs branch A with half the lanes masked, then branch B with the other half masked — both paths cost full time, about a 2x penalty (l05-gpu-execution-model.md:57-68). An attention mask is the same branch decision for every thread in the warp: no divergence. A per-token early exit makes each thread decide from its own data: the warp splits and pays both paths. Uniform control flow is free. Data-dependent control flow is not (l05:361-363).

**Q13 (L05).** 32 threads in a warp each read one FP32 from scattered addresses. Quantify the wasted bandwidth and the fix.
**A.** Each 4-byte read lands in a different 128-byte segment, so the hardware issues 32 transactions of 128 bytes: 4,096 bytes fetched for 128 useful — 97% waste (l05:137-145, QA ~354). Fix: rearrange data so adjacent threads touch adjacent elements; the same 32 reads become one 128-byte transaction with zero waste (l05 QA ~354-355).

## L06 — FlashAttention (Q14-Q16)

**Q14 (L06, depth: FlashAttention tiling).** Work the online-softmax rescaling with numbers: one row, tiles with scores [1,2] and [3,4]. Show the running statistics converge to the true softmax.
**A.** True softmax: max 4, L = e^-3+e^-2+e^-1+1 = 0.0498+0.1353+0.3679+1 = 1.553 (l06-flash-attention.md:288-292). Tiled: tile 1 local max 2, L(1) = e^-1+1 = 1.368; tile 2 max 4, L(2) = 1.368 (l06:293-295). Running max becomes 4; rescale tile 1's sum by e^(2-4) = 0.1353: 1.368 x 0.1353 = 0.185; add tile 2's 1.368: total 1.553 (l06:296-297). The running statistics converge to the true max and sum, tile by tile, so the output is exactly standard attention (l06:298-303).

**Q15 (L06).** Why is the running max tracked at all, instead of only the running sum?
**A.** Numerical stability: without subtracting the max, e^x overflows for large scores — e^100 is not representable (l06:305-311). The max is a whole-row quantity like the sum, so it gets the same online treatment: keep the running max and rescale earlier tiles' exponentials by e^(old max - new max) when it moves (l06 QA on running max). Both statistics cost O(1) per row; the N-by-N matrix cost O(N^2) (l06:312-317).

**Q16 (L06).** What did FlashAttention-2 fix over FA1, in measured numbers, and why does each GPU generation need a rewrite?
**A.** FA1 parallelized over batch and heads but scheduled sequence-length work suboptimally, causing excess reads and writes. FA2 parallelizes over sequence length too, uses CUTLASS 3 primitives, and trims non-matmul overhead like rescaling (l06:366-377). Measured on A100: 50-73% of peak throughput versus 25-40% for FA1 — roughly double the training throughput from scheduling alone (l06:378-380). Each generation needs a rewrite because the hardware's fast path changes (async copy engines, new matrix instructions, new precisions); the algorithm is stable but the implementation must be re-ported (l06 QA ~700).

## L07 — Memory-Efficient Networks (Q17-Q19)

**Q17 (L07, depth: quantization formats).** Contrast NVFP4 and MXFP4 on format, block size, and scales; work the 70B memory number; state the deployment decision rule.
**A.** NVFP4: E2M1 values (1 sign, 2 exponent, 1 mantissa), block size 16, E4M3 scales per block. MXFP4 (OCP standard): E2M1 values, block size 32, E8M0 power-of-two scales per block (l07-memory-efficient-networks.md:450-458). 70B in NVFP4: 70B x 0.5 bytes = 35 GB of weights against 140 GB in FP16 (l07:460-463). Decision rule: NVFP4 when serving on Blackwell with good calibration; INT4 GPTQ/AWQ when serving on older GPUs or consumer cards (l07:464-470).

**Q18 (L07).** Why does naive quantization break past ~6B parameters, and what are the two fixes' opposite strategies?
**A.** Outlier features: about 0.1% of dimensions develop values tens of times larger than the rest, so one scale factor cannot cover both outliers and normal values (l07:388-396). LLM.int8() splits the problem: compute the outlier dimensions in FP16, quantize the rest to INT8 (l07:398-401). SmoothQuant moves the difficulty: multiply activations by a per-channel smoothing factor s and divide the weights by s, so both quantize cleanly to INT8 (l07:406-413). Split the problem, or rebalance it.

**Q19 (L07).** Write the linear-quantization mapping, and derive the integer-only matmul from it.
**A.** r = S(q - Z), with S and Z solved from rmin/rmax and the bit-width range (for N bits, q in [-2^(N-1), 2^(N-1)-1]) (l07:341-351). Substituting Y=Sy(qY-ZY), W=Sw(qW-Zw), X=Sx(qX-Zx) into the matmul: qY = (Sw Sx / Sy)(qW - Zw)(qX - Zx) + Zy — all integer math with one rescale to N bits, so weights and compute are both integer: storage and speed improve together (l07:353-363).

## L08 — Fine-tuning and PEFT (Q20-Q22)

**Q20 (L08, depth: LoRA serving).** Compare Punica and S-LoRA: what problem does each solve, with the lesson's numbers, and what does Compressed Multi-LoRA add?
**A.** Punica solves batched compute: per-request adapters break the batched GEMM (request i needs Y_i = X_i @ (W + B_i A_i)), and its SGMV kernel gathers each request's adapter weights into one launch — 12x higher throughput than 2023 serving systems, +2ms per token, ~2% overhead (l08-finetuning-and-peft.md:515-536). S-LoRA solves memory capacity: unified paging puts KV cache and adapter weights in one pool, serving 2,000 adapters on one A100 80GB at 7.6+ req/s (30x HuggingFace PEFT), paging cold adapters to CPU since 10-50 MB adapters transfer negligibly over PCIe (l08:542-556). Compressed Multi-LoRA shrinks the adapters: joint diagonalization factorizes B_i A_i into U Sigma_i V^T with U, V shared, so per-adapter params fall from 2dr to r — 1.6x throughput, 99%+ quality kept, in vLLM (l08:562-567).

**Q21 (L08).** Work the 10x fine-tuning memory figure for a 7B model.
**A.** FP16 weights: 14 GB. FP16 gradients: 14 GB. Adam keeps three FP32 states per parameter — master weight copy 28 GB, momentum 28 GB, variance 28 GB = 84 GB (l08:326-331). Subtotal 112 GB, already 8x the weights; add activations for the backward pass and it exceeds 140 GB — over 10x (l08:331-334).

**Q22 (L08).** What is PPO-based RLHF's four-model memory bill, and which component does GRPO delete?
**A.** PPO holds four models plus each one's optimizer state: the policy being trained, the frozen reference policy (for the KL penalty), the reward model, and the value model (critic) (l08:184-191). GRPO samples a group of responses per prompt and computes advantages by comparing within the group, so the learned value model is gone — the group baseline replaces the critic, deleting one model from memory (l08:270-280).

## L09 — Efficient Architectures (Q23-Q25)

**Q23 (L09, depth: MoE parallelism).** What is node-limited routing, what does M=4 buy in traffic math, and what is V3's auxiliary-loss-free alternative to the load-balance loss?
**A.** V3 routes each token to 8 of 256 experts; unconstrained, those 8 could sit on 8 different nodes. Node-limited routing caps each token's experts at M=4 nodes, halving the worst-case cross-node fan-out from 8 to 4 nodes per token per MoE layer (l09-efficient-architectures.md:412-425). Combined with DeepEP's custom all-to-all kernels, the paper reports near-zero all-to-all overhead (l09:420-425). Instead of an auxiliary load-balance loss (which trades quality for balance), V3's auxiliary-loss-free strategy adjusts a per-expert bias from observed utilization: over-used experts get bias lowered, under-used raised — no loss term, no quality tradeoff (l09:447-457).

**Q24 (L09).** Why does DeepSeek-V3 train with no tensor parallelism, and how does DualPipe replace it?
**A.** TP all-reduces after every attention and MLP block, judged too expensive at 671B scale; instead V3 uses 16-way pipeline parallelism with the DualPipe schedule (l09:482-490). DualPipe duplicates the pipeline so micro-batches flow from both ends: the fill of one direction overlaps the drain of the other, shrinking the bubble toward zero, and the expert-parallelism all-to-all hides inside the overlapped compute (l09:486-496). Data parallelism runs ZeRO-1 underneath (l09:497-505).

**Q25 (L09).** Why do convolutional sequence models fail at recall while attention succeeds? Work the toy.
**A.** Recall needs the mixing between tokens to depend on content. Attention's mixing matrix is input-dependent — queries and keys are functions of the input — so on "C 8 ... what is the value of C?", the query from "C?" matches the key from "C" and the value 8 routes to the output (l09:259-287). A convolution's mixing matrix is diagonal-constant: position i always mixes i-k with weight w_k regardless of tokens, so no fixed pattern can route arbitrary content lookups — the failure is structural (l09:283-293).

## L10 — Parallelism (Q26-Q30)

**Q26 (L10, depth: ZeRO stages).** Work ZeRO stages 1-3 for 10B parameters on Nd=8 GPUs: the formulas and the numbers.
**A.** Baseline 16P = 160 GB per GPU: fits nowhere (l10-parallelism.md:156-165, 194-198). Stage 1 (optimizer state): 4P + 12P/Nd = 40 + 15 = 55 GB (l10:167-170, 199-201). Stage 2 (+ gradients): 2P + 14P/Nd = 20 + 17.5 = 37.5 GB (l10:172-175, 202-203). Stage 3 (+ parameters): 16P/Nd = 20 GB, at the price of about 1.5x more communication volume from the parameter all-gathers (l10:176-180, 204-206). Rule: match the stage to the memory gap, not the maximum (l10:216-220).

**Q27 (L10, depth: ring all-reduce).** Why is ring all-reduce bandwidth-optimal? Work it for N=4, X=1 GB, and N=16, X=4 GB.
**A.** The ring runs 2(N-1) iterations: N-1 scatter-reduce rounds then N-1 all-gather rounds (l10:239-248). Each node sends 2(N-1)X/N bytes, so per-node traffic approaches 2X as N grows — independent of GPU count, which is the bandwidth-optimality claim (l10:252-262). N=4, X=1 GB: 6 iterations, 2x3x1/4 = 1.5 GB per node, 15 ms at 100 GB/s NVLink (l10:250-259). N=16, X=4 GB: 30 iterations, 2x15x4/16 = 7.5 GB per node, approaching the 2X = 8 GB limit (l10 QA on N=16). Caveat: 2(N-1) iterations pay latency, so small messages want fewer, fatter hops — bandwidth-optimal is not latency-optimal (l10:260-262, QA follow-up).

**Q28 (L10, depth: context parallelism).** When does context parallelism beat the other splits? Work the 128K activation math and describe DeepSpeed-Ulysses' communication pattern.
**A.** Reach for it when activations dominate the memory bill — the long-context regime (l10:403-412). Worked: one layer at batch 1, N=128K, d=7168 stores about 34 x N x d = 31.9 GB per layer; 61 layers = about 1.95 TB of activations, exceeding the 1.3 TB FP16 weight bill; split 8 ways, each GPU holds 244 GB (l10:433-444). Ulysses splits the sequence across GPUs, all-to-alls Q/K/V so each GPU holds the full sequence for a subset of heads, computes attention per head, then all-to-alls the output back — a few all-to-alls per layer, near-linear strong scaling 64 to 256 GPUs at 131K (l10:414-422).

**Q29 (L10).** Work the pipeline bubble for 4 stages and 8 microbatches, and name the two tuning knobs and their trades.
**A.** Fill and drain take 3 steps each: 8+4-1 = 11 total steps, 3 idle; bubble = (p-1)/(m+p-1) = 3/11 = 27% of GPU time wasted (l10:341-349). Doubling microbatches to 16: 3/19 = 16% (l10:350-353). Knobs: microbatch size (larger = higher arithmetic intensity, smaller = smaller bubble) and schedule (GPipe all-forwards-then-all-backwards vs 1F1B interleaving, which shrinks the bubble and the activation memory GPipe holds) (l10:355-366).

**Q30 (L10).** Walk through Megatron's column-row trick on Y = GeLU(X W1) W2 and state the per-layer communication price of tensor parallelism.
**A.** Split W1 by columns across GPUs: each computes half the hidden units, no communication needed because GeLU is elementwise (l10:298-308). Split W2 by rows: each GPU holds the rows matching its hidden half and produces a partial output; one all-reduce sums the partials (l10:306-310). Attention follows the same pattern (Q,K,V column-split by heads, output projection row-split): two matmuls, one synchronization per block — two all-reduces per layer, every layer, every step (l10:308-313).

---

## Verdict

**PASS: 30/30 answerable from the lesson text.**

All 30 questions were written against facts, numbers, and mechanisms stated verbatim in the ten lesson files, with exact lesson:line citations above. No question required information outside the text; no question is marked FAIL. Coverage: 2 questions each for L01, L02, L05; 3 for L03, L06, L07, L08, L09; 4 for L04; 5 for L10; plus the 10 mandated depth follow-ups (roofline Q5, KV cache math Q8, speculative decoding Q9, FlashAttention tiling Q14, quantization formats Q17, LoRA serving Q20, MoE parallelism Q23, ZeRO stages Q26, ring all-reduce Q27, context parallelism Q28). No lesson fixes were made.
