# CS336 EXAM — Gate E1

**Course:** CS336: Language Modeling from Scratch (Spring 2026)
**Role:** EXAMINER (gate E1). 36 depth interview questions, 2 per lesson.
**Rule:** every question is answerable FROM THE TEXT ONLY. Each item names the lesson section/line that contains the answer. No invention.

---

## Lesson 1 — Tokenization (l01-tokenization)

### L01-Q1

**Question:** BPE and WordPiece both grow a vocabulary by merging pairs, but their first merge differs. On a corpus containing the words "low", "lower", "lowest", which pair does BPE merge first, which pair does WordPiece merge first, and what is the one-sentence mechanism behind WordPiece's choice?

**Answer:** BPE merges the pair with the highest raw frequency, so it merges (l, o) first. WordPiece scores each candidate pair as count(pair) / (count(first) * count(second)) and merges (s, t) first, because the score is 1.00 (s and t never appear apart in the corpus). The mechanism: "WordPiece grows by likelihood lift" — it prefers pairs whose co-occurrence is surprising under independence, while BPE grows by raw frequency.

**Source:** l01-tokenization — section "WordPiece grows by likelihood lift" (BPE merges by frequency vs WordPiece's likelihood-lift scoring, worked on the "low / lower / lowest" toy).

### L01-Q2

**Question:** A serving operation handles 10M requests per day with 500 words per request, priced at $2.50 per million tokens. Work the daily cost under two tokenizers: fertility 1.3 vs fertility 1.6. What is the annual gap?

**Answer:** At fertility 1.3: 10M x 500 x 1.3 = 6.5B tokens/day; 6.5B / 1M x $2.50 = $16,250/day. At fertility 1.6: 8.0B tokens/day = $20,000/day. The daily gap is $3,750, which annualizes to about $1.37M per year. The lesson: "tokenizer choice at serving scale" is a dollars-and-cents decision, not an academic one.

**Source:** l01-tokenization — section "tokenizer choice at serving scale" (the serving-arithmetic worked example, fertility 1.3 vs 1.6 at $2.50/M tokens).

---

## Lesson 2 — Resource Accounting (l02-resource-accounting)

### L02-Q1

**Question:** The lesson gives the 6ND attention correction: the quadratic attention term dominates the 6ND estimate when the correction ratio N/(6d) exceeds 1. For d_model = 4096, compute the ratio at N = 4,096, 32,768, and 131,072, and state the crossover sequence length in terms of d_model.

**Answer:** N/(6d) with d = 4096: at N = 4,096 the ratio is 0.17 (attention negligible); at N = 32,768 it is 1.33; at N = 131,072 it is 5.3. The crossover is near N = 6 x d_model (about 24K tokens): beyond that point "6ND breaks" and attention, not the linear matmuls, dominates training FLOPs.

**Source:** l02-resource-accounting — section "when 6ND breaks" (the N/(6d) correction ratio, worked at three sequence lengths, crossover at 6 x d_model).

### L02-Q2

**Question:** You have 8 H100 GPUs (80GB each). The lesson's rule: 2 bytes (bf16 weights) + 2 bytes (bf16 gradients) + 4 bytes (fp32 master) + 4 bytes (Adam states) = 12 bytes per parameter. What is the largest model trainable under this accounting, and how does the ceiling change with 8-bit Adam (6 bytes per parameter)?

**Answer:** Total memory: 8 x 80GB = 640GB. 640e9 / 12 bytes/param = 53.3e9, so the ceiling is about 53B parameters ("the 53B calculation"). With 8-bit Adam the per-parameter cost drops to 6 bytes, so the ceiling doubles to about 106B parameters.

**Source:** l02-resource-accounting — sections "the 53B calculation" (12 bytes/param, 53.3B ceiling on 8 H100s) and "optimizer zoo" (8-bit Adam halves the per-parameter bytes).

---

## Lesson 3 — Architecture (l03-architecture)

### L03-Q1

**Question:** The lesson quantifies MLA's KV-cache advantage. At 8-bit precision with 128 KV heads of dimension 128 (standard MHA), the cache costs about 64KB per token. Work MLA's cost with a 512-dim latent plus a 64-dim decoupled RoPE vector, state the compression factor, and explain why RoPE must be decoupled from the compressed latent.

**Answer:** MHA: 2 x 128 x 128 x 2 bytes = 64KB per token. MLA: 512-dim latent + 64-dim decoupled RoPE = 576 dims x 2 bytes = about 1.1KB per token — roughly 57x smaller. RoPE must be decoupled because rotation does not survive low-rank compression: compressing the rotated keys would destroy the relative-position signal, so the rotation is applied outside the compressed path.

**Source:** l03-architecture — section "MLA compresses by rank" (the 512+64-dim arithmetic, the ~57x factor, and why RoPE is decoupled).

### L03-Q2

**Question:** RoPE encodes position by rotating Q and K so the attention score depends only on relative angle. Why must the combination be a multiplication (inner product) rather than an addition, and what happens to the frequencies when Llama 3 raises the RoPE base from 10,000 to 500,000?

**Answer:** The inner product of rotated vectors is a function of the relative angle only; adding embeddings would introduce cross terms that leak absolute position. The frequencies are theta_i = base^(-2i/128), so raising the base to 500k lowers the slowest-rotating frequencies, stretching the position encoding over longer contexts — this is how Llama 3 extends context length.

**Source:** l03-architecture — sections "RoPE: position via rotation" (why multiply, not add; cross terms leak absolute position) and "the frequencies, worked" (theta_i = base^(-2i/128), base 10,000 -> 500,000 context stretch).

---

## Lesson 4 — Linear Attention + MoE (l04-linear-attention-moe)

### L04-Q1

**Question:** Mamba-2's recurrent state update is S_t = gamma(t) . S_{t-1} + k_t v_t^T. Why must gamma be input-dependent rather than state-dependent, and what breaks if you make the wrong choice?

**Answer:** Gamma is the forget gate: "Mamba-2: learn when to forget." It must be input-dependent (a function of the current token) so the parallel scan can compute it from the inputs alone. If gamma depended on the state, the recurrence would go sequential and kill parallel training — the whole point of the scan formulation.

**Source:** l04-linear-attention-moe — section "Mamba-2: learn when to forget" (the S_t update, input-dependent gating, and the parallel-scan constraint).

### L04-Q2

**Question:** The lesson works an F x P load-balancing loss on a toy: expert assignment fractions F = [0.7, 0.2, 0.08, 0.02] and routing probabilities P = [0.6, 0.25, 0.1, 0.05]. Compute the loss value L, and explain which direction the gradient pushes the router probabilities P and why that fixes the OlMoE-style collapse.

**Answer:** L = F . P = 0.7x0.6 + 0.2x0.25 + 0.08x0.1 + 0.02x0.05 = 0.479. The gradient with respect to P is F, so it pushes router probability mass off the most-used experts (the ones with high F) toward underused ones. In the OlMoE collapse, routing piles onto 2 of the experts; the aux loss penalizes that concentration and rebalances the load.

**Source:** l04-linear-attention-moe — section "the F x P loss, worked" (the toy arithmetic, L = 0.479, gradient w.r.t. P equals F, and the OlMoE 2-expert collapse).

---

## Lesson 5 — GPUs (l05-gpus)

### L05-Q1

**Question:** The lesson defines wave quantization: with 108 SMs and tiles of 256x128, a matrix needing 1,792 tiles runs in one wave but 1,793 tiles runs in two waves with 96 SMs idle. Derive the idle count for 1,793 tiles and give the general formulas for waves and idle SMs.

**Answer:** 1,792 tiles fill the 108 SMs' capacity in wave one (the toy uses 98 tiles per wave in its block geometry; 1,793 crosses the boundary). General rules: waves = ceil(T / S) where T is tiles and S is SMs; idle SMs in the final wave = S - (T mod S). The second wave for the extra tile leaves almost all SMs idle — hence "wave quantization, worked."

**Source:** l05-gpus — section "wave quantization, worked" (waves = ceil(T/S), idle = S - (T mod S), the 1792 vs 1793 toy).

### L05-Q2

**Question:** Work the online softmax from the lesson: tile 1 sees values [3, 1], tile 2 sees [5, 2]. Track the running max m and the denominator l through both tiles, and explain why the max must be tracked at all.

**Answer:** Tile 1: m = 3, l = 1 + e^(1-3) = 1.135. Tile 2: new max 5; rescale l = 1.135 x e^(3-5) = 0.153, then add the tile-2 terms 1 + e^(2-5) = 1.050, giving l = 1.203. The true denominator is 178.6; the rescaled value is 1.203 x e^5 = 178.5 (match). The max must be tracked because e^90 overflows fp32 — without per-tile max-tracking the naive single-pass softmax is numerically dead.

**Source:** l05-gpus — section "FlashAttention: the victory lap" (the [3,1] / [5,2] online-softmax toy, the e^90 overflow argument).

---

## Lesson 6 — Triton Kernels (l06-triton-kernels)

### L06-Q1

**Question:** The lesson compares a naive softmax (5 kernels) against a one-row-per-block Triton fused softmax for an MxN input. Count the naive version's reads and writes, state the fused version's traffic, and give the reduction factor.

**Answer:** Naive: 5 kernels, 5MN reads + 3MN writes = 8MN total memory traffic. Triton fused (one row per thread block, everything in SRAM): 2MN (one read, one write) in a single kernel. That is 4x fewer bytes moved.

**Source:** l06-triton-kernels — section "the savings, worked" (5MN reads + 3MN writes = 8MN vs 2MN, the 4x reduction).

### L06-Q2

**Question:** Shared memory has 32 banks of 4 bytes. If 32 threads each read one float from the same column of a 32x32 tile, how many times does the access serialize, and what is the lesson's swizzle trick to fix it?

**Answer:** All 32 threads hit the same bank (column stride maps every thread to one bank), so the access serializes 32x. The fix is swizzling: XOR the row index into the column index so consecutive rows land in different banks ("swizzling, the idea"), restoring full bandwidth.

**Source:** l06-triton-kernels — sections "Break 3" (32 banks, the same-column 32x serialization) and "swizzling, the idea" (XOR row into column).

---

## Lesson 7 — Parallelism (l07-parallelism)

### L07-Q1

**Question:** In Megatron-style tensor parallelism, a transformer block's two matmuls are split column-wise then row-wise, with a single all-reduce per pair. Why this wiring order and not the reverse, and what breaks in the reverse?

**Answer:** Column-split the first matmul (each GPU computes its shard of the hidden activations), apply the elementwise nonlinearity locally (no communication needed), then row-split the second matmul and all-reduce once. Reversed, the sync would have to happen before the nonlinearity, adding an extra all-reduce and forcing synchronization at the wrong point: "keeps the elementwise nonlinearity local."

**Source:** l07-parallelism — section "wiring a transformer block" (column-then-row split, one all-reduce, nonlinearity stays local).

### L07-Q2

**Question:** GPipe and 1F1B pipeline schedules differ in bubble cost and activation memory. Give both bubble fractions in terms of p pipeline stages and m micro-batches, state how many micro-batches each stage must hold in memory, and evaluate the bubble for p=4, m=8.

**Answer:** GPipe: bubble (p-1)/(m+p-1), paid twice (forward + backward flushes), and each stage holds m micro-batches in memory. 1F1B: bubble (p-1)/m, paid once, each stage holds p micro-batches. At p=4, m=8: GPipe bubble = 3/11 ~ 27%, paid twice (~37.5% of steady-state is the lesson's once-vs-twice comparison); 1F1B bubble = 3/8 = 37.5% paid once. The lesson's headline: 1F1B pays the bubble once instead of twice while holding fewer micro-batches per stage.

**Source:** l07-parallelism — section "GPipe vs 1F1B" (the (p-1)/(m+p-1) vs (p-1)/m bubbles, twice vs once, m vs p micro-batches in memory).

---

## Lesson 8 — 4D Parallelism (l08-4d-parallelism)

### L08-Q1

**Question:** The ZeRO ladder reduces memory but changes communication. State the communication cost multiplier for stages 1/2 vs stage 3 (in units of all-reduce equivalents P), explain why stage 3 costs 3P (what the two extra all-gathers and the reduce-scatter are for), and give the lesson's concrete payoff: a 50B model on A100s where DDP dies at 7B.

**Answer:** Stages 1 and 2 cost 2P (a reduce-scatter plus an all-gather = one all-reduce equivalent, so the overhead is near-free and can be overlapped). Stage 3 costs 3P: two all-gathers (parameters must be gathered before every forward and every backward pass) plus one reduce-scatter for the gradients. The payoff: under DDP, a 7B model dies on an 80GB A100 (112GB > 80GB), while ZeRO-3 with overlap plus sweep-and-free frees make a 50B model fit ("near-free" relative to the memory unlocked).

**Source:** l08-4d-parallelism — sections "The ZeRO ladder" (stages, the 50B vs 7B/112GB>80GB payoff) and "ZeRO stage costs" (2P vs 3P, the two all-gathers + one reduce-scatter, overlap + sweep-and-free).

### L08-Q2

**Question:** The lesson's exact activation-memory formula has a floor of 34sbh/t, where 34 = 24 + 10. Name the two components (what the 24 and the 10 count), explain which parallelism divides each, and state what activation checkpointing (recompute) deletes.

**Answer:** The 24 counts the MLP matmul terms; the 10 counts layernorm, dropout, and residual terms. Tensor parallelism divides the 24; sequence parallelism divides the 10. Recompute deletes the 5as^2/h quadratic term (the attention activation), which dominates at long sequence length. Worked: s=8192, b=8, h=8192, t=8 gives 4.6GB.

**Source:** l08-4d-parallelism — sections "where the 34 comes from" (24 = MLP matmuls, 10 = layernorm/dropout/residual; TP divides 24, SP divides 10) and "Activation memory, exactly" (recompute deletes the 5as^2/h quadratic; the 4.6GB worked example).

---

## Lesson 9 — Scaling Laws (l09-scaling-laws)

### L09-Q1

**Question:** The lesson tells "The Kaplan-Chinchilla saga": Kaplan's team concluded one scaling recipe and Chinchilla overturned it. Name the three methodological flaws in Kaplan's setup that the lesson identifies, and state why the IsoFLOP approach is the reliable default.

**Answer:** Kaplan (1) excluded the unembedding parameters from the parameter count, (2) used warmup that was too short for small models, and (3) used a fixed suboptimal batch size for small models. IsoFLOP is the reliable default because it holds compute fixed and varies model size and data directly, measuring the loss frontier instead of extrapolating fitted curves.

**Source:** l09-scaling-laws — section "The Kaplan-Chinchilla saga" (the three flaws: unembedding exclusion, short warmup, suboptimal batch) and the IsoFLOP recommendation.

### L09-Q2

**Question:** Training-optimal is about 20 tokens per parameter, but the lesson argues you should "overtrain for serving" because serving FLOPs dwarf training FLOPs. Give the three token-per-parameter ratios cited (Qwen3 235B, Llama 3 405B, DeepSeek-V3) and the one-line logic for why overtraining wins.

**Answer:** Qwen3 235B at ~153 tokens/param, Llama 3 405B at ~39, DeepSeek-V3 at ~22 — all past 20. The logic: training is a one-time cost, but a model served billions of times spends far more FLOPs in inference than training; a smaller, overtrained model pays less per query forever.

**Source:** l09-scaling-laws — section "Overtrain for serving" (the 20 tok/param optimum, the three cited ratios, serving FLOPs dwarf training FLOPs).

---

## Lesson 10 — Inference (l10-inference)

### L10-Q1

**Question:** The lesson works speculative-decoding arithmetic: K=4 draft tokens, draft cost c=1/20 of a big-model pass. Compute expected big-model passes when all 4 are accepted, when 2 are accepted, and when 0 are accepted. Which case is slower than baseline?

**Answer:** Expected passes = (K x r)/(1 + K x c) structure; the lesson's worked numbers: accept all 4 -> ~3.3x speedup (1.2 effective passes for 4 tokens); accept 2 -> ~1.7x; accept 0 -> 0.83x, i.e., SLOWER than baseline (the draft cost with zero accepted tokens is pure overhead).

**Source:** l10-inference — section "the speculative decoding arithmetic" (K=4, c=1/20, the 3.3x / 1.7x / 0.83x ladder).

### L10-Q2

**Question:** Work one decode step for a 70B model from the lesson: 140GB of weights read at 3.35TB/s gives what per-step latency and per-GPU tokens/sec? At what batch size does the KV cache overtake the weights (per-sequence cache 1.3GB), and why can batching never fix the attention component?

**Answer:** 140GB / 3.35TB/s = 42ms per step, about 24 tokens/sec per GPU. The cache overtakes the weights when B x 1.3GB > 140GB, i.e., B > 108. Batching cannot fix attention because each sequence has its own KV cache and the attention becomes B independent matvecs — batching amortizes the weight reads, not the per-sequence attention.

**Source:** l10-inference — sections "the decode step, byte by byte" (140GB, 42ms, 24 tok/s, the B>108 cache crossover) and "Intensity, precisely" (batching cannot fix attention: per-sequence caches, B independent matvecs).

---

## Lesson 11 — Scaling Advanced (l11-scaling-advanced)

### L11-Q1

**Question:** The lesson compares WSD against cosine schedules on "reuse arithmetic." With a budget of 5 training runs, count the total full-run cost of the cosine approach vs the WSD approach, and state the rewind rule (how far back and re-decay for what fraction).

**Answer:** Cosine: 5 full runs plus one restart from scratch = 6 full runs (each new horizon needs a retrain). WSD (warmup-stable-decay): about 5.1 full runs — the stable phase is horizon-independent, so you rewind to before the decay and re-decay at 10-20% of the budget for each new horizon. The saving: ~(0.9 full runs) per horizon change.

**Source:** l11-scaling-advanced — section "WSD vs cosine, the reuse arithmetic" (5 runs + restart = 6 vs ~5.1; rewind + re-decay at 10-20%).

### L11-Q2

**Question:** The lesson's muP rule for Adam: the per-layer learning rate is 1/fan-in. Compute it for a layer with fan-in 4096 and one with fan-in 512, state the ratio between them, and name the two muP invariants (A1, A2) that this rule protects.

**Answer:** 1/4096 vs 1/512 — the smaller layer gets an 8x larger learning rate. The rule protects the A1/A2 invariants (A1: activation scales stay order-1 across width; A2: update sizes stay order-1 across width), so that widening the model does not silently change the training dynamics.

**Source:** l11-scaling-advanced — section "the Adam per-layer rule, worked" (LR = 1/fan-in, the 4096 vs 512 toy, A1/A2 invariants).

---

## Lesson 12 — Evaluation (l12-evaluation)

### L12-Q1

**Question:** Work the lesson's ELO example: player X at 1200, player Y at 1000. What is Y's expected score, how many points does Y gain for an upset win vs an expected win, and what is the lesson's one-line moral about upsets?

**Answer:** Expected score for Y = 1/(1 + 10^(200/400)) = 0.24. Upset win: +24 points. Expected win: +8 points. Moral: upsets move ratings 3x more than expected wins — ELO learns most from the surprising result.

**Source:** l12-evaluation — section "ELO, worked" (X=1200 vs Y=1000, expected 0.24, +24 vs +8, upsets move 3x more).

### L12-Q2

**Question:** A model scores 30% pass@1 but 85% pass@100 on reasoning benchmarks. The lesson names this "the verifier gap." Explain in one paragraph what the gap means about the model's knowledge vs its selection ability, and why the lesson says "the business is in the gap."

**Answer:** The model knows the answer (it appears in 85% of 100 samples) but cannot reliably select it on the first try (30%). The knowledge is present; the selection is broken. The business is in the gap because whoever builds the verifier / reranker that closes pass@1 toward pass@100 captures most of the model's latent value without training a bigger model.

**Source:** l12-evaluation — section "reasoning evals: MATH, AIME, pass@k" (pass@1 30% vs pass@100 85%, the verifier gap, the business is in the gap).

---

## Lesson 13 — Training Data (l13-training-data)

### L13-Q1

**Question:** The lesson structures copyright risk into three layers — access, copying, training. For each layer, name the rule or number the lesson gives (robots.txt/ToS; the Anthropic $1.5B settlement arithmetic; the fair-use status of training itself).

**Answer:** Access: robots.txt and Terms of Service govern whether you may fetch the data. Copying: the Anthropic $1.5B settlement over ~500K books works out to ~$3,000 per book — "copying was the crime," and the lesson has you "work the Anthropic arithmetic." Training: the act of training on the data was ruled fair use (narrowly). The 2025 picture: access is governed by ToS, the copying is where the liability landed, training itself survived.

**Source:** l13-training-data — sections "The 2025 picture" and "work the Anthropic arithmetic" (three layers; $1.5B / ~500K books ~= $3,000/book; copying was the crime, training ruled fair use).

### L13-Q2

**Question:** Work the DCLM funnel from the lesson: 240T tokens in, keep 1.4% — how many tokens come out? What classifier setup does the lesson prescribe (fastText on what positives), and what was the surprise finding about which positives beat Wikipedia?

**Answer:** 240T x 0.014 = ~3.4T, reported as ~3T tokens out. The setup: fastText classifiers trained with instruction-shaped positives (OpenHermes + ELI5). The surprise: instruction-shaped data beat Wikipedia-style positives as the classifier's positive set. The funnel rule is "rules first, then classifiers."

**Source:** l13-training-data — section "work the DCLM funnel" (240T in, 1.4% keep, ~3T out; fastText on OpenHermes + ELI5; instruction-shaped positives beat Wikipedia; rules-first then classifiers).

---

## Lesson 14 — Data Pipeline (l14-data-pipeline)

### L14-Q1

**Question:** The lesson states MinHash LSH's core property (P(collision) = Jaccard) and works a toy: A={1,2,3,4}, B={3,4,5,6}. Compute the Jaccard similarity, state the S-curve threshold formula (1/b)^(1/r), and evaluate the lesson's tuning example: b=20 bands, r=450 rows gives what threshold and what does that mean?

**Answer:** Jaccard = |intersection|/|union| = 2/6 = 0.33. The S-curve threshold is (1/b)^(1/r); with b=20, r=450 the threshold is 0.993 — an extremely steep S-curve that only admits near-duplicate pairs, i.e., the transition point of "tune the S-curve" set for aggressive dedup.

**Source:** l14-data-pipeline — sections "MinHash LSH" (P(collision) = Jaccard; the {1,2,3,4} vs {3,4,5,6} toy, J = 0.33) and "tune the S-curve" (threshold (1/b)^(1/r); b=20, r=450 -> 0.993).

### L14-Q2

**Question:** Work the lesson's "50-epoch trap": 10T low-quality tokens + 10B high-quality tokens, 1T training budget, uniform sampling. How many epochs does the high-quality data see, and how does UniMax fix it (the cap and the reallocation)?

**Answer:** Uniform sampling splits the 1T budget proportionally: 500B low-quality + 500B high-quality. 500B / 10B = 50 epochs on the high-quality data — the trap. UniMax caps any source at ~4 epochs and reallocates the freed budget back to the other sources ("UniMax, mechanized").

**Source:** l14-data-pipeline — sections "work the 50-epoch trap cleanly" (10T + 10B, 1T budget, 50 epochs) and "UniMax, mechanized" (~4-epoch cap, reallocation).

---

## Lesson 15 — Post-Training (l15-post-training)

### L15-Q1

**Question:** The lesson gives "Schulman's calibration argument, in full": why must calibration be policy-dependent, and why does a human-written "I do not know" dataset fail to calibrate the model?

**Answer:** Calibration must be policy-dependent because the model's uncertainty is a property of its own rollouts, not of any human demonstrator. A human-written "I do not know" dataset teaches abstention on the demonstrator's uncertainty, not the model's — the policy never learns when IT is uncertain. RL (SFT teaches confident answers; RL trains on the policy's own rollouts) is what ties the "I don't know" to the policy's actual uncertainty.

**Source:** l15-post-training — section "Schulman's calibration argument, in full" (calibration must be policy-dependent; human-written abstention data fails; RL trains on the policy's own rollouts).

### L15-Q2

**Question:** Reconstruct the DPO derivation chain from the lesson: start from the nonparametric assumption, go through the KL-constrained optimum's closed form, invert it to the implied reward, and land on the DPO loss. Then state the gradient rule from the lesson's toy ("work the DPO gradient on a toy").

**Answer:** Chain: the nonparametric assumption gives the KL-constrained reward objective a closed-form optimum — the policy proportional to the reference tilted by exp(reward/beta). Invert it: the implied reward equals beta times the log-ratio of policy over reference. Plug that implied reward into the Bradley-Terry preference model and you get the DPO loss ("unpack the nonparametric assumption"). Gradient rule: push the winner's log-probability up, the loser's down, with the update scaled by the surprise (how wrong the current preference ranking is).

**Source:** l15-post-training — sections "the DPO derivation" and "unpack the nonparametric assumption" (the full chain) plus "work the DPO gradient on a toy" (up on winner, down on loser, scaled by surprise).

---

## Lesson 16 — RLVR (l16-rlvr)

### L16-Q1

**Question:** The Dr. GRPO critique (Liu et al.) names two normalization failures in GRPO. Work both: (a) the std-0 trap — what happens when all 8 rollouts are correct (or all wrong), and what is the fix? (b) the length-normalization hack — how does dividing the loss by response length reward "wrong-but-long"?

**Answer:** (a) All 8 correct: rewards all 1, mean 1, std 0 — GRPO divides by the std, so the update explodes (division by zero/epsilon) on a problem the model already mastered. All 8 wrong: same explosion on a problem with zero signal. Fix: drop the std normalization; advantage = reward minus group mean, no division — then both degenerate groups give advantages of 0 and no update ("the std-0 trap"). (b) Dividing the loss by response length means a wrong answer spread over 1000 tokens gets a smaller per-token penalty than the same wrong answer in 100 tokens — once the model knows it will fail, blabbing dilutes the penalty. Fix: stop dividing by length, and the ever-growing chain of thought caps off.

**Source:** l16-rlvr — sections "Where GRPO breaks: interrogate the normalizations" and "the std-0 trap" (std-0 explosion, the fix) and "the aha moment was already there" (length hack, wrong-but-long).

### L16-Q2

**Question:** Work the lesson's RLOO toy: four rollouts with rewards [1, 0, 0, 1]. Compute each rollout's leave-one-out baseline and its advantage, verify the advantages sum to 0, and state the lesson's one-line relationship between RLOO and GRPO.

**Answer:** Rollout 0's baseline = mean of the other three = (0+0+1)/3 = 0.33, advantage 0.67. Rollout 1: baseline (1+0+1)/3 = 0.67, advantage -0.67. Rollout 2: same as 1, -0.67. Rollout 3: same as 0, +0.67. Advantages [0.67, -0.67, -0.67, 0.67] sum to exactly 0. The lesson's line: "RLOO is GRPO with the std division deleted" (GRPO on the same toy gives mean 0.5, std 0.5, advantages [1, -1, -1, 1]).

**Source:** l16-rlvr — section "RLOO, worked" (the four-rollout toy, baselines, advantages summing to 0, the GRPO comparison).

---

## Lesson 17 — Multimodality (l17-multimodality)

### L17-Q1

**Question:** The lesson frames SigLIP as a one-line change from CLIP's 2N-way softmax to a binary sigmoid loss. Explain the statistical consequence (why CLIP's expected loss moves with batch size but SigLIP's does not), the batch-size ceiling the lesson gives for SigLIP (32K, and why), and the systems win (what devices do, and the TPU-day comparison).

**Answer:** CLIP's softmax couples every pair in the batch: making a negative worse helps the positive win even if the positive is not close, so the objective itself shifts with batch size. SigLIP judges each of the N^2 pairs independently (binary decision), so the expected loss is the same at any batch size. The 32K ceiling: beyond 32K negatives the extra pairs add no signal — the model already separates easy negatives (gradients ~0) and the remaining signal is the limited supply of hard negatives. Systems win: devices compute local losses and rotate text embeddings (ring exchange) to cover off-diagonal blocks — no global softmax — giving 5 days on 32 TPUv4 versus CLIP's 10 days on 256 TPUv3.

**Source:** l17-multimodality — sections "softmax vs sigmoid (the one-line difference)" (statistical consequence, 32K ceiling) and "the systems win" (local losses, rotate embeddings, 5 days/32 TPUv4 vs 10 days/256 TPUv3).

### L17-Q2

**Question:** A 1344x1344 document under AnyRes costs how many image tokens (show the arithmetic), why is the 336x336 alternative illegible (~10 pixels per character), and how does Qwen3-VL's 2x2 merger change the token bill? What is the lesson's binding constraint on VLM resolution?

**Answer:** 16 crops of 336x336 plus 1 overview = 17 encodings x 576 tokens = 9,792 image tokens before the question is even asked. At 336x336 a page of text is ~10 pixels per character — below the 14x14 patch size — so the encoder's features contain no character information ("the 10-pixels-per-character failure"). Qwen3-VL's 2x2 merger compresses each crop's 576 tokens to 144, so 17 x 144 = 2,448 tokens (a quarter of the AnyRes bill). The binding constraint on VLM resolution is never the encoder: it is the LLM's context window.

**Source:** l17-multimodality — sections "the token budget of resolution" (9,792 tokens) and "the 10-pixels-per-character failure," plus the QA "A 1344x1344 document costs 9,792 image tokens. Walk me through the alternatives." (merger arithmetic, context window as binding constraint).

---

## Lesson 18 — Serving Inference (l18-inference)

### L18-Q1

**Question:** Parcae loops transformer blocks to get more flops without more parameters. Work the stability constraint the lesson gives: what goes wrong with an unconstrained loop (use the A = 1.2, 4-loop numbers), what is the state-space-theory fix (spectral radius, the negative diagonal, the -0.9 numbers), and what is the scaling-law finding about today's models?

**Answer:** Unconstrained loop: each pass multiplies activations by the block's effective matrix A; with A = 1.2 on the diagonal, 4 loops grow activations by 1.2^4 = 2.07x per pass, compounding across layers to NaN. Fix: constrain A to a negative diagonal matrix with spectral radius under 1 (entries like -0.9) — magnitudes shrink 10% per pass (bounded forever) while the alternating signs keep the network computing. Scaling finding: as data grows, optimal recurrence grows too; today's models have zero recurrence with tons of data — the far left of the curve, possibly under-looping. Inference bonus: fewer parameters means a smaller decode tax, more KV cache, and less cross-GPU communication.

**Source:** l18-inference — sections "Parcae: flops without parameters," "the spectral radius, defined" and "work the constraint" (1.2^4 = 2.07x vs -0.9 bounded), "the under-looping claim," and "the inference bonus."

### L18-Q2

**Question:** The lesson claims "two lines of routing code" buy 40% faster serving via cache-aware disaggregation. State the routing rule (what fraction is estimated, what goes where), work the win (the book-paste vs 90 chat requests), and name the cost that the 40% figure is net of.

**Answer:** Routing rule: estimate each request's cache-hit fraction; route low-hit-rate (fresh, ~10% of traffic, e.g., a book-paste at ~0%) to the cold prefill pool and warm requests (mid-conversation, ~90%+) to the warm pool. Win, worked: in one mixed pool the 5-second book-paste prefill blocks 90 chat requests (P99 of 5+ seconds); separated, the warm pool serves the 90 chats with P99 under a second — 40% faster overall. The 40% is net of the KV cache transfer cost (shipping gigabytes per request from prefill fleet to decode fleet).

**Source:** l18-inference — sections "Disaggregation: split the fleets" and "cache-aware routing (the 40% win)" (the two lines, the book-paste/90-chat worked win) and "the KV cache transfer problem" (40% net, not gross).

---

## Verdict

**PASS — 36/36.** All 36 questions are answered from the lesson text, each with its quoted section. No FAIL items: no question required material absent from the text.

**Coverage notes (no FAILs):** L18 is titled "Serving Inference" (guest lecture by Dan Fu), distinct from L10's inference lesson — both were examined separately with non-overlapping question sets. One honest caveat from the text itself: l18's lecturer warned his slides were AI-generated and fine details may be wrong; this exam follows his spoken claims as the lesson does.
