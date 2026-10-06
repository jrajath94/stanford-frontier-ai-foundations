# Chapter-plate build audit: cs229 l13–l17
Built 2026-10-06. Builder role: figure stage (chapter plates).
21 plates total: 4 + 4 + 5 + 4 + 4. All Python-rendered SVG,
warm paper #F7F4EE, sans stack on root, dense four-region layout
matching the mse435 l01 bar exactly (left = cost without the rule,
center = the stored object, right = cost with the rule, bottom =
tradeoff in one line, footer = the one connection in one sentence).
Every number was recomputed in the generator scripts with asserts;
three lesson-text discrepancies were caught by the code and are
flagged below.

Generators: `build/chap-plates/gen_l13.py`, `gen_l14.py`,
`gen_l15.py`, `gen_l16.py`, `gen_l17.py` (asserts live in the
scripts). Inserter: `build/chap-plates/insert_plates.py`
(anchor-uniqueness asserted per plate). build/build.py was NOT run.
l01–l12 untouched.

## l13 — contrastive learning + RAG (4 plates)

1. `content/v2/cs229/assets/plate-l13-chap-contrastive.svg`
   Concept: InfoNCE contrastive learning + family (end of "The
   family" section, before "Where it breaks").
   Left: classifiers per concept, labels are the bottleneck.
   Center: the InfoNCE loss toy (pos 0.9, 100 negs at 0.1, tau
   0.1: fraction 0.9676, loss 0.033; hard neg at 0.85 drops it
   to 0.61).
   Right: SimCLR batch 4,096 (8,190 negatives/anchor, 33.5M
   sims/step, 8.3-nat ceiling), MoCo 65k queue, CLIP 400M pairs.
   Bottom: the negative set buys the information ceiling at
   33.5M similarity computations per step.

2. `content/v2/cs229/assets/plate-l13-chap-hardneg.svg`
   Concept: hard negatives + mining (end of "Hard negatives",
   before "Embedding geometry").
   Left: easy negatives (anchor cat-soccer, truck at 0.05,
   loss ~0, nothing learned).
   Center: the mined hard negative (FIFA text vs cat photo;
   gradient focuses: weight 0.3 vs 0.001 = 300x harder push).
   Right: three mining generations (in-batch, MoCo queue,
   global), 2% false-negative poison, dedup gate at 0.95 cosine.
   Bottom: harder negatives teach finer distinctions; dedup
   above 0.95 cosine before mining.

3. `content/v2/cs229/assets/plate-l13-chap-ann.svg`
   Concept: ANN search at 10M vectors (end of "Search variants",
   before "RAG").
   Left: exact search (10M x 768 = 7.7B ops/query, recall 1.0,
   cost fatal).
   Center: the index (IVF 10k clusters, HNSW layered graph,
   PQ 96 groups of 8, sharding past 100M).
   Right: IVF nprobe 10 (7.7M ops, recall 0.95), HNSW
   (~50k distances, recall 0.98), PQ (30 GB to 1 GB, recall
   ~0.90).
   Bottom: every recall point past 0.90 is bought with latency.

4. `content/v2/cs229/assets/plate-l13-chap-rag.svg`
   Concept: RAG pipeline (end of "RAG", before "How embeddings
   are evaluated").
   Left: fine-tuning the docs (training run, hosted weights,
   opaque, unleaky).
   Center: the pipeline (embed question, ANN retrieves 100,
   cross-encoder reranks to 5, generate grounded, cite).
   Right: refund toy priced (3 x 500 = 1,500 tokens/query,
   128k fits ~85 retrievals, hybrid +3-8 recall, rerank
   0.94 vs 0.12, middle-of-context -10-20 pts).
   Bottom: retrieval failures become answer failures; modular,
   governable, deletable.

## l14 — transformers (4 plates)

1. `content/v2/cs229/assets/plate-l14-chap-attention.svg`
   Concept: attention by hand (end of "Attention, by hand",
   before "Multi-head attention").
   Left: fixed windows (5 neighbors, fixed weights, cannot
   reach 20 back).
   Center: the toy audited (scores 1,1,2; weights
   0.212/0.212/0.576, softmax sums to 1; new frame
   [0.788, 0.788]; query [1,0] gives 0.422/0.155/0.422).
   Right: quadratic (N x N scores, 16.7M at N=4096;
   1/sqrt(d): d=64 softmax ratio 8e13 -> 54.6).
   Bottom: every token sees every token; that is the power
   and the quadratic bill.

2. `content/v2/cs229/assets/plate-l14-chap-positions.svg`
   Concept: position encodings (end of "Position" section,
   before "Attention variants").
   Left: no order ("dog bites man" = "man bites dog",
   permutation-invariant).
   Center: the decision table (sinusoidal, learned, RoPE,
   ALiBi, NoPE).
   Right: 2026 default RoPE (theta 500,000, dot keeps
   cos((m-n)theta)), NoPE 2.1 vs 250 perplexity, ALiBi 2.3
   stable to ~1.8k tokens ([uncertain: secondary analysis],
   carried from the lesson).
   Bottom: RoPE is the default; ALiBi extrapolates; NoPE is
   free only with a causal mask.

3. `content/v2/cs229/assets/plate-l14-chap-block.svg`
   Concept: the transformer block counted (end of "The block",
   before "Decoding").
   Left: post-norm wiring (stream rescales every block, depth
   capped).
   Center: one block (attention 4d^2 = 67.1M, MLP 8d^2 =
   134.2M, total 201.3M; pre-norm + residuals).
   Right: 32 blocks (6.44B + 204.8M embeddings = ~6.6B, a 7B
   model; the 12d^2 sizing rule).
   Bottom: depth buys composition; pre-norm buys depth.

4. `content/v2/cs229/assets/plate-l14-chap-decoding.svg`
   Concept: decoding (end of "Decoding", before "Training").
   Left: greedy (argmax every step, repetition loops, no
   backtrack).
   Center: the distribution (logits [3,2,1]; T=1:
   0.665/0.245/0.090; T=0.5: 0.867/0.117/0.016; T=2:
   0.506/0.307/0.186).
   Right: top-p 0.9 (keep 0.526/0.316/0.158, drop the 0.05
   token; code low T, prose high T).
   Bottom: decoding tunes the draw; it cannot fix the
   distribution (exposure bias: 1%/token -> 63% of 100-token
   answers err).

## l15 — efficiency, ICL, SFT (5 plates; the lesson has five
named concept pillars, so it gets five)

1. `content/v2/cs229/assets/plate-l15-chap-kvcache.svg`
   Concept: the KV cache (end of "The KV cache", before
   "Attention variants: shrink the cache").
   Left: recompute per token (token t costs t^2, sum to 4096
   = 2.3e10 scores).
   Center: the cache (512 KB/token, 2 GB per 4k sequence,
   batch of 10 = 20 GB + 14 GB weights = 34 GB of 80 GB).
   Right: O(t) per token (sum 8.4M, 2,730x cheaper, cubic to
   quadratic).
   Bottom: bytes, not big-O, set the batch size.

2. `content/v2/cs229/assets/plate-l15-chap-diets.svg`
   Concept: cache diets + serving stack (end of "Attention
   variants: shrink the cache", before "PagedAttention").
   Left: full MHA (512 KB/token, batch of 10, memory-bound
   decode).
   Center: the diets (GQA-8 128 KB = 4x, MQA 16 KB = 32x,
   MLA latent ~16x, CLA-2 halves, fp8 halves, eviction).
   Right: the production stack (GQA-8 + fp8 + evict, paged
   12% to 96% util, speculative 2-3x, INT8 2x batch,
   $0.28/M tokens).
   Bottom: shrink the bytes, then stop wasting them.

3. `content/v2/cs229/assets/plate-l15-chap-moe.svg`
   Concept: mixture of experts (end of "Mixture of experts",
   before "Quantization for serving").
   Left: dense (every parameter every token).
   Center: the router (logits [2.0,1.0,0.5,-1.0], softmax
   0.609/0.224/0.136/0.030, top-2 renormalized 0.731/0.269).
   Right: DeepSeek-V3 (256 experts top-8 + 1 shared, 671B
   params, 37B active = 5.5%; Mixtral 2 of 8).
   Bottom: flat per-token compute for a systems project.

4. `content/v2/cs229/assets/plate-l15-chap-icl.svg`
   Concept: in-context learning (end of "The shock: ICL",
   before "SFT").
   Left: the old way (train a model per task, months per
   task).
   Center: frozen weights (sea/sky/cheese -> fromage, no
   gradient step, attention continues the pattern).
   Right: the price (emerges past ~10B, reliable at 175B;
   order/wording brittleness; 20 x 100 = 2,000 tokens
   overhead per query forever).
   Bottom: prototype with ICL, ship with SFT.

5. `content/v2/cs229/assets/plate-l15-chap-sft.svg`
   Concept: SFT (end of "SFT: teach the format", before
   "The honest price").
   Left: the base model (completes text, not an assistant).
   Center: the pair (x = instruction, y = answer; loss on y
   ONLY; labels [-100,-100,-100,mer]; mask the instruction).
   Right: the price (1,000 curated beats 50,000; LoRA 131k
   vs 16.8M = 128x fewer; forgetting 1-3 pts; LR 1e-5,
   1-3 epochs).
   Bottom: spend the budget on quality, not count.

## l16 — reinforcement learning (4 plates)

1. `content/v2/cs229/assets/plate-l16-chap-bandits.svg`
   Concept: exploration vs exploitation (end of the bandit
   section, before "The MDP").
   Left: pure exploitation (greedy on A forever, 0.3/pull,
   misses C's 0.7).
   Center: the dial (epsilon 0.1, decay 1.0 -> 0.01, UCB
   value + sqrt(2 ln t / n) bonus, Thompson sampling).
   Right: UCB worked (bonuses 0.74/1.16/2.32, scores
   1.04/1.66/2.32, pull C by optimism, no random pulls).
   Bottom: exploration costs reward now for knowledge later.

2. `content/v2/cs229/assets/plate-l16-chap-values.svg`
   Concept: value functions / Bellman (end of "Value
   functions", before "Dynamic programming").
   Left: greedy on immediate reward (wanders, discounted
   total -10).
   Center: the Bellman chain (V(9)=8.0 ... V(3)=-0.43;
   value now = reward + gamma x value later; contraction
   shrinks error 0.9x per sweep).
   Right: act on V* (Q(3,right) = -0.43 beats Q(3,left) =
   -2.25; pi* = argmax Q*).
   Bottom: greedy on V* is optimal; greedy on R is ruin.

3. `content/v2/cs229/assets/plate-l16-chap-experience.svg`
   Concept: learning from experience (end of "Learning from
   experience", before "The key question").
   Left: DP (model given, policy iteration in 2 rounds,
   exact and unusable).
   Center: the estimators (MC unbiased/slow; TD biased/fast/
   online, V(9) 0 -> 0.9 in one step; n-step dial).
   Right: Q-learning (0 -> 4.5 -> 6.75 -> 9, max is
   off-policy; SARSA honest about exploration; DQN adds
   replay + target net).
   Bottom: tables converge; networks negotiate.

4. `content/v2/cs229/assets/plate-l16-chap-reinforce.svg`
   Concept: REINFORCE + variance (end of "Where it breaks:
   variance", before "Reward shaping").
   Left: no gradient (values cannot improve pi_theta,
   reward not differentiable).
   Center: the trick (grad E[R] = E[R grad log pi], p
   cancels; total 3 pushes right up 0.12; stochastic
   policies only).
   Right: calm it (baseline -4.5 gives 7.5/-7.5 updates;
   reward-to-go; actor-critic: TD error 1.2, theta
   0.847 -> 0.883, V(8) 5.0 -> 5.12).
   Bottom: subtract the predictable; learn from the surprise.

## l17 — RL for LLMs (4 plates)

1. `content/v2/cs229/assets/plate-l17-chap-advantages.svg`
   Concept: advantages (end of "Advantages", before "PPO").
   Left: raw 0/1 reward (500 thinking tokens, one bit, 490
   good steps punished).
   Center: A = return - baseline (V=0.5; correct +0.5,
   wrong -0.5; expectation unchanged).
   Right: GAE (deltas +0.5/-0.2/+0.3, lambda 0.95,
   A_0 = 0.563; group mean free; critic doubles forward
   cost).
   Bottom: the expectation is unchanged; the variance drops.

2. `content/v2/cs229/assets/plate-l17-chap-ppo.svg`
   Concept: PPO clipping (end of "PPO", before "RLHF").
   Left: on-policy (one step per fresh batch, stale data,
   r=3 counts one trajectory triple).
   Center: the clipped ratio [0.8, 1.2] (A>0 r=1.5 clipped
   flat; A>0 r=0.5 full push; A<0 r=1.5 full correction
   unclipped; A<0 r=0.5 clipped flat).
   Right: the plumbing (eps 0.2, 4 epochs, LR 1e-5 to 3e-6,
   GAE 0.95, KL leash to SFT ref; 10% objective, 90%
   plumbing).
   Bottom: watch the ratio histogram, not the loss.

3. `content/v2/cs229/assets/plate-l17-chap-rlhf.svg`
   Concept: RLHF (end of "RLHF", before "GRPO").
   Left: no verifier (open-ended writing, taste not
   checkable).
   Center: Bradley-Terry (A 2.0 vs B 0.5, P = 0.818, loss
   0.20; 4-9 responses/prompt; ~70% annotator agreement).
   Right: InstructGPT recipe (SFT -> reward model -> PPO,
   KL leash; Goodhart past the peak; DPO loss 0.644, no
   reward model, offline, never explores).
   Bottom: the approximation is the whole game.

4. `content/v2/cs229/assets/plate-l17-chap-grpo.svg`
   Concept: GRPO + RLVR (end of "Verifiable rewards",
   before "The honest price").
   Left: the critic (second forward pass, doubles RL step
   cost, bias infects advantages).
   Center: the group (G=4, rewards [1,1,0,0], mean 0.5
   std 0.5, advantages +1/+1/-1/-1; no critic, no GAE).
   Right: DeepSeek-R1 (R1-Zero no SFT, AIME 15.6% -> 71.0%;
   R1 79.8% matches o1; aha moment; rule rewards, no
   neural RM).
   Bottom: scale G with difficulty; never trust a rising
   reward curve alone (5% blind spot).

## l17-verify.svg font fix

`content/v2/cs229/assets/svg/l17-verify.svg`: root
font-family changed from `Georgia, serif` to the spec sans
stack (Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex
Sans', sans-serif). The only formula line, `reward = 1 :
0`, now carries `font-family="Source Serif 4, Newsreader,
serif"` per the spec (serif only for pure formula lines).
Verified: still parses as XML. Referenced only by
`content/v2/cs229/index.md` (line 152), as stated.

## Number discrepancies caught by the code (flagged for the
figure auditor / content auditor)

1. l14 decoding subchapter: the lesson's T=2 row prints
   [0.468, 0.284, 0.248] for logits [3,2,1]. The code
   computes softmax([1.5,1,0.5]) = [0.506, 0.307, 0.186].
   The lesson row does not recompute. The plate uses the
   code-computed values.
2. l15 MoE router subchapter: the lesson prints the
   renormalized top-2 as [0.735, 0.265], obtained by
   renormalizing its rounded softmax [0.61, 0.22]. Exact
   from the logits: softmax [0.609, 0.224, 0.136, 0.030],
   renormalized [0.731, 0.269]. The plate uses the exact
   code-computed values.
3. l16 terminal-reward convention: the Bellman-chain
   subchapter uses V(10)=10 (V(9) = -1 + 0.9*10 = 8.0),
   while the Q-learning/TD subchapter uses R=9 net with
   V(10)=0 (Q(9,right) converges to 9). Same quantity,
   two values (8.0 vs 9.0) in different conventions. Each
   plate uses its own section's numbers (both internally
   consistent and audited); the cross-section inconsistency
   is a content issue for the auditor.
4. Minor: l15's "batch of 10: 20 GB" is GiB (20 GiB =
   21.47 decimal GB); the lesson's paging-utilization math
   (12% vs 96%) is consistent in decimal GB.

## Spec compliance notes

- Reject list: no robot/brain/glowing network/stock/clip
  art; no gradient, glow, shadow, watermark, or logo.
  Flat fills only: #F7F4EE ground, #FFFDF8/#E7F1F8/#E7F4EF
  panels, ink #1B2838, muted #5C6B7A, teal #1F7A72.
- 8px grid: all coordinates multiples of 8; stroke 1.5 on
  panels, 2 on the right (with-the-rule) panel, matching
  the bar.
- Captions match the bar format verbatim:
  `Chapter plate L<nn>-C<n>. Left: ... Center: ... Right:
  ... Bottom: ... Dense chapter plate. Source: original
  synthesis of the session. Project: Stanford Frontier AI.`
- Every plate placed at the end of its concept's section,
  verified to sit immediately before the next `##` header.
- Labeled `[uncertain]` carry-over: l14 positions plate
  keeps the lesson's own `[uncertain: secondary analysis]`
  note on the NoPE perplexity numbers.
- Deliberately folded (no plate): l13 embedding geometry
  (cosine/dot, modality gap) and evaluation sections; l16
  reward shaping; l17 process supervision and the
  reward-hacking zoo. The figure auditor may request plates
  for these.
