# cs336 crash course , all 18 units

Condensed review of the full course. Each unit: the one-line idea,
the key numbers, and the most interview-tested concept. Numbers
marked toy come from the course's executed numpy scripts, numbers
marked HYP are hypothetical. U01-U09 by the first builder.
U10-U18 in this build.

## U01 , tokenization

Text becomes integers. BPE merges the most frequent pair, the
pretokenization regex decides what can merge. Compression is
bytes per token: ~4.2 for English web text, ~2.5 for code (toy).
Mismatch between training and inference tokenizers silently
changes every count.

## U02 , tensor programming

Shapes are contracts. einsum states them, strides explain views.
Matmul costs 2*m*n*k FLOPs, backward ~2x forward. Arithmetic
intensity decides whether a kernel is memory-bound. Optimizer
state (Adam: 2 moments) doubles the parameter memory.

## U03 , transformer block

Embed, project Q/K/V, scaled softmax attention, causal mask,
multi-head reshape, residual, pre-norm, SwiGLU FFN, RoPE, output
head. Pre-norm stabilizes deep stacks. RMSNorm drops the
recenter. RoPE rotates pairs by position, theta sets the ruler.

## U04 , attention and MoE

GQA trades KV heads for memory: fewer heads, same quality, much
smaller cache. MLA compresses the cache further. MoE routes each
token to top-k experts, the router needs a load-balance loss or
experts collapse. Active compute is per-token, not per-parameter.

## U05 , optimization

AdamW decouples weight decay. Warmup then cosine (or WSD).
Gradient clipping bounds the step, not the loss. Batch size and
LR scale together up to the critical batch size. Checkpoint
weights, optimizer moments, RNG state, and data position, or the
resume is not a resume.

## U06 , GPU/TPU

The roofline: min(peak FLOPs, bandwidth x intensity). Decode is
bandwidth-bound (1 FLOP/byte), prefill is compute-bound at long
T. MFU measures achieved vs peak. Measure before you optimize.
vendor claims are branches to verify.

## U07 , kernels

Tile for the memory hierarchy, keep hot data in SRAM. Online
softmax makes FlashAttention possible: one pass, no materialized
NxN. Triton is the portable kernel language. Correctness first
against a reference, benchmark claims are branches.

## U08 , data sharding

Data-parallel averages gradients with all-reduce. ZeRO shards
optimizer state, then gradients, then parameters. Overlap
communication with backward compute via bucketing. Mixed
precision needs loss scaling. Fault tolerance is a first-class
design item, not an afterthought.

## U09 , model parallel

Tensor-parallel splits layers (needs fast interconnect).
pipeline-parallel splits depth (pays the bubble). 1F1B schedules
shrink the bubble. Hybrid DP/TP/PP is the standard large-run
recipe. Activation checkpointing trades recompute for memory.

## U10 , scaling laws and extrapolation

Loss follows L = E + A*(C/C0)^{-a} (toy: a=0.34, recovered
0.340, RMS 0.0031). 6ND estimates compute. The isoFLOP optimum
for the toy law: N=7.0e10, D=1.4e12, D/N=20. Preregister every
extrapolation: prediction, band, rule, date. Report bootstrap
bands, not point estimates. Extrapolation fails on form change,
noise regime, and range.

## U11 , inference and serving

Decode is bandwidth-bound: each token re-reads the weights.
KV cache: 2*L*h_kv*d_h*bytes/token (toy: 128 KB/token, batch 8 x
8192 = 8.59 GB). Continuous batching beats static batching.
Speculative decoding: speedup = E/(k*c+1) (toy: 1.95x). Serve
with SLOs: TTFT, TPOT, p99. The cost/latency/quality triangle
rules every decision.

## U12 , evaluation and validity

Perplexity = e^{loss} (toy: e^2.3 = 9.97). Always report CIs:
SE = sqrt(p(1-p)/n) (toy: 0.72 +- 0.039 at n=500). Dedup train
against eval: n-gram overlap + canaries (toy: 100% overlap
found, recall 1.00). Internal evals steer, external evals judge.
never tune on the external. Judge agreement needs kappa, not raw
percent. Metrics must carry cost.

## U13 , data sourcing

WARC stores the crawl. CDX indexes it. Extraction yield = kept /
raw bytes (toy: 1.9/6.6 GB = 28.6%). Hazards: PDFs, code files,
book chapters. Provenance: URL + crawl date + license. Language
ID: the non-English-called-English rate is the metric (toy:
2.8%). PII: detect, redact, never train on it. Version with
content hashes (toy id 613575a7e17d78eb), sort, seed, pin,
hash to prove determinism.

## U14 , filtering, dedup, mixing

Dedup first, then filter: scores computed once per unique doc.
Bloom: (1-e^{-kn/m})^k, no false negatives (toy: k=6, FPR
0.0216 at 8 bits/item). Near-dup: shingles -> MinHash ->
LSH bands. P(match) = 1-(1-s^r)^b (toy: Jaccard 0.113 vs 0.125).
Train-test dedup at n-gram level (toy: 13.3% overlap). Mixing:
reweight by target/source ratio, interaction terms move the
optimum (toy argmin shifted to the boundary, then interior).
Repetition trades recall for memorization (toy logistic).

## U15 , midtraining and SFT

Stages: pretrain (LM), midtrain (continued LM, new mix),
post-train (behavior). RoPE theta 50x stretches the ruler but
needs long documents. Train/serve templates must match
byte-exactly. SFT loss on assistant tokens only. Packing cuts
padding waste 47.6% -> 22.0% (toy, 344 bins) with boundary
masks. LoRA: 2*d*r per pair (toy: 8.39M at r=16). Measure
forgetting: mean +0.15 nats over 6 checkpoints (toy). Full
checkpoints carry weights, optimizer, RNG, data position.

## U16 , alignment and RL

BT: P(pref) = sigmoid(r_w - r_l). KL keeps the policy near the
reference. PPO: min(r*A, clip(r)*A) (toy terms [0.7, 0.9, -1.0,
1.2, 1.2]). GRPO: group mean/std as baseline (toy: [1,-1,1,-1]
vs [0,0,0,0]). DPO: r = beta*log(pi/pi_ref) inside the BT loss
(toy: margin 0.139, loss 0.626). Verifiable rewards beat learned
rewards where checkable. Reward hacking: proxy up, truth down.
the held-out truth metric reveals it. Collapse = all-equal
advantages = zero signal.

## U17 , multimodality and systems

Images become tokens: 16x16 patches -> 256 tokens each. Budget
tokens against the context (toy: 8 images + 512 text = 2560, 80%
image). Interleaved streams mix modalities with boundary tokens.
Interfaces: tok/s/GPU, GB/s, bytes/step. Separate rollout and
training fleets, async with bounded queues (toy: 83.3% util,
1200/hr backlog at 120 vs 100/min). Lineage: id_n =
sha256(id_{n-1} + stage_n). Amdahl: new total = total - share +
share/speedup (toy: 143 ms, backward 38.5%, 2x -> 19.2%).

## U18 , guests and defense

Guest content is unknown: no artifacts inspected, no inference
from speaker names. The honesty protocol: name the source, state
the inspected extent, state the unknown. Replication passes when
an independent rerun lands inside all seed bands. Ablations:
full 2^n, fractional 2^{n-k}, OAT 1+n (toy: 16/8/5 for 4
factors). Keep a failure diary: date, symptom, hypothesis, fix,
lesson. Report cost honestly (toy: $1,200 for 480 GPU-hr. 25%
failure share). The oral defense ladder: define, toy, derive,
implement, compare, debug, critique, design.

## Capstone A , research replication (honest failure)

Replicate the U10 power-law fit on a new synthetic law, then test a
preregistered extension at 4x noise. Replication: true a=0.41,
fitted 0.412, PASS. Extension: observed 0.405 against the
preregistered band [0.30, 0.38], FAILED. The failure is
experimenter error, not noise: the band was anchored to the old
law (a=0.34), not the new one (a=0.41). The post-hoc diagnosis is
labeled post-hoc. The preregistered test stands as failed. Lesson:
preregister the band against the law under test, and report the
miss exactly as registered.

## Capstone B , applied serving plan (all inputs HYPOTHETICAL)

Fictional SaaS support chat on a 7B bf16 model. SLOs (HYP): p99
TTFT < 2000 ms, p99 TPOT < 50 ms at 40 peak RPS (HYP). Sizing
(computed from HYP inputs): 1.7 rps/GPU, 32 GPUs at 30% headroom,
14.9 GB of 80 GB per GPU, $57,600/month (HYP). Rollout: 1% canary
24h, then 10%, 50%, 100% on green gates. Rollback on +0.5pt error rate
or +20% p99 latency. Chosen for v1: full-precision 7B at 32 GPUs,
quantize only after U12 evals. Nothing here is measured.

## The arc in one paragraph

Tokens become tensors (U01-U02), tensors become a transformer
(U03-U04), trained by Adam on sharded hardware (U05-U09),
following scaling laws (U10), served with caches and batching
(U11), measured with valid evals (U12), fed by sourced,
deduped, mixed data (U13-U14), adapted by midtraining and SFT
(U15), aligned by RL (U16), extended to images and run as a
system (U17), and defended as honest research (U18).
