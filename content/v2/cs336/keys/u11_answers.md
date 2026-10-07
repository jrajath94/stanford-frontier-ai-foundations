# U11 answer key

All numeric claims from `../visuals/compute_u11.py` (executed
2026-10-06). Reference: L=32, GQA 8 KV heads, dh=128, bf16. Claim
class: REQUESTED-BRANCH.

## R1 (remediation)

14.0 GB per decode token: the full weights are read once per token.

## A1

(a) Prefill: whole prompt at once, compute-bound. Decode: one token
at a time, memory-bound. (b) 1.0 vs ~2048 FLOP/byte, 14.0 GB per
token. (c) Prefill attention O(T^2) dominates: chunk the prefill.

## A2

(a) 2*L*h_kv*d_h*bytes per token. (b) 128.0 KB/token. 0.27, 8.59,
68.72 GB. (c) 2*80*8*128*2 = 327,680 bytes/token, x8x4096 =
10.7 GB.

## A3

(a) K/V are deterministic in the tokens, share the blocks. (b)
Unshared 2.62 GB, shared 0.26 GB, saving 2.36 GB. (c) Thirds at
100/500/2000 tokens, 100 requests: shared once per length class:
saving = sum over classes of (n_c - 1) * len_c * 128 KB.

## A4

(a) Iteration-level scheduling: evict finished, insert waiting.
(b) Static 800 wall steps vs 775 slot-steps over 2 slots, plus
immediate starts. (c) The arrival rate vs service rate: gain needs
queueing and length variance.

## A5

(a) Fixed blocks with a block table, waste < 1 block per sequence.
(b) 50.0% vs 0.0% on the toy. (c) Uniform 2048: contiguous waste
0%, paged waste 0%: paging buys nothing, keep it simple.

## A6

(a) E = (1-alpha^{k+1})/(1-alpha), speedup = E/(k*c+1). (b) 1.95x
at 0.7. 1.09x/1.44x/2.65x at 0.3/0.5/0.9. (c) E = (1-0.85^6)/0.15
= 3.39, cost = 5*0.2+1 = 2.0, speedup 1.69x: yes, it wins.

## A7

(a) Accept with min(1, p/q), else resample from norm(max(0, p-q)).
(b) Token 1: 1.0, token 2: 0.6, mixture recovers p. (c) No:
temperature 0 gives a point mass, support collapses and exactness
fails.

## A8

(a) Temperature reshapes, top-p/top-k truncate. (b) 0.865/0.66,
0.644/1.37, 0.455/1.80 bits, top-p 0.9 keeps 3. (c) High
temperature for diversity, many seeds: pass@100 wants broad
sampling, e.g. T=1.0, top-p 0.95.

## A9

(a) Extend the cache, prefill only new tokens. (b) 100 vs 1600
token-equivalents. 192 MB saved per turn. (c) 10k * 800 * 128 KB
= 1.02 TB: too big for one node, shard or evict idle sessions.

## A10

(a) 14.0/7.0/3.50 GB, quality must be measured per slice, never
assumed. (b) The decode win: weight traffic halves. (c) int8:
70B = 70 GB fits, eval math, code, and safety slices before
shipping.

## A11

(a) TTFT, TPOT, p99. (b) 120/533 ms, 8.0/38.1 ms, p99 ~4.6x mean
for exponential service. (c) Chunk the prefill, add capacity to
cut queueing.

## A12

(a) Load, quality, cost, latency is the slack. (b) 0.8 s decode
per request-slot. 10 RPS / 2 slots = 5 GPUs at the mean, plus
headroom. (c) Take the quality edge (smaller/quantized model) or
the cost edge (more GPUs), latency cannot give: state which and
the SLO risk.
