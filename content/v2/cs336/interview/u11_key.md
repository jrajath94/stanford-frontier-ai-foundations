# U11 interview key

## B1-B6

B1. Decode re-reads weights per token (1 FLOP/byte), prefill
reuses them across T (~T FLOP/byte).
B2. 2 * L * h_kv * d_h * bytes per token.
B3. Iteration-level insert/evict, no slot idles while work waits.
B4. Speedup = E/(k*c+1), E=(1-alpha^{k+1})/(1-alpha).
B5. Temperature, top-p, top-k.
B6. Time to first token, time per output token, 99th percentile.

## L1

L1.1 K/V appended per step per layer per head.
L1.2 128 KB/token. 8.59 GB.
L1.3 2*32*8*128*2 = 131,072 bytes.
L1.4 `kv_bytes`, doubling b or T doubles.
L1.5 Recompute is quadratic, debug: MHA is 4x (512 KB), critique:
no sharing, experiment: sweep h_kv.

## L2

L2.1 Small model drafts k tokens, target verifies in one pass.
L2.2 E=2.53, speedup 1.95x.
L2.3 Geometric series of acceptances.
L2.4 The loop, output distribution equals target alone.
L2.5 Medusa needs head training, debug: style mismatch kills
alpha, critique: independence, experiment: sweep k.

## E1

(a) 2*80*8*128*2 = 327,680 B/token, x8x4096 = 10.7 GB. (b)
Weights 140 GB need 2 GPUs, cache 10.7 GB fits alongside: 2 GPUs
(minimum). (c) GQA with fewer KV heads, quantization of the
cache. Rubric: (a) 2 pts, (b) 1 pt, (c) 1 pt. Red flag: ignoring
weights.

## E2

(a) Exponential-ish service (p99/mean ~4.6). (b) Chunk the
prefill (attacks TTFT), add capacity to cut queueing (attacks
both). Rubric: (a) 1 pt, (b) 2 pts.

## D1

Bug: `seq.blocks` is not cleared after eviction, so a reused
sequence object aliases freed blocks. Fix: clear the block list
on evict (or allocate a fresh list on schedule). Rubric: find 2
pts, fix 1 pt, name the aliasing 1 pt.

## S1

Without sharing: 100*2000*128KB = 25.6 GB just for prefixes.
With sharing: 0.26 GB. A prefix GPU (or shared block pool) is
clearly worth it: ~25 GB saved.

## S2

Cost = 3*0.5+1 = 2.5. E = 2.53, speedup 1.01x: the scheme is
dead. Draft must be much cheaper than the target.

## R1

Gaps: (1) no batch size: throughput undefined, report it. (2) no
length distribution: cache behavior unknown, report it. (3) no
latency: throughput-only claims hide tails, report p99.
