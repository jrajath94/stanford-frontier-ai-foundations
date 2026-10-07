# keys.md, U03 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

C = 6PD + 12LnTD: dense term plus the attention term over
D tokens in sequences of length T.

## E02

6*1664*31 = 309504. Attention: 12*2*8*4*31 = 23808.
Total 333312.

## E03

The attention term at T = 16384 is large: the estimate
used 6PD only. Add 12LnTD, the profiler counts it.

## E04

T = 512, n = 4096: attention/dense = (12*512)/(72*4096)
= 0.021, about 2 percent. Negligible.

## E05

Hypothesis: step time vs T fits a + bT + cT^2 with the
quadratic term significant past ~8K, 6PD-only (no T^2)
underfits. Report both coefficients.

## E06

Per token: 2P + 4LTn FLOPs (weights plus cached-key
scores).

## E07

2*13e9*50 = 1.3e12 FLOPs dense, attention add-on is
small. About 1.3e12.

## E08

Same FLOPs but different bytes: one model moves fewer
bytes (quantized weights, smaller cache). Decode is
bandwidth-bound, so bytes set speed.

## E09

Yes: 2P per token still holds per sequence, batch 64
reuses weights 64x, raising intensity toward
compute-bound. FLOPs per token unchanged, time per
token falls.

## E10

Hypothesis: per-token time correlates with bytes (R^2
high) and weakly with FLOPs (R^2 low) at batch 1.
Report both R^2.

## E11

KV bytes = 2 * B * T * L * n * bytes-per-number (keys
plus values, fp16: x2).

## E12

2*1*32768*32*4096*2 = 17179869184 bytes = 16.0 GiB.

## E13

The KV cache: it grows with T while weights are fixed.
Fixes: smaller batch, shorter context, paged/blocked
cache, or quantization of the cache.

## E14

KV heads h_kv = 8: cache = 2*B*T*L*(h_kv*d)*bpe =
2*B*T*L*n*(h_kv/h)*bpe. Here 1/4 of the MHA cache.

## E15

Hypothesis: max batch B_max(T) = C/T with C from the
cache byte budget. Fit B_max vs 1/T, report C and R^2.

## E16

Prefill is the parallel forward pass over the prompt
that builds the KV cache before decode starts.

## E17

Dense term: 2*7e9*8192 = 1.15e14 FLOPs.

## E18

The T^2 score term: naive attention materializes
(65536)^2 scores per head. Tile or chunk the prefill.

## E19

Chunked into 4: peak score memory per chunk is
(16384)^2 per head-set, 1/16 of the full matrix.

## E20

Hypothesis: time-to-first-token is quadratic in T with
naive attention and linear with tiled attention, the
bend point is where scores exceed fast memory. Report
the bend T.

## E21

Prefill intensity ~T FLOP/byte, decode intensity ~B
FLOP/byte (batch 1: ~1).

## E22

Prefill T=64: I=64 < 150, memory-bound. Decode B=200:
I=200 > 150, compute-bound.

## E23

Decode is compute-bound at this batch: bandwidth is not
the limit, so more bandwidth does nothing. Check the
batch against the ridge.

## E24

int8 weights halve the bytes again: decode intensity
~2B per byte-count... precisely, I doubles to ~2B
(FLOP per byte), since bytes halve at fixed FLOPs.

## E25

Hypothesis: tokens/s rises linearly with batch
(bandwidth-bound) then flattens (compute-bound), the
knee sits near B = ridge point. Report the knee B.

## E26

Batching trades latency for throughput: one weight read
serves B requests.

## E27

Toy model: latency = 0.020*(1+0.05*15) = 0.035 s.
Throughput = 16/0.035 = 457 tok/s total.

## E28

Diagnose: past the compute roof. Batching adds latency
with no throughput gain. Reduce batch to the knee.

## E29

0.020*(1+0.05*(B-1)) <= 0.025 gives B-1 <= 5, so max
batch 6 in the toy model.

## E30

Hypothesis: throughput saturates with batch while p99
latency keeps rising, the best B is at the saturation
knee. Report both curves.

## E31

c: draft/target per-token time ratio. gamma: drafts per
round. a: per-token acceptance probability.

## E32

E[k] = (1-0.8^6)/0.2 = 3.689. Speedup =
3.689/(5*0.05+1) = 2.95x.

## E33

Causes: (1) low acceptance a (measure it, formula E[k]
collapses), (2) slow draft c (measure draft vs target
step time).

## E34

Speedup = E[k]/1 = E[k]: every round costs one target
step. Larger gamma always helps until verification
slows (C08 slope).

## E35

Hypothesis: measured speedup vs gamma rises then falls,
the peak gamma matches argmax of E[k]/(gamma*c+1).
Report both peaks.

## E36

Draft tokens are known inputs, so the target computes
all gamma+1 distributions in one parallel forward pass,
like a prefill.

## E37

1 + 0.05*8 = 1.4 target steps.

## E38

The verification pass is compute-heavy (long context
or huge gamma): the ~1 step model broke. Model as
1 + slope*gamma.

## E39

Drafts must be re-encoded: verification becomes a
prefill over prompt + drafts, costing more than one
step. Keep drafts in the cache.

## E40

Hypothesis: verification time is linear in gamma with
a small slope, the slope sets the optimal gamma.
Report the fitted slope.

## E41

Accept x_i with probability min(1, p(x_i)/q(x_i)).

## E42

0.2/0.7 = 0.286.

## E43

Bug class: the resample distribution is wrong (not
norm(max(0, p-q))) or the bonus token mishandled, so
the output is not p-distributed.

## E44

q = p: min(1, p/p) = 1, so a = 1. Every draft
accepts.

## E45

Hypothesis: over 50K rounds the output histogram
matches p within sampling noise. Report max bin
deviation.

## E46

(1) Same tokenizer/vocabulary for draft and target.
(2) Exact resample distribution norm(max(0, p-q)).

## E47

a = 1.

## E48

Different tokenizers mean different distributions over
different spaces, the rejection-sampling identity no
longer applies. Outputs diverge from p.

## E49

No: dropping on reject without resampling loses the
residual mass max(0, p-q). Output is not p.

## E50

Hypothesis: with mismatched tokenizers the output
distribution diverges measurably from p (toy only).

## E51

E[k] = (1 - a^(gamma+1)) / (1 - a).

## E52

(1 - 0.6^5)/0.4 = (1-0.07776)/0.4 = 2.306.

## E53

The constant-a assumption broke: real acceptance
falls with position (later drafts condition on earlier
ones). Fit per-position a_i.

## E54

gamma + 1 tokens per round (all drafts plus the bonus
token).

## E55

Hypothesis: per-position acceptance a_i declines with
i. Report the fitted a_i curve.

## E56

Speedup = E[k] / (gamma*c + 1).

## E57

E[k] = (1-0.9^8)/0.1 = (1-0.4305)/0.1 = 5.695. Cost:
7*0.1+1 = 1.7. Speedup 3.35x.

## E58

Causes: (1) verification slower than 1 step (long
context), (2) measured a below the assumed a on this
workload.

## E59

Speedup = E[k], push gamma up until verification cost
(1 + slope*gamma) bends the curve.

## E60

Hypothesis: formula vs measured speedup agree within
20 percent, deviations trace to the verification
slope. Report the error map over (a, c, gamma).
