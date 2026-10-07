# keys.md, U08 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

Iteration-level scheduling: the batch is re-formed
every decode iteration, admitting arrivals without
waiting for the batch to finish.

## E02

Continuous completions: 1, 2, 3, 4 (mean 2.5).
Static: 2, 2, 4, 4 (mean 3.0). Mean latencies:
1.0 vs 1.5.

## E03

The scheduler inserts whole prefills. Add chunked
prefill (or separate the phases).

## E04

No. with all arrivals at t=0 there is no wait to
remove. Static pairs finish at mean 1.5 vs
continuous 2.5 in the toy. The win was about
formation waiting.

## E05

Hypothesis: continuous cuts mean latency and the
gap grows with burstiness. Report both.

## E06

Admission control decides who enters the running
set. Backpressure signals upstream to slow down
when the queue fills.

## E07

Peak queue 6, max wait 2.0 s, mean wait 0.8 s.

## E08

Arrival rate exceeds service rate. The queue never
drains. Waits grow without bound.

## E09

The server rejects everything immediately. Pure
fail-fast admission.

## E10

Hypothesis: mean wait explodes past the stability
point (arrival = service rate). Report the knee.

## E11

Prefill: one parallel prompt pass, compute-bound.
Decode: token-by-token, memory-bound.

## E12

Prefill 0.256 s, decode token 0.02 s, 2-way chunk
0.128 s.

## E13

Missing chunked prefill. Whole prefills stall the
running decodes.

## E14

The distinction collapses. Treat all tokens alike. 
the scheduler still batches, but phases no longer
differ.

## E15

Hypothesis: decode stall falls as 1/chunks until
bookkeeping dominates. Report the curve.

## E16

Bytes per token = 2 * L * n * bytes per float.

## E17

524,288 bytes = 0.5 MiB per token. B is evicted.

## E18

Thrash. The working set exceeds the lot, so every
access evicts the next need.

## E19

The shared 200-token prefix parks twice: 100 MiB
wasted in the toy.

## E20

Hypothesis: LRU beats FIFO under locality, ties
otherwise. Report both hit rates.

## E21

p99: the latency 99% of requests beat. Little's
law: concurrency = throughput * mean latency.

## E22

Mean 1.2 s, p50 1.0 s, p99 3.0 s, concurrency 60.

## E23

Look at the slow requests' attributes: prompt
length, cache hit, arrival burst. One cause
usually dominates.

## E24

p50 = p99 = 1.2 s, concurrency 60. The tail
vanishes. The mean is now honest.

## E25

Hypothesis: one cause (e.g. long prefill)
dominates the tail. Report the split.

## E26

g = softmax(W_g x). Keep top-k logits,
renormalize. Output = sum w_i E_i(x).

## E27

Experts 0 and 2, weights 0.646 and 0.354.

## E28

Gate collapse. One expert won early and starved
the rest.

## E29

k = 8 is a dense model: every expert runs for
every token.

## E30

Hypothesis: usage entropy dips early then
recovers with the balance loss. Report the curve.

## E31

Capacity = (tokens * k / experts) * factor.

## E32

Capacity 20. Expert 0 drops 20 of 40. Global:
20/128 = 15.6%.

## E33

Its drops are heavy: it trains on a biased
subset of its routed tokens.

## E34

Buffers grow 8x. Drops vanish but memory may OOM. 
the factor is not free.

## E35

Hypothesis: eval is flat until drops pass ~5%,
then falls. Report the knee.

## E36

Imbalance = max count / mean count.

## E37

Mean 8, max 40, ratio 5.0.

## E38

The counts are skewed: one expert holds ~5x the
mean and sets the step time.

## E39

All 64 to one expert: ratio 64/8 = 8.0.

## E40

Hypothesis: imbalance vs task score is U-shaped
in the balance weight. Report the sweet spot.

## E41

Bytes one way = tokens * k * d * bytes per float.

## E42

One way 1 MiB, round trip 2 MiB, 32 layers 64
MiB per step.

## E43

Dispatch communication dominates. The FLOP count
is equal. The wire is not.

## E44

Dispatch halves. Quality risk rises: one expert
per token means routing mistakes have no backup.

## E45

Hypothesis: step time grows with k through
dispatch bytes. Report the slope.

## E46

Ring all-reduce per GPU: 2*(n-1)/n * S.
All-to-all per GPU: ~S each way.

## E47

All-reduce: 1.75 MiB/GPU. All-to-all: ~1 MiB/GPU
each way.

## E48

Incast congestion: all GPUs blasting one peer
collapses the effective bandwidth.

## E49

n = 2: all-reduce = 1 MiB/GPU. All-to-all ~0.5
MiB/GPU each way.

## E50

Hypothesis: time is linear in S with different
slopes. Report both.

## E51

R(B) = B / (c0 + c1*B). Asymptote: 1/c1.

## E52

142.9, 307.7, 381.0 tok/s. Gains: 2.15x then
1.24x.

## E53

KV-cache memory. The formula's best B does not
fit.

## E54

R = 1/c1 = 500 always. Batching buys nothing.

## E55

Hypothesis: measured R follows B/(c0+c1*B) with
fitted constants. Report the fit.

## E56

Cost per 1M = 1e6 / R * (dollars per hour) /
3600.

## E57

$1.67 at 500 tok/s, $0.83 at 1000 tok/s. A
640-token request: $0.00107 at the base rate.

## E58

Utilization below 100%. The toy assumed full
use. Real GPUs idle.

## E59

Unchanged. Both halves cancel in the division.

## E60

Hypothesis: utilization dominates the gap between
the toy and the bill. Report both.
