# Lab 08: Serving and sparse MoE

Unit: cs229s-U08. Date: 2026-10-06.
Run: `python3 verify_lab08.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-08.md`.

## Objective

Verify U08 mechanisms on CPU with numpy: continuous vs
static completion means, the admission wait distribution,
prefill/decode/chunk times, KV bytes per token and the LRU
eviction, the tail percentiles with Little's law, top-2
routing weights, expert capacity and drops, the 5.0
imbalance, dispatch bytes, the all-reduce formula, the
batching curve with shrinking gains, and the cost-per-
million-tokens arithmetic.

## Setup

```bash
python3 verify_lab08.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Continuous vs static means and latencies.
2. Admission waits under a burst.
3. Phase times and chunking.
4. KV bytes and LRU eviction.
5. Tail percentiles and Little's law.
6. Top-2 routing.
7. Capacity and drops.
8. Imbalance ratio.
9. Dispatch bytes.
10. Collective costs.
11. Batching curve.
12. Cost per request.

## Replication proposal (PROPOSED, not executed)

Question: does continuous batching cut mean latency on a
bursty trace, and does the MoE dispatch tax exceed the
dense baseline at matched FLOPs? Hypothesis: continuous
wins by the formation-wait gap, and MoE wins only when
dispatch stays under the FLOP saving. Method: event
simulation for the scheduler. Byte-count model plus
measured all-to-all for MoE. Budget: CPU hours. Failure
criterion: no latency gap on bursty traces (then the
model is miscalibrated).
