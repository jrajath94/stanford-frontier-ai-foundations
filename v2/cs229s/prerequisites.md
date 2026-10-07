# prerequisites.md: cs229s prerequisite map

Date: 2026-10-06.

Shared bridge modules live at `v2-pack/shared/prerequisites/`.
This file links them. It does not rebuild them. Each unit lesson
carries local remediation for the exact objects it uses.

## Bridge index

| ID | File | Used by |
|---|---|---|
| P04 | `../shared/prerequisites/p04_spectral.md` | U05 (conditioning, low-rank, quantization error) |
| P10 | `../shared/prerequisites/p10_ml_foundations.md` | U01, U06, U10 (evaluation, train/test splits, metrics) |
| P11 | `../shared/prerequisites/p11_neural_nets.md` | U01 (forward/backward, autodiff) |
| P12 | `../shared/prerequisites/p12_pytorch.md` | U01, U02, U04, U05 (tensors, dtypes, devices) |
| P13 | `../shared/prerequisites/p13_language.md` | U01 (sequence modelling, perplexity) |
| P14 | `../shared/prerequisites/p14_transformer.md` | U01, U03 (attention, KV cache) |
| P15 | `../shared/prerequisites/p15_hardware.md` | U02, U03, U04, U05 (memory hierarchy, roofline) |
| P16 | `../shared/prerequisites/p16_distributed.md` | U07, U08, U09 (collectives, many-machine cost reasoning) |
| P17 | `../shared/prerequisites/p17_rl.md` | U06 (rewards, value computation, RLHF/DPO mechanics) |
| P19 | `../shared/prerequisites/p19_retrieval.md` | U10 (sparse/dense retrieval, quality measurement) |
| P22 | `../shared/prerequisites/p22_experiments.md` | U10 (experiment design, paper reading, rubrics) |
| P24 | `../shared/prerequisites/p24_production_ml.md` | U10 (production systems, gates, rollback) |

## Per-unit local remediation

- U01: re-derives RNN recurrence, attention shapes, parameter
  count, and activation memory from the tensor level up. No
  assumed fluency with transformer blocks.
- U02: re-derives arithmetic intensity and the roofline from
  FLOP and byte counts. GPU hierarchy is rebuilt from
  registers to HBM.
- U03: re-derives KV-cache memory from attention shapes and
  speculative decoding acceptance from a Bernoulli toy.
- U04: rebuilds the CUDA execution model from threads to
  grids, then derives online softmax and FlashAttention
  tiling from the memory budget.
- U05: rebuilds linear quantization from rounding, then
  sparsity formats and structured matrices from small
  examples.
- U06: re-derives scaling-law fits from two anchor
  points, RLHF reward mechanics and DPO from preference
  pairs, and LoRA merge math from rank factors. No
  assumed fluency with reinforcement learning or
  fine-tuning methods.
- U07: re-derives convolution and the SSM recurrence from
  the kernel view, and fp16 overflow from exponent range.
  No assumed fluency with state-space models.
- U08: re-derives the prefill/decode split, KV-cache
  eviction budgets, and tail-latency percentiles from
  request traces. No assumed fluency with serving
  infrastructure.
- U09: re-derives collective costs, bubble fractions, and
  ZeRO sharding from byte counts. No assumed fluency
  with distributed training.
- U10: re-derives ANN recall/latency tradeoffs and
  retrieval quality metrics from small index toys. No
  assumed fluency with search or production ML.

## Diagnostic

`diagnostics/diagnostic-cs229s.md` tests P04, P11, P12, P13, P14,
P15 readiness with 12 questions. Keys in
`diagnostics/keys-cs229s.md`. A learner who misses more than 4
works the linked bridge modules before U01.
