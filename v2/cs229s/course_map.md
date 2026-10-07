# course_map.md: cs229s U01-U10 map

Date: 2026-10-06. Baseline: October 6, 2026.

## Unit list (full course)

| Unit | Title | Prereqs | Bridge content | Calendar anchor |
|---|---|---|---|---|
| U01 | Sequence models and transformer workload | P11, P12, P13, P14 | Neural networks and autodiff, PyTorch, tensors, and numerical stability, Language and sequence modelling, Transformer mechanics | Week 1, Sep 27: Transformer Architecture |
| U02 | Hardware-aware design and compilers | P12, P15 | PyTorch, tensors, and numerical stability, Hardware and computer architecture | Week 2, Sep 30: Hardware Aware Algorithm Design |
| U03 | Transformer accounting and speculative inference | P14, P15 | Transformer mechanics, Hardware and computer architecture | Week 2, Oct 04: Analyzing the Performance of Transformers |
| U04 | CUDA and efficient attention | P12, P15 | PyTorch, tensors, and numerical stability, Hardware and computer architecture | Week 3, Oct 07 and Oct 11: CUDA/GPU programming, Efficient Attention |
| U05 | Quantization, sparsity, structured operators | P04, P12, P15 | Spectral and numerical linear algebra, PyTorch, tensors, and numerical stability, Hardware and computer architecture | Week 4, Oct 14: Memory Efficient Neural Networks |
| U06 | Training, adaptation, and data pipelines | P10, P11, P14, P17 | ML foundations and evaluation, Neural networks and autodiff, Transformer mechanics, Reinforcement learning | Week 4, Oct 18 / Week 5, Oct 21 / Oct 25: Adapting Large Language Models, Data in AI Pipelines. Oct 23 guest Tim Dettmers |
| U07 | Linear attention, SSMs, and FFT | P04, P12, P14, P15 | Spectral and numerical linear algebra, PyTorch, tensors, and numerical stability, Transformer mechanics, Hardware and computer architecture | Week 6, Oct 28 / Nov 01: Efficient Attention-Free Architectures, State Space Models |
| U08 | Serving and sparse MoE | P14, P15, P16 | Transformer mechanics, Hardware and computer architecture, Distributed systems and networking | Week 7, Nov 04: LLM Serving Efficiency. Nov 08: Sparse Mixture-of-Experts. Nov 11 guest Dylan Patel |
| U09 | Parallelism, clusters, and scheduling | P15, P16 | Hardware and computer architecture, Distributed systems and networking | Week 7, Oct 25: Parallelism Fundamentals. Oct 30 guest Deepak Narayanan. Nov 15 Parallelism. Nov 18 Cluster Scheduling |
| U10 | Retrieval systems, guests, and project synthesis | P19, P22, P24 | Retrieval and information access, Experimental method and research literacy, Production ML and stakeholder foundations | Week 10, Dec 02: Efficient Retrieval Systems. Nov 22 guest Albert Gu. Dec 06 poster session. Dec 09 final reports |

## Concept rows per unit

12 rows each, 120 total. Row IDs and names follow the ledger:

- U01: C01 RNN/transformer comparison, C02 pretraining/fine-tuning,
  C03 attention dimensions, C04 forward/backward, C05 parameter count,
  C06 activation memory, C07 autoregressive workload, C08
  training/inference contrast, C09 MLP versus attention cost, C10
  sequence scaling, C11 model quality, C12 baseline.
- U02: C01 GPU hierarchy, C02 kernels, C03 compilation/fusion, C04
  arithmetic intensity, C05 roofline, C06 layouts, C07
  vectorization, C08 launch costs, C09 concurrency, C10 tensor
  cores, C11 precision, C12 memory-bound/compute-bound tests.
- U03: C01 training FLOPs, C02 decode FLOPs, C03 KV-cache memory,
  C04 prefill, C05 arithmetic intensity, C06 batching, C07 draft
  model, C08 verification, C09 rejection/correction, C10 exactness
  assumptions, C11 acceptance rate, C12 latency tradeoff.
- U04: C01 threads/warps/blocks, C02 coalescing, C03 tiling, C04
  synchronization, C05 occupancy, C06 reductions, C07 attention
  bottleneck, C08 approximation versus exact IO savings, C09
  online softmax, C10 FlashAttention, C11 backward recomputation,
  C12 benchmark validity.
- U05: C01 scale/zero point, C02 rounding/clipping, C03
  per-channel/group choices, C04 activation versus weight
  quantization, C05 calibration, C06 structured/random sparsity,
  C07 pruning, C08 realized kernel speed, C09 butterfly matrices,
  C10 Monarch matrices, C11 error propagation, C12 quality
  validation.
- U06: C01 scaling laws, C02 zero/few-shot, C03 emergence
  caveats, C04 instruction following, C05 RLHF/RLAIF, C06
  constitutional feedback, C07 PEFT/LoRA, C08 data loading, C09
  preprocessing, C10 shuffling/packing, C11 throughput, C12
  reproducibility.
- U07: C01 convolution versus recurrence, C02 quadratic
  attention, C03 kernelized linear attention, C04 state-space
  recurrence, C05 convolution view, C06 scan, C07 FFT, C08
  numerical stability, C09 causal constraints, C10 expressivity
  tradeoffs, C11 hardware realization, C12 workload comparisons.
- U08: C01 continuous batching, C02 admission/backpressure, C03
  prefill/decode scheduling, C04 cache management, C05
  throughput/tail latency, C06 expert routing, C07 expert
  capacity, C08 load imbalance, C09 token dispatch, C10
  communication cost, C11 batching effects, C12 cost per request.
- U09: C01 data/tensor/pipeline/expert splits, C02 collective
  cost, C03 interconnect placement, C04 stragglers, C05 bubbles,
  C06 sharding, C07 checkpointing, C08 resource allocation, C09
  preemption, C10 fairness, C11 utilization, C12 failure
  recovery.
- U10: C01 index build versus query, C02 embedding storage, C03
  ANN recall/latency, C04 hybrid retrieval, C05
  filtering/reranking, C06 cluster economics guests, C07 SSM guest
  evidence, C08 project proposal/milestone, C09 profiling
  hypothesis, C10 reproducible optimization, C11 quality
  regression, C12 operational defense.

## Dependency flow

U01 feeds U03 (workload accounting needs the transformer
workload) and U06 (adaptation builds on the workload). U02
feeds U04 (GPU model before CUDA detail), U05 (hardware costs
before compression), and U09 (hardware before clusters). U03
uses U02 arithmetic intensity and feeds U08 (serving builds on
prefill/decode). U04 uses U03 KV-cache analysis. U05 uses U02
roofline and precision. U06 feeds U10 (reproducibility,
quality gates). U07 feeds U08 (MoE dispatch uses collectives)
and U10 (guest evidence grading). U08 uses U03 and U09
collectives. U09 uses U08 dispatch math. U10 synthesizes all:
retrieval, evidence grading, and the project arc.

## Scoping decisions

- Interview quotas apply per unit (each unit is a major lesson):
  8 breadth questions, two 8-rung deep ladders, 2
  analytical/quantitative exercises, 1 implementation/debug task,
  2 changed-constraint scenarios, 1 research-critique question.
- Per concept: the 15-item contract, with breadth recall (5
  exercises per concept, E01-E60 per unit), one
  numerical/analytical exercise, failure diagnosis,
  counterfactual comparison, and an evidence-based research
  question inside the lesson exercises. Deep oral ladders live
  at unit level in the interview bank.
- Capstones: each lab file ends with a unit-scope replication
  proposal, labeled PROPOSED, not executed. Course-level
  consolidation in `capstones/`: (a) executed SSM replication +
  falsifiable extension with a negative result reported
  honestly. (b) applied/FDE RAG serving design with labeled
  hypothetical numbers. Course-level transfer sets (10) and
  oral defenses (10 ladders x 8) in `interview/`.
