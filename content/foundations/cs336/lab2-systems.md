---
page_id: cs336-lab2
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 92
nav: "Lab 2 · Systems"
title: "Lab 2: Systems, Kernels, Parallelism"
summary: "Roofline analysis, MFU estimation, and choosing a parallelism strategy. 45 minutes."
sources:
  - tag: assignment
    label: "CS336 Assignment 2: Systems"
  - tag: video
    label: "Lecture 6 video"
    url: https://www.youtube.com/watch?v=xnDHaNUvHBg
---

Background: [Lecture 2](l02-pytorch-resource-accounting.html), [Lecture 5](l05-gpus-tpus.html), [Lecture 6](l06-kernels-triton-xla.html), [Lecture 7](l07-parallelism-1.html), [Lecture 8](l08-parallelism-2.html).

## Exercise 1: Roofline on paper (15 min)

An H100 has 989.5 TFLOP/s dense BF16 and 3.35 TB/s HBM bandwidth.

1. Compute its arithmetic intensity (FLOP per byte).
2. A linear layer does Y = XW with X: (2048, 8192), W: (8192, 8192). Compute the arithmetic intensity of this op. Is it compute-bound or memory-bound?
3. Now run the same layer as decoding with batch 1, sequence length 1. What changes? What is the bottleneck now?
4. Name two techniques that move decode toward the compute-bound region.

> [!INTERVIEW] This is the single most common systems question in LLM interviews. Do it until the arithmetic is automatic.

## Exercise 2: Estimate training time (15 min)

You train a 7B-parameter Transformer on 1T tokens with 128 H100s.

1. Estimate total training FLOPs using the 6ND rule.
2. At 45% MFU, how many days does training take? Show every step.
3. Your MFU is 25%, not 45%. Name the three most likely causes, in order, and one diagnostic for each.

## Exercise 3: Pick a parallelism strategy (15 min)

You have a 70B model, 2048-token sequences, and 64 H100s across 8 nodes (8 GPUs per node, NVLink inside the node, InfiniBand between nodes).

1. The model does not fit on one GPU. Which parallelism do you apply first, and why?
2. Where do you place tensor parallelism: inside the node or across nodes? Justify with bandwidth numbers.
3. Sketch the full 4D strategy (DP, TP, PP, and one more). State what each dimension shards.
4. What breaks first if you double the sequence length to 4096?

> [!INTERVIEW] Interviewers want the reasoning chain: memory constraint first, then bandwidth hierarchy, then the strategy. State numbers, not adjectives.
