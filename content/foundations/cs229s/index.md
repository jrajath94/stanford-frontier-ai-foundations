---
title: "CS229S: Systems for Machine Learning"
course: cs229s
type: course-index
instructor: Azalia Mirhoseini
term: Fall 2024
---

The systems companion to CS336. Where CS336 taught you how to build a language model, CS229S teaches you how to think about the machine it runs on: arithmetic intensity, the memory hierarchy, FLOP counting, and the art of making the same model run 10x faster.

This course is built as a bridge. Its topics overlap heavily with CS336, so these lessons teach what is new, the systems methodology and the analytical frameworks, and link to the CS336 lessons for the mechanics. Nothing is taught twice.

## Lessons

1. [Introduction: the systems mindset](l01-introduction.html), the stack, the 10x gap, the toolkit
2. [Hardware-aware algorithm design](l02-hardware-aware-design.html), arithmetic intensity, roofline, GPU memory hierarchy
3. [Analyzing transformer performance](l03-transformer-performance.html), FLOP counting, KV cache efficiency, speculative decoding
4. [Efficient attention](l04-efficient-attention.html), bottlenecks, FlashAttention from the systems angle
5. [Memory-efficient networks](l05-memory-efficient-networks.html), quantization, sparsity, pruning, distillation
6. [Parallelism fundamentals](l06-parallelism-fundamentals.html), data/tensor/pipeline, ZeRO, FSDP, when to use which
7. [Guest: Tim Dettmers](l07-guest-dettmers.html), efficient training and inference, from the builder of bitsandbytes and QLoRA
8. [Guest: Deepak Narayanan](l08-guest-narayanan.html), parallelism in practice

## Sources

- Course site: [CS229S Fall 2024](https://cs229s.stanford.edu/fall2024/) (calendar, slides)
- Guest talks: Stanford MLSys Seminars YouTube
- Note: no public lecture video playlist exists for this offering. Lessons are built from the official slide decks and the two guest talk recordings.
