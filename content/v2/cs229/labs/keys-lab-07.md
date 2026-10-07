# keys-lab-07.md

Date: 2026-10-06. Reference outputs for lab 07.

Computed with numpy 1.26.x, float64, on this machine.
Tolerances below absorb platform float differences.

## Task 1

SIMCLR loss (B = 4, d = 6, seed 9): 3.3532.
Per-example terms: 1.0557, 0.8086, 0.6405,
0.8484. Tolerance: 0.01. After moving each
positive pair 10 percent closer: 3.2695
(tolerance 0.01). The loss falls when
positive inner products rise, confirming the
monotonicity of SL-04: the objective rewards
positive alignment and punishes negative
alignment.

## Task 2

Square layer (4096x4096): dense 16,777,216.
LoRA counts: r=4: 32,768 (512.0x), r=8:
65,536 (256.0x), r=16: 131,072 (128.0x),
r=64: 524,288 (32.0x). Rectangular layer
(11008x4096): dense 45,088,768. LoRA counts:
r=4: 60,416 (746.3x), r=8: 120,832
(373.2x), r=16: 241,664 (186.6x), r=64:
966,656 (46.6x). Tolerance: exact integers.
Memory nuance: LoRA shrinks the trainable
parameters, gradients, and optimizer states,
but the frozen base weights and the
activations are unchanged, so training-time
memory savings are limited where they
dominate.

## Task 3

Ranked list [d1, d2, d3, d4, d5, d6], gold
{d2, d4, d5}. Recall@3 = 1/3 = 0.3333.
Recall@5 = 3/3 = 1.0000. DCG@5 = 0/log2(2)
+ 3/log2(3) + 0/log2(4) + 2/log2(5) +
1/log2(6) = 0 + 1.8928 + 0 + 0.8614 +
0.3869 = 3.1410. IDCG@5 = 3/1 + 2/log2(3) +
1/2 + 0 + 0 = 4.7619. NDCG@5 = 3.1410 /
4.7619 = 0.6596. Tolerance: 0.001. NDCG
punishes the d1-at-rank-1 miss (the top
position carries the largest discount
weight). Recall@5 does not (it ignores
ranking entirely).
