---
page_id: cs229s-l07
course_slug: cs229s
course_name: "CS229S: Systems for Machine Learning"
course_order: 3
order: 7
nav: "L07 · Memory Efficiency"
title: "Lecture 7: Memory-Efficient Neural Networks"
summary: "Shrink the model three ways: pruning and sparsity, quantization from FP32 down to integers, and knowledge distillation. All numbers worked by hand."
date: "2024-10-14"
instructor: "Azalia Mirhoseini"
offering: "Fall 2024"
concepts: [pruning, sparsity, structured-sparsity, nm-sparsity, quantization, kmeans-quantization, linear-quantization, floating-point, knowledge-distillation]
sources:
  - tag: slides
    label: "Memory-Efficient Neural Networks slide deck (Fall 2023 headers)"
  - tag: paper
    label: "Han et al., Learning both Weights and Connections for Efficient Neural Networks (2015)"
    url: https://arxiv.org/abs/1506.02626
  - tag: paper
    label: "Han et al., Deep Compression (ICLR 2016)"
    url: https://arxiv.org/abs/1510.00149
  - tag: paper
    label: "Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (2017)"
    url: https://arxiv.org/abs/1712.05877
  - tag: paper
    label: "Hinton et al., Distilling the Knowledge in a Neural Network (2014)"
    url: https://arxiv.org/abs/1503.02531
  - tag: supplement
    label: "MIT 6.5940 TinyML and Efficient Deep Learning Computing (Prof. Song Han)"
    url: https://hanlab.mit.edu/courses/2023-fall-65940
---

## How to read this lesson

**Level 1 (Core)** gives the three shrink methods and their
tradeoffs: pruning, quantization, distillation. **Level 2 (Deep)**
works the numbers: the FP32 bit layout, k-means storage math, and
the linear-quantization scale equations. The lecture draws on MIT
6.5940 (Song Han); the classical schemes here complement
[CS336 L02](../cs336/l02-resource-accounting.html)'s precision
ladder.

Note: the Fall 2024 calendar lists "Butterfly and Monarch matrices"
under this lecture's sparsity topics. The shipped slide deck (Fall
2023) does not cover them. They are flagged below, not taught.

## Level 1: Three ways to shrink a model

Memory is expensive. Lecture 5 taught reducing memory *accesses*;
this lecture reduces the memory *required*.

**Pruning and sparsity.** Discard a subset of parameters after
training. The result is a sparse network.

**Quantization.** Use a subset of the real numbers: fewer bits per
value instead of the full 32-bit float.

**Knowledge distillation.** Train a small student model to mimic a
large teacher.

Pruning and quantization compose: the Deep Compression results show
the combination beats either alone, with accuracy falling as
compression rises.

![Memory chapter plate](assets/media-generation-plate-chapter-l07-memory-0-e99a1302-0e56-463a-917c-27764af8b66a.webp "Shell: synthesis. Prune, quantize, distill: three cuts, one smaller model. Source: original chapter plate.")

## Level 1: Pruning

Formal goal: given original weights W, find pruned weights Wp
minimizing loss L(x; Wp) subject to ||Wp||_0 < T, where ||.||_0
counts nonzeros and T is the target parameter count. An ideal
pruning process accelerates training or inference on real hardware,
retains accuracy, and generalizes across models.

**Patterns decide hardware value.** Unstructured (fine-grained)
pruning removes individual weights: higher accuracy, but the
surviving nonzeros scatter randomly, so you still read and write
every block. Structured (coarse-grained) pruning removes whole
rows, channels, or blocks: lower accuracy, but memory access stays
regular and hardware stays fast.

![N:M sparsity](assets/slide-l07-nm-sparsity.png "2:4 structured sparsity: 2 of every 4 contiguous values are zero. Ampere sparse tensor cores exploit it. Source: Stanford slides.")

NVIDIA Ampere's sparse tensor cores exploit **2:4 structured
sparsity**: in every contiguous group of 4 values, exactly 2 are
zero. The pattern is regular enough for low metadata overhead, and
50% of weights disappear while accuracy holds. The interview point:
sparsity only speeds up inference if the pattern matches what the
hardware accelerates.

**How to choose what to remove.** Direct one-shot pruning is hard;
iterative prune-and-fine-tune works better (Han et al., 2015).
Heuristics: fine-grained magnitude pruning keeps weights with
large absolute values (W = [5, 0.3, -0.4] keeps 5); coarse-grained
magnitude pruning keeps rows with large sums of absolute values.
Regression-based pruning (He et al., 2017) picks rows to minimize
the layer-output reconstruction error ||Z - Zhat||_F^2 subject to a
nonzero budget, then relearns the remaining weights.

**Sensitivity analysis** sets the per-layer ratio: layers differ,
so prune each layer at increasing ratios in magnitude order and
measure accuracy. Early layers tend to be sensitive; fully
connected layers tend not to be. One global sparsity number is
wrong; per-layer numbers are right.

> [!QA]
> Q: Why does unstructured pruning often fail to speed up inference?
> A: Because the surviving weights scatter randomly across memory. The hardware still reads and writes every block to find the nonzeros, so the memory traffic barely drops. Structured patterns like 2:4 N:M keep accesses regular, which is what sparse tensor cores need. Accuracy favors fine-grained; speed favors structured.
> Follow-up: How do you pick the sparsity ratio per layer?
> A: Sensitivity analysis: prune each layer at increasing ratios in magnitude order and measure accuracy. Layers have different sensitivity, with early layers usually more fragile. Assign aggressive ratios only where accuracy survives.

## Level 1: Quantization, the idea

Quantization maps values from a large set to a smaller set. The
quantization error is |true value - quantized value|. Motivation in
two lines:

```
Storage = number of parameters x bits per parameter
Add cost: O(N) in bit-width N | Multiply cost: O(N^2)
```

Halving the bits halves the storage and quarters the multiply
cost. That is the whole economic case.

## Level 1: How numbers live in bits

**Integers.** Unsigned n-bit: 0 to 2^n - 1. Signed: one bit for
sign. Two common signed layouts: sign-magnitude (a sign bit plus
magnitude) and two's complement (negation by bit-flip plus one;
-70 as 10111010 versus sign-magnitude's 11000110 for the same
magnitude pattern the slides show).

**Fixed point.** Fixed split between integer and fraction bits.
The slides' 8-bit example with 4 fraction bits reads -4.375.

**Floating point (IEEE 754).** Value = (-1)^sign x (1 + fraction)
x 2^(exponent - bias). The exponent buys dynamic range: the gap
between the largest and smallest representable values. Bias
(127 for FP32's 8-bit exponent) shifts the exponent range to cover
small values.

Worked FP32 example from the slides: exponent bits sum to 120,
bias 127, fraction 2^-2 = 0.25. Value = 1.25 x 2^(120-127) =
1.25 x 2^-7 = 0.0097656.

![FP32 example](assets/slide-l07-fp32-example.png "Sign, 8-bit exponent, 23-bit mantissa. The worked value is 0.0097656. Source: Stanford slides.")

Edge cases: exponent all zeros gives subnormals (value =
(-1)^sign x fraction x 2^(1-127)); all-ones exponent with zero
fraction is infinity; nonzero fraction is NaN. You cannot
represent infinitely small values: the format has a floor.

**FP16 versus BF16.** FP16: 5 exponent bits, 10 fraction bits.
BF16: 8 exponent bits, 7 fraction bits. BF16 trades precision for
dynamic range, which stabilizes training: gradients span many
orders of magnitude, and range matters more than fine resolution.
This is why BF16 is the default training format.

## Level 1: K-means quantization

Han et al., Deep Compression (ICLR 2016 best paper). Inference
procedure:

1. Cluster the FP32 weight values into S clusters.
2. Store per weight a log2(S)-bit cluster index, plus the S
   FP32 centroids.
3. At compute time, reconstruct weights through the index map.
   Computation stays FP32.

![K-means quantization](assets/slide-l07-kmeans-quant.png "Cluster FP32 weights, store 2-bit indices plus 4 centroids, reconstruct at compute time. Source: Stanford slides, Han et al. 2016.")

Storage math for a D by D FP32 matrix: 32 x D^2 bits originally.
After: log2(S) x D^2 index bits + S x 32 centroid bits. At D = 4,
S = 4: 64 bytes become 20 bytes, 3.2x compression, and the ratio
improves as S stays small relative to D. There are no compute
savings: the math is still FP32.

Fine-tuning trains the shared centroids, not individual weights:
dL/dCk sums dL/dWij over all weights assigned to centroid k, then
C updates by gradient descent. Huffman coding on the indices
exploits nonuniform cluster frequencies for 20%+ further savings
on AlexNet's last layer.

> [!QA]
> Q: Work out the k-means storage saving for a 4x4 FP32 matrix with 4 clusters.
> A: Original: 32 bits x 16 weights = 64 bytes. After: each weight stores a 2-bit cluster index (log2 4), so 16 x 2 bits = 4 bytes, plus 4 FP32 centroids = 16 bytes. Total 20 bytes. Saving: 64 - 20 = 44 bytes, 3.2x compression. Compute is unchanged because weights reconstruct to FP32 before the matmul.
> Follow-up: When would you prefer linear quantization instead?
> A: When you also want faster math. K-means only shrinks storage; the compute stays floating point. Linear quantization maps values to integers with a scale and zero-point, so the matmul itself runs in integer arithmetic.

## Level 1: Linear quantization

Jacob et al. (2017). Map real values r to integers q with a scale
S and zero-point Z:

r = S (q - Z)

Given rmin and rmax from the weight matrix and qmin, qmax from the
bit-width (for N bits, -2^(N-1) to 2^(N-1) - 1), solve the two
equations rmax = S(qmax - Z) and rmin = S(qmin - Z) for S and Z.
The slides' 2-bit example maps weights like 2.09 to integer 1
with S = 1.07.

Integer-only matmul follows by substitution. With Y = Sy(qY - ZY),
W = Sw(qW - Zw), X = Sx(qX - Zx):

qY = (Sw Sx / Sy) (qW - Zw)(qX - Zx) + Zy

All integer math, with one rescale to N bits. Weights and compute
are both integer: storage and speed improve together.

![Linear quantization](assets/slide-l07-linear-quant-sz.png "Solve for scale S and zero-point Z from the value ranges, then run the matmul in integers. Source: Stanford slides, Jacob et al. 2017.")

## Level 1: Knowledge distillation

Small models are wanted (phones, IoT, laptops) but hard to train:
they underfit. Distillation trains a small student against a large
teacher that already learned well.

The design space aligns student and teacher on: output logits
(Hinton et al., 2014), intermediate weights, intermediate
features, gradients of feature maps, or sparsity patterns.

Logits matching, worked: teacher says plane 5, car 1, giving
softmax probabilities 0.982 and 0.017. Student says plane 3, car
2: 0.731 and 0.269. The distillation loss (cross-entropy
E(-pt log ps) or L2 on probabilities) pulls the student's
distribution toward the teacher's. The student learns the
teacher's uncertainty, not just the hard label: that dark
knowledge is the point.

![Distillation logits](assets/slide-l07-distill-logits.png "Teacher 5 vs 1, student 3 vs 2: the loss aligns the probability distributions. Source: Stanford slides.")

## Level 2: Butterfly and Monarch matrices [uncertain]

The Fall 2024 calendar lists "Butterfly and Monarch Matrices"
under this lecture's sparsity bullets. The shipped deck does not
contain them, so this is a pointer, not a derivation. The idea, at
the level the calendar implies: structured matrix families that
represent dense transforms with far fewer parameters than a full
matrix, giving the hardware-friendly regularity that unstructured
sparsity lacks. Treat any interview claim about them as
[uncertain] until checked against the 2024 lecture recording.

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/slide-l07-nm-sparsity.png" alt="N:M sparsity">
<div class="rc-body">
<strong>1. Patterns decide sparsity value</strong>
<p>Unstructured pruning keeps accuracy but scatters memory access.
2:4 structured sparsity matches Ampere tensor cores: 50% fewer
weights, real speedup.</p>
<p class="rc-num">Key: 2 of every 4 are zero</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l07-nm-sparsity.png" alt="Pruning method">
<div class="rc-body">
<strong>2. Prune iteratively, per layer</strong>
<p>Iterative prune-and-fine-tune beats one-shot. Magnitude and
regression heuristics pick victims; sensitivity analysis sets
per-layer ratios.</p>
<p class="rc-num">Key: min L s.t. nonzeros below T</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l07-fp32-example.png" alt="FP32">
<div class="rc-body">
<strong>3. FP32 is sign, exponent, mantissa</strong>
<p>Value = (-1)^s (1+f) 2^(e-127). The exponent buys dynamic
range. Worked value: 0.0097656. BF16 trades precision for range.</p>
<p class="rc-num">Key: 1 + 8 + 23 bits</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l07-kmeans-quant.png" alt="K-means quantization">
<div class="rc-body">
<strong>4. K-means shrinks storage, not compute</strong>
<p>Cluster weights, store log2(S)-bit indices plus centroids.
4x4 FP32 with 4 clusters: 64 to 20 bytes, 3.2x. Math stays FP32.</p>
<p class="rc-num">Key: indices + centroids</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l07-kmeans-quant.png" alt="Storage math">
<div class="rc-body">
<strong>5. Storage math: 32D^2 minus the map</strong>
<p>Saving = 32D^2 - (log2(S) D^2 + 32S). Huffman on skewed
indices adds 20%+. Fine-tune centroids via summed gradients.</p>
<p class="rc-num">Key: 64B - 20B = 44B</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l07-linear-quant-sz.png" alt="Linear quantization">
<div class="rc-body">
<strong>6. Linear quantization runs integer math</strong>
<p>r = S(q - Z). Solve S, Z from the ranges. The matmul becomes
qY = (SwSx/Sy)(qW-Zw)(qX-Zx) + Zy: all integer.</p>
<p class="rc-num">Key: scale and zero-point</p>
</div>
</div>
<div class="recap-card">
<img src="assets/slide-l07-distill-logits.png" alt="Distillation">
<div class="rc-body">
<strong>7. Distillation teaches uncertainty</strong>
<p>The student matches the teacher's probability distribution,
not just labels. Align on logits, features, or gradients.</p>
<p class="rc-num">Key: dark knowledge transfers</p>
</div>
</div>
<div class="recap-card">
<img src="assets/media-generation-plate-chapter-l07-memory-0-e99a1302-0e56-463a-917c-27764af8b66a.webp" alt="Chapter plate">
<div class="rc-body">
<strong>8. Shrink three ways, compose them</strong>
<p>Prune structure, quantize values, distill knowledge. Each
trades a little accuracy for a lot of memory. Together they win.</p>
<p class="rc-num">Key: pruning + quantization compose</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Memory-Efficient Neural Networks slide deck (Fall 2023 headers).
- MIT 6.5940 TinyML (Song Han): the lecture's stated inspiration.

**Further reading:**
- Han et al., "Deep Compression" (ICLR 2016): k-means plus pruning plus Huffman.
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2017).
- Dettmers et al., "LLM.int8()" (2022): what breaks at LLM scale and the outlier fix.
- Xiao et al., "SmoothQuant" (2022): the deck's motivating LLM quantization figure.

**Caveats from these sources.** The deck's quantization examples
are small-matrix illustrations; LLM-scale behavior adds outlier
features that break naive schemes (see LLM.int8()). The 2:4
sparsity speedup needs Ampere-or-newer sparse tensor cores and
TensorRT-style kernels; unstructured sparsity rarely speeds up
dense GEMM. Butterfly and Monarch matrices are calendar-listed but
not in the shipped deck: [uncertain].

## Connections to the other courses

- **CS336 L02:** the precision ladder (FP32 to FP4) and what each format costs.
- **CS229S L04:** quantization shrinks the weights that memory-bound decoding re-reads.
- **CS229S L08:** LoRA as an alternative to full fine-tuning memory cost.
- **CS229S L01:** the memory wall: why shrinking matters at all.

> [!CHEAT]
> **Memory efficiency cheatsheet.** Prune: min L(x;Wp) s.t. ||Wp||0 < T. Ideal: faster on HW, keeps accuracy, generalizes. Unstructured: accurate, slow. Structured: fast, coarser. 2:4 N:M: 2 of 4 zero, Ampere sparse tensor cores, 50% weights gone. Iterative prune + fine-tune > one-shot. Magnitude (fine/coarse), regression (min ||Z-Zhat||F^2), sensitivity per layer. Quantization: storage = params x bits; add O(N), multiply O(N^2) in bit-width. FP32: (-1)^s(1+f)2^(e-127), bias 127. Subnormal: e=0; Inf/NaN: e=all-1. FP16: 5+10; BF16: 8+7 (range for training). K-means: store log2(S)-bit idx + S centroids; 4x4/S=4: 64B->20B, 3.2x; compute stays FP32; Huffman +20%. Linear: r=S(q-Z); qY=(SwSx/Sy)(qW-Zw)(qX-Zx)+Zy, integer-only. Distillation: student matches teacher logits/features/gradients; loss E(-pt log ps) or L2.

> [!MEMORY]
> **Shrink in one line.** Remove weights that do not matter, spend fewer bits on the rest, and teach a small model what the big one knows.
