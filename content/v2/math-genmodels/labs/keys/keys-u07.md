# Lab answers: U07

Unit: math-genmodels-U07. Date: 2026-10-06. Baseline: October 6, 2026.

## E1

Joint sums to 1.000000 over the 27 sequences. p(abc) = 0.05,
log p = -2.9957, NLL = 2.9957, perplexity = 2.7144.

## E2

Row 2 weights (0.3302, 0.6698), output (3.3395, 4.3395)
confirmed. With Q scaled by 10 and no 1/sqrt(2): scores become
(0, 7.071), weights collapse to about (0.0008, 0.9992): the
head attends almost entirely to position 2. Lesson: without
the scale, dot products grow with dimension, the softmax
saturates at init, and gradients vanish before training
starts.

## E3

T=0.1: (0.0000, 0.0000, 1.0000) to 4 decimals. T=0.5: (0.0159, 0.1173,
0.8668). T=1: (0.0900, 0.2447, 0.6652). T=2: (0.1863, 0.3072,
0.5065). T=10: (0.3006, 0.3322, 0.3672). Seed-12 sample
of 2,000 at T=2 gives (0.182, 0.304, 0.514), matching within
noise.

## E4

Mults per head: L=128: 1.0M. 512: 16.8M. 2048: 268M. 8192:
4.3B. KV bytes per layer: 64KB, 256KB, 1MB, 4MB. Derivation:
2 x L x d x 4 bytes: L = 128, d = 64 gives 2 x 128 x 64 x 4 =
65,536 bytes = 64KB. Quadratic confirmed: 4x L gives 16x mults.
The scores matrix (fp32) exceeds 1 GB at L = 16384
(16384^2 x 4 bytes = 1.07 GB).

## E5

x_t = 0.5, target velocity 2.0. Losses: 0.0, 0.04, 1.0.
Endpoints: t=0 gives 0.0 = x_0, t=1 gives 2.0 = x_1.

## E6

Pseudo-likelihood sums to 1.0639 over the 27 sequences, not
1.0. PL(abc) = 0.0352 versus the true 0.05. This proves the
masked conditionals do not define a joint law: their product
is unnormalized, so pseudo-perplexity is not a likelihood and
cannot be compared with AR NLL.
