# Answer keys, lesson 06 (VQ-VAE and discrete latents)

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4, float64,
seed 0 where RNG is used. Ground truth: compute_run5a.py.

## E01

Shape (4, 2). Each entry owns its Voronoi region: the
set of z_e nearer to it than to any other entry.

## E02

[0.05, 1.45, 3.65, 2.25]. Winner: index 0, e_1 = [1,
0]. Margin over runner-up: 1.45 - 0.05 = 1.40.

## E03

Recon ||x - decoder(z_q)||^2: moves the decoder (and
the encoder via STE). Codebook ||sg[z_e] - e_k||^2:
moves e_k, freezes z_e. Commitment beta ||z_e -
sg[e_k]||^2: moves z_e, freezes e_k.

## E04

Total: 0.05 + 0.05 + 0.0125 = 0.1125. Smallest: the
commitment term, 0.0125.

## E05

STE grad: 2(z_q - target) = [0, -1]. True grad: [0, 0]
(the argmin is locally constant). The STE substitutes
the gradient the loss would have without quantization.

## E06

Perturbing z_e by 1e-6 leaves the winner unchanged, so
the loss is unchanged: [0, 0] exactly.

## E07

N_new = 0.99 * [10, 2, 0, 1] + 0.01 * [3, 2, 0, 1] =
[9.93, 2.0, 0.0, 1.0].

## E08

The unguarded update divides by N_new = 0 and produces
NaN for the dead code's mean. The guard keeps the
stale vector [-1, 0].

## E09

Dead: code 3 (e_3 = [-1, 0]). Effective K = 3.

## E10

[0.625, 0.25, 0.0, 0.125]. Bits per code: 1.2988
(entropy of the fitted distribution).

## E11

The uniform prior samples the dead code 25 percent of
the time. The decoder never saw e_3 during training,
so those samples are out-of-distribution for it.

## E12

Decoder error 0 (identity decoder), quantization
error 0.05. All 0.05 is quantization.

## E13

K = 1: 0 bits, 0.91625. K = 2: 1 bit, 0.19125. K =
4: 2 bits, 0.06625. First bit buys 4.8x, second bit
2.9x.

## E14

Nominal rate: log2(8) = 3 bits. Effective rate:
log2(3) = 1.585 bits. The stakeholder sees the
effective rate.

## E15

No. Without a fitted prior over indices it is a
compressor, not a generative model.

## E16

4 distinct outputs max. 3 reachable (code 3 dead).

## E17

d^2 to e_1: 0.58. To e_2: 0.58. Exact tie on the
bisector. Argmin picks the first (e_1), arbitrarily.

## E18

The encoder drifts away from the codes and the
quantization error grows. The STE copy becomes a
worse approximation.

## E19

K sets whole bits per code: hard, countable, exact.
Beta prices KL continuously: soft, smooth, with no
exact bit count. VQ gives the bill. VAE gives the
dial.

## E20

Varied: codebook learning rule (EMA vs gradient).
Fixed: seed, data, encoder/decoder init and
architecture, K, beta, steps. Metrics: codebook
entry displacement per step (stability) and final
distortion.

## L01

Vector quantization: replace each input with its
nearest codebook entry. Toy: distances [0.05, 1.45,
3.65, 2.25], winner e_1. Derivation: the argmin is
piecewise constant, so d z_q / d z_e = 0 almost
everywhere and undefined at boundaries. Implement:
quantize as in the lesson. Compare: rounding is VQ
with a fixed grid. Learned VQ adapts cell geometry.
Debug: flicker means z_e sits on a bisector. The fix
is a tie rule plus checking encoder stability.
Critique: nearest-neighbor is optimal only for the
squared-error distortion. Other distortions want
other rules. Design: run the cosine-distance variant
and compare winners.

## L02

STE: forward the quantized value, backward copy the
gradient as if quantization were identity. Toy: [0,
-1] vs [0, 0]. Derivation: z_e + sg[z_q - z_e]
forwards to z_q and backprops to z_e. Implement:
ste_quantize as in the lesson. Compare: STE is
biased, zero extra cost. Gumbel-softmax is biased
with temperature control. REINFORCE is unbiased
with high variance. Debug: rising quantization
error means the copied gradient optimizes a fantasy. Shrink the encoder-code gap (raise beta) or revive
codes. Critique: acceptable when the quantization
gap stays small and ablations confirm it. Not
acceptable as a claimed unbiased estimator.
Design: measure the angle between STE and
finite-difference gradients versus ||z_e - z_q||.

## L03

EMA: the codebook tracks running means of assigned
encodings with horizon 1/(1-gamma). Toy: counts to
[9.93, 2.0, 0.0, 1.0]. Derivation: N and M updates
in C05. E_k = M_k / N_k. Implement: ema_update as in
the lesson. Compare: EMA needs no learning rate and
is stable. Gradient codebook loss is simpler code but
couples to the optimizer. Debug: NaN means a dead
code divided by zero. Add the guard. Critique: never
reviving dead codes is a flaw for capacity use but a
feature for stability (no thrash). Design: restart
dead codes at live encodings plus noise and measure
usage after 100 batches.

## L04

Rate/distortion: bits per code (log2 K) versus mean
squared quantization error. Toy: (0, 0.91625), (1,
0.19125), (2, 0.06625). Derivation: nested
codebooks can only reassign points to nearer codes,
so distortion never rises with K. Implement:
rate_distortion as in the lesson. Compare: VQ is
hard bits, VAE-beta is a soft dial. Debug: nominal
8 vs effective 3 means dead codes. Report effective
K. Critique: squared error need not match the
application's perceptual or business metric. The
curve is honest only for the metric it measures.
Design: compute the elbow K from second
differences at K = 1, 2, 4, 8.

## L05

Two-stage story: stage 1 learns the discrete
bottleneck. Stage 2 learns a prior over the codes. Generation samples the prior and decodes. Toy prior:
[0.625, 0.25, 0.0, 0.125], 1.2988 bits vs 2.0
uniform. Derivation: entropy of the fitted unigram
vs log2 K. Implement: fit_unigram_prior as in the
lesson. Compare: uniform is simple and wrong. Unigram is matched. Autoregressive captures spatial
structure at a second model's cost. Debug: bad
samples with a good autoencoder means the prior is
mismatched. Refit it on final indices. Critique:
the prior often dominates sample quality more than
the autoencoder does. Design: the uniform-vs-fitted
sample comparison from mechanism C shell 9.

## T1

Missing: the straight-through idiom. Corrected line:
`return ze + (zq - ze), k` with the difference under
stop-gradient in an autograd framework (in plain
numpy, document that the backward copy is the
contract). Even without the fix, the commitment loss
still trains the encoder (its gradient w.r.t. z_e is
exact: 2 beta (z_e - e_k)).

## S1

The EMA mean of assigned encodings is not unit norm,
so the update breaks the sphere constraint.
Projection: renormalize e_k <- e_k / ||e_k|| after
each update. The distance table changes because all
entries keep unit norm: distances become 2 - 2 cos
(angle), i.e. cosine geometry.

## S2

Mitigations: (1) product quantization: split D into
sub-vectors with small sub-codebooks. Approximation:
independent sub-code choice. (2) Approximate nearest
neighbor (e.g. IVF index). Approximation: occasional
wrong winner. Both trade exactness for speed.

## R1

The toy refutes it: the forward pass is exact but
the backward gradient [0, -1] differs from the true
[0, 0]. The bias is exactly this gap. Quantify it by
the angle or norm difference between the STE gradient
and the finite-difference gradient of the true
quantized loss. The bias is small when ||z_e - z_q||
is small, i.e. when the commitment loss keeps the
encoder near its codes.
