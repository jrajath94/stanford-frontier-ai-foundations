# Interview bank, U06 discrete latent modelling and VQ-VAE

Date: 2026-10-06. Questions only. Keys in
interview/keys-u06.md. Closed-book. Do not read the keys
first. The running toy: codebook E = [[1,0],[0,1],
[-1,0],[0,-1]], z_e = [0.9, 0.2], identity decoder,
beta = 0.25.

## Breadth (6)

B1. Define vector quantization. State the
quantization rule on the toy.
B2. Write the three VQ losses. State what each
one moves.
B3. Define the straight-through estimator. Give
the toy's STE gradient and true gradient for
||z_q - target||^2, target = [1, 0.5].
B4. Run one EMA update by hand: counts [10, 2, 0,
1], batch assigns [3, 2, 0, 1], gamma = 0.99.
B5. Define a dead code. Name the dead code on the
toy 8-encoding batch and the effective K.
B6. Fill the rate/distortion table for K = 1, 2,
4 on the toy encodings.

## Deep ladder D1, learning through the argmin (5 follow-ups)

D1.1. Define the argmin's gradient problem in one
sentence.
D1.2. Toy: give the STE gradient and the true
gradient. Show the numbers.
D1.3. Derive the z_e + sg[z_q - z_e] idiom: what
the forward computes and where the backward
gradient goes.
D1.4. Implement/debug: encoder weights never
change during training. Name the missing piece
and the one-line fix.
D1.5. Changed constraint: the codebook must live
on the unit sphere. Name what breaks in EMA and
the fix.

## Deep ladder D2, codebook health (5 follow-ups)

D2.1. Define codebook health in one sentence.
D2.2. Toy: usage [4, 3, 0, 1]. Name the dead code
and state the consequence for a uniform prior.
D2.3. Derive the EMA update as a running mean.
Explain the dead-code guard.
D2.4. Implement/debug: NaN appears in the
codebook after an update. Diagnose exactly.
D2.5. Research critique: "More codes are always
better." Attack with nominal vs effective K.

## Analytical/quantitative (2)

Q1. K = 8 with 5 dead codes. Compute the nominal
rate, the effective rate, and state which one the
rate/distortion curve should use.
Q2. The fitted prior on [0,0,1,0,3,0,1,0] gives
1.2988 bits per code vs 2.0 uniform. Compute the
saving per code and explain what the skew buys.

## Implementation/debug (1)

T1. This code intends the VQ forward pass:

```python
import numpy as np
def vq_forward(ze, E):
    d2 = ((E - ze) ** 2).sum(1)
    k = int(np.argmin(d2))
    zq = E[k]
    return zq, k
```

It trains, but the encoder weights never change.
Name the missing piece (one idiom), write the
corrected return line, and state which loss term
still trains the encoder even without the fix.

## Changed-constraint scenarios (2)

S1. K = 65536, D = 256. The O(K D) scan per
vector is too slow. Name two mitigations and the
approximation each introduces.
S2. The decoder must be linear (no nonlinearity).
State the maximum number of distinct outputs and
what this implies for reconstruction detail.

## Research-critique (1)

R1. "Straight-through estimation is unbiased
because the forward pass is exact." Attack with
the toy gradients. Name the measurement that
quantifies the bias and the condition under which
the bias is small.
