# keys.md, U07 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

Recurrence: x_t = A x_{t-1} + B u_t, O(T) time, O(1)
memory, sequential. Convolution: y_t = sum k_i
u_{t-i}, O(T K) time, parallel over T.

## E02

Recurrence states: [1.0, 0.9, 1.81, 1.629].
Convolution outputs: [1, 0.5, 1.25, 0.5].

## E03

The kernel window is shorter than the dependency
distance. Position 1000 never enters any output.

## E04

K = T costs O(T^2) and gains the full receptive
field. It becomes as expensive as attention with
less flexibility.

## E05

Hypothesis: accuracy is flat until K reaches the
dependency distance, then jumps. Report the curve.

## E06

FLOPs per head: 2 T^2 d (scores) + 2 T^2 d
(values). Score memory: T^2 floats.

## E07

34.4 TFLOPs per head with values. Score memory:
67.1M * 4 bytes = 256 MiB per head.

## E08

T = 32768: T^2 = 1.07e9 floats = 4.29 GB per head.
Eight heads: 34.4 GB of scores.

## E09

FLOPs double (linear in d). Score memory does not
move (no d in T^2).

## E10

Hypothesis: wall time grows 4x per T doubling.
Report the ratios.

## E11

Output = phi(Q) (phi(K)^T V). State S = phi(K)^T V
has shape (d, d).

## E12

Scores: 17.2 TFLOPs. Linear: 0.268 GFLOPs. Ratio:
64x.

## E13

The feature map is too flat for sharp selection.
Exact copy needs near-one-hot weights. Elu+1
blurs them.

## E14

No. linear costs T d^2, quadratic costs T^2 d.
With d > T, linear is more expensive. The
boundary is d = T.

## E15

Hypothesis: sharper phi wins on copy, flatter phi
wins on smooth tasks. Report both.

## E16

h_t = A h_{t-1} + B x_t, y_t = C h_t. A: (N,N),
B: (N,1), C: (1,N), h: (N,).

## E17

h = [1.0, 0.9, 1.81, 1.629]. Pulse x=[1,0,0,0]:
h_3 = 0.9^3 = 0.729.

## E18

Fixed A cannot forget on command. The old
document's state leaks into the new document.

## E19

A = 1 never forgets. With constant input the state
grows without bound: a pure integrator.

## E20

Hypothesis: leakage decays as 0.9^gap between
documents. Report the curve.

## E21

K_k = C A^k B. Y = K * x equals the recurrence:
same computation, streaming vs parallel form.

## E22

K = [1, 0.9, 0.81, 0.729]. Both ways: y = [1.0,
0.9, 1.81, 1.629].

## E23

The scan survives. The convolution view needs a
fixed kernel.

## E24

K becomes time-varying: no single kernel exists.
The convolution view collapses.

## E25

Hypothesis: conv-view beats the loop past the T
where FFT overhead pays off. Report the
crossover.

## E26

(a2,b2) o (a1,b1) = (a2*a1, a2*b1 + b2).
Associative: affine-map composition.

## E27

Final state 1.9333 both ways (tree total equals
sequential final state).

## E28

Associativity failed. A non-associative combine
makes the tree order matter.

## E29

Pad to a power of 2 or carry the odd element.
Depth becomes ceil(log2 n).

## E30

Hypothesis: the scan wins in a middle T band:
loop too serial below, FFT overhead too high
above. Report the band.

## E31

conv(x, k) = IFFT(FFT(x) * FFT(k)). Convolution
in time is multiplication in frequency.

## E32

Naive: 1,048,576. FFT path: 3*10240 + 1024 =
31,744. Ratio: 33.0x.

## E33

FFT overhead (bit-reversal, twiddles, three
passes) dominates at small n. Direct wins below
the crossover.

## E34

Direct convolution wins. Three taps cost 3n vs
the FFT's n log n machinery.

## E35

Hypothesis: direct beats FFT below n = 128-512
on this CPU. Report the measured crossover.

## E36

Growth: A^n. Discretization: A_d = exp(step *
A_c). Stable needs |A| < 1 (scalar).

## E37

0.99^1000 = 4.32e-5. 1.01^1000 = 20,959. Fp16
death step: 1115.

## E38

A drifted past 1 during training. Short evals
hide it. Long evals explode.

## E39

A_d = e^0.1 = 1.105: unstable by construction.

## E40

Hypothesis: free A explodes at 10x train length,
exp-parameterized A holds. Report both.

## E41

Centered: y_t reads x_{t+1} (future). Causal: y_t
reads only x_t, x_{t-1}, ....

## E42

Centered y_1 = 4.0, leaking x_2 = 3. Causal y_1 =
2.5, clean.

## E43

The future leaked into training (centered kernel
or wrong-side padding). Eval cheats. Generation
cannot.

## E44

No. fill-in-the-blank sees the full sequence at
inference, so bidirectional is legal.

## E45

Hypothesis: centered trains to lower loss but
generates garbage. Report both.

## E46

Exact copy needs T distinguishable slots.
Softmax has T^2. A linear state has d^2 numbers.

## E47

Need: 8192*128 = 1,048,576. Have: 128^2 =
16,384. Shortfall: 64x.

## E48

The linear kernel blurs. Sharp retrieval needs
near-one-hot selection, which the smooth state
cannot hold.

## E49

No. d = 1024 gives 1,048,576 state numbers =
need. The bound stops biting at T = 8192.

## E50

Hypothesis: linear accuracy falls with passkey
depth, softmax holds. Report the curves.

## E51

Attention: T^2 * bytes per float of score
traffic. SSM: T*d stream bytes + N*d state
bytes resident.

## E52

134,217,728 bytes scores. 2,097,152 stream. 
32,768 state. Ratio: 64x.

## E53

The recurrence ran as an unfused eager loop,
materializing every state to HBM.

## E54

The fused state spills to HBM. Fusion breaks
first. Traffic climbs back toward the loop
cost.

## E55

Hypothesis: the fused kernel moves ~64x fewer
score-equivalent bytes than the loop. Report
both.

## E56

| model | train FLOPs | state | range |
| transformer | 34.4 T | O(T d) | full |
| SSM | 0.268 G | 64 KiB | full (faded) |
| CNN K=512 | 1.07 T | O(K d) | 512 |

## E57

34,359,738,368. 268,435,456. 1,073,741,824.
Transformer/SSM ratio: 128x.

## E58

Expected. Range 512 < distance 5000: the kernel
never reaches the passkey.

## E59

The transformer wins. At T = 512 the T^2 cost is
affordable and exactness is free.

## E60

Hypothesis: measured time and recall-probe ratios
match the table within 2x. Report all.
