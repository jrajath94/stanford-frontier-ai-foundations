# keys-u07.md: interview bank U07 answers

Date: 2026-10-06. Closed-book reference answers with
rubrics. Keep separate from the questions file.

## Breadth answers

B1. Recurrence: x_t = A x_{t-1} + B u_t, O(T),
sequential. Convolution: y_t = sum k_i u_{t-i},
O(T K), parallel.

B2. 2 T^2 d FLOPs per matmul, T^2 floats of scores.
Doubling T multiplies both by 4.

B3. Output = phi(Q)(phi(K)^T V). State S =
phi(K)^T V, shape (d, d).

B4. H_t = A h_{t-1} + B x_t, y_t = C h_t. A: (N,N)
controls memory decay and stability.

B5. K_k = C A^k B, y = K * x. Collapses when A
varies with the input (no fixed kernel).

B6. (a2,b2) o (a1,b1) = (a2*a1, a2*b1+b2).
Associativity lets the tree reorder safely.

B7. Conv(x,k) = IFFT(FFT(x)*FFT(k)). Direct wins
for short kernels or small n (overhead).

B8. Growth A^n. Keep A stable via exp
discretization of a negative continuous A, or
eigenvalues inside the unit disk.

## Deep ladder 1 answers

L1a. Every query meets every key: the T x T score
matrix.

L1b. Scores: 17.2 TFLOPs. Score memory: 256 MiB
per head fp32.

L1c. (2T)^2 = 4 T^2: both FLOPs and memory scale
with T^2.

L1d. Phi(Q)(phi(K)^T V): O(T d^2) time, O(d^2)
state.

L1e. FLOP ratio 64x (17.2 TFLOPs vs 0.268
GFLOPs). State: 256 MiB vs 64 KiB.

L1f. The kernel is too flat for sharp selection.
First fix: try a sharper phi, or admit the task
needs softmax.

L1g. Assumption: phi approximates the softmax
kernel. Counterexample: exact copy needs one-hot
rows. Elu+1 blurs them.

L1h. Sweep phi in {elu+1, relu, sharp-approx} on
copy. Expect sharper phi to rank higher on copy.

Rubric: must compute the ratio, not quote it. Red
flag: "linear attention is always better." Fix:
the expressivity price.

## Deep ladder 2 answers

L2a. All three compute the linear recurrence
output: streaming, kernel, and tree forms.

L2b. States [1.0, 0.9, 1.81, 1.629]. Kernel [1,
0.9, 0.81, 0.729]. Outputs equal the states
(C=1).

L2c. Y_t = sum_{k<=t} C A^k B x_{t-k} by induction
on the recurrence. The kernel taps are the
impulse response.

L2d. Combine as above. Work O(n), depth O(log n).

L2e. Convolution view for fixed A (FFT-fast).
Scan for input-dependent A (no kernel exists).

L2f. Associativity failed. The tree order then
matters and the total is wrong.

L2g. Assumption: A fixed, so one kernel fits all
positions. Counterexample: selective SSM with
input-dependent A: no single K exists.

L2h. Time conv-view vs loop vs scan across T.
Expect: loop at small T, scan in the middle, FFT
conv-view at large T.

Rubric: must derive K_k, not assert it. Red flag:
confusing the scan with the convolution. Fix:
which assumption each needs.

## Analytical answers

A1. 0.99^1000 = 4.32e-5. 1.01^1000 = 20,959.
Search n from 1000 upward for 1.01^n > 65,504:
first hit at n = 1115 (65,816).

A2. Scores: 134,217,728 bytes. Stream: 2,097,152.
State: 32,768. Ratio 64x. An unfused eager loop
destroys it by materializing every state to HBM.

## Debug answer

D1. Bug: the raw FFT product is the circular
convolution, not the linear one. Fix: zero-pad
both to length 2n before transforming:
np.fft.fft(x, 2n). The theorem needs linear
convolution. Padding makes the circular result
coincide with it.

## Changed-constraint answers

S1. With infinite SRAM the traffic argument dies,
but FLOPs do not: attention still costs T^2 d vs
T d^2, and exact pairwise routing still needs the
matrix. SSM wins on FLOPs at long T. Attention
wins on exact recall. The choice becomes
FLOPs-vs-expressivity, not memory.

S2. Generation is gone, so causality relaxes:
bidirectional kernels and full attention are
legal. The cheapest accurate choice is usually a
bidirectional encoder-style model or chunked
attention, not a causal SSM.

## Research critique answer

R1. Five audits: (1) recall probes included?
Invalidate: only smooth tasks tested. (2) sharp
selection tasks? Invalidate: no copy/passkey
probe. (3) matched FLOPs? Invalidate: the linear
model used more parameters. (4) length
generalization? Invalidate: tested at train
length only. (5) seeds? Invalidate: one run per
model.
