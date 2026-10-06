# Answer keys, interview bank U06

Date: 2026-10-06. Ground truth: compute_run5a.py.
Interview provenance: role-derived practice, not employer
material. Format per answer: strong answer, red flags,
rubric, remediation.

## B1

Replace each input with its nearest codebook
entry. Toy: z_q = e_1 = [1, 0], d^2 = 0.05. Strong answer: the operation plus the toy values. Red flags:
"quantization averages the codebook". Rubric: 2/2
operation plus values. 1/2 operation only. Remediation:
U06-C01, C02.

## B2

Recon ||x - decoder(z_q)||^2 moves the decoder
(and encoder via STE). Codebook ||sg[z_e] -
e_k||^2 moves e_k. Commitment beta ||z_e -
sg[e_k]||^2 moves z_e. Strong answer: all three losses
with what each moves. Red flags: "one loss trains
everything". Rubric: 2/2 three losses plus targets. 1/2
two. Remediation: U06-C03.

## B3

Forward the quantized value. Backward copy the
gradient as if quantization were identity. Toy:
STE [0, -1], true [0, 0]. Strong answer: the forward
rule, the backward rule, and the toy pair. Red flags:
"STE equals the true gradient". Rubric: 2/2 rules plus
toy pair. 1/2 rules only. Remediation: U06-C04.

## B4

0.99*[10,2,0,1] + 0.01*[3,2,0,1] = [9.93, 2.0,
0.0, 1.0]. Strong answer: the EMA arithmetic with the
result. Red flags: dropping the 0.01 term. Rubric: 2/2
arithmetic plus result. 1/2 result only. Remediation:
U06-C05.

## B5

A dead code is never selected (usage 0). Toy:
code 3 (e_3 = [-1, 0]). Effective K = 3. Strong answer:
the definition, the toy code, and the effective K. Red flags: "nominal K is what counts". Rubric: 2/2 definition
plus effective K. 1/2 definition only. Remediation:
U06-C06.

## B6

K = 1: 0 bits, 0.91625. K = 2: 1 bit, 0.19125.
K = 4: 2 bits, 0.06625. Strong answer: all six numbers.
Red flags: "more codes always help a lot". Rubric: 2/2
all numbers. 1/2 partial. Remediation: U06-C09.

## D1

D1.1. The argmin is piecewise constant, so its
gradient is 0 almost everywhere and undefined at
boundaries. Strong answer: the constancy plus where the
gradient dies. Red flags: "the gradient is small but
usable". Rubric: 2/2 constancy plus the zero. 1/2 one.
Remediation: U06-C02, C04.
D1.2. STE: 2(z_q - target) = [0, -1]. True:
[0, 0]. Strong answer: both gradients. Red flags:
"they match". Rubric: 2/2 both. 1/2 one. Remediation:
U06-C04.
D1.3. Forward: z_e + (z_q - z_e) = z_q. Backward:
sg kills the correction term's gradient, so
dL/dz_e = dL/dz_q. Strong answer: the forward identity
plus the backward copy. Red flags: "sg is optional".
Rubric: 2/2 both directions. 1/2 one. Remediation:
U06-C04.
D1.4. Missing: the straight-through wrapper.
Fixed line: `return ze + (zq - ze), k` with the
difference under stop-gradient in autograd. Strong answer: the missing idiom plus the fixed line. Red flags:
"detach zq instead". Rubric: 2/2 idiom plus line. 1/2
idiom only. Remediation: U06-C04.
D1.5. The EMA mean of assigned encodings is not
unit norm. Fix: renormalize e_k after each
update. Strong answer: the cause plus the fix. Red flags:
"scale the learning rate". Rubric: 2/2 cause plus fix.
1/2 cause only. Remediation: U06-C05.

## D2

D2.1. Codebook health: every entry wins a
nonzero share of assignments and the
quantization error stays small. Strong answer: both
health criteria. Red flags: "low error alone is enough".
Rubric: 2/2 both criteria. 1/2 one. Remediation: U06-C01,
C06.
D2.2. Dead: code 3. A uniform prior samples it
25 percent of the time although the decoder never
saw it: out-of-distribution samples. Strong answer: the
dead code plus the sampling consequence. Red flags: "the
prior never picks dead codes". Rubric: 2/2 code plus
consequence. 1/2 code only. Remediation: U06-C06, C07.
D2.3. N_k and M_k are exponential moving
averages of counts and assigned-encoding sums. E_k = M_k / N_k. The guard keeps the stale vector
when N_k = 0 instead of dividing by zero. Strong answer:
the EMA definitions plus the guard. Red flags: "divide
anyway". Rubric: 2/2 definitions plus guard. 1/2
definitions only. Remediation: U06-C05.
D2.4. The guard is absent: a dead code's mean
divided by N_k = 0. Add the N_k > 0 guard. Strong answer: the failure plus the guard. Red flags: "add
epsilon to the denominator". Rubric: 2/2 failure plus
guard. 1/2 failure only. Remediation: U06-C05.
D2.5. Nominal K = 8 claims 3 bits. Effective K =
3 gives 1.585 bits. Dead codes are paid capacity
with no distortion gain. The curve should use
effective K. Strong answer: both bit counts plus the
paid-capacity point. Red flags: "plot nominal K". Rubric:
2/2 counts plus point. 1/2 counts only. Remediation:
U06-C06, C09.

## Q1

Nominal: log2(8) = 3 bits. Effective: log2(3) =
1.585 bits. Use effective K. Strong answer: both numbers
plus the rule. Red flags: "3 bits is the rate". Rubric:
2/2 numbers plus rule. 1/2 numbers only. Remediation:
U06-C06, C09.

## Q2

Saving: 2.0 - 1.2988 = 0.7012 bits per code. The
skew buys cheaper transmission: frequent codes
cost fewer bits under an optimal code. Strong answer:
the saving plus the skew mechanism. Red flags: "uniform
codes are optimal". Rubric: 2/2 saving plus mechanism.
1/2 saving only. Remediation: U06-C07, C09.

## T1

Missing: the straight-through idiom. Corrected:
`return ze + (zq - ze), k` with stop-gradient on
the difference in an autograd framework. The
commitment loss still trains the encoder (its
gradient 2 beta (z_e - e_k) is exact). Strong answer:
the idiom, the corrected line, and why the encoder still
learns. Red flags: "the encoder gets no gradient". Rubric:
2/2 line plus reason. 1/2 line only. Remediation:
U06-C03, C04.

## S1

(1) Product quantization: split D into
sub-vectors with small sub-codebooks. Approximation: independent sub-code choice. (2)
Approximate nearest neighbor index. Approximation: occasional wrong winner. Strong answer: both schemes with their approximations. Red flags: "no approximation involved". Rubric: 2/2 both
schemes plus approximations. 1/2 one. Remediation:
U06-C02, C10.

## S2

At most K distinct outputs (4 on the toy, 3
reachable). Fine detail between codes is lost. The linear decoder can only memorize per-code
outputs. Strong answer: the output cap plus the lost
detail. Red flags: "the decoder interpolates". Rubric:
2/2 cap plus detail. 1/2 cap only. Remediation: U06-C10,
C11.

## R1

Toy: [0,-1] vs [0,0]. The forward pass being
exact says nothing about the backward gradient.
Quantify by the angle or norm between the STE
gradient and the finite-difference gradient of
the true quantized loss. Small when ||z_e - z_q||
is small. Strong answer: the toy pair, the forward-vs-
backward point, and the quantification. Red flags: "exact
forward means exact training". Rubric: 2/2 pair plus
quantification. 1/2 pair only. Remediation: U06-C04, C12.
