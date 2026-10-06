# Answer keys, interview bank U04

Date: 2026-10-06. Ground truth: compute_run4.py.
Interview provenance: role-derived practice, not employer
material. Format per answer: strong answer, red flags,
rubric, remediation.

## Breadth

B1. A coupling is a joint distribution with
marginals p and q: rows sum to p, columns to q.
Different plans have different costs (1.0 vs
2.0 on the toy), so the distance must take the
minimum to be a single number. Strong answer: the
definition plus why the minimum matters, with the toy
numbers. Red flags: "any coupling gives the distance".
Rubric: 2/2 definition plus the min argument. 1/2
definition only. Remediation: U04-C01, C02.
B2. W1 = max_{L(f) <= 1} E_p[f] - E_q[f].
Assumptions: metric ground cost, f
1-Lipschitz everywhere, max over all such f.
Breaking Lipschitz first gives overestimates:
f(x) = -2x reports 2.0 on a true 1.0. Strong answer:
the dual, the three assumptions, and the overestimate
example. Red flags: dropping the Lipschitz bound.
Rubric: 2/2 dual plus assumptions. 1/2 dual only.
Remediation: U04-C02, C03.
B3. L(f) <= product of top singular values
times 1 per ReLU. Toy: 2.2361 * 1.1180 = 2.5.
Not an equality: the true constant is the max
attained slope (1.0 here), the product only
bounds it. Strong answer: the bound, the toy product,
and why it is not an equality. Red flags: "the product
is the Lipschitz constant". Rubric: 2/2 bound plus the
inequality warning. 1/2 the product only. Remediation:
U04-C04.
B4. Gap = 0.01 * 1 = 0.01, a 100x bias. Cost
one: 90.9 percent of weights pile on the clip
boundary, collapsing the function class. Cost
two: deep stacks shrink activations and
gradients by c^depth (1e-10 at 5 layers). Strong answer: the gap number plus both costs. Red flags:
"clipping is harmless". Rubric: 2/2 number plus both
costs. 1/2 number only. Remediation: U04-C05, C07.
B5. lambda * E[(||grad f(xhat)|| - 1)^2],
xhat on real-fake segments. At 0.3, lambda =
10: two-sided 4.9, one-sided 0.0. Blind spot:
segments only. A spike at x = 5 carries
gradient 26.7 with penalty 5.1e-22. Strong answer:
the formula, the two penalty values, and the blind
spot with its numbers. Red flags: "the penalty sees
everything". Rubric: 2/2 formula plus blind spot. 1/2
formula only. Remediation: U04-C06.
B6. Ratio = gap / W_exact on a reference pair.
0.5: the critic lags, the loss is a story.
1.4: impossible under the bound. The critic
breaks Lipschitz. Strong answer: the ratio definition
and the reading of both values. Red flags: "1.4 means
a great critic". Rubric: 2/2 definition plus both
readings. 1/2 definition only. Remediation: U04-C07,
C12.

## D1

D1.1. W1 is the cheapest cost of moving one
distribution's mass onto the other's. Strong answer:
the transport intuition in one sentence. Red flags:
"it is a pointwise distance". Rubric: 2/2 with the
cost-minimum idea. 1/2 vague. Remediation: U04-C01,
C02.
D1.2. Cost matrix [[1,3],[1,1]]. Diagonal
0.5*1+0.5*1 = 1.0. Crossed 0.5*3+0.5*1 = 2.0. Strong answer: the matrix plus both plan costs. Red flags:
picking the crossed plan. Rubric: 2/2 matrix plus both
costs. 1/2 one cost. Remediation: U04-C01.
D1.3. Only coupling delta_{(0,theta)}. Cost
|theta|. Strong answer: the coupling and its cost. Red flags: "cost is theta squared". Rubric: 2/2 both. 1/2
one. Remediation: U04-C01, C02.
D1.4. Likeliest cause: the crossed coupling.
Fix: take the min over legal plans (or check
the plan is the diagonal one). Strong answer: cause plus
fix. Red flags: "the cost matrix is wrong". Rubric: 2/2
cause plus fix. 1/2 cause only. Remediation: U04-C01.
D1.5. W1 = 0 for every pair. The dual sup is
0. The distance is vacuous. Strong answer: the zero
distance and the zero sup. Red flags: "the sup is still
positive". Rubric: 2/2 both facts. 1/2 one. Remediation:
U04-C02, C03.

## D2

D2.1. The Lipschitz constant is the maximum
slope: |f(a)-f(b)| <= L|a-b|. Strong answer: the
inequality with the slope reading. Red flags: "it is
the average slope". Rubric: 2/2 inequality plus reading.
1/2 inequality only. Remediation: U04-C04.
D2.2. Two-sided: 10*(0.3-1)^2 = 4.9 and
10*(2.5-1)^2 = 22.5. One-sided: 0.0 and
22.5. Strong answer: all four numbers. Red flags:
penalizing the one-sided 0.3 case. Rubric: 2/2 all four.
1/2 two. Remediation: U04-C06.
D2.3. P(|w| > 0.01) = 1 - 0.02/0.2 = 0.9. Strong answer: the number with the arithmetic. Red flags:
"most weights stay inside". Rubric: 2/2 number plus
arithmetic. 1/2 number only. Remediation: U04-C05.
D2.4. The spike sits off the interpolation
segments. The penalty never samples it.
Diagnose by evaluating the penalty near the
spike. Strong answer: the location argument plus the
diagnostic. Red flags: "raise lambda". Rubric: 2/2
location plus diagnostic. 1/2 location only.
Remediation: U04-C06.
D2.5. The guarantee omits the scale: the
bound is c-Lipschitz, so the gap is
c-scaled, and it omits capacity: the box
starves the function class. Exposed by the
calibration ratio (0.01 on a true 1.0). Strong answer:
both omissions plus the exposing ratio. Red flags:
"clipping gives the true W1". Rubric: 2/2 both omissions
plus ratio. 1/2 one. Remediation: U04-C05, C11.

## Analytical/quantitative

Q1. Ratio = 0.02 / 2.0 = 0.01. Training
minimizes the critic's gap. You want the true
distance. The 100x miss is invisible without
the ratio. Strong answer: the ratio plus why training
hides the miss. Red flags: "a small gap means a good
distance". Rubric: 2/2 ratio plus the training point.
1/2 ratio only. Remediation: U04-C07, C12.
Q2. Scale factor 0.01^5 = 1e-10. The
generator's gradient flows through the
critic's activations, so it shrinks by the
same product: the signal vanishes. Strong answer: the
factor plus the flow-through argument. Red flags: "only
the critic suffers". Rubric: 2/2 factor plus argument.
1/2 factor only. Remediation: U04-C05, C07.

## Implementation/debug

T1. Bug one: the sign is flipped for a loss to
minimize. The critic loss should be
-(f(xp).mean() - f(xq).mean()), i.e. minimize
E_q - E_p. Bug two: no Lipschitz enforcement,
so f grows unbounded and 3.0 > W1 = 1.0 is
not a distance. Rules: the loss sign must
match max-versus-min roles. The dual gap is a
distance only under the bound. Strong answer: both bugs
plus both rules. Red flags: fixing only the sign. Rubric:
2/2 both bugs and rules. 1/2 one bug. Remediation:
U04-C02, C04.

## Changed-constraint scenarios

S1. The max gap over 2-Lipschitz f equals 2 *
W1: toy gaps double to 2.0, 4.0, 0.02. The
penalty target becomes (||grad|| - 2)^2. Strong answer:
the scaled gaps plus the new penalty target. Red flags:
"the penalty target stays 1". Rubric: 2/2 gaps plus
target. 1/2 gaps only. Remediation: U04-C06.
S2. The reversal then erases task signal along
with domain signal: features become invariant
to a label the task needs. Smallest repair:
condition the domain classifier on the task
label (per-class domain confusion), or drop
lambda where the correlation is task load
bearing. Strong answer: the failure plus the smallest
repair. Red flags: "more reversal helps". Rubric: 2/2
failure plus repair. 1/2 failure only. Remediation:
U04-C09, C11.

## Research-critique

R1. One: clipping at c = 0.01 scales the gap,
so 0.001 is a scaled artifact (C05). Two:
n_critic = 1 leaves the critic stale, so the
gap is a loose lower bound (C07, C08). Three:
one seed cannot separate method from luck.
same-seed multi-seed reruns are required
(C12). Four: the gap is not a sample-quality
certificate. Coverage diagnostics can fail
while the gap passes (C11). Strong answer: all four
attacks with their concept refs. Red flags: accepting
the gap as a certificate. Rubric: 2/2 four attacks. 1/2
two. Remediation: U04-C05, C07, C11, C12.
