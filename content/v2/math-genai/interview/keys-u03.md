# keys-u03.md, U03 interview answer keys

Date: 2026-10-06. Closed-book answers for the U03 bank.
Keep separate from the questions. All numbers computed
2026-10-06, numpy 1.26.4, float64 (compute_run3.py).
Interview provenance: role-derived practice, not employer
material. Format per answer: strong answer, red flags,
rubric, remediation.

## Breadth

B1. V(D,G) = E_{p_data}[log2 D(x)] + E_z[log2(1 -
D(G(z)))]. D maximizes, G minimizes. Strong answer:
states the full value function and names which player
maximizes and which minimizes. Red flags: swapping the
roles, dropping the log base. Rubric: 2/2 with both roles
right. 1/2 formula only. Remediation: U03-C01, C02.
B2. D*(x) = p_data(x)/(p_data(x)+p_g(x)). Pointwise max
of a log y + b log(1-y) gives y = a/(a+b). Strong answer: gives the ratio and the pointwise maximizer
argument. Red flags: "D* = 1 where p_data wins". Rubric:
2/2 ratio plus argument. 1/2 ratio only. Remediation:
U03-C03.
B3. max_D V = 2 JS(p_data || p_g) - 2. Assumptions: D
ranges over all functions, densities exist on the same
space, the inner max is attained. Strong answer: the
identity plus all three assumptions. Red flags: claiming
the identity holds for a finite net. Rubric: 2/2 identity
plus assumptions. 1/2 identity only. Remediation: U03-C04.
B4. Minimax: -0.001 (dead). Non-saturating: -0.999
(alive). Fixes saturation. Does not fix log-boundary
explosion or misleading signal from a too-strong critic.
Strong answer: both numbers plus what the fix does and
does not fix. Red flags: "non-saturating fixes training".
Rubric: 2/2 numbers plus the limits. 1/2 numbers only.
Remediation: U03-C05, C07.
B5. Mode collapse: the generator assigns ~0 mass to a
data mode. Diagnoses: left-mode mass 0.0232 versus
0.4952. Hides: JS = 0.2809 bits looks modest. Sample
quality looks fine. Strong answer: the definition, the
mass diagnostic, and why JS and samples hide it. Red flags: "low JS means no collapse". Rubric: 2/2 with the
mass numbers. 1/2 definition only. Remediation: U03-C06,
C12.
B6. Saturation -> non-saturating loss. Explosion ->
eps clip on D. Misleading signal -> update balance
(k, TTUR). Strong answer: all three pathology-fix pairs.
Red flags: one fix for all three. Rubric: 2/2 all three
pairs. 1/2 two. Remediation: U03-C05, C07, C11.

## D1

D1.1. Two players fight over one number: the
discriminator maximizes it by calling out fakes, the
generator minimizes it by making better fakes. Strong answer: the minimax roles in one clean sentence. Red flags: "the generator maximizes". Rubric: 2/2 both roles
right. 1/2 one. Remediation: U03-C01, C02.
D1.2. a = gauss(1, 0,1) = 0.2420, b = gauss(1, 1,0.5)
= 0.7979. D* = 0.2420/1.0399 = 0.2327. Strong answer:
all three numbers with the ratio. Red flags: adding the
densities wrong. Rubric: 2/2 all numbers. 1/2 partial.
Remediation: U03-C03.
D1.3. The four-line derivation is E07 of the lesson
keys: substitute D*, write m = (pd+pg)/2, collect KL
terms, read off 2 JS - 2. Strong answer: the four steps
in order. Red flags: skipping the m substitution. Rubric:
2/2 four steps. 1/2 two. Remediation: U03-C04.
D1.4. The clip is absent: 1 - D* underflows to 0 at
the grid tails and log2(0) = -inf. Fix: clip D* to
[1e-15, 1-1e-15] before the logs. Strong answer: the
mechanism (underflow) plus the exact clip. Red flags:
"add 1e-15 inside the log only". Rubric: 2/2 mechanism
plus fix. 1/2 fix only. Remediation: U03-C07.
D1.5. JS = 1 bit (maximum), gradient exactly 0 in the
generator parameters. The game gives G no direction:
the value is flat. This is the honest reason for U04.
Strong answer: the number, the zero gradient, and the
link to U04. Red flags: "JS = 1 means G learns fast".
Rubric: 2/2 number plus the flat-value reading. 1/2
number only. Remediation: U03-C04, C07.

## D2

D2.1. Saturation: the minimax generator gradient -D
vanishes exactly where D is confident and G most needs
signal. Strong answer: where it vanishes and why that
location hurts most. Red flags: "the gradient is small
everywhere". Rubric: 2/2 location plus reason. 1/2
location only. Remediation: U03-C05, C07.
D2.2. d/dt log(1-sig(t)) = -sig(t) = -0.001. D/dt
-log sig(t) = -(1-sig(t)) = -0.999. Strong answer: both
derivatives with values. Red flags: sign errors. Rubric:
2/2 both right. 1/2 one. Remediation: U03-C05.
D2.3. -log D* = log(2m/pd). E_pg splits into KL(pg||pd)
- KL(pg||m) + 1 (E10 of the lesson keys). Strong answer:
the split with the three terms. Red flags: dropping the
minus KL term. Rubric: 2/2 all three terms. 1/2 two.
Remediation: U03-C04, C05.
D2.4. +inf means D hit 0 on some fake: the eps clip is
missing or D is frozen perfect. Pathology (2) of C07.
Check the D output histogram, add the clip, unfreeze D.
Strong answer: the two causes plus the three-step check.
Red flags: "more training fixes it". Rubric: 2/2 causes
plus check. 1/2 causes only. Remediation: U03-C07.
D2.5. The fix changes the objective: at optimal D it
minimizes KL(pg||pd) - KL(pg||m) + 1, not JS. The
minus term adds mode-seeking pressure. Keep the
minimax form when the JS story matters (analysis,
comparisons to f-GAN) or when you need the value to
mean a divergence. Strong answer: the changed objective
and when to keep minimax. Red flags: "non-saturating is
strictly better". Rubric: 2/2 objective plus the keep
condition. 1/2 objective only. Remediation: U03-C04, C05.

## Analytical/quantitative

Q1. V = -1.3537194326677915, JS = 0.32314028366610423,
2JS-2 matches V to 4.4e-16. The clip is load-bearing
because 1e-300 underflows: 1-(1-1e-300) == 0 in
float64, so without the clip the tail integral is
-inf (errors.md E-011). Strong answer: all numbers, the
match, and the underflow mechanism. Red flags: "the clip
is cosmetic". Rubric: 2/2 numbers plus mechanism. 1/2
numbers only. Remediation: U03-C04, C07.
Q2. Loss contribution: log2(1e-9) = -29.90 bits.
Gradient scale: 1/(1e-9 ln 2) = 1.44e9. The eps = 1e-7
clip bounds the loss at -23.25 bits and the gradient at
1.44e7: bounded spikes at the cost of bias. Strong answer: all four numbers plus the bias tradeoff. Red flags: "clipping is free". Rubric: 2/2 numbers plus
tradeoff. 1/2 numbers only. Remediation: U03-C05, C07.

## Implementation/debug

T1. Bug 1: np.log is the natural log. The course uses
log base 2, so every value is off by the factor ln 2.
Bug 2: no (0,1) assert: D = 0 or D = 1 gives -inf
silently instead of failing loud. Rules enforced:
state the log base at first use. Assert the domain
before the function that needs it. Strong answer: both
bugs plus both rules. Red flags: fixing only the log
base. Rubric: 2/2 both bugs and rules. 1/2 one bug.
Remediation: U03-C02, C07.

## Changed-constraint scenarios

S1. Survives: the players, the game structure, the
equilibrium concept. Breaks first: log D at D = 0 is
-inf, so V is undefined on any mistake and the D*
derivation fails. Smallest repair: randomized response
(emit the hard decision with probability 1-eps), which
restores (0,1) outputs. Strong answer: what survives,
what breaks first, and the repair. Red flags: "nothing
breaks". Rubric: 2/2 all three. 1/2 two. Remediation:
U03-C02, C08.
S2. Parameter sharing: one conditional GAN shares all
weights across classes. 1000 separate GANs share
nothing (1000x the parameters, roughly). Rare classes:
the conditional model borrows structure. Separate GANs
starve. Evaluation: the conditional model needs
per-class metrics (1000x the metric cost either way),
but training and serving one model beats 1000. Strong answer: sharing, rare-class, and evaluation arguments.
Red flags: "1000 GANs are fine". Rubric: 2/2 all three
arguments. 1/2 two. Remediation: U03-C08, C12.

## Research-critique

R1. Sentence 1: the number is the empirical game value
with a 2-layer critic, so the approximation gap is
unknown and the number measures critic weakness, not
distribution closeness (C04). Sentence 2: N = 500 lets
the inner max overshoot on lucky draws, so 0.001 may
be finite-sample noise. A sample-split would expose it
(C07/U02-C10). Sentence 3: even true JS = 0.001 would
not certify samples, because one number cannot see
coverage. The three-metric table plus a sample sheet
would expose the missing mode (C12). Strong answer: one
attack per sentence plus the exposing measurement. Red flags: accepting any sentence at face value. Rubric: 2/2
three attacks with measurements. 1/2 two. Remediation:
U03-C04, C07, C12.
