# keys.md, U03 lesson answer keys

Date: 2026-10-06. Closed-book answers for lesson 03. Keep
separate from the lesson file. All numbers computed
2026-10-06, numpy 1.26.4, float64, seed 0 where RNG is used
(compute_run3.py).

## E01

V(D, G) = E_{x ~ p_data}[log2 D(x)] + E_{z}[log2(1 -
D(G(z)))]. D controls the referee's bets. G controls the
fake samples. D maximizes, G minimizes.

## E02

The fake pile is the single point x = 1. D* learns D*(1) =
0 (all mass there is fake) and D* = 1 elsewhere. G's
output is constant, so its gradient through D is zero
wherever it sits: moving the constant slightly changes
nothing about the pile it produces. G never improves.

## E03

p_g = p_data, D(x) = 1/2 everywhere. V = log2(1/2) +
log2(1/2) = -1 + -1 = -2 bits.

## E04

V = log2(0.5) + log2(0.5) = -2 for every G. G gets zero
signal: the objective does not depend on its parameters.
This is why the inner max must be real.

## E05

For fixed x, maximize a log y + b log(1-y) with a =
p_data(x), b = p_g(x), y in (0,1). d/dy: a/y - b/(1-y) =
0. Multiply by y(1-y): a(1-y) - by = 0, so y = a/(a+b).
Second derivative -a/y^2 - b/(1-y)^2 < 0: a maximum.

## E06

x = 0.5. a = gauss(0.5, 0, 1) = 0.3521. b = gauss(0.5, 1,
0.5) = 0.4839. D* = 0.3521/(0.3521+0.4839) = 0.4211.

## E07

V(G,D*) = E_pd[log(pd/(pd+pg))] + E_pg[log(pg/(pd+pg))].
Write m = (pd+pg)/2, so pd+pg = 2m. First term:
E_pd[log(pd/m)] - log 2 = KL(pd||m) - 1. Second:
KL(pg||m) - 1. Sum: KL(pd||m) + KL(pg||m) - 2 = 2 JS - 2.

## E08

Equilibrium needs p_g = p_data exactly. Here the means
match (0 = 0) but the scales differ (0.5 versus 1.0), so
JS = 0.1338 > 0 and V = -1.7324 > -2. Matching one
parameter is not matching the distribution.

## E09

t with sigmoid(t) = 0.001: t = -6.907. Minimax: d/dt
log(1-sig(t)) = -sig(t) = -0.001. Non-saturating: d/dt
-log(sig(t)) = -(1-sig(t)) = -0.999.

## E10

-log D* = log((pd+pg)/pd) = log(2m/pd). E_pg of that:
E_pg[log(pg/pd)] - E_pg[log(pg/(2m))] = KL(pg||pd) -
(KL(pg||m) - log2 2) = KL(pg||pd) - KL(pg||m) + 1. The
minus sign rewards p_g for putting mass where m is small
relative to p_g: concentration, i.e. sharpness.

## E11

Predicted: P(N(2,1) < 0) = Phi(-2) = 0.0228. Measured
(seed 0, N = 10000): 0.0232. SE = 0.0015, so the
measurement is within 1 SE: the prediction holds.

## E12

First, 0.2809 bits is not near zero relative to the
scale: the max JS on disjoint supports is 1 bit, so the
generator used up 28 percent of the divergence budget.
Second, JS does not name the failure: the left-mode mass
0.0232 versus 0.5 says a whole mode is gone, which
"0.2809" never states.

## E13

(1) Saturation: minimax G gradient vanishes at confident
D. Non-saturating fixes it. (2) Explosion: log-boundary
gradients hit 1e9 scale. Non-saturating does not fix it.
the eps clip does. (3) Misleading signal from a too
strong critic. Neither loss fixes it. Update balance
(C11) manages it.

## E14

log2(1e-9) = -29.90 bits. d/dD log2 D = 1/(D ln 2) =
1.44e9 at D = 1e-9. The clip to [1e-7, 1-1e-7] bounds the
gradient at 1.44e7 and the loss at -23.25 bits: bounded
spikes, at the cost of bias near the boundary.

## E15

V = E_{(x,y)~p_data}[log D(x,y)] + E_{z,y}[log(1 -
D(G(z,y),y))]. At (x=-2, y=1): p_data(x|y=1) at -2 is
~0.0540*... precisely, pd(-2|1) = N(-2, 2, 1) = 0.00013,
pg(-2|1) ~ 0. So D* = a/(a+b) = 1.0: the sample is
certainly real data carrying a wrong label.

## E16

Pooled metrics still pass (the marginal game is
unchanged). Every per-class metric fails: class-conditional
means, per-class coverage. Diagnostic rule: report
metrics per y, never pooled only.

## E17

fc: 100*4096 + 4096 = 413696. dc1: 128*256*16 + 128 =
524416. dc2: 64*128*16 + 64 = 131136. dc3: 3*64*16 + 3 =
3075. Total 1072323. Shapes: (4,4,256) -> (8,8,128) ->
(16,16,64) -> (32,32,3).

## E18

V = E_{x~p_data}[log D(x,E(x))] + E_z[log(1-D(G(z),z))].
At optimum the pair discriminator cannot distinguish, so
p_G(x,z) = p_E(x,z). Integrate over z: p_g(x) = p_data(x).

## E19

D trains 100 steps per G step and becomes perfect. Under
the minimax loss G's gradient is 0 to float noise. The
loss is flat. Nothing moves. It looks stable and learns
nothing. Diagnostic: D accuracy 1.0 on held-out piles for
many straight rounds with zero G parameter movement.

## E20

JS: G2 (0) beats G1 (0.2809). Mean gap: G2 (0.0119)
beats G1 (2.0063). Left-mode mass: G2 (0.4952) beats G1
(0.0232). No metric proves G2 is good: each proves G2
passed that metric. G1 passes quality-only metrics while
failing coverage.

## L01

Minimax: min_G max_D V(D,G), D up, G down, same V. Toy:
max_D V = -1.3537 at the bad start. Derivation: pointwise
max of a log y + b log(1-y) gives y = a/(a+b) (E05).
Code: the C03 snippet. Assert |V + 1.3537| < 1e-3 on seed
0, N = 100000. O(1) per x. Compare: the game needs
samples only. Likelihood needs densities. Debug: constant
G output gives D* = 0 at that point and zero G gradient.
Critique: the unrestricted-D assumption breaks first. A
net D reaches an approximation. Design: sigmoid(a x + b)
D, grid over (a,b), test max V -> -1.3537 as the grid
refines.

## L02

Saturation: the minimax G gradient -D dies where D is
confident. Toy: at D = 0.001, -0.001 versus -0.999.
Derivation: d/dt log(1-sig(t)) = -sig(t). D/dt -log
sig(t) = -(1-sig(t)). Code: the C05 snippet. Check both
are -0.5 at t = 0. Compare: minimax dead at D -> 0,
non-sat dead at D -> 1. Debug: +inf G loss means D hit 0
on fakes: the critic, not the loss formula, is at fault.
check the D output histogram. Critique: the fix assumes D
stays in (0,1) and trainable. Design: linear D, confident
start, count G steps to |mu_g| < 0.1 under each loss.

## L03

Mode collapse: the generator assigns ~0 mass to a data
mode. Toy: 0.0232 versus 0.4952 left-mode mass.
Derivation: G's gradient flows only through samples G
emits. No fake at the missing mode means no gradient
there. Code: the C06 diagnostic. Assert |measured -
0.0228| < 3*SE. Compare: VAEs punish missing mass via KL
(blurrier, better coverage). GANs risk collapse for
sharpness. Debug: coverage passes but tails gone means
partial collapse. The threshold metric is blind to it.
Critique: x < 0 assumes the mode boundary is known.
Design: a two-sample statistic (e.g. energy distance)
between G samples and held-out data, no hand threshold.

## L04

Conditional game: G(z,y), D(x,y), one minimax game per
class sharing weights. Toy: D*(-2,0) = 0.3333,
D*(-2,1) = 1.0. Derivation: the objective splits as E_y
of per-class games. Code: the C08 snippet. Check the
cross term is 1.0. Compare: conditioning shares weights
across classes. K separate GANs share nothing. Debug:
identical per-class means means G dropped y. Critique:
pooled metrics hide class-wise failure. Design: class
weights 0.9/0.1, predict the rare class degrades first,
measure per-class D* estimation error at fixed N.

## L05

Evaluation: ranking generators with no likelihood. Toy:
G1 versus G2 on JS, mean gap, left-mode mass.
Derivation: a metric is a sample mean. Passing it proves
only that. Code: the C12 snippet with SEs. Assert G2's
mean gap is within 2 SE of 0. Compare: fixed metrics are
dumb and honest. Learned metrics adapt but can be gamed.
Debug: JS = 0.001 on N = 500 with a small critic measured
the critic (overshoot + approximation gap), not the
generator. Critique: no single number certifies samples.
Design: shippable report = metric values + SEs + support
verdict + a hand-inspected sample sheet.

## Implementation and debug task

Correct version:

```python
import numpy as np
def gan_value(D_fn, xs_data, xs_fake):
    Dd = np.asarray(D_fn(xs_data), float)
    Df = np.asarray(D_fn(xs_fake), float)
    assert np.all((Dd > 0) & (Dd < 1)), "D left (0,1) on data"
    assert np.all((Df > 0) & (Df < 1)), "D left (0,1) on fakes"
    return float(np.log2(Dd).mean() + np.log2(1.0 - Df).mean())

rng = np.random.default_rng(0)
def Dstar(x):
    a = np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
    b = np.exp(-0.5*((x-1)/0.5)**2)/(0.5*np.sqrt(2*np.pi))
    return a/(a+b)
xd = rng.normal(0, 1, 100000)
xf = rng.normal(1, 0.5, 100000)
print(gan_value(Dstar, xd, xf))  # -1.3537, seed 0
```

The bugs: (1) np.log is the natural log. The course
convention is log base 2, so every value is off by a
factor of ln 2. (2) No (0,1) assert: D = 0 gives -inf
silently instead of a loud failure. The fix: np.log2 and
the assert before any log.

## Changed-constraint scenarios

S1. Hard D in {0,1}. Survives: the game structure, the
players, the equilibrium concept. Breaks first: log D(x)
at D = 0 is -inf, so V is undefined on any mistake. The
D* derivation (which needs interior (0,1)) fails.
Smallest repair: randomized response (output the hard
decision with probability 1-eps, flip with eps), which
restores (0,1) outputs and a finite V.

S2. D minimizes, G maximizes. The equilibrium flips: D
wants D = 0 on data and D = 1 on fakes (the worst
referee), G wants to be caught. New equilibrium: D(x) =
0 on data, 1 on fakes, V = -inf... precisely, the game
is unbounded below for D, so no finite equilibrium
exists. G learns to be maximally detectable: the opposite
of generation. The sign assignment is load-bearing.

## Research-critique question

Flaw 1: the number is the empirical game value, not JS.
With a 2-layer critic the approximation gap is unknown
(C04 assumption 1 broken). Expose: widen the critic and
watch the number move.
Flaw 2: N = 500 gives finite-sample overshoot of the
inner max (C07 pathology family, U02-C10 estimation
gap). Expose: sample-split the max and the evaluation.
Flaw 3: even true JS = 0.001 would not certify samples:
JS is one number and quality-only readings miss
coverage (C12). Expose: report the three-metric table
plus a hand-inspected sample sheet.
