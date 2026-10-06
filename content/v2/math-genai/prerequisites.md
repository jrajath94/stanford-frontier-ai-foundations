# prerequisites.md, math-genai prerequisite graph

Date: 2026-10-06. The shared 24-module bridge (P01-P24) already
exists at ~/workspace/stanford-frontier-ai/v2-pack/shared/prerequisites/.
This file gives the course-level graph plus self-contained local
remediation for the P03, P06, P07, and P08 essentials that U01 needs.
Notation follows the shared modules (log base 2 in bits, I(X, Y)
with a comma).

## Module index (from the prerequisite atlas)

P01 numeracy/algebra/notation. P02 Python and scientific software.
P03 vectors/geometry/linear maps. P04 spectral and numerical linear
algebra. P05 scalar and multivariable calculus. P06 probability.
P07 statistical estimation and uncertainty. P08 information theory.
P09 optimization and constrained problems. P10 ML foundations and
evaluation. P11 neural networks and autodiff. P12 PyTorch, tensors,
numerical stability. P13 language and sequence modelling.
P14 transformer mechanics. P15 hardware and computer architecture.
P16 distributed systems. P17 reinforcement learning. P18 Bayesian
inference, latent variables, sampling. P19 retrieval. P20 tools, APIs,
agent state. P21 security and privacy. P22 experimental method.
P23 economics. P24 production ML.

## Dependency chains into this course

- U01 needs P03 (needs P01), P06 (needs P01), P07 (needs P04, P06),
  P08 (needs P06, P07).
- U02 needs P05 (needs P01, P03), P08, P09 (needs P04, P05),
  P18 (needs P06, P07, P08, P09).
- U03 needs P08, P09, P11 (needs P02, P03, P05, P09, P10).
- U04 needs P04 (needs P03), P08, P09.
- U05 needs P08, P11, P18.
- U06 needs P04, P09, P11, P18.
- U07 needs P05, P06, P08, P18.
- U08 needs P09, P11, P12 (needs P02, P11), P18.
- U09 needs P05, P08, P13 (needs P06, P08, P10),
  P14 (needs P11, P12, P13), P18.
- U10 needs P12, P14, P15 (needs P01, P02), P17 (needs P06, P09,
  P10).

## Local remediation: P03 essentials used by U01

### R1, a file is a vector

A file with d numbers is a point in R^d. A 4-bit pattern 0101 is
the vector [0, 1, 0, 1] in R^4. Shape (4,).
Concrete: x = [0, 1, 0, 1]. x_2 = 1 (index, not exponent).

### R2, dot product

Multiply matching items, then add.
Concrete: [0, 1, 0, 1] dot [1, 1, 1, 1] = 0 + 1 + 0 + 1 = 2.
Rule: the dot product counts shared 1s for binary vectors.

### R3, vector mean

Add the vectors, divide by the count.
Concrete: [0, 0, 0, 0] and [1, 1, 1, 1] have mean [0.5, 0.5, 0.5, 0.5].
Check: computed 2026-10-06, numpy 1.26.4, float64.

## Local remediation: P06 essentials used by U01

### R4, sample space and events

The sample space lists every possible outcome. An event is a subset.
Concrete: sample space of one coin flip is {heads, tails}. The event
"heads" has probability 1/2 for a fair coin.

### R5, probability mass function

A probability mass function assigns one number to each outcome.
The numbers are in [0, 1] and sum to 1.
Concrete: fair coin: P(heads) = 0.5, P(tails) = 0.5. Sum = 1.
Misconception: a probability is not a count. The count 3 out of 8 is
data. The probability 3/8 is the model of that data.

### R6, expectation

The expectation is the probability-weighted average.
Concrete: for a fair coin paying 1 for heads and 0 for tails,
E = 0.5 * 1 + 0.5 * 0 = 0.5.

## Local remediation: P07 essentials used by U01

### R7, maximum likelihood on a coin

Pick the parameter that makes the observed data most probable.
Concrete: 7 heads in 10 flips. Candidate theta = 0.7 gives
log-likelihood 7 log2(0.7) + 3 log2(0.3) = -8.81 bits. Candidate
theta = 0.5 gives -10.0 bits. Theta = 0.7 wins.
Computed 2026-10-06.

### R8, small-sample wobble

An estimate from few samples wobbles. One head in one flip gives
theta = 1, which claims a sure thing from one observation.
Rule: distrust estimates whose sample count is tiny.

## Local remediation: P08 essentials used by U01

### R9, logs and bits

log2(x) asks: 2 to what power gives x. One bit is one binary choice.
Concrete: log2(8) = 3, log2(1/4) = -2.
Rule: log of a product is the sum of logs. This is why
log-likelihoods add across independent items.

### R10, logs of small numbers

log2 of a tiny probability is a large negative number. A model that
assigns probability 2^-30 to the data pays 30 bits for that item.
Rule: probabilities multiply, so log-probabilities add, and a near-zero
probability dominates the sum.

## Local remediation: P05 essentials used by U02

### R11, derivative as a slope

The derivative df/dx is the slope of f at x. Concrete: f(x) =
x^2, df/dx = 2x. At x = 3 the slope is 6. Check: (3.001^2 -
3^2)/0.001 = 6.001, close to 6.

### R12, partial derivative and gradient

Hold all but one variable fixed, take the slope in that
variable. The gradient collects the partials. Concrete: g(a,
b) = a^2 + 3b. dg/da = 2a, dg/db = 3. Gradient at (1, 5) is
[2, 3].

### R13, convex function

A function is convex when every chord between two graph
points lies above the graph. Concrete: x^2 is convex.
sqrt(x) is concave (chords below). x^3 - 3x is neither.
Rule: convexity decides the Jensen direction.

## Local remediation: P09 essentials used by U02

### R14, gradient descent

Step against the gradient to lower a function. Concrete:
minimize (x-3)^2 from x = 0, step 0.1: x <- x - 0.1*2(x-3).
First step: x = 0.6. It walks toward 3.

### R15, supremum

The supremum is the least upper bound: no member exceeds it,
and members approach it. Concrete: sup of {1 - 1/n} is 1,
never attained. The dual maximizes over functions. The sup
may sit at a limit point.

## Local remediation: P18 essentials used by U02

### R16, Monte Carlo

Estimate an expectation by a sample mean. Concrete: 1000
draws from Bernoulli(0.7) with seed 0 give mean 0.678 (numpy
1.26.4, computed 2026-10-06).
Error scale: sigma/sqrt(N). Rule: predict the error bar
before measuring.

### R17, ELBO in one line

log p(x) >= E_q[log p(x,z) - log q(z)]. The gap is KL(q(z) ||
p(z|x)). Concrete: the C03 toy gives bound -1.4490, gap
0.4490, sum -1.0 = log p(x). Computed 2026-10-06.

### R18, importance identity

E_p[g] = E_q[g * p/q] when supports allow. Concrete: with r
= [0.5, 1.75], E_q[r] = 1.0 exactly. Rule: ratios average to
1 under q. Use it as the first check of any ratio estimate.

## Local remediation: P09 essentials used by U03

### R19, min-max games and saddle points

min_x max_y f(x, y): read inside out. For each x, y plays
its best, giving g(x) = max_y f(x, y). Then minimize g
over x. A saddle point (x*, y*) is a minimum in x and a
maximum in y at the same point. Concrete: f(x, y) = (x -
y)^2 - y^2... simpler: f(x, y) = x*y. For fixed x, max
over y is unbounded unless y is bounded: restrict y to
[-1, 1]. Then g(x) = |x|, minimized at x* = 0 with y* =
sign(x). The GAN game is min_G max_D V: D's best response
first, then G. Rule: the inner max must track the outer
player, or the number means nothing.

## Local remediation: P11 essentials used by U03

### R20, a neuron

A neuron is an affine map plus a nonlinearity: out =
sigma(w . x + b). w is a weight vector, b a bias scalar,
sigma a fixed function (sigmoid, ReLU). Concrete: w =
[2], b = -1, x = 1, sigma = sigmoid: out = sigmoid(1) =
0.7311. Shape: x in R^d, w in R^d, out a scalar. A layer
stacks neurons. A net is layers composed.

### R21, backprop is the chain rule

To train, we need d loss / d w for every weight. The chain
rule: d loss/dw = (d loss/d out)(d out/dw). Backprop
applies it layer by layer, from the loss backward. It
computes every gradient in one forward pass plus one
backward pass: O(params), not O(params^2). Concrete: loss
= (out - 1)^2, out = sigmoid(w x + b), x = 1, w = 2, b =
-1. d loss/d out = 2(out-1) = -0.5379. d out/dw =
sigmoid'(1) * 1 = 0.1966. Product: -0.1058. Check by
finite differences: (loss(2.001) - loss(1.999))/0.002 =
-0.1058. Computed 2026-10-06.

### R22, the training loop

Repeat: sample a batch, forward pass (compute the loss),
backward pass (gradients), step against the gradient
(R14). For GANs there are two loops interleaved: ascend
on D, descend on G (U03-C02 pseudocode). The loop is the
same. Only the sign and the player change.

## Local remediation: P04 essentials used by U04

### R23, Lipschitz maps and the spectral norm

A map f is L-Lipschitz when |f(a) - f(b)| <= L |a - b| for
all a, b. L is the maximum slope. Concrete: f(x) = 2x is
2-Lipschitz. f(x) = -x is 1-Lipschitz. For an affine map
x -> W x, the Lipschitz constant is the largest singular
value of W. Concrete: W = [[2], [-1]] has sv = sqrt(5) =
2.2361, so x -> W x is 2.2361-Lipschitz. Composition
multiplies constants: L(f composed with g) <= L(f) L(g).
ReLU is 1-Lipschitz. Check: computed 2026-10-06, numpy
1.26.4, float64.

### R24, gradient norm as a local slope

For smooth f, |grad f(x)| is the local slope at x. The
Lipschitz constant is the max of |grad f| over the whole
space. Concrete: f(x) = 2.5x has grad 2.5 everywhere, so
it is 2.5-Lipschitz. Finite-difference check:
(f(x+h) - f(x-h)) / (2h) with h = 1e-6 gives
2.49999999996. Rule: a penalty on |grad f| at samples
controls the slope only where it is measured.

## Diagnostic routing

diagnostics/diagnostic-01-genai-foundations.md probes R1-R10 and the
U01 concepts. A score below 70 percent sends the learner back to the
matching R-section before U02. R11-R18 are probed inside the U02
lesson exercises (E01-E03 assume R13, E14-E15 assume R16, E16-E17
assume R11-R12, E18-E19 assume R15). A learner failing those is
sent to the matching R-section before the U02 ladders. R19-R22 are
probed inside the U03 lesson exercises (E03 assumes R19, E09
assumes R21, E19 assumes R14+R22, E02 assumes R20). A learner
failing those is sent to the matching R-section before the U03
ladders. R23-R24 are probed inside the U04 lesson exercises
(E07-E08 assume R23, E11-E12 assume R24, E05 assumes R15+R19).
A learner failing those is sent to the matching R-section
before the U04 ladders. R25-R26 are probed inside the U05
lesson exercises (E03 assumes R25, E02 assumes R26) and
reused by U07 (E08 assumes R25). A learner failing those is
sent to the matching R-section before the U05 ladders.
R27 is probed inside the U08 lesson exercises (E09-E10
assume R27 tensor shapes, E11 assumes R27). A learner
failing those is sent to R27 before the U08 ladders.
R28-R29 are probed inside the U09 lesson exercises
(E07-E08 assume R28, E10-E12 assume R29). A learner
failing those is sent to the matching R-section before
the U09 ladders. R30-R31 are probed inside the U10
lesson exercises (E04-E05 and E07-E08 assume R30, E10-E13
assume R31). A learner failing those is sent to the
matching R-section before the U10 ladders.

### R25, Gaussian facts for P18 (used by U05, U07)

N(mu, sigma^2) has density (2 pi sigma^2)^-0.5
exp(-(x - mu)^2 / (2 sigma^2)). E[x] = mu, Var(x) =
sigma^2. Linear transforms stay Gaussian: a x + b ~
N(a mu + b, a^2 sigma^2). Sums of independent Gaussians
are Gaussian with summed means and variances. The KL
between N(mu, s2) and N(0, 1) is 0.5 (mu^2 + s2 - 1 -
log s2) nats. Conditioning two Gaussians gives a
Gaussian (used in the DDPM posterior). Check: computed
2026-10-06, numpy 1.26.4, float64.

### R26, the ELBO five-line proof (U02 to U05 bridge)

Line 1: log p(x) = log int p(x | z) p(z) dz.
Line 2: multiply and divide by q(z | x) inside.
Line 3: log p(x) = log E_q[ p(x | z) p(z) / q(z | x) ].
Line 4: Jensen (log concave): >= E_q[ log p(x | z) +
log p(z) - log q(z | x) ].
Line 5: group the last two terms as -KL(q || p).
Result: log p(x) >= E_q[log p(x | z)] - KL(q || p).
The gap is KL(q || p(z | x)). Check: the toy gap
1.3040 nats, computed 2026-10-06.

### R27, tensor shapes for P12 (used by U08)

A tensor has a shape: a list of axis lengths. The
toy U-Net uses (C, H, W): channels, height, width.
A conv 1->8 k3 on 1x8x8 gives 8x8x8: channels
change, space stays (stride 1, padding 1). A
matmul (m, k) @ (k, n) gives (m, n): the inner
axes must match. Broadcasting: (4,) + (3, 4)
works (the 4 aligns), (3,) + (4,) fails.
no_grad/detach: marks a tensor as not needing
gradients (used for targets like the drawn eps,
and for the frozen x_T in the determinism
check). dtype bytes: fp32 4, fp16 2, int8 1.
Check: (2, 3) @ (3, 2) -> (2, 2), computed
2026-10-06, numpy 1.26.4, float64.

### R28, sequence modelling basics for P13 (used by U09)

A token is an integer id for a text piece, the
vocab lists all of them (V = 4 in the toy: a, b,
c, d). A sequence is a list of ids. The AR
factorization: p(x_1..x_n) = prod p(x_i |
x_<i). Teacher forcing: train each factor on
the true prefix. Perplexity: 2^{NLL/n}, the
effective branching factor, in [1, V].
Embeddings: each id becomes a vector (the E
matrix). Check: the bigram NLL 1.73697 bits
and perplexity 1.82574, computed 2026-10-06.

### R29, attention basics for P14 (used by U09)

Queries, keys, values: three projections of the
input. Scores: QK^T/sqrt(d). Weights: softmax
over keys per query (rows sum to 1). Output: the
weighted average of V. The causal mask zeroes
the future (upper triangle) so position i sees
only 0..i. Scale 1/sqrt(d) keeps the softmax
out of saturation. Check: the 3x3 toy weights
rows [1,0,0], [0.3302,0.6698,0],
[0.2483,0.2483,0.5035], computed 2026-10-06.

### R30, hardware basics for P15 (used by U10)

Bytes: params times bytes-per-element (fp32 4,
fp16 2, int8 1, int4 0.5). Bandwidth: bytes
per second the memory delivers. Latency: time
per op. Decode is memory-bound: each token
reads all weights once, so tok/s <= bandwidth
/ model bytes (the roofline). Prefill is
compute-bound: O(n^2) FLOPs dominate. Always
roofline both limits, quoting FLOP/s for a
memory-bound workload is true and irrelevant.
Check: 7B fp16 = 14e9 bytes, at 2e12 B/s the
ceiling is 142.9 tok/s. Authored arithmetic,
computed 2026-10-06.

### R31, RL basics for P17 (used by U10)

An MDP: states, actions, rewards, transitions.
A policy pi(a|s): the action distribution. The
return: discounted sum of rewards. The value:
expected return from a state. The policy
gradient trick: grad E[return] = E[grad log
pi(a|s) * advantage]: actions better than
average get more probable. PPO: clip the
policy ratio to [1-e, 1+e] so steps stay
small. The Bradley-Terry model: P(A beats B)
= sigmoid(r_A - r_B). For an AR-LM: state =
tokens so far, action = next token, reward =
preference score. Check: the bandit toy: two
actions, rewards 1 and 0, pi = softmax([0,
0]) = [0.5, 0.5]. d/dlogits log pi(a1) =
[0.5, -0.5], with advantage 1 the update
raises pi(a1). Computed 2026-10-06.
