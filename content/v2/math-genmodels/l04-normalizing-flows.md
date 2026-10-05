---
page_id: math-genmodels-l04
course_slug: math-genmodels
course_name: "Mathematical Foundations of Generative Models"
course_order: 12
order: 4
nav: "L04 · Normalizing Flows"
title: "Lecture 4: Normalizing Flows — The Warper"
summary: "The warper answers the one question by bending noise into data with an invertible map, giving exact densities through the change-of-variables formula. The Jacobian determinant worked on 1-D and 2-D toys, and the price: rigidity."
date: "2026-10-05"
instructor: "Prof. Prathosh A P"
offering: "2025"
concepts: [normalizing-flows, change-of-variables, jacobian-determinant, invertibility, coupling-layer, exact-likelihood]
sources:
  - tag: paper
    label: "Papamakarios et al., Normalizing Flows for Probabilistic Modeling and Inference (2021)"
    url: https://arxiv.org/abs/1912.02762
  - tag: paper
    label: "Dinh, Krueger & Bengio, NICE: Non-linear Independent Components Estimation (2014)"
    url: https://arxiv.org/abs/1410.8516
---

## The question, for warps

Restate the one question: learn the rule behind samples, draw
fresh samples from it. The warper's answer: start with simple
noise, say uniform numbers. Learn a warping function that bends
the noise into data. The warp must be **invertible**: every noise
point maps to exactly one data point and back. Because the warp
is invertible, the machine computes exact probabilities through
the change-of-variables formula. No lower bound. No
approximation. Exact density is the warper's prize.

## First attempt: bend noise, then count

Take the simplest warp. Noise z is uniform on [0,1]: p_z(z) = 1.
The warp stretches and shifts: x = 2z + 1. So z = 0 becomes x =
1, z = 1 becomes x = 3. The outputs fill [1,3].

What is the density of x? Probabilities are conserved: the chance
that x lands in a small interval equals the chance that z lands in
the corresponding pre-interval.

```ascii
z:  |--dz--|                    p_z = 1
         | warp: x = 2z + 1
         v
x:  |----dx----|                dx = 2 dz
```

A dz-wide slice of z maps to a dx-wide slice of x, with dx = 2
dz. The same probability mass spreads over twice the width, so the
density halves:

```ascii
p_x(x) dx = p_z(z) dz
p_x(x) = p_z(z) x |dz/dx| = 1 x 1/2 = 0.5,  for x in [1,3]
```

Check: the density 0.5 over a width-2 interval integrates to 1.
The formula p_x(x) = p_z(z) |dz/dx| is the **change of variables**:
density transforms by the inverse slope. Stretch by 2, density
halves. This is exact, not a bound.

## Where general warps break, part 1: folds

The formula assumed one z per x. Drop that and watch it break.
Take z uniform on [-1,1], so p_z = 0.5, and the warp x = z^2.
Now z = 0.5 and z = -0.5 both give x = 0.25. The warp folds.

Each branch contributes its own term. Near x = 0.25:

```ascii
branch 1: z = +sqrt(x),  |dz/dx| = 1/(2 sqrt(x)) = 1.0
branch 2: z = -sqrt(x),  |dz/dx| = 1/(2 sqrt(x)) = 1.0

p_x(0.25) = 0.5 x 1.0 + 0.5 x 1.0 = 1.0
```

Check: p_x(x) = 0.5/sqrt(x) on [0,1] integrates to
[sqrt(x)]_0^1 = 1. The math works, but the price is visible: the
density is a sum over pre-images, one term per fold. A deep neural
warp folds thousands of times. The branch count explodes, and the
exact formula becomes unusable. Invertibility is not a luxury. It
is what keeps the density to a single term.

## Where general warps break, part 2: the determinant bill

Move to 2-D. A warp maps (z1, z2) to (x1, x2). The slope becomes
a matrix, the **Jacobian**: J[i,j] = dx_i/dz_j. The density
formula uses its determinant, which measures volume change:

```ascii
p_x(x) = p_z(z) / |det J|
```

Work a toy. Warp: x1 = 2 z1, x2 = z1 + z2. The Jacobian:

```ascii
J = [ dx1/dz1  dx1/dz2 ]   =   [ 2  0 ]
    [ dx2/dz1  dx2/dz2 ]       [ 1  1 ]

det J = 2 x 1 - 0 x 1 = 2
```

The unit square (area 1) warps to a parallelogram of area 2, so
density halves: p_x = 1/2 = 0.5 with p_z = 1. The determinant is
the 2-D version of the stretch factor.

Now the bill. A general d x d determinant costs O(d^3)
operations. For d = 1,000 (a tiny image patch), that is 10^9
multiply-adds per density evaluation, versus one forward pass of
the network. Exact density through a free-form neural warp is
exact but unaffordable. The warper needs warps whose determinants
cost almost nothing.

## The key question

What if the warp is invertible by construction, with a Jacobian
whose determinant reads off in O(d) time?

## The new idea: coupling layers

Split the variables into two halves. Transform one half with a
function of the other half, and leave the other half untouched.
This is a **coupling layer** (Dinh et al., 2014, the design the
Papamakarios review systematizes):

```ascii
forward:   x_a = z_a
           x_b = z_b x s(z_a) + t(z_a)

inverse:   z_a = x_a
           z_b = (x_b - t(x_a)) / s(x_a)
```

s and t are small neural networks (scale and shift). The inverse
is free: x_a recovers z_a directly, then z_b follows by
arithmetic. No iteration, no solving.

The Jacobian is triangular, because x_a does not depend on z_b:

```ascii
J = [ I    0        ]
    [ *    diag(s)  ]

det J = product of s(z_a) entries
```

A triangular determinant is the product of its diagonal: O(d)
operations. Watch it on numbers. z = [0.5, 1.0], s(z_a) = 2,
t(z_a) = 1:

```ascii
forward:  x_a = 0.5,  x_b = 1.0 x 2 + 1 = 3.0
inverse:  z_a = 0.5,  z_b = (3.0 - 1) / 2 = 1.0   (round trip works)
log det J = log 2 = 0.693
p_x([0.5, 3.0]) = p_z([0.5, 1.0]) / 2
```

Sampling runs the warp forward: draw z from a standard bell
curve, apply the layers, get x. Density runs it backward: map x
to z, evaluate p_z(z), divide by the determinant. Both directions
are exact. Stack many coupling layers, alternating which half
gets transformed, and the warp gains expressivity while every
layer stays invertible with a cheap determinant.

## Mapping back: what each property fixes

| General-warp failure | Coupling-layer answer | How |
|---|---|---|
| Folds: density sums over exploding branches | Invertible by construction | One z per x, always; the inverse is arithmetic |
| Determinant costs O(d^3): 10^9 ops at d = 1,000 | Triangular Jacobian | det = product of diagonal entries, O(d) |
| No sampling rule | Fixed base distribution N(0,I) | Draw z, warp forward |

## The honest price: rigidity, demonstrated

The coupling constraint is strict: each layer leaves half the
variables untouched. One layer cannot change z_a at all. Work it:
with z = [0.5, 1.0] and the mask above, x_a = 0.5 no matter what
s and t learn. The layer's expressive power is confined to one
half. Real flows stack dozens of layers with alternating masks
and permutations so every variable gets transformed
eventually, but the constraint never lifts: the architecture
family is a small subset of all neural maps.

Two more prices, stated plainly. First, dimension is frozen: the
warp maps R^d to R^d, so flows cannot compress to a small latent
space the way the VAE's sculptor does. A 256x256 image needs a
196,608-dimensional warp. Second, the triangular structure
biases the function class: variables transformed early never see
variables transformed late within one layer. Expressivity
studies show flows need more depth than free-form networks for
the same fit.

## Tying to CS336: what flows cannot use

CS336 builds the transformer block: residual stream, layer norm,
multi-head attention, MLP. Every piece is free-form. A residual
block x + f(x) is not invertible in general: two different inputs
can map to one output, which is exactly the folding failure from
the toy. So flow architectures cannot borrow the transformer
block as-is. They use restricted pieces: coupling layers,
invertible 1x1 convolutions (Glow), activation normalization with
tractable determinants. The warper's exactness is bought with
architectural discipline that the storyteller's transformer never
needs.

> [!QA]
> Q: Why must a normalizing flow be invertible?
> A: Because the density formula p_x(x) = p_z(z) / |det J| assumes exactly one pre-image z per x. The folding toy shows the alternative: with x = z^2, both z = 0.5 and z = -0.5 give x = 0.25, and the density becomes a sum over branches (0.5 x 1.0 + 0.5 x 1.0 = 1.0). A deep network folds thousands of times, so the branch sum explodes. Invertibility keeps the formula to one term.
> Follow-up: Can a flow still be universal if every layer is so restricted?
> A: In the limit of many layers, yes: coupling layers with alternating masks are universal approximators of distributions under mild conditions. In practice, depth is the price: flows stack dozens of restricted layers where a free-form network might use a few. Exactness trades against parameter efficiency, not against ultimate expressivity.

> [!QA]
> Q: What is the Jacobian determinant actually measuring?
> A: How much the warp stretches volume. In the 1-D toy, x = 2z + 1 stretches by 2, so density halves: p_x = 0.5. In the 2-D toy, det J = 2 means the unit square becomes an area-2 parallelogram, so density halves again: p_x = 0.5. Probability mass is conserved. The determinant is the exchange rate between z-volume and x-volume.
> Follow-up: Why does a triangular Jacobian make the determinant cheap?
> A: The determinant of a triangular matrix is the product of its diagonal entries: d multiplications instead of O(d^3). The coupling layer's Jacobian is triangular because x_a = z_a does not depend on z_b, so the upper-right block is zero. At d = 1,000, that is 1,000 operations versus 10^9.

> [!QA]
> Q: If flows give exact likelihoods, why did diffusion models take over image generation?
> A: Exactness is not the same as sample quality per parameter. The rigidity price bites: coupling layers leave half the variables fixed per layer, dimensions cannot shrink, and matching diffusion's sample quality takes very deep stacks. Diffusion pays a different price (1,000 sampling steps) but uses free-form denoiser architectures with no invertibility constraint, which fit image data better in practice.
> Follow-up: Where are flows still the right tool?
> A: Wherever exact density matters more than raw sample quality: anomaly detection (exact p(x) flags outliers), lossless compression (bits-back coding needs exact probabilities), and scientific simulators that need calibrated uncertainties. The warper is the density instrument. The restorer is the sample artist.

![Chapter plate: exact densities bought with architectural rigidity](assets/plate-l04.png "Chapter plate. Exact densities bought with architectural rigidity. Source: original plate for Stanford Frontier AI.")

## Recap: the whole lesson on one screen

1. **The question, for warps.** Learn the rule. Draw fresh samples. Bend noise into data with an invertible map.
2. **Change of variables, by hand.** z ~ Uniform(0,1), x = 2z + 1: p_x = 1 x 1/2 = 0.5 on [1,3]. Stretch by 2, density halves. Exact.
3. **Folds break it, demonstrated.** x = z^2 on [-1,1]: two branches give p_x(0.25) = 1.0. Branch sums explode for deep nets.
4. **The determinant bill, demonstrated.** 2-D toy det J = 2, p_x = 0.5. General d x d determinants cost O(d^3): 10^9 ops at d = 1,000.
5. **The key question.** What if the warp is invertible by construction with an O(d) determinant?
6. **Coupling layers.** x_b = z_b x s(z_a) + t(z_a), inverse by arithmetic, triangular Jacobian. Toy: [0.5, 1.0] -> [0.5, 3.0], log det = 0.693, round trip exact.
7. **The price: rigidity.** Each layer freezes half the variables. Dimensions cannot shrink. The architecture family is a strict subset of free-form nets.
8. **The CS336 tie.** Residual blocks x + f(x) fold, so flows cannot use transformer blocks as-is. Exactness buys discipline.

## Official sources and further reading

**Official:**
- Papamakarios et al., Normalizing Flows for Probabilistic Modeling and Inference (2021): https://arxiv.org/abs/1912.02762 — the review this chapter follows: coupling, autoregressive, and residual flow designs with their determinant costs.
- Dinh, Krueger & Bengio, NICE (2014): https://arxiv.org/abs/1410.8516 — the coupling layer origin.

**Further reading:**
- Dinh, Sohl-Dickstein & Bengio, RealNVP (2016): alternating masks and multi-scale stacking.
- Kingma & Dhariwal, Glow (2018): invertible 1x1 convolutions for images.

**Caveats from these sources.** [uncertain]: normalizing flows do not appear in the Prathosh playlist's topic sequence (weeks 1-12 cover GANs, VAE, DDPM, AR models, RL). This lesson is grounded in the Papamakarios review and the NICE/RealNVP papers instead. The universality claim for coupling stacks holds under smoothness conditions stated in the review, not unconditionally.

## Connections to the other courses

- **CS336:** the transformer block: free-form where flows are disciplined. The residual x + f(x) folding argument.
- **math-genai (sibling):** exact vs. approximate density: flows and autoregressive models are the only families with exact likelihoods. The ELBO lessons show what everyone else settles for.
- **CS229 L11:** the change-of-variables formula generalizes the Gaussian conditioning algebra: densities transform, parameters follow.
- **L05/L06 (this course):** continuous-time flows (neural ODEs, flow matching) reunite the warper with the restorer. The determinant bill is why diffusion won the sampling race.
