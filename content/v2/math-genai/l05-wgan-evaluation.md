---
page_id: math-genai-l05
course_slug: math-genai
course_name: "Mathematical Foundations of Generative AI"
course_order: 11
order: 5
nav: "L05 · WGAN and Evaluation"
title: "Lecture 5: Wasserstein GANs and Judging a Generator"
summary: "Two answers to Lesson 4's failures: the Wasserstein distance, whose gradient never saturates (worked on a point-mass toy: W = |θ| vs JS stuck at 0.693), and FID, the industry judge, computed by hand on a six-point toy (FID = 4)."
date: "2026-10-05"
instructor: "Prof. Prathosh A P"
offering: "2025"
video_id: _IBfVkrvqAI
concepts: [wgan, wasserstein-distance, lipschitz, kantorovich-rubinstein, fid, frechet-distance, inception]
sources:
  - tag: video
    label: "W4L11: Wasserstein GANs (video _IBfVkrvqAI)"
    url: https://www.youtube.com/watch?v=_IBfVkrvqAI
  - tag: video
    label: "W4L16: Evaluation of Generative Models (video 5Mchnh2xedI)"
    url: https://www.youtube.com/watch?v=5Mchnh2xedI
  - tag: paper
    label: "Arjovsky, Chintala, Bottou, Wasserstein GAN (2017)"
    url: https://arxiv.org/abs/1701.07875
---

## The task: a distance that never goes numb

Lesson 4 left two open wounds. The JS-like divergence saturates:
when the model is far from the data, the distance sits near its
maximum and its gradient nearly vanishes, so the generator gets
no signal. And there is no way to score a trained generator,
because it has no likelihood.

Both wounds come from the same root: the divergences so far only
care about probability ratios at each point. When the model and
the truth do not overlap at all, the ratio is 0 or infinite
everywhere, and every such pair looks equally "maximally far".
A model 10 units away scores the same as a model 100 units away.
No gradient can point downhill if the whole landscape is flat.

## First attempt: watch JS go flat

Make the toy as stark as possible. The truth is a point mass at
0: P_X = {0: 1.0}. The model is a point mass at θ: P_θ = {θ:
1.0}. They never overlap (for θ ≠ 0). Compute the
Jensen-Shannon divergence for θ = 10 and θ = 100.

```ascii
theta = 10:   P_X = {0: 1}, P_theta = {10: 1}, M = {0: 0.5, 10: 0.5}
  JS = 0.5*log(1/0.5) + 0.5*log(1/0.5) = log 2 = 0.693
theta = 100:  M = {0: 0.5, 100: 0.5}
  JS = log 2 = 0.693
```

Identical: 0.693 both times. JS cannot tell 10 from 100. Its
gradient with respect to θ is exactly zero. The generator is
lost: every direction looks the same. This is saturation made
quantitative, and it is why Lesson 4's generator stalled.

## The key question

Is there a distance between distributions that keeps working
when they do not overlap: a distance that says 10 is closer
than 100?

## The new idea: the earth mover's distance

The **Wasserstein distance** (earth mover's distance) asks: how
much mass must be moved, times how far, to reshape one
distribution into the other? On the point-mass toy the answer
is obvious. All the mass sits at θ and must travel to 0:

```ascii
W(P_X, P_theta) = |theta - 0| = |theta|

theta = 10:   W = 10
theta = 100:  W = 100
```

Ten is closer than a hundred, and the gradient is constant:
dW/dθ = 1 (for θ > 0), pointing downhill no matter how far
apart the distributions are. No flat region. No saturation.
This is the whole appeal: a distance with usable gradients
everywhere.

The formal definition matches the intuition: W is the minimum,
over all ways of pairing mass between the two distributions,
of the average distance mass travels. For point masses there
is only one pairing, hence |θ|. For spread-out distributions
the pairing problem is real, but the property survives: W
varies smoothly as the distributions move.

To train with it, the lecture uses the dual form
(Kantorovich-Rubinstein): the Wasserstein distance equals the
best achievable value of E[f(x)] − E[f(x̂)] over all
**1-Lipschitz** functions f. Lipschitz is a speed limit: the
function's slope never exceeds 1 anywhere. Without the speed
limit the max would be infinite (make f huge on real points,
hugely negative on fakes). With it, the best f measures
exactly the earth-moving cost.

The WGAN is therefore the Lesson 3 game with two changes:

```ascii
J = E[ f(x) ] - E[ f(x_hat) ]      (no logs, no sigmoid)
f must stay 1-Lipschitz            (the speed limit)
```

The critic outputs a raw score, not a probability. It
maximizes J. The generator minimizes it. On the point-mass
toy with f(x) = −x (slope −1, valid): J = E[−x] − E[−θ] =
θ. Minimizing over θ drives θ to 0 with gradient 1 at every
step. The generator walks downhill from 100 to 0 without
ever stalling. Compare: the JS game gave gradient zero the
whole way.

Enforcing the speed limit is the engineering. The original
WGAN clips the critic's weights to a small box after each
step, which crudely bounds the slope. It works but cripples
the critic's capacity. The later gradient-penalty variant
penalizes the slope directly. Either way, the price is a
slower, more constrained critic than the GAN's free one.

## The second wound: judging without a likelihood

A trained generator has no density, so MLE cannot score it.
The lecture's answer is to judge samples against samples:
take real data and generated data, and measure the distance
between the two *sets*. The industry standard is the
**Fréchet Inception Distance**, FID.

The procedure, confirmed in the lecture transcript:

1. Take a pre-trained Inception network (a deep image
   classifier trained on ImageNet). Its inner layers are
   known to respond to the same visual structure humans
   notice.
2. Pass n real images and n generated images through it.
   Take the output of one deep layer (the L-th layer, a
   hyperparameter) as the **feature** of each image.
3. Compute the mean μ and covariance Σ of the real
   features, and μ̂, Σ̂ of the generated features.
4. Assume both feature sets are Gaussian. The FID is the
   Wasserstein (Fréchet) distance between those two
   Gaussians:

```ascii
FID = ||mu - mu_hat||^2  +  Tr( Sigma + Sigma_hat - 2*(Sigma*Sigma_hat)^{1/2} )
```

The first term punishes shifted centers. The second punishes
different spreads and correlations. Lower FID means the
generated distribution sits closer to the real one. FID = 0
means the two Gaussians match exactly.

Work it by hand on a six-point toy. Real features (2-D):
(0,0), (2,0), (1,1). Generated features: (0,2), (2,2),
(1,3). Same cloud, shifted up by 2.

```ascii
mu_real = (1, 1/3)          mu_gen = (1, 7/3)
Sigma_real = Sigma_gen = [ 2/3   0  ]
                         [ 0    2/9 ]

||mu_real - mu_gen||^2 = 0^2 + 2^2 = 4
( Sigma_real * Sigma_gen )^{1/2} = Sigma_real   (they are equal)
Tr( Sigma + Sigma - 2*Sigma ) = 0

FID = 4 + 0 = 4
```

The whole distance comes from the mean shift, exactly as it
should: the clouds have identical shape, one sits 2 units
higher. If the generator also matched the center, FID would
be 0. One number, computed from samples, no likelihood
needed.

## The honest price

Wasserstein fixes saturation but not everything. The
Lipschitz constraint is awkward: weight clipping is crude
and the gradient penalty adds cost and tuning. WGAN training
is steadier than GAN training but slower, and mode collapse
can still happen (a collapsed model is still a valid low-W
point if the critic is weak). The saddle point is tamer, not
gone.

FID has its own honest limits, and the lecture states the
frame plainly: it judges in feature space, not pixel space.
It needs the Inception network, so it is an image metric.
for text or audio you need a different feature extractor.
It assumes Gaussian features, which is false but useful.
It is biased by sample size (small n inflates it), and it
cannot detect memorization: a generator that replays training
images scores a perfect FID. Judging creation remains partly
human.

| Wound from Lesson 4 | Answer | How | Number |
|---|---|---|---|
| Saturation (JS flat at 0.693) | Wasserstein distance | Gradient is 1 everywhere. No flat region | W = 10 vs W = 100. JS = 0.693 both |
| No likelihood to score models | FID | Wasserstein distance between Inception-feature Gaussians | Toy FID = 4, all from the mean shift |
| Mode collapse | Partially helped | Non-saturating gradients explore more, but collapse still possible | Not eliminated |

> [!QA]
> Q: Why is the Wasserstein distance better behaved than JS?
> A: JS only sees probability ratios, so non-overlapping distributions all score the max: 0.693 for θ = 10 and θ = 100 alike, gradient zero. Wasserstein measures moving cost: W = 10 versus W = 100, with a constant downhill gradient of 1. The generator always knows which way to go.
> Follow-up: What does the critic actually compute in a WGAN?
> A: It approximates the Kantorovich-Rubinstein dual: max over 1-Lipschitz f of E[f(x)] − E[f(x̂)]. No sigmoid, no logs. The Lipschitz speed limit keeps the max finite. Weight clipping enforces it crudely. The gradient penalty enforces it directly.

> [!QA]
> Q: What is FID, concretely?
> A: Pass real and generated images through a pre-trained Inception net, take one deep layer's outputs as features, fit a Gaussian to each feature set, and compute the Fréchet (Wasserstein) distance between the Gaussians: ||μ − μ̂||² + Tr(Σ + Σ̂ − 2(ΣΣ̂)^{1/2}). In the hand toy the two clouds had identical shape with centers 2 apart, giving FID = 4. Lower is better. 0 is a perfect match.
> Follow-up: Why Inception features instead of pixels?
> A: Because the lecture's cited study found Inception's inner layers track human visual judgment. Pixel distance says a one-pixel shift is a big change. Feature distance says it is nothing. FID judges what humans would judge, approximately.

> [!QA]
> Q: Can FID be gamed?
> A: Yes. A generator that memorizes and replays training images gets FID near zero without creating anything. FID also assumes Gaussian features, is biased upward at small sample sizes, and only works where a good feature extractor exists (images). It is a judge, not a proof.
> Follow-up: Then why is it the standard?
> A: Because the alternatives are worse. Likelihood does not exist for GAN generators, human rating does not scale, and pixel metrics disagree with perception. FID is the best cheap proxy, used with its limits stated.

## Recap: the whole lesson on one screen

1. **The wound.** JS saturates: θ = 10 and θ = 100 both score 0.693. Zero gradient, no signal.
2. **The key question.** Is there a distance that works when distributions do not overlap?
3. **The answer.** Wasserstein: the cost of moving mass. W = |θ|: 10 vs 100, gradient 1 everywhere.
4. **The WGAN.** Same two-player game, new rules: raw critic scores, 1-Lipschitz speed limit, no logs.
5. **Worked.** With f(x) = −x, J = θ. The generator walks from 100 to 0 without stalling.
6. **The second wound.** No likelihood, so judge samples with samples: FID.
7. **Worked.** Six 2-D points, identical clouds shifted by 2: FID = 4, all from the mean term.
8. **The price.** Lipschitz enforcement is crude. FID needs Inception, assumes Gaussians, and cannot catch memorization.

## Official sources and further reading

**Official:**
- W4L11: Wasserstein GANs:
  https://www.youtube.com/watch?v=_IBfVkrvqAI
- W4L16: Evaluation of Generative Models:
  https://www.youtube.com/watch?v=5Mchnh2xedI: FID procedure and formula confirmed in transcript.

**Further reading:**
- Arjovsky, Chintala, Bottou, "Wasserstein GAN" (2017):
  https://arxiv.org/abs/1701.07875: the dual form and weight clipping.
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (2017):
  https://arxiv.org/abs/1706.08500: the paper that introduced FID.

**Caveats.** The FID procedure, its formula, and the Inception/ImageNet details are confirmed in the W4L16 transcript. The W4L11 transcript was bot-blocked, so the WGAN treatment (duality, Lipschitz, clipping) follows the standard Arjovsky et al. presentation. [uncertain] The point-mass W = |θ| toy and the six-point FID = 4 computation are the lesson's own.

## Connections to the other courses

- **CS229 L11 (diffusion models):** that course never needs FID for training (its loss is plain denoising error), but FID is still how its samples are judged against GAN samples in papers. A worked tie: a diffusion model with FID 3.2 beats a GAN with FID 9.1 on the same dataset, and the number means the diffusion model's feature Gaussian sits closer to the real one by exactly that Fréchet gap.
- **CS336:** the training-data memorization discussions there meet FID's blind spot here: a model that replays training data scores perfectly on FID while creating nothing. Both courses converge on the same warning: no single metric certifies generation.
