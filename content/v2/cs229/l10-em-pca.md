---
page_id: cs229-l10
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 10
nav: "L10 · EM and PCA"
title: "Lecture 10: EM Algorithm and Principal Component Analysis"
summary: "Jensen's inequality builds the ELBO, EM alternates E and M steps on GMMs, and PCA finds the axes of variation."
date: "2026-05-06"
instructor: "Chris Ré"
offering: "Spring 2026"
duration: "1:20:00"
video_id: sUS-eTa0l6s
video_title: "Lecture 10: GMM (EM), PCA"
video_caption: "Original lecture. Chris Ré derives EM via the ELBO and presents PCA as the workhorse of dimensionality reduction."
concepts: [EM, expectation-maximization, ELBO, Jensen-inequality, latent-variable, E-step, M-step, PCA, eigenvalues, dimensionality-reduction]
sources:
  - tag: video
    label: "Lecture 10 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=sUS-eTa0l6s
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: fit the mixture when the labels are hidden

Lecture 9 set up the Gaussian mixture: the data is k Gaussians
mixed together, each point secretly from one of them. The
**latent variable** z^(i) is that secret: which cluster point i
came from. If we knew the z's, fitting would be trivial: group the
points by label and average (lecture 5's GDA). We do not know them.
That is the whole difficulty.

## First attempt: guess the labels, fit, repeat

The naive idea is hard EM by hand: guess the labels (k-means),
fit Gaussians to each guessed group, reassign each point to its
most likely Gaussian, repeat. Watch it on a toy. Two 1-D Gaussians,
true means 0 and 10, 6 points: {0.5, -0.5, 1.0} from cluster 1,
{9.5, 10.5, 10.0} from cluster 2. Guess labels by k-means: correct
here. Fit: mu_1 = 0.33, mu_2 = 10.0. Reassign: unchanged. Done.

Now corrupt the start: guess that {0.5, -0.5} are cluster 1 and
{1.0, 9.5, 10.5, 10.0} are cluster 2. Fit: mu_1 = 0, mu_2 = 7.75.
Reassign 1.0: distance to 0 is 1.0, to 7.75 is 6.75, so cluster 1.
Reassign 9.5: cluster 2. Converges to mu_1 = 0.17, mu_2 = 10.0:
recovered anyway. But hard assignment throws away doubt: the point
at 5.0, exactly between, gets forced into one cluster at full
weight and drags its mean. The soft version, which weighs points by
their responsibilities, is what we want. The question is how to
derive it instead of inventing it.

## The key question

MLE says: maximize the probability of the observed data. The
observed data is x. The labels z are hidden. The likelihood sums
over all possible labelings:

```ascii
L(theta) = product over i of sum over z of p(x^(i), z; theta)
```

The log of a sum. The log cannot reach inside the sum, so no closed
form, no clean gradient. The key question: can we maximize this
without ever differentiating through the sum?

## Jensen builds a lower bound

**Jensen's inequality**: for a concave function like log, the
function of an average is at least the average of the function.
Picture log(x): the curve sags below every chord between two of its
points. So log(E[X]) >= E[log(X)]. The log of the average beats the
average of the logs.

Apply it to the log likelihood. Introduce any distribution Q_i over
the hidden labels of example i. Then:

```ascii
log sum_z p(x, z) = log sum_z Q(z) * p(x, z)/Q(z) >= sum_z Q(z) * log(p(x, z)/Q(z))
```

The right side is the **ELBO**, the evidence lower bound. It is a
lower bound on the log likelihood, and it is a sum, not a log of a
sum: tractable. The bound is tight (equality) when Q(z) equals the
posterior p(z|x. Theta): Jensen is an equality when the chord
collapses, i.e., when Q puts all its structure where the posterior
is. So: set Q to the posterior, and maximizing the ELBO pushes the
true likelihood up.

![ELBO](assets/svg/l10-elbo.svg "The ELBO. Jensen's inequality turns the log of a sum into a sum: a tractable lower bound. Tight when Q equals the posterior. Source: original plate for Stanford Frontier AI.")

## EM: alternate two easy steps

**Expectation-Maximization** alternates. The **E-step**: fix the
parameters, set Q_i(z) = p(z | x^(i). Theta), the responsibilities.
This makes the ELBO touch the likelihood at the current parameters.
The **M-step**: fix Q, maximize the ELBO over theta. With Q fixed,
the ELBO is a weighted log likelihood: each example counted with
fractional weights, and weighted MLE for Gaussians is closed form.

For GMMs the steps are concrete. E-step: compute each point's
responsibilities under the current Gaussians. M-step: update each
Gaussian's mean to the responsibility-weighted average of points,
its covariance to the weighted spread, its mixing weight to the
average responsibility. Then repeat. The likelihood rises every
round: the E-step tightens the bound to touch the current value,
the M-step climbs the bound, so the true likelihood cannot fall.

Run the toy. Start mu_1 = 0, mu_2 = 7.75, equal weights, variance
1. E-step for x = 1.0: responsibility for cluster 1 proportional to
exp(-(1-0)^2/2) = 0.61, for cluster 2 exp(-(1-7.75)^2/2) ~ 0.
Nearly hard. For a point at 5.0 (add one): cluster 1 gives
exp(-12.5) ~ 0, cluster 2 gives exp(-3.8) = 0.022: soft, mostly
cluster 2 but not entirely. M-step: mu_1 becomes the weighted
average, pulled slightly by the 5.0 point's small responsibility.
Converge: mu_1 = 0.33, mu_2 = 10.0. The soft weights handled the
boundary point honestly instead of forcing it.

EM is k-means grown up: E-step softens the assignment into
responsibilities, M-step softens the mean update into weighted
averages. Same alternating skeleton, uncertainty preserved.

## The honest price of EM

EM climbs to a local maximum of the likelihood, not the global one.
Initialization matters: start the Gaussians badly and EM polishes
the wrong answer confidently. It can also crawl: near the top, the
steps shrink and convergence slows to a walk. And the likelihood it
maximizes can be degenerate: one Gaussian collapsing onto a single
point sends the likelihood to infinity (variance -> 0, density ->
infinite). Practitioners bound the variances away from zero or use
priors. EM is a hill-climber with no view of the landscape.

## PCA: the axes of variation

New job, same lecture. The customer records have 200 features.
Plotting is impossible, distance computations are expensive, and
most features are redundant (yearly spend and visit frequency move
together). **PCA** (principal component analysis) finds the few
directions that carry the most variation and projects the data onto
them.

The lecture's toy: SUVs plotted by two features, say length and
weight. The cloud is a diagonal cigar: long SUVs are heavy. The
**first principal component** is the direction of the cigar's long
axis: the single direction along which the data varies most.
Project every SUV onto that line and you keep most of the spread
with one number instead of two.

Formally: center the data (subtract the mean), form the covariance
matrix Sigma, take its **eigenvectors**. The eigenvector with the
largest **eigenvalue** is the first principal component. The
eigenvalue is the variance along it. The second component is the
next eigenvector, perpendicular to the first, and so on. Keep the
top k, drop the rest.

Work a toy. Four points: (1,1), (2,2), (3,3), (4,4), already
centered at (2.5, 2.5) -> (-1.5,-1.5), (-0.5,-0.5), (0.5,0.5),
(1.5,1.5). Covariance: each point contributes [a^2, a^2. A^2, a^2]
with a in {1.5, 0.5}: Sigma = [[2.5, 2.5],[2.5, 2.5]] (dividing by
4). Eigenvalues: 5 and 0. Eigenvectors: (1,1)/sqrt(2) and
(1,-1)/sqrt(2). The first component explains 5/5 = 100 percent of
the variance: the data lives on the diagonal, and one number (the
position along it) captures everything. The second eigenvalue is 0:
no variation perpendicular to the diagonal, so dropping it loses
nothing.

In practice: eigenvalues (5.0, 3.0, 0.4, 0.05, ...). Keep components
until the kept eigenvalues sum to, say, 95 percent of the total.
That is the dimensionality reduction: 200 features down to 12, with
95 percent of the variation intact.

![PCA](assets/svg/l10-pca.svg "PCA. The first principal component is the cigar's long axis: the direction of maximum variance. Eigenvalues say how much each axis carries. Source: original plate for Stanford Frontier AI.")

## The honest price of PCA

PCA is linear: it finds straight axes. Data curved like a Swiss
roll has no good straight summary. The roll's two intrinsic
dimensions need nonlinear methods. PCA also chases variance, not
meaning: the highest-variance direction might be sensor noise while
the signal hides in a small one. It is sensitive to scaling: measure
one feature in millimeters instead of meters and it dominates the
covariance, so standardize features first. And the eigendecomposition
costs O(d^3): at d = 100,000 (raw pixels), exact PCA is dead and
randomized approximations take over.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| ELBO via Jensen | Log of a sum has no closed form | log(E) >= E[log]: lower bound that is a sum; tight at Q = posterior |
| E-step | Labels hidden | Set Q to responsibilities under current params; bound touches likelihood |
| M-step | Bound still needs maximizing | Weighted MLE: closed-form weighted averages; likelihood rises every round |
| Soft vs hard | Hard assignment forces boundary points | 5.0 point gets soft weight, not forced; k-means is the hard limit |
| PCA | 200 features, mostly redundant | Top eigenvectors of covariance; toy: eigenvalue 5 of 5 = 100 percent on the diagonal |

> [!QA]
> Q: What is the ELBO and why does EM maximize it instead of the likelihood?
> A: The evidence lower bound: sum_z Q(z) log(p(x,z)/Q(z)), a lower bound on the log likelihood built by Jensen's inequality. The true log likelihood is a log of a sum over hidden labelings: intractable to differentiate. The ELBO is a sum: tractable. EM alternates: the E-step sets Q to the posterior, making the bound tight at the current parameters. The M-step maximizes the bound, which is weighted MLE with closed forms. Since the bound touches the likelihood before each M-step, climbing the bound climbs the likelihood. It never decreases.
> Follow-up: When is the bound tight?
> A: When Q(z) = p(z|x. Theta), the posterior under the current parameters. Jensen's inequality is an equality when the distribution inside is degenerate relative to the function: here, when Q matches the posterior, the "average" the log wraps is exact. That is precisely the E-step's choice.

> [!QA]
> Q: How is EM different from k-means?
> A: Same alternating skeleton, soft instead of hard. K-means E-step: assign each point to one cluster. EM E-step: compute responsibilities, the posterior probability per cluster. K-means M-step: plain means. EM M-step: responsibility-weighted means, covariances, and mixing weights. K-means minimizes distortion; EM maximizes likelihood. Let the Gaussians' variances go to zero and EM's soft steps harden into k-means.
> Follow-up: What can go wrong with EM?
> A: Three things. Local maxima: bad initialization gets polished, not fixed. Slow crawl near convergence. Degenerate likelihood: a Gaussian collapsing onto one point drives variance to zero and likelihood to infinity, so bound variances or use priors. None of these affect the guarantee that likelihood never decreases. They affect which maximum you reach and how fast.

> [!QA]
> Q: What does PCA actually compute, and how do you choose k?
> A: Center the data, form the covariance matrix, take its eigendecomposition. The eigenvectors are the principal components, ordered by eigenvalue: variance along that axis. Project onto the top k. Choose k by explained variance: keep components until their eigenvalues sum to a target like 95 percent of the total. In the toy, eigenvalues (5, 0) meant k=1 keeps 100 percent.
> Follow-up: When does PCA fail?
> A: When the structure is nonlinear (Swiss roll), when variance is not meaning (the top component is noise), or when features are unscaled (millimeters dominate meters: standardize first). It also costs O(d^3), so exact PCA dies around d = 100,000 and randomized methods take over.

## Recap: the whole lesson on one screen

1. **The job.** Fit k Gaussians when each point's label is a
   hidden secret.
2. **First attempt.** Guess labels, fit, reassign, repeat. Hard
   forcing of boundary points. No uncertainty.
3. **The key question.** Maximize the likelihood without
   differentiating through a log of a sum.
4. **Jensen.** log(E) >= E[log]. The ELBO: a tractable lower
   bound, tight at Q = posterior.
5. **EM.** E-step: responsibilities. M-step: weighted MLE.
   Likelihood rises every round. Toy converges to mu = 0.33, 10.0.
6. **EM's price.** Local maxima, slow crawl, degenerate collapse.
   Initialization matters.
7. **PCA.** Top eigenvectors of the covariance: the axes of
   variation. Toy: (5, 0), one axis keeps 100 percent.
8. **PCA's price.** Linear only, chases variance not meaning,
   scale-sensitive, O(d^3).

## Official sources and further reading

**Official:**
- Lecture 10 video, Stanford Online YouTube:
  https://www.youtube.com/watch?v=sUS-eTa0l6s — Chris Ré derives
  the ELBO from Jensen's inequality, runs EM on GMMs, and presents
  PCA on the SUV toy.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the full
  EM derivation and PCA eigendecomposition.

**Caveats from these sources.** The lecture states Jensen only for
concave functions (the log case) and draws the chord picture live.
The general statement is in the notes. The SUV toy is the lecture's
own illustration. The diagonal 4-point miniature in this lesson is
an original with the lecture's eigenstructure. The degenerate
likelihood warning is standard practice around EM, presented here
as the practitioner's caveat.

## Connections to the other courses

- **CS229 L05:** GDA: the same Gaussians with labels known; EM
  handles them hidden.
- **CS229 L09:** k-means: the hard limit of EM as variances go to
  zero.
- **CS229 L11:** diffusion models: latent-variable generative
  modeling with the ELBO grown to neural scale.
- **CS229 L13:** embeddings: PCA's linear ancestor of learned
  representations.
- **CS336:** randomized PCA and scale: dimensionality reduction at
  training scale.
