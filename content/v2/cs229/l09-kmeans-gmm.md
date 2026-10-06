---
page_id: cs229-l09
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 9
nav: "L09 · K-Means and GMM"
title: "Lecture 9: Clustering, K-Means and Gaussian Mixtures"
summary: "The unsupervised turn: k-means as the intuitive ad-hoc algorithm, k-means++ seeding, and GMMs as the soft probabilistic version."
date: "2026-05-04"
instructor: "Chris Ré"
offering: "Spring 2026"
duration: "1:16:24"
video_id: bSmIGBCoffA
video_title: "Lecture 9: K-Means and GMM (non-EM)"
video_caption: "Original lecture. Chris Ré introduces unsupervised learning with k-means and sets up Gaussian mixture models."
concepts: [unsupervised-learning, clustering, k-means, k-means++, distortion, local-minima, GMM, elbow-method]
sources:
  - tag: video
    label: "Lecture 9 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=bSmIGBCoffA
  - tag: video
    label: "Explainer: StatQuest, K-means clustering"
    url: https://www.youtube.com/watch?v=4b5d3muPQmA
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: a million unlabeled customer records

A store has one million customer records: age, yearly spend, visit
frequency. No labels. Nobody tagged anyone "bargain hunter" or
"whale". The job: find the natural groups, so marketing can treat
each group differently. This is **clustering**, the flagship
unsupervised task. No y anywhere. The machine must invent the
categories from the shape of the data.

## First attempt: k-means

The intuitive algorithm: pick k group centers, assign each customer
to the nearest center, move each center to its group's average,
repeat. That is **k-means**, and the lecture calls it ad hoc on
purpose: it is not derived from a probability model, it is just the
obvious thing, and it works.

The two steps, precisely. **Assignment**: each point joins the
cluster whose center mu_j is closest. **Update**: each center moves
to the mean of its assigned points. Repeat until assignments stop
changing.

Watch it on a toy. Six points on a line: {1, 2, 3, 10, 11, 12}.
k = 2. Start centers at mu_1 = 1, mu_2 = 12 (bad luck: the extremes).

```ascii
start:   mu_1 = 1, mu_2 = 12
assign:  {1,2,3} -> mu_1,  {10,11,12} -> mu_2
update:  mu_1 = 2, mu_2 = 11
assign:  unchanged. Done in 1 round.
```

Lucky start. Now start at mu_1 = 1, mu_2 = 2 (both in the left
clump):

```ascii
start:   mu_1 = 1, mu_2 = 2
assign:  {1} -> mu_1,  {2,3,10,11,12} -> mu_2
update:  mu_1 = 1, mu_2 = 7.6
assign:  {1,2,3} -> mu_1,  {10,11,12} -> mu_2
update:  mu_1 = 2, mu_2 = 11.  Done in 2 rounds.
```

Same data, different start, same good answer here. But the lecture
stresses the general truth: the final clusters depend on where the
centers start. K-means converges (the **distortion**, the sum of
squared distances to centers, falls every round and cannot fall
forever), but it converges to a **local minimum**, not the global
one. Different seeds, different answers. The NP-hardness result the
lecture cites says no efficient algorithm guarantees the global
optimum.

### Subchapter: one round, audited

Trust the toy, then verify it. First round, lucky start: assigned
{1,2,3} to mu_1 = 1. The update moves mu_1 to the mean: (1+2+3)/3
= 2. Assigned {10,11,12} to mu_2 = 12: mean (10+11+12)/3 = 11. Now
audit the elbow's k=1 number. Mean of all six: 39/6 = 6.5.
Distortion: (1-6.5)^2 + (2-6.5)^2 + (3-6.5)^2 + (10-6.5)^2 +
(11-6.5)^2 + (12-6.5)^2 = 30.25 + 20.25 + 12.25 + 12.25 + 20.25 +
30.25 = 125.5. (An earlier draft of this lesson wrote 121.5: wrong.
The arithmetic is above.) k=2 gives 4.0: the drop from 125.5 to 4.0
is the elbow's signal, and it survives the correction.

![Round audit](assets/plate-l09-round-audit.webp "One round, audited. Assign {1,2,3} to center 1, {10,11,12} to center 12. Update to means 2 and 11. The k=1 distortion is 125.5, not 121.5. Source: original audit for the toy round. Project: Stanford Frontier AI.")

### Subchapter: the mean is the minimizer

Why the mean, and not the median or the midpoint? Fix the
assignments and minimize the distortion over mu_j. The term for
cluster j is sum of (x - mu_j)^2 over its points. Derivative with
respect to mu_j: -2 * sum(x - mu_j) = 0. So sum(x) = n * mu_j, and
mu_j = mean of the points. One line of calculus, no choice
involved. The median would minimize the sum of absolute distances,
a different objective (k-medians). K-means uses squared distances,
so it gets the mean. The algorithm's two steps are both optimal for
their subproblem: assignment is optimal per point, update is
optimal per center. That is why distortion never rises.

## Where it breaks: the seed decides

Demonstrate with numbers. Eight points: four at x = 0 (call them
A), four at x = 10 (B), and the true groups are {A} and {B}. Start
centers at 0 and 0.1 (both inside A). Round 1: all eight points go
to mu_2 = 0.1 (closer than 0 to every B point? B at 10: distance to
0.1 is 9.9, to 0 is 10: yes, mu_2 wins everything). mu_1 gets zero
points. The implementation must handle the empty cluster (common
fix: reinitialize it randomly). Depending on the fix, you can end
with both centers inside A and B unclustered: distortion far above
optimal. The seed decided the answer. On real data with hundreds of
dimensions, bad seeds are the norm, not the exception.

### Subchapter: the seed lottery

Two seeds, same data, different bills. Seeds at 1 and 12: one
round, distortion 4.0, done. Seeds at 0 and 0.1: round 1 assigns
all eight points (four at 0, four at 10) to mu_2 = 0.1, because 9.9
< 10 for every B point. mu_1 gets zero points: the empty-cluster
case. Reinitialize mu_1 randomly and you may land inside A again,
ending with both centers in A and B unclustered: distortion stuck
near 200 instead of 4.0. The lottery is the algorithm: k-means is
deterministic given the seed, and the seed is luck. Production
practice: k-means++ seeding plus 10 restarts, keep the best
distortion. Sklearn does this by default (n_init=10).

![Seed lottery](assets/plate-l09-seed-lottery.webp "The seed lottery. Seeds at 1 and 12: one round, distortion 4.0. Seeds at 0 and 0.1: a cluster strands, distortion near 200. Source: original plate for the seed dependence. Project: Stanford Frontier AI.")

## The key question

Can we seed the centers so cleverly that the local minimum is
probably a good one, with a guarantee?

## k-means++: seed far apart

**K-means++**, from Stanford graduate students (Arthur and
Vassilvitskii), seeds one center at a time: pick the first center
uniformly at random among the points. Pick each next center with
probability proportional to its squared distance from the nearest
existing center. Points far from all centers are likely seeds.
Points near a center are unlikely.

On the toy {1,2,3,10,11,12}: first center lands somewhere, say 2.
Squared distances: 10 is 64 away, 11 is 81, 12 is 100. 1 is 1, 3 is
1. The next seed is overwhelmingly likely to come from {10,11,12}.
The two clumps get one seed each with high probability. The lecture
reports the payoff: k-means++ guarantees an expected approximation
ratio of O(log k) to the optimal distortion. Not optimal (NP-hard
forbids that), but provably close, from seeding alone. It is the
default in sklearn.

## How many clusters? The elbow

K-means needs k up front. The **elbow method**: run k-means for
k = 1, 2, 3, ..., plot the final distortion. Distortion always falls
as k rises (k = n gives distortion 0: every point its own center).
Look for the **elbow**, the k where the curve bends: gains slow
down after it. Toy distortions: k=1: 125.5, k=2: 4.0, k=3: 2.7,
k=4: 1.5. The elbow is at k = 2: the drop from 125.5 to 4.0 dwarfs
everything after. The lecture's honest note: elbows are often
ambiguous on real data. It is a heuristic, not a rule.

![Elbow method](assets/svg/l09-elbow.svg "The elbow method. Distortion falls with k. The bend at k = 2 marks the natural cluster count. Source: original plate for Stanford Frontier AI.")

## Softening: Gaussian mixture models

K-means makes hard assignments: each point belongs to exactly one
cluster. Real groups overlap. A **Gaussian mixture model** (GMM)
softens everything: the data is a mix of k Gaussians, each point has
a probability of belonging to each one.

The model: pick a cluster j with probability phi_j, then draw x
from Gaussian(mu_j, Sigma_j). The **responsibility** gamma_j(x) is
the posterior probability that point x came from cluster j: how much
cluster j "claims" x. A point between two clumps might be 70 percent
cluster 1, 30 percent cluster 2, instead of k-means' all-or-nothing.

K-means is the limiting case: let every Sigma shrink toward zero
and the responsibilities harden to 0 or 1. The lecture frames GMM as
the probabilistic grown-up of the ad-hoc algorithm: same spirit,
with uncertainty quantified.

![GMM](assets/svg/l09-gmm.svg "Gaussian mixture model. Each point carries responsibilities across clusters: 70 percent cluster 1, 30 percent cluster 2. K-means is the hard limit. Source: original plate for Stanford Frontier AI.")

### Subchapter: responsibilities, worked

Take two 1-D Gaussians: cluster 1 at mu = 2, cluster 2 at mu = 11,
both sigma = 1, equal priors phi = 0.5. Point x = 3. Density under
cluster 1: exp(-(3-2)^2/2)/sqrt(2*pi) = 0.242. Under cluster 2:
exp(-(3-11)^2/2)/sqrt(2*pi), about 1e-15: essentially zero.
Responsibility gamma_1(3) = 0.5*0.242 / (0.5*0.242 + 0) = 1.0. The
point is fully claimed. Now x = 6.5, the midpoint. Both densities
equal exp(-(4.5)^2/2)/sqrt(2*pi): identical. gamma_1 = gamma_2 =
0.5. The midpoint splits evenly. Between 3 and 6.5 the
responsibility slides smoothly from 1.0 to 0.5: that slide is the
softness k-means cannot express.

![Responsibilities worked](assets/plate-l09-responsibility.webp "Responsibilities, worked. Point 3: cluster 1 claims it fully, 1.0 vs 0.0. Point 6.5, the midpoint: 0.5 and 0.5. Source: original toy for the responsibility arithmetic. Project: Stanford Frontier AI.")

### Subchapter: the interview trap, crescents

Draw two interleaved half-moons: crescent A opening right,
crescent B opening left, nested. Ask k-means with k=2. It draws
one straight bisector and splits both crescents in half: each
cluster is half of each moon. Distortion is locally minimal and
completely wrong. The failure is structural: k-means only knows
Euclidean balls. The interview trap is asking "when would you not
use k-means" and accepting "when k is unknown". The strong answer:
non-convex shapes. The fix is density or graph methods (DBSCAN,
spectral clustering), which follow the moons. Decision rule: plot
a 2-D projection first. If you see crescents, spirals, or rings,
do not run k-means.

![Crescents](assets/plate-l09-crescents.webp "Know when to quit k-means. Spherical clusters: k-means wins. Interleaved crescents: the bisector splits both moons in half. Source: original plate for the non-convex failure. Project: Stanford Frontier AI.")

## The honest price

K-means buys simplicity and pays in guarantees: local minima, seed
dependence, and no notion of uncertainty. It assumes spherical
clusters of similar size: stretch one cluster into a long ellipse
and k-means splits it wrongly, because Euclidean distance is the
only geometry it knows. Choosing k is a heuristic. GMM buys soft
assignments and pays in fitting: the cluster labels are hidden, so
MLE has no closed form, and the next lecture's EM algorithm must
iterate. Both assume you know the right k and the right distance.
When clusters are non-convex (two interleaved crescents), both fail
and density methods take over.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| K-means | No labels, need groups | Alternate assignment and mean-update; distortion falls every round; toy converges in 1-2 rounds |
| Local-minima diagnosis | Same data, different answers per seed | Convergence is to a local minimum; NP-hard globally; seeds at 0 and 0.1 can strand a cluster |
| K-means++ | Bad seeds are the norm | Seed proportional to squared distance; O(log k) expected approximation ratio; sklearn default |
| Elbow method | k is unknown | Distortion 121.5, 4.0, 2.7, 1.5: the bend at k=2; heuristic, often ambiguous |
| GMM | Hard assignments lie about overlap | Responsibilities: 70/30 splits; k-means is the zero-variance limit |

> [!QA]
> Q: Why does k-means converge, and to what?
> A: Each round has two steps and neither raises the distortion (sum of squared distances to centers). Assignment moves each point to its nearest center, lowering or holding its term. Update moves each center to its points' mean, which is the unique minimizer of its term. Distortion falls every round, is bounded below by 0, so it converges. It converges to a local minimum: no single reassignment or mean-move improves it, but a different seed could reach a better one. The seed-dependence is fundamental, not a bug.
> Follow-up: What do you do about empty clusters?
> A: Common fixes: reinitialize the empty center to a random data point (often the point farthest from its center), or drop it and continue with k-1. The lecture's implementation note: handle it explicitly, because bad seeds produce empty clusters routinely in high dimensions.

> [!QA]
> Q: How does k-means++ seeding work, and what does it guarantee?
> A: Seed centers one at a time. First center uniform at random. Each next center chosen with probability proportional to its squared distance from the nearest existing center: far-apart points are likely seeds. On {1,2,3,10,11,12} with first seed at 2, the squared distances (64, 81, 100 for the right clump vs 1, 1 for the left) make the second seed land in the right clump with high probability. Guarantee: expected distortion within O(log k) of optimal. It cannot promise optimal: the problem is NP-hard.
> Follow-up: Why squared distance and not distance?
> A: Because the objective is squared distance (distortion). Seeding proportional to squared distance samples proportionally to each point's current contribution to the objective, which is what the approximation proof needs. Linear distance would under-seed far outliers relative to their cost.

> [!QA]
> Q: What is a responsibility in a GMM?
> A: The posterior probability that a point came from each cluster: gamma_j(x) = phi_j * Gaussian(x. Mu_j, Sigma_j) / sum over clusters. It is how much cluster j claims the point. A point dead center in cluster 1 has responsibility near 1 for it. A point between clusters splits, e.g., 0.7 and 0.3. K-means' hard assignment is the limit as cluster variances go to zero: responsibilities collapse to 0 or 1.
> Follow-up: When does k-means fail where GMM succeeds?
> A: Overlapping clusters of different sizes or shapes. K-means draws a hard bisector and assumes spheres. A small dense clump next to a big diffuse one gets mis-split. GMM's responsibilities and per-cluster covariances model the overlap and the shapes. The price: fitting needs EM (lecture 10), and you still choose k.

## Recap: the whole lesson on one screen

1. **The job.** One million unlabeled customers. Find the natural
   groups. No y anywhere.
2. **K-means.** Assign to nearest center, move centers to means,
   repeat. Toy {1,2,3,10,11,12} converges in 1-2 rounds.
3. **Where it breaks.** Seeds decide: centers at 0 and 0.1 strand
   a cluster. Local minima, NP-hard globally.
4. **The key question.** Can seeding alone guarantee a good local
   minimum?
5. **K-means++.** Seed proportional to squared distance. O(log k)
   expected approximation ratio. Sklearn default.
> [!QA]
> Q: Walk me through the mechanism: prove the update step must be the mean.
> A: Fix the assignments. Cluster j's distortion term is sum over its points of (x - mu_j)^2. Differentiate with respect to mu_j: -2 * sum(x - mu_j) = 0. So sum(x) = n_j * mu_j, and mu_j = (1/n_j) * sum(x): the mean. One line. The median would minimize the sum of absolute deviations, a different algorithm (k-medians). K-means minimizes squares, so the update is forced to be the mean.
> Follow-up: Both steps are optimal for their subproblem. Why is the whole algorithm not optimal?
> A: Because the steps optimize different variables alternately: assignment fixes centers, update fixes assignments. Each step is the best move given the other half frozen. Alternating best-moves converges to a local minimum of the joint problem, like descending into the nearest valley while the deepest valley sits across a ridge. Coordinate descent shares this property.

> [!QA]
> Q: Applied design: one million customers, features are age, yearly spend, visit frequency. Marketing wants segments. Walk through your choices.
> A: First, standardize: spend is in dollars (thousands), visits in counts (tens). Without scaling, Euclidean distance is spend distance: a $100 spend gap dwarfs a 5-visit gap, and clusters become spend brackets. Standardize each feature to unit variance, or use domain weights. Second, k-means++ seeding with 10 restarts, keep the best distortion. Third, choose k by elbow plus silhouette score, and show marketing the cluster profiles (mean age, spend, visits per cluster) so the segments are actionable. Fourth, sanity check: tiny clusters (under 1% of customers) are usually outliers, not segments.
> Follow-up: Marketing asks for exactly 5 segments because the campaign has 5 creatives. Elbow says 3. What do you do?
> A: Run k=5 and k=3, profile both, and show the trade: k=5 splits a natural group to fill the quota. If the split is interpretable (e.g., high-spend splits into frequent and infrequent), take 5. If it is arbitrary, push back with the profiles: forcing k manufactures segments that do not exist, and the campaign personalizes on noise.

> [!QA]
> Q: When do you pay for EM and a GMM instead of k-means?
> A: When the answer needs uncertainty or shape. Overlapping customer tiers where a point is genuinely 60/40: k-means forces a hard label, GMM reports the split. Clusters with different covariances: a tight dense clump beside a broad diffuse one. K-means' spherical bias mis-splits them. GMM fits a covariance per cluster. Downstream weighting: if the next stage weights points by cluster confidence, you need responsibilities, not labels. The price: EM is slower, has worse local optima than k-means, and still needs k.
> Follow-up: Can GMM fail where k-means succeeds?
> A: Yes, on well-separated spherical clusters with little data: GMM fits full covariances (d^2 parameters per cluster) and overfits, while k-means' rigidity is a virtue. More parameters need more data. The bias-variance trade from lecture 6 applies to clustering too.

> [!QA]
> Q: K-means++ guarantees O(log k) expected approximation. What does that actually promise on the toy?
> A: k=2, so log 2 is a constant: the guarantee says the expected distortion is within a small constant factor of optimal. Optimal here is 4.0. The guarantee promises the seeding lands near 4.0 in expectation, not that any single run hits it. On the toy, first seed anywhere, second seed lands in the opposite clump with probability proportional to squared distances (64+81+100 vs 1+1): overwhelmingly. The guarantee is about the seeding distribution, and k-means' own local descent can only improve on the seeded start.
> Follow-up: Why can no algorithm promise optimal efficiently?
> A: The lecture cites NP-hardness: finding the global minimum distortion is NP-hard in general dimension. Any efficient algorithm must settle for approximation. K-means++ is the best cheap answer: provably close from seeding alone.

6. **The elbow.** Distortions 125.5, 4.0, 2.7, 1.5: bend at k=2.
   Heuristic, often ambiguous.
7. **GMM.** Soft assignments via responsibilities: 70/30 splits.
   K-means is the hard limit.
8. **The honest price.** Local minima, spherical bias, heuristic k,
   EM needed for GMM, both die on crescents.
9. **The audit.** One round verified: means 2 and 11. k=1
   distortion is 125.5 (corrected from 121.5).
10. **The mean.** Derivative of the squared term forces the mean.
    Both steps optimal, the joint problem local.
11. **Responsibilities, worked.** Point 3: 1.0 vs 0.0. Midpoint
    6.5: 0.5/0.5. The slide is the softness.
12. **Crescents.** Non-convex shapes break k-means structurally.
    Plot first, then choose the algorithm.
13. **The lottery.** Seeds decide. K-means++ plus 10 restarts in
    production.

## What is used where

**K-means runs everywhere labels are absent and shapes are
tame.** Customer segmentation is the canonical job. Image color
quantization: cluster pixels into k colors to compress. Embedding
pipelines: k-means over word or product vectors to build
candidate sets. Sklearn's KMeans defaults to k-means++ seeding
with 10 restarts: the production configuration is the lesson's
prescription. GMMs appear wherever soft assignment matters:
speaker diarization (who spoke when, with overlap) and
background/foreground separation.

## Watch next

<div class="video-block"><div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/4b5d3muPQmA" title="StatQuest: K-means clustering" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div><p class="video-cap">Explainer: StatQuest, K-means clustering. Josh Starmer works the algorithm step by step with his usual diagrams. Watch after the toy rounds.</p></div>

## Official sources and further reading

**Official:**
- Lecture 9 video, Stanford Online YouTube:
  - [Chris Ré runs](https://www.youtube.com/watch?v=bSmIGBCoffA)
  k-means live, proves convergence to local minima, presents
  k-means++ with its approximation guarantee, and sets up GMMs.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the full
  k-means and GMM treatment.

**Caveats from these sources.** The lecture calls k-means "ad hoc"
deliberately: it is the intuitive algorithm, not a derived one.
The k-means++ attribution (Stanford graduate students Arthur and
Vassilvitskii) and the sklearn-default status are the lecture's.
The toy runs in this lesson are original miniatures of the
lecture's live demos. GMM fitting (EM) is deferred to lecture 10
by the lecture's own ordering.

## Connections to the other courses

- **CS229 L05:** GDA's Gaussians with known labels. GMM is GDA
  with the labels hidden.
- **CS229 L10:** EM: the algorithm that fits GMMs, and PCA for
  visualizing clusters.
- **CS229 L06:** the elbow as model selection. Distortion vs k as
  a bias-variance curve.
- **CS224N:** clustering word vectors: k-means on embeddings.
