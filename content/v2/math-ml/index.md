# Mathematical Foundations of Machine Learning

Course home for math-ml: a deep crash course on the mathematics behind
machine learning. The course follows the NPTEL/IISc course noc26-cs02,
"Mathematical Foundations of Machine Learning", by Prof. Prathosh A P
(IISc 2021-2026, IIT Delhi 2017-2021). Twelve weeks. The first offering
ran Jan 19, 2026 to Apr 10, 2026, with the exam on Apr 25, 2026. The
second offering began in August 2026.

Recordings live in the YouTube playlist
PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu. The playlist was enumerated on
2026-10-06: 89 entries (1 intro, 69 lectures Lec 01-Lec 69, 19
tutorials).

## The ten units

| Unit | Topic | Lesson pages |
|---|---|---|
| U01 | Mathematical language and computational setup: sets, functions, notation, units, sums, logs, vectors, honest plots, assertions, precision, reproducibility | [Lesson 01](lessons/u01/lesson-01-mathematical-language.html), [keys](lessons/u01/keys.html) |
| U02 | Linear geometry and decompositions: subspaces, rank, projections, eigenvectors, SVD, covariance, conditioning | [Lesson 02](lessons/u02/lesson-02-linear-geometry-and-decompositions.html), [keys](lessons/u02/keys.html) |
| U03 | Calculus and optimization: partials, chain rule, convexity, gradient descent, Newton, constraints, KKT, learning rates | [Lesson 03](lessons/u03/lesson-03-calculus-and-optimization.html), [keys](lessons/u03/keys.html) |
| U04 | Probability, density, and estimation, in three source blocks: sample spaces and variables (a), entropy/KL/MLE (b), risk framework, Bayes, EM, nonparametric methods (c) | [Block a](lessons/u04/lesson-04a-first-source-block.html), [Block b](lessons/u04/lesson-04b-second-source-block.html), [Block c](lessons/u04/lesson-04c-third-source-block.html), [keys a](lessons/u04/keys-04a.html), [keys b](lessons/u04/keys-04b.html), [keys c](lessons/u04/keys-04c.html) |
| U05 | Risk, regularization, and generalization: empirical and population risk, bias and variance, penalties, priors, CV, leakage, shift | [Lesson 05](lessons/u05/lesson-05-risk-regularization-generalization.html), [keys](lessons/u05/keys.html) |
| U06 | Linear models, kernels, and margins: OLS, logistic, softmax, GLMs, kernels, Gram matrices, SVM, the dual | [Lesson 06](lessons/u06/lesson-06-linear-models-kernels-margins.html), [keys](lessons/u06/keys.html) |
| U07 | Trees and ensembles: decision regions, impurity, pruning, bagging, random forests, boosting, imbalance, baselines | [Lesson 07](lessons/u07/lesson-07-trees-ensembles.html), [keys](lessons/u07/keys.html) |
| U08 | Neural and sequence architectures: MLP, backprop, CNNs, RNNs, gates, attention, transformers, normalization | [Lesson 08](lessons/u08/lesson-08-neural-sequence-architectures.html), [keys](lessons/u08/keys.html) |
| U09 | Unsupervised and latent models: k-means, PCA, reconstruction, held-out evaluation (the EM and GMM leaves live in the U04c block) | [Lesson 09b](lessons/u09/lesson-09b-unsupervised-leaves.html), [keys](lessons/u09/keys-09b.html) |
| U10 | Generative bridge and mathematical synthesis: VAEs, GANs, objectives, samples versus densities, research critique, source-gap audit | [Lesson 10](lessons/u10/lesson-10-generative-bridge-synthesis.html), [keys](lessons/u10/keys.html) |

## How to use this course

1. Take the [placement diagnostic](diagnostics/diagnostic-01-math-language.html) first. It tells you which units to read slowly and which to skim.
2. Read the units in order. U04 runs as three source blocks (a, b, c). U09 spreads across the U04c block (EM, GMMs, latent models) and the U09b lesson (k-means, PCA).
3. Answer keys live in separate files, one per lesson. Attempt every exercise before you open the key.
4. Labs give three computation tasks per unit. The [interview bank](interview/interview-u02-questions.html) gives breadth questions, deep ladders, quantitative drills, and oral defenses per unit, with keys separate. Two [capstones](capstones/capstone-research-kernel-margins.html) close the course.
5. The [crash course](crash-course.html) compresses all ten units into thirty minutes. The [cheatsheet](cheatsheet.html) holds every formula, shape, and trap on one dense page.
6. Notation is fixed in [notation and shapes](notation_and_shapes.html). Terms are fixed in the [glossary](glossary.html).

## Prerequisites

Per-unit prerequisites come from the shared bridge modules P01-P24, in
`~/workspace/stanford-frontier-ai/v2-pack/shared/prerequisites/`:

| Unit | Prerequisite modules |
|---|---|
| U01 | P01 numeracy, algebra, notation, P02 Python and scientific software |
| U02 | P03 vectors, geometry, linear maps, P04 spectral and numerical linear algebra |
| U03 | P05 scalar and multivariable calculus, P09 optimization and constrained problems |
| U04 | P06 probability, events to distributions, P07 statistical estimation and uncertainty, P08 information theory and density objectives |
| U05 | P07 statistical estimation and uncertainty, P09 optimization and constrained problems, P10 ML foundations and evaluation |
| U06 | P03 vectors, geometry, linear maps, P07 statistical estimation and uncertainty, P09 optimization and constrained problems |
| U07 | P06 probability, events to distributions, P07 statistical estimation and uncertainty, P10 ML foundations and evaluation |
| U08 | P11 neural networks and autodiff, P12 PyTorch, tensors, numerical stability, P13 language and sequence modelling, P14 transformer mechanics |
| U09 | P04 spectral and numerical linear algebra, P07 statistical estimation and uncertainty, P18 Bayesian inference, latent variables, sampling |
| U10 | P08 information theory and density objectives, P18 Bayesian inference, latent variables, sampling, P22 experimental method and research literacy |

Read the modules in dependency order: P01 first, then P02-P06, then the
rest. Local remediation items R01-R78 for this course live in
[prerequisites](prerequisites.html).

## Course state (2026-10-06)

- 120 coverage rows: all taught and assessed.
- Source attribution: pending. Playlist references are title-level
  only. No transcript inspection is claimed.
- RUN 6 adversarial audit: FAIL (3 major, 7 minor findings). Fix
  attempt 1 of 3 repaired F-01 through F-09. F-10 was rejected as a
  false positive with evidence. A re-audit of the touched files was in
  progress.
- 25 figures, all computed with matplotlib from the render scripts in
  the build root. No AI-generated images.
