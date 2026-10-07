# Answer keys: U08

Unit: math-genmodels-U08. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers below. Red flags name the common failure. Rubrics say
what earns full credit.

## B1

Strong: held-out likelihood scores points the model did not
train on. Model A: -10.1073 over the 5 points. Red flag:
reporting training likelihood. Rubric: the definition plus the
number.

## B2

Strong: log p = ELBO + KL(q||posterior). Toy: -1.7371 =
-1.8621 + 0.125. Red flag: calling the bound the likelihood.
Rubric: the identity and the three numbers.

## B3

Strong: fraction of true modes with samples inside the radius.
Model B: 1/2 = 0.5. Red flag: confusing coverage with
precision. Rubric: the definition and the number.

## B4

Strong: precision = fraction of samples on the manifold.
recall = fraction of modes covered. A: 0.4708, 1.0. B:
0.9555, 0.5. Red flag: reporting one without the other.
Rubric: both definitions and all four numbers.

## B5

Strong: truth bimodal (mean 0, var 4.25), model A Gaussian
with the same moments: 1-D FID = 0, shapes completely
different. Red flag: "FID 0 means identical." Rubric: the
construction with the FID computation.

## B6

Strong: nearest training distance per sample. A pile at zero
signals a copy. Memorizer 0.0066, honest model 0.4666. Red flag:
eyeballing samples instead. Rubric: the test and both
numbers.

## L1 ladder

1. Sum of log densities on unseen points.
2. A: -10.1073, B: -70.949.
3. B assigns near-zero density to left-mode points (0.0006
   at x = 0.1 versus 0.1933).
4. Exact -1.7371 versus bound -1.8621: the bound's truth is
   unknown within the 0.125 gap.
5. Compare exact with exact, bound with bound, never a bound
   against an exact number without labeling.

## L2 ladder

1. Precision on-manifold fraction, recall mode fraction.
2. A: 0.4708/1.0, B: 0.9555/0.5.
3. FID = 0 with matched moments but wrong shape: moments
   cannot see bimodality.
4. NN distances 0.0066 (copying) versus 0.4666 (honest).
5. Each metric has a blind spot: likelihood misses
   sharpness, precision misses coverage, FID misses shape,
   samples hide copying. The suite covers the blind spots.

## A1

Strong: 1-D Frechet between N(m1,s1) and N(m2,s2) is
(m1-m2)^2 + (s1-s2)^2. With matched moments: 0. Discrete
law: mass 1/3 each at -sqrt(6.375), 0, +sqrt(6.375): mean 0,
variance 4.25, three spikes. Red flag: claiming the Gaussian
is the only FID-0 model. Rubric: the formula plus a valid
discrete construction.

## A2

Strong: SE of the mean = sqrt(var/n) = sqrt(4.25/5000) =
0.0292. Observed -0.0317 is 1.09 standard errors from 0.
Consistent with truth. Red flag: calling -0.0317 a bias.
Rubric: the SE computation and the ratio.

## D1

Strong: bug: Inception features were trained on natural
images. They do not represent X-ray structure, so FID
measures distance in an irrelevant feature space. The score
is blind to the artifacts humans see. Fix: replace the
feature extractor with a domain model (or use
precision/recall on domain features), and add a human rating
protocol. Test: the FID-0-style trap: construct two X-ray
sample sets humans rank differently and confirm the metric
separates them. If not, the metric is void for this domain.
Red flag: tuning the generator to the Inception FID. Rubric:
the feature-mismatch mechanism, the fix, the separation
test.

## T1

Strong: the NN memorization test needs the training set.
Redesign: use a held-out reference split as the manifold
proxy for precision/recall, plus membership-inference
probes that only need model outputs. Lose: direct copying
detection (a model can memorize train while looking clean
against held-out). State the residual risk explicitly.
Red flag: claiming the proxy is equivalent. Rubric: the
redesign plus the named loss.

## T2

Strong: choose sample precision/recall as a pair, or if
forced to one number, held-out likelihood with exact
densities where available. Defense: likelihood is the only
metric that scores probability where data lands. Hides:
sample sharpness and mode dropping (likelihood rewards
coverage, B's failure is invisible in the average). State
the hiding explicitly beside the number. Red flag: picking
FID without the caveat. Rubric: the choice, the defense,
the named hiding.

## R1

Strong: attack 1: FID sees only feature means and
covariances: the unit's FID-0 counterexample proves a
perfect score with the wrong shape. Attack 2: "best
generative model" needs samples, coverage, cost, and the
task: FID is one projection. Fair experiment: fixed
features, precision/recall plus held-out likelihood where
densities exist, compute-matched budgets, uncertainty over
seeds. Red flag: the FID as verdict. Rubric: both attacks
plus the suite.
