# AUDIT — CS229 L01-L06 chapter plates (builder report)
# Built 2026-10-06 by the chapter-plate builder subagent.
# Role: BUILDER, figure stage (per build/PIPELINE.md). Auditor is a different agent.

## What was built

24 dense chapter plates (4 per lesson, one per major concept), all in
`content/v2/cs229/assets/`, all 960x600 SVG matching the mse435 bar
(`content/v2/mse435/assets/plate-l01-chap-*.svg`): warm paper #F7F4EE,
left panel #FFFDF8, center #E7F1F8, right #E7F4EF, tradeoff band,
one-connection footer, caption format identical to the bar.

Generator: `build/chap_plates_cs229_l01_06.py`. It recomputes every
number from the lesson text in python3 and asserts before writing any
plate (all asserts pass; run output: "all number checks passed").
Overflow guard: every row measured with PIL against its panel width,
every panel's last row asserted inside the panel. 7 plates visually
checked via cairosvg PNG renders.

Caption lines inserted into the six lessons at the end of each
concept's section (verified: 4 per lesson, anchors unique, captions
match the bar's title string verbatim).

## L01 — l01-introduction.md

| # | File | Concept (section) | Left: cost without | Center: stored object | Right: cost with | Bottom: tradeoff |
|---|---|---|---|---|---|---|
| C1 | plate-l01-chap-definitions.svg | Samuel 1959 / Mitchell 1997 ("The job") | hand rules: 500 engineer-hours, 10 rules/day vs 100 evasions/day, 63% false positives | Mitchell's T, E, P: sort email, 10,000 labels, accuracy 94% then 99% on held-out mail | measured learning: precision vs recall is a product choice | adaptability costs data: 10,000 labels = 5 labeler-days |
| C2 | plate-l01-chap-paradigms.svg | The three paradigms | labels cost 5 labeler-days per 10,000, labels are the ceiling | labeled pairs = supervised, raw data = unsupervised, actions+rewards = reinforcement | regression $200/sq ft, clusters split at 10.5, RL 29M games or $500,000 in falls | RL fits where trials are cheap |
| C3 | plate-l01-chap-ruleloss.svg | Why the old way broke | 500 engineer-hours to stand up, 25 per wave forever | humans 10/day vs spammers 100/day, 63% false positives at 1,000 rules | retrain on this week's mail, no new code, 10,000 labels = 5 labeler-days | rules still win under 100 rules in a still world |
| C4 | plate-l01-chap-price.svg | The honest price | one clever engineer, microseconds per run, dead every Monday | data, compute, trust | filter survives Monday, failures unpredictable | a learned model adapts, but nobody can read its rules back |

## L02 — l02-linear-regression.md

| # | File | Concept (section) | Left | Center | Right | Bottom |
|---|---|---|---|---|---|---|
| C1 | plate-l02-chap-setup.svg | Supervised setup + loss chip | no score: hand-picked line, knobs with no judge | J(theta) = 1/(2m) sum of squared errors | 3-house toy J = 6.33, then 3.32 after one step | one $1M miss contributes 10^12 and drags the line |
| C2 | plate-l02-chap-gd.svg | Gradient descent + learning rate | calculus only: needs a matrix inverse, dies on neural networks | error x feature summed: (0,0) to (0.167, 0.383) | alpha 0.01 crawls, 0.1 converges, 1.5 explodes (4,-8,16,-32,64) | when the loss bounces, alpha is too high |
| C3 | plate-l02-chap-sgd.svg | SGD + mini-batch | full passes: 80B multiply-adds per step at 2B examples | one example per step, unbiased: (-2+-6)/2 = -4 | m times cheaper, mini-batch 32-256 the production middle | noise is the price of speed; shuffle every epoch |
| C4 | plate-l02-chap-normaleq.svg | Normal equations | hundreds of steps, alpha tuned by hand | theta = (X'X)^-1 X'y: det 6, theta (1/3, 3/2), J = 0.028 exact | O(n^3): 10^12 ops at n = 10,000; singular on redundant features | no alpha, no iterations, but only small n, full rank, linear models |

## L03 — l03-logistic-regression.md

| # | File | Concept (section) | Left | Center | Right | Bottom |
|---|---|---|---|---|---|---|
| C1 | plate-l03-chap-mle.svg | MLE (coin) | line + 0.5 threshold: one outlier flips a diagnosis 1.75 to 2.9 | coin scoreboard L(phi) = phi^7(1-phi)^3 peaks at 0.7, 2.3x the fair coin | log version: -6.11 beats -6.93; logs cure 0.5^10000 underflow | write P(data|knobs) and maximize it |
| C2 | plate-l03-chap-gaussian.svg | Gaussian noise to squares | least squares felt arbitrary | y = theta'x + noise: log likelihood = const - sum errors^2 | maximizing likelihood = minimizing squared errors | change the noise assumption, get a different loss |
| C3 | plate-l03-chap-sigmoid.svg | Sigmoid + logistic regression | line predicts 0.27 and 0.55, no walls at 0/1 | g(-2)=0.12, g(0)=0.5, g(2)=0.88, smooth monotone | gradient (h-y)x; tumor toy size knob to 0.25 | sigmoid+squared loss is the flat-gradient trap; sigmoid+cross-entropy is the design |
| C4 | plate-l03-chap-newton.svg | Newton / IRLS | GD: 500-2,000 tuned iterations, ~10^9 ops | theta := theta - J'/J'': one step on theta^2 vs GD's 38 | weighted least squares at O(nd^2+d^3): 408,000 ops at d=20, 10^27 at d=1B | few expensive steps lose to many cheap steps |

## L04 — l04-glms-softmax.md

| # | File | Concept (section) | Left | Center | Right | Bottom |
|---|---|---|---|---|---|---|
| C1 | plate-l04-chap-expfam.svg | Exponential family | least squares and logistic felt like separate inventions | p(y; eta) = b(y) exp(eta'T(y) - a(eta)) | Gaussian and Bernoulli both fit; mean = a'(eta): phi 0.8 gives eta 1.386 | the mean falls out of the normalizer by differentiation |
| C2 | plate-l04-chap-glm.svg | GLM recipe | deriving each model from scratch | 1. pick distribution, 2. eta = theta'x, 3. predict a'(eta) | Gaussian -> least squares, Bernoulli -> logistic, Poisson doubles 100 to 200 hits | eta must be linear in x: curved boundaries impossible |
| C3 | plate-l04-chap-softmax.svg | Softmax | three sigmoids (0.88,0.73,0.62) sum to 2.23 | e^z_j / sum e^z_c: (0.63,0.23,0.14) | four whys: GLM-dictated, smooth, max-entropy, convenient; temperature dial | O(k) per prediction; k = 50,000 words is the bottleneck |
| C4 | plate-l04-chap-xent.svg | Cross-entropy | no multi-class loss | -log(p_true): 0.46 at 0.63, 3.00 at 0.05 | gradient (p - one-hot) x, errors sum to zero | unbounded: mislabeled example at p=10^-6 costs 13.8 |

## L05 — l05-gda-naive-bayes.md

| # | File | Concept (section) | Left | Center | Right | Bottom |
|---|---|---|---|---|---|---|
| C1 | plate-l05-chap-genframe.svg | Generative framing | averages only: 40 kg closer to 4 than 4,000, so "cat" (wrong) | Bayes' rule: pick max p(x|y) p(y) | two roads to the same line at y = 1; diverge when class models are wrong | class models cost assumptions, buy one-pass training |
| C2 | plate-l05-chap-gda.svg | GDA | logistic draws the boundary directly | per-class Gaussians, class averages, one pass; bells cross at x = 22 | w = Sigma^-1(mu_1 - mu_0); 5,251 vs 10,301 knobs at d = 100 | shared spread buys linearity and half the knobs |
| C3 | plate-l05-chap-naivebayes.svg | Naive Bayes | GDA needs real features; words are indicators | words independent given class, fit by counting: spam 0.32 | spam wins 64 to 1; 10M emails/day on one core | independence lie double-counts; ranking survives, calibration does not |
| C4 | plate-l05-chap-laplace.svg | Laplace smoothing | unseen word scores 0 in both classes: blind | add 1 to every count: (0+1)/(10+2) = 1/12 | no vetoes; scored in log space (-1.14, -5.30) | the price of humility is small |

## L06 — l06-bias-variance.md

| # | File | Concept (section) | Left | Center | Right | Bottom |
|---|---|---|---|---|---|---|
| C1 | plate-l06-chap-biasvar.svg | Bias-variance | fit harder: degree-10 train 0.00, test 4.7 | error = bias^2 + variance + noise: 0.01 and 0.027 on the toy | U-curve: degree 3 wins at 0.18, degree 15 falls to 4.9 | data fights variance; flexibility fights bias |
| C2 | plate-l06-chap-doubledescent.svg | Double descent | the U-canon: big models must fail | interpolation threshold p ~ n: knife-edge fit | 15% spikes to 25% then falls to 8% | worst place to sit: just past the sweet spot |
| C3 | plate-l06-chap-split.svg | Train/dev/test | training error lies: polynomial scored 0.00 and lied | train fits, dev compares, test reports once; 6000/2000/2000 | dev 0.18, test 0.21; k-fold mean 0.21 when data is scarce | every decision made on a dataset contaminates it |
| C4 | plate-l06-chap-ridge.svg | Ridge | whipsawing coefficients; X'X singular at det near -0.0001 | J = squared loss + rho||theta||^2; always invertible for rho > 0 | test error 4.7 at rho 0, 0.9 at rho 1, 2.1 at rho 100 | ridge costs bias on purpose; tune rho on dev, never training |

## L02 fixes (done, same files, no extra round)

1. **plate-l02-feature-scaling.webp re-rendered.** Old render: PIL footer
   caption overlapped by the valley ellipse and zigzag strokes (footer
   text crossed by figure strokes at bottom). New render
   (`build/regen_feature_scaling.py`): all figure content confined to
   y <= 1000, clean footer band at y 1060-1280, caption stamped centered
   inside the band. Pixel-level self-check in the script asserts zero
   dark pixels in the gutter (y 1000-1060). Visually verified via read
   of the webp: title, two panels, valley ellipse with zigzag fully
   inside, circle with straight arrow, center pill, clean footer.
   Content unchanged (living area 1000-3000, bedrooms 1-5, narrow
   valley, round bowl, one alpha for every knob).
2. **L226 ASCII block trimmed to 12 lines.** Dropped the two blank lines
   after the "knobs" line and after the "start" line; the single trace
   (houses -> knobs -> start -> errors/grads -> step -> after) is intact.
   Verified 12 lines with sed/awk.

## Compliance notes for the figure auditor

- F4 (chapter plate per concept): 24 plates, 4 per lesson, each at the
  end of its concept's section.
- F5: captions name the source ("original synthesis of the lesson") and
  the shell is not applicable to chapter plates (bar uses no shell on
  chapter plates either); the bar's exact caption string format is used.
- F6: reject list enforced — no robot/brain/glowing network/stock/clip
  art; no gradient/glow/shadow (flat fills only); no watermark/logo;
  every number code-computed (asserts in the generator).
- Spec: warm paper #F7F4EE, sans stack with serif only for pure formula
  lines, dense four-region layout (left = cost without the rule,
  center = the stored object, right = cost with the rule, bottom =
  tradeoff in one line), footer = the one connection in one sentence.
- l07-l17 untouched. build/build.py not run. Media pipeline not used
  (plates are code-generated SVGs, permitted for image generation).

## Files touched

- 24 new SVGs in `content/v2/cs229/assets/`
- caption lines added to the 6 lesson markdowns (4 each)
- `content/v2/cs229/assets/plate-l02-feature-scaling.webp` re-rendered
- `content/v2/cs229/l02-linear-regression.md` ASCII block trimmed
- `build/chap_plates_cs229_l01_06.py` (plate generator + number asserts)
- `build/insert_chap_plates_l01_06.py` (caption insertion)
- `build/regen_feature_scaling.py` (feature-scaling re-render)
