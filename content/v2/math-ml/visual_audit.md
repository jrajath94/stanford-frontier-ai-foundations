# visual_audit.md, math-ml RUN 1

Date: 2026-10-06. Audit table per the visual spec. Medium ladder applied:
the first medium that passes the four tests is used. All plates are
original toys with computed numbers. No source figure was copied.

## Unit-to-figure map (U01 lesson)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u01 | one input gives one output | domain set | image set | f01 | ASCII | original |
| u02 | every symbol has a contract | bare symbols | symbol table | f02 | table | original |
| u03 | units must match | bare numbers | tagged equation | f03 | equation block | original |
| u04 | accumulation folds items one by one | list [2,5,7] | 14 and 70 | f04 | ASCII | original |
| u05 | log space keeps tiny products usable | 0.0 underflow | -921.034 finite | f05 | table | original |
| u06 | arrows chain head to head | v, w | v + w | f06 | ASCII | original |
| u07 | reduction deletes the named axis | shape (2,3) | shape (3,) / (2,) | f07 | ASCII | original |
| u08 | known answers judge code | three inputs | three asserted outputs | f08 | code block | original |
| u09 | honest axes start at zero | six rows | scatter, axes at 0 | f09 | PNG | original, rendered |
| u10 | guards name bad values | three inputs | pass/fail verdicts | f10 | table | original |
| u11 | fewer digits, larger error | float32 | float64 | f11 | table | original |
| u12 | seed fixes the sequence | seed 7 | identical outputs | f12 | table | original |
| u13 | naming and checking beats silent numbers | costs without rule | costs with rule | f13 | ASCII | original |

## Render status

- f09: rendered with matplotlib 3.6.3 (Agg), dpi 150, file
  visuals/u01/f09_honest_scatter.png, opened and read 2026-10-06:
  six dots visible, axes start at 0, labels carry units. Status: verified.
- f01-f08, f10-f13: embedded in the lesson text. Status: present, not
  separately rendered.

## Logged visual decisions

1. SVG lesson plates per visual_system_generic.md are not built in RUN 1.
   ASCII, table, equation, and code media are the first media on the
   ladder that pass the four tests for these claims. The spec's ladder
   rule makes them compliant. SVG plates for architecture-heavy units
   (U08) are planned for later runs.
2. No interactive toys in RUN 1. No canvas control is needed for these
   claims.
3. Symbol reuse: the vector chip v = [3, 4] keeps its values across f06,
   f08, and the norm checks, per the object-path rule.
4. No figure was copied from any instructor or third-party source.
   Copying is not permitted without a license check. All plates are
   original.

## Page audit result (lesson-01)

Every heading maps to exactly one figure id above. No unit is text-only.
No architecture unit appears in U01, so no before/after architecture
plate is required yet. Status: pass.

## RUN 2 audit (2026-10-06)

### Unit-to-figure map (U02 lesson)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u02-c01 | dependent arrows share one world | two arrows | one line, dim 1 | f01 | ASCII | original |
| u02-c02 | rank 1 keeps one line, kills one | input plane | output line | f02 | ASCII | original |
| u02-c03 | dot product measures agreement | v, w | cos 0.6 | f03 | ASCII | original |
| u02-c04 | shadow plus residual equals the arrow | b, line a | p + r, r.a = 0 | f04 | PNG | original, rendered |
| u02-c05 | a matrix moves every corner | unit square | rotated square | f05 | PNG | original, rendered |
| u02-c06 | determinant is the area scale | unit square | area 3 / area 0 | f06 | ASCII | original |
| u02-c07 | undo works, or best-possible undo | D x = b | x, residual 0 | none (code) | code block | original |
| u02-c08 | some directions survive the matrix | E, arrows | stretch 3 and 1 | f07 | ASCII | original |
| u02-c09 | two stretch factors, one kept | sigma [5, 3] | error 3.0 | f08 | PNG | original, rendered |
| u02-c10 | eigenvalue signs name curvature | P vs Q | bowl vs saddle | f09 | ASCII | original |
| u02-c11 | co-movement has directions | 4 points | eigendirections | f10 | PNG | original, rendered |
| u02-c12 | kappa amplifies input wobble | b, b + db | x, x + dx | none (code) | code block | original |

Note: the lesson file labels figures f01-f10 per lesson scope. The
audit maps them to u02-c01..c12. C07 (inverse) and C12 (conditioning)
carry their checkable state in code blocks (residual 0.0, bound check
1999.9 <= 4002.0), which the honest-medium rule prefers over a
drawing. C08's eigenvector plate is f07.

### Unit-to-figure map (U04a source block, Lec 02-10)

Figure labels are the lesson's own (sb01-sb04). Sections without a
labeled figure carry their checkable state in code or equations.

| Section | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| SB01 | a random variable maps outcomes to numbers | image | 0/1 | sb01 | ASCII | original |
| SB02 | PMF rows sum to 1. CDF differences recover PMF | raw X | table, total 1.0 | none (code check) | code block | original |
| SB03 | sum to forget, renormalize to zoom | joint table | marginals, conditionals | sb02 | ASCII | original |
| SB04 | expectation is the weighted center | die faces | 3.5, var 35/12 | none (equations) | equation block | original |
| SB05 | a sample is not the distribution | 8 draws | mean 0.5 vs 0.3 | none (code) | code block | original |
| SB06 | IID licenses generalization | draws | product form | sb03 | PNG f01_iid_sample.png | original, rendered |
| SB07 | the data picks the parameter | flips | p_hat 0.75 | none (code) | code block | original |
| SB07b | histogram density from 8 points | 8 points | counts [2,2,2,2] | sb04 | PNG f02_histogram_density.png | original, rendered |
| SB08 | finite n leaves a gap | p(x) truth | p_hat, 1/sqrt(n) | none (equation) | equation block | original |

### Render status

- visuals/u02/f04_projection.png, f05_transform.png, f08_svd_values.png,
  f10_covariance.png: rendered matplotlib 3.6.3 (Agg), dpi 150, opened
  and read 2026-10-06. f04: arrows b, a, p, r visible, r.a = 0 label.
  f05: square and rotated square visible. f08: bars 5 and 3 with error
  label. f10: 4 points and both eigendirections visible. Status: verified.
- visuals/u04/f01_iid_sample.png, f02_histogram_density.png: rendered
  same toolchain, opened and read 2026-10-06. f01: truth bars (0.7,
  0.3) vs sample bars (0.5, 0.5). f02: 4 bins with counts. Verified.
- ASCII/equation/code figures: embedded in lesson text. Present, not
  separately rendered.

### Logged visual decisions

1. Eigenvector plate (C08) stays ASCII: the state change is two
   numbers (stretches 3 and 1) on named directions. A PNG adds no new
   checkable state. Logged per the honest-medium rule.
2. Inverse/pseudoinverse (C07) and conditioning (C12) plates are code
   blocks: the claim is a computed residual/bound, best checked as
   numbers, not as a drawing.
3. SB02/SB04 plates are equation blocks: the PMF sum and the 35/12
   identity are the checkable state.
4. No figure copied from any instructor or third-party source.

## RUN 3 audit (2026-10-06)

### Unit-to-figure map (U03 lesson)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u03-c01 | freeze-frame slopes agree paper vs code | analytic 7, 8 | fd 7.0, 8.0 to 8 digits | f01 | table | original |
| u03-c02 | two slopes multiply through the chain | x=2, links | dy/dx = 150 | f02 | ASCII | original |
| u03-c03 | one toy, three slope objects | f, g | grad, J, H tables | f03 | equation block | original |
| u03-c04 | each term eats part of the curve | T1 error 0.1487 | T2 error 0.0237 | f04 | table | original |
| u03-c05 | convexity is one bowl | one bowl | two bowls + saddle | f05 | PNG | original, rendered |
| u03-c06 | each point votes with error strength | w = 1 | grad -2.25, w* = 1.3 | f06 | equation block | original |
| u03-c07 | GD glides, SGD wobbles to the same bowl | J = 9 | GD J = 0.1038, SGD J = 3.2022 | f07 | PNG | original, rendered |
| u03-c08 | Newton reaches the bowl in 4 steps | x0 = 1.0 | 1.224745 | f08 | PNG | original, rendered |
| u03-c09 | the free bottom is forbidden | x = 0 free | x* = 1 on the fence | f09 | ASCII | original |
| u03-c10 | four checks certify the fenced optimum | unchecked | 4 passes, gap 0 | f10 | equation block | original |
| u03-c11 | step size picks the regime | one eta | slow, fast, exploded | f11 | PNG | original, rendered |
| u03-c12 | two evaluations judge the gradient | analytic | rel err 1.6e-10 | f12 | code block | original |

Note: the lesson file labels figures f01-f12 per lesson scope. The
audit maps them to u03-c01..c12. C01, C04, C06, C10, C12 carry their
checkable state in tables, equation blocks, or code, which the
honest-medium rule prefers over a drawing. C02 and C09 are ASCII:
the state change is a labeled chain and a labeled fence.

### Unit-to-figure map (U04b source block, Lec 11-19)

Figure labels are the lesson's own (sb05-sb12). Sections without a
labeled figure carry their checkable state in code or equations.

| Section | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| SB09 | the fair coin is the most surprising | two coins | H 0.6931 vs 0.5623 | sb05 | table | original |
| SB10 | KL is the surprise gap, direction matters | p, q | 0.0823 vs 0.0872 | sb06 | PNG f03_entropy_kl.png | original, rendered |
| SB11 | the scan minimum sits at the truth | free t | (0.7, 0.0) | sb07 | code block | original |
| SB12 | the data votes 3 to 1 | flips | theta_hat 0.75 | sb08 | ASCII | original |
| SB13 | the data picks the bell | 4 points | N(2.2, 0.05) | sb09 | PNG f04_mle_gaussian.png | original, rendered |
| SB14 | the estimate is the empirical share | counts 4,2,4 | [0.4, 0.2, 0.4] | sb10 | table | original |
| SB15 | the cloud gives its center and stretch | 6 points | mu, Sigma, eig | sb11 | equation block | original |
| SB16 | two humps, one valley | mixture | p(2.5) = 0.01753 | sb12 | equation block | original |

### Render status

- visuals/u03/f01_gd_path.png, f02_convex.png, f03_newton.png,
  f04_learning_rates.png: rendered matplotlib 3.6.3 (Agg), dpi 150,
  opened and read 2026-10-06. f01: GD line and SGD dots visible,
  labels carry the measured J values. f02: one bowl and two bowls
  with minima and saddle dots visible. f03: h(x) curve with numbered
  Newton points 0-5 visible. f04: three traces on symlog, eta labels
  visible. Status: verified.
- visuals/u04/f03_entropy_kl.png, f04_mle_gaussian.png: rendered
  same toolchain, opened and read 2026-10-06. f03: p and q bars with
  both KL directions in the footer. f04: histogram, fitted bell, and
  mu_hat marker visible. Status: verified.
- One label correction during render: f01 SGD legend first read
  J=3.2143, recomputed J=3.2022, re-rendered and re-read.
  See errors.md E-009.
- ASCII/equation/table/code figures: embedded in lesson text.
  Present, not separately rendered.

### Logged visual decisions

1. C01/C04 plates stay tables: the claim is numeric agreement
   between two methods, best checked as numbers.
2. C02/C09 plates stay ASCII: a two-link chain and a one-line fence
   need labels, not pixels.
3. C03/C06/C10/SB15/SB16 plates are equation blocks: the tables
   themselves are the checkable state.
4. C12 plate is a code block: the six-line check is the claim.
5. SB11 plate is a code block: the scan result is the claim.
6. No figure copied from any instructor or third-party source.

## RUN 4 audit (2026-10-06)

### Unit-to-figure map (U05 lesson)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u05-c01 | the scoreboard is not the future | 4 points | R_hat 0.25, R 0.25 by luck | none (numbers) | equation block | original |
| u05-c02 | the loss chooses the optimum | y = [0,1,1] | 0.2222 vs 0.3333 vs 0.6365 | none (table) | table | original |
| u05-c03 | assumptions do the work, not n | n = 4 | MSE 0.0525 vs 0.21 | none (numbers) | equation block | original |
| u05-c04 | bias trades against variance | MLE 0.0525 | Laplace 0.02778 | f01 | PNG | original, rendered |
| u05-c05 | the tax moves the answer | w 0.9286 | w 0.8125, objective 3.4375 | none (table) | table | original |
| u05-c06 | the prior is the penalty | ridge 0.8125 | MAP 0.8125 | none (numbers) | equation block | original |
| u05-c07 | the locked model scores worse unseen | train 0.4821 | test 0.8653 | none (table) | table | original |
| u05-c08 | the penalty wins on held-out folds | lambda 0 | lambda 0.5, CV 1.1806 | f04 | PNG | original, rendered |
| u05-c09 | train falls, test turns up | degree 1 | degree 3, test 28.56 | f02 | PNG | original, rendered |
| u05-c10 | more data shrinks the gap | gap 0.31 at n=3 | gap 0.11 at n=40 | f03 | PNG | original, rendered |
| u05-c11 | the perfect score measured the leak | train 1.0 | deploy 0.6 | none (table) | table | original |
| u05-c12 | a new x range multiplies the error | train 0.0986 | shift 0.5548 | none (table) | table | original |

Note: the lesson file labels figures f01-f04 per lesson scope. The
audit maps them to u05-c04 (f01), u05-c09 (f02), u05-c10 (f03),
u05-c08 (f04). C01-C03, C05-C07, C11, C12 carry their checkable
state in tables or equation blocks, which the honest-medium rule
prefers over a drawing.

### Unit-to-figure map (U04c source block, Lec 15-16, 20-27, Tut 3-9)

Figure labels are the lesson's own (sb13-sb22 continue the u04
sequence. SB17-SB26 in text). Sections without a labeled figure
carry their checkable state in code or tables.

| Section | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| SB17 | ERM is a well-posed verdict | two rules | h2 wins, R_hat 0.0 | none (table) | table | original |
| SB18 | 0.1587 is the floor | two Gaussians | Bayes error 0.1587 | none (table) | table | original |
| SB19 | every ERM claim survives four steps | data | R_hat A 0.1667, B 0.0 | none (steps) | numbered steps | original |
| SB20 | the threshold is a dial | scores | minmax/NP/ROC table, AUC 0.9167 | none (table) | table | original |
| SB21 | the log sits outside the sum | 6 points | loglik -9.8125 | none (code) | code block | original |
| SB22 | the bound makes the climb monotone | mu [1,4] | -13.0089 to -3.3320 | none (table) | table | original |
| SB23 | the start decides the destination | two starts | -3.3320 vs -14.1109 | none (table) | table | original |
| SB24 | the swap changes nothing | mu [0,5] | loglik -9.8125 both ways | none (table) | table | original |
| SB25 | every point votes with a puff | 4 points | p_hat(1.5) = 0.2535 | none (code) | code block | original |
| SB26 | memory with a vote | 4 points | 1-NN train error 0.0 | none (code) | code block | original |

### Render status

- visuals/u05/f01_bias_variance.png, f02_capacity_gap.png,
  f03_learning_curves.png, f04_cv_folds.png: rendered matplotlib
  3.6.3 (Agg), dpi 150, opened and read 2026-10-06. f01: bias^2
  falls, variance rises, U minimum at degree 2. f02: log-scale bars,
  all six labels readable, legend clear of labels. f03: gap arrows
  0.31 and 0.11, noisy single-seed curves labeled as such. f04:
  fold bars with CV means 1.8515 and 1.1806 in the legend.
  Status: verified.
- Two label fixes during render: f02 first used a linear y-axis
  (train bars invisible, degree-3 bar clipped). Re-rendered with a
  log y-axis. The "2.88" label then hid behind the legend twice.
  fixed by moving labels inside bars and the legend to lower left.
  See errors.md E-013.
- Table/equation/code figures: embedded in lesson text. Present,
  not separately rendered.

### Logged visual decisions

1. C01-C03 plates stay numbers/equations: the claims are two
   scalars each. A drawing adds no checkable state.
2. C02/C05/C07/C11/C12 plates stay tables: the claim is a numeric
   comparison across regimes.
3. C06 plate stays an equation block: the equivalence is one
   number matching another.
4. SB17-SB26 plates stay tables/code/steps: the source block's
   claims are computed verdicts (risks, errors, likelihoods), best
   checked as numbers. No source figure was available to adapt
   (G2 open), and no figure was copied.

## RUN 5 audit (2026-10-06)

### Unit-to-figure map (U06 lesson)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u06-c01 | the OLS line minimizes RSS | 4 points | line 0.3+0.8x, RSS 1.8 | f01 | PNG | original, rendered |
| u06-c02 | the fit implies a noise belief | RSS 1.8 | sigma2 0.45, pred var 1.665 | none (numbers) | equation block | original |
| u06-c03 | one gradient step lifts the likelihood | w=0, ll -1.3863 | w=0.25, ll -1.2691 | f02 | PNG | original, rendered |
| u06-c04 | the votes sum to one | logits [2,1,0] | p sums to 1.0, CE 0.4076 | none (numbers) | equation block | original |
| u06-c05 | logit and sigmoid invert | p=0.7 | 0.8473 and back | none (numbers) | equation block | original |
| u06-c06 | the corrected map matches the kernel | 43 vs 49 | 49 = 49 | none (numbers) | equation block | original |
| u06-c07 | the Gram is PSD | 3 points | eig 0.39/0.97/1.64 | f04 | PNG | original, rendered |
| u06-c08 | the boundary maximizes the margin | 4 points | width 1.4142, C prices slack | f03 | PNG | original, rendered |
| u06-c09 | primal equals dual | two points | a*=0.25, 0.25=0.25 | none (numbers) | equation block | original |
| u06-c10 | the step moves through the kink | w=[1,1] | w=[0.9,0.9] | none (code) | code block | original |
| u06-c11 | scaling rounds the needle | kappa 3.35e6 | kappa 2.618 | none (numbers) | equation block | original |
| u06-c12 | the hinge resists noise | 9 flips | 0.7167 vs 0.7833 | none (numbers) | equation block | original |

Note: the lesson file labels figures f01-f04 per lesson scope. The
audit maps them to u06-c01 (f01), u06-c03 (f02), u06-c08 (f03),
u06-c07 (f04). C02, C04-C06, C09, C11, C12 carry their checkable
state in numbers or equations. C10 in code. The honest-medium rule
prefers these over drawings.

### Unit-to-figure map (U09 remaining leaves, lesson 09b)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u09-c01 | Lloyd moves centers to cluster means | 6 points, centers at pts | centers at means, J 0.1333 | f01 | PNG | original, rendered |
| u09-c02 | the ruler picks the neighbor | A,B,C raw | flip 1->2->1 under scaling | none (numbers) | equation block | original |
| u09-c03 | one direction keeps 99.72% | 5 points | PC1, eig 6.54/0.018 | none (numbers) | equation block | original |
| u09-c04 | the error is the dropped eigenvalue | point, PC1 | zoomed pink segment, MSE 0.009172 | f02 | PNG | original, rendered |
| u09-c09 | the noisy sensor buys a direction | 3 sensors, one noisy | eig 7.33/3.83/0.0085 | none (numbers) | equation block | original |
| u09-c12 | the elbow is at k=2 | k in {1,2,3,5} | held-out 14.18->0.44->0.34->0.29 | f03 | PNG | original, rendered |

The 09b lesson does not re-teach C05-C08, C10, C11 (lesson-04c).
No figure repeats their content.

### Render status

- visuals/u06/f01_ols_fit.png, f02_logistic_step.png,
  f03_margin.png, f04_kernel_gram.png: rendered matplotlib 3.6.3
  (Agg), dpi 150, opened and read 2026-10-06. f01: line, 4
  points, 4 residual segments, legend values match the compute
  script. f02: first render placed dots off the sigmoid curve
  (plotted at x=0,1 instead of z=w x). Re-rendered with dots at
  z=0 and z=0.25 sitting on the curve, re-read. f03: boundary,
  two margin lines, width label 1.4142. Only the -1 side has
  support vectors (honest asymmetry, noted in the lesson). f04:
  3x3 cells readable, eigenvalues in the footer. Status:
  verified. See errors.md E-014.
- visuals/u09/f01_kmeans.png, f02_pca.png, f03_heldout.png:
  rendered same toolchain, opened and read 2026-10-06. f01:
  first render clipped the "center 1" label at the edge. 
  re-rendered with the label moved, re-read. f02: full-frame
  render hid the tiny dropped error. Re-rendered zoomed on the
  first point with the pink segment visible, re-read. f03:
  first title claimed "held-out picks k=2" with a U-shape, but
  measured held-out keeps falling (0.44, 0.34, 0.29). Retitled
  honestly to the elbow claim and re-rendered, re-read. Status:
  verified. See errors.md E-015, E-016.
- Table/equation/code figures: embedded in lesson text.
  Present, not separately rendered.

### Logged visual decisions

1. C02/C04-C06/C09/C11/C12 (U06) plates stay numbers: the claims
   are scalar identities, best checked as numbers.
2. C10 plate is a code block: the step is the claim.
3. C02/C03/C09 (U09b) plates stay numbers/equations: distances,
   eigenvalues, and the factor comparison are numeric.
4. f03 (u06) keeps the asymmetric support vectors: inventing a
   symmetric toy would be prettier and less honest.
5. No figure copied from any instructor or third-party source.

## Completion-run audit (2026-10-06)

### Unit-to-figure map (U07 lesson)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u07-c01 | a tree cuts the plane into rectangles | six points | three rectangles, two fences | f01 | PNG | original, rendered |
| u07-c02 | three impurity scores agree at pure nodes | counts [4,1] | 0.32 / 0.5004 / 0.2 | f02 | PNG | original, rendered |
| u07-c03 | one threshold wins the scan | five candidates | t=0.5, gain 0.5 | none (table) | table | original |
| u07-c04 | alpha sets the exchange rate | full tree | prune at alpha 0.1 | none (table) | table | original |
| u07-c05 | bootstrap worlds miss points | four points | three replicates | none (table) | table | original |
| u07-c06 | the rho floor survives any B | single 1.0 | 0.37 / 0.604 / 0.208 | none (equations) | equation block | original |
| u07-c05/c06 | averaging stumps cuts validation error, then flattens | single stump 0.0421 | bagged 0.0278 at B=5, non-monotone (0.0344 at B=25) | f04 | PNG | original, rendered |
| u07-c07 | OOB votes are honest | three replicates | OOB table | none (table) | table | original |
| u07-c08 | the missed point gets half the weight | 0.25 each | [1/6,1/6,1/6,1/2] | f03 | PNG | original, rendered |
| u07-c09 | one round fits the leftovers | SSE 2.0 | SSE 0.5 | none (equations) | equation block | original |
| u07-c10 | accuracy hides the skew | 0.95 accuracy | F1 0.5714 | none (table) | table | original |
| u07-c11 | credit is arbitrary under correlation | gain 0.5/0.0 | permute drops | none (table) | table | original |
| u07-c12 | every score needs a dumb reference | no baseline | 0.5 vs 1.0 | none (table) | table | original |

Note: the lesson file labels figures f01-f04 per lesson scope. The
audit maps them to u07-c01 (f01), u07-c02 (f02), u07-c08 (f03),
u07-c05/c06 (f04: bagged-stump val MSE vs B). C03, C04, C07 and
C09-C12 carry their checkable state in tables or equations, which
the honest-medium rule prefers over drawings.

### Unit-to-figure map (U08 lesson)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u08-c01 | shapes compose through the stack | x (2,) | out 0.0 | f03 | PNG | original, rendered |
| u08-c02 | the backward walk matches differences | analytic | fd agrees to 8 digits | none (table) | table | original |
| u08-c03 | the kernel marks change | [1,2,3,4] | [-2,-2] | none (ASCII) | ASCII | original |
| u08-c04 | sharing divides the bill | 102,500 | 208 | none (table) | table | original |
| u08-c05 | the kernel must fit inside | 28x28 | 24x24 then 12x12 | none (equations) | equation block | original |
| u08-c06 | the state is a running note | x_0 = 1.0 | h_0, h_1 | none (table) | table | original |
| u08-c07 | one scalar decides the gradient fate | w^10 | 9.77e-4 vs 57.67 | f02 | PNG | original, rendered |
| u08-c08 | gates guard the cell | x = 0.7 | f/i/g/o/c/h values | none (table) | table | original |
| u08-c09 | attention is a soft lookup | Q, K | weights, outputs | f01 | PNG | original, rendered |
| u08-c10 | shapes must line up at every add | X (5,8) | nine-row trace | none (table) | table | original |
| u08-c11 | tame, then restore scale | [1,2,3] | [-1.2247,0,1.2247] | none (table) | table | original |
| u08-c12 | Adam erases the scale ratio | g = [0.5,-0.3] | [0.01,-0.01] vs SGD | f04 | PNG | original, rendered |

### Unit-to-figure map (U10 lesson)

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u10-c01 | draw z first, then x given z | mixture | ten samples | f02 | PNG | original, rendered |
| u10-c02 | the encoder pays for its claims | q = N(0.4, 0.5) | KL 0.3981, ELBO -1.3283 | none (equations) | equation block | original |
| u10-c03 | D ascends, G descends | D(fake) 0.3 | V -0.5798 -> -0.9163 | none (table) | table | original |
| u10-c04 | each objective has its failure | p, q | JSD 0.0462 | none (table) | table | original |
| u10-c05 | ten samples give a coarse CDF | samples | 0.3 vs 0.5 at x = 0 | f03 | PNG | original, rendered |
| u10-c06 | three jobs, three knobs | beta free | collapse at beta = 10 | none (table) | table | original |
| u10-c07 | the proof needs four assumptions | log p(x) | gap 0.1438 | f01 | PNG | original, rendered |
| u10-c08 | loose claims die on toys | three claims | three killer numbers | none (table) | table | original |
| u10-c09 | predicted 4.0, measured 4.2887 | theory | seeded trials | none (table) | table | original |
| u10-c10 | attack with build numbers | three claims | honest versions | none (table) | table | original |
| u10-c11 | defend without notes | prompts | rubric | none (list) | list | original |
| u10-c12 | gaps stay visible | G1-G7 | status table | none (table) | table | original |

### Render status

- visuals/u07/f01_regions.png, f02_impurity.png,
  f03_adaboost_weights.png, f04_bagging_mse.png: rendered
  matplotlib 3.6.3 (Agg), dpi 150, opened and read 2026-10-06.
  f01: three regions, six points, two fences, footer fixed
  after a first-read clip. f02: three curves with the
  [4,1] values marked. f03: weight bars 0.25 -> 1/6 and
  0.5. f04: val MSE 0.0513/0.0278/0.0344/0.0335 with the
  single-stump line at 0.0421. the non-monotone curve is
  kept honest. Status: verified.
- visuals/u08/f01_attention.png, f02_grad_powers.png,
  f03_mlp_forward.png, f04_adam_sgd.png: rendered same
  toolchain, opened and read 2026-10-06. f01: 2x3 weights
  readable. f02: log-scale curves 0.5^t and 1.5^t. f03:
  three boxes with shapes and values. f04: SGD and Adam
  arrows on the quadratic contours. Status: verified.
- visuals/u10/f01_elbo_gap.png, f02_mixture_samples.png,
  f03_cdf.png: rendered same toolchain, opened and read
  2026-10-06. f01: bars -0.7340/-0.8779 with the gap
  arrow. tick labels fixed after a first-read overlap.
  f02: density curve with ten sample dots. f03: step vs
  smooth CDF. Status: verified.
- Table/equation/ASCII/list figures: embedded in lesson
  text. Present, not separately rendered.

### Logged visual decisions

1. C03/C05/C07/C10-C12 (U07) plates stay tables: the claims
   are candidate scores and vote tables, best checked as
   numbers.
2. C06/C09 (U07) plates are equation blocks: the variance
   formula and the SSE drop are the claims.
3. C02-C06/C08/C10/C11 (U08) plates stay tables/equations/
   ASCII: gradients, counts, shapes, and gate values are
   numeric. the slide diagram needs labels, not pixels.
4. C02-C04/C06/C08-C12 (U10) plates stay tables/equations/
   lists: the synthesis claims are verdicts and audits.
5. f04 (u07) keeps the non-monotone bagging curve: a
   smoothed invented curve would be prettier and less
   honest (repeats E-016).
6. No figure copied from any instructor or third-party
   source.

## Fix-builder caption register (2026-10-06, F-02)

Every rendered PNG now carries, in its lesson's "Rendered figures"
appendix: a markdown image embed with alt text, and a one-sentence
caption naming the source and the russian-doll shell. Source is
"original" for all 35 PNGs. Shell is 3 (computed before/after) for
all 35: each figure shows its concept's toy with the computed
before/after numbers that the claim rests on. Paths are relative to
the build root. The lesson embeds use ../../visuals/<unit>/...

| PNG | Lesson figure id | Caption (one sentence) | Alt text |
|---|---|---|---|
| visuals/u01/f09_honest_scatter.png | f09 | Six dots on honest axes that start at zero, each axis labeled with units. | Scatter plot of six dots with both axes starting at zero and unit labels |
| visuals/u02/f04_projection.png | f04 (u02-c04) | Vectors b, a, projection p, and residual r with r orthogonal to a. | Vectors b, a, projection p, and residual r with r orthogonal to a |
| visuals/u02/f05_transform.png | f05 (u02-c05) | The unit square mapped by the matrix to a rotated square. | Unit square mapped by the matrix to a rotated square |
| visuals/u02/f08_svd_values.png | f08 (u02-c09) | Bar chart of singular values 5 and 3 with the rank-1 error labeled. | Bar chart of singular values 5 and 3 with the rank-1 error labeled |
| visuals/u02/f10_covariance.png | f10 (u02-c11) | Four data points with both eigendirections of the covariance drawn. | Scatter of four data points with both covariance eigendirections drawn |
| visuals/u03/f01_gd_path.png | f07 (u03-c07) | GD glides to J 0.1038 while SGD wobbles to J 3.2022 on the same bowl. | Teal GD line gliding to J 0.1038 and SGD dots wobbling to J 3.2022 |
| visuals/u03/f02_convex.png | f05 (u03-c05) | One bowl beside two bowls with a saddle point marked. | One bowl beside two bowls with a saddle point marked |
| visuals/u03/f03_newton.png | f08 (u03-c08) | Numbered Newton iterates converge on the h(x) curve. | h(x) curve with numbered Newton iterates converging |
| visuals/u03/f04_learning_rates.png | f11 (u03-c11) | Three J traces on a symlog axis for slow, fast, and exploded step sizes. | Three J traces on a symlog axis for slow, fast, and exploded step sizes |
| visuals/u04/f01_iid_sample.png | sb03 (SB06) | Truth bars 0.7 and 0.3 against sample bars 0.5 and 0.5 from 8 draws. | Truth bars 0.7 and 0.3 against sample bars 0.5 and 0.5 from 8 draws |
| visuals/u04/f02_histogram_density.png | sb04 (SB07b) | Four histogram bins with counts 2, 2, 2, 2 from 8 points. | Four histogram bins with counts 2, 2, 2, 2 from 8 points |
| visuals/u04/f03_entropy_kl.png | sb06 (SB10) | Bars for p and q with both KL directions labeled in the footer. | Teal bars for p and q with both KL directions labeled in the footer |
| visuals/u04/f04_mle_gaussian.png | sb09 (SB13) | Histogram of four points with the fitted Gaussian and mu_hat marker. | Histogram of four points with the fitted Gaussian bell and mu_hat marker |
| visuals/u05/f01_bias_variance.png | f01 (u05-c04) | Bias squared falls and variance rises across degrees, with the U minimum at degree 2. | Bias squared falling and variance rising across degrees with the U minimum at degree 2 |
| visuals/u05/f02_capacity_gap.png | f02 (u05-c09) | Log-scale bars show train error down and test error up at degree 3. | Log-scale bars with train error down and test error up at degree 3 |
| visuals/u05/f03_learning_curves.png | f03 (u05-c10) | Learning curves with gap arrows 0.31 at n 3 and 0.11 at n 40. | Learning curves with gap arrows 0.31 at n 3 and 0.11 at n 40 |
| visuals/u05/f04_cv_folds.png | f04 (u05-c08) | Grouped fold bars with CV means 1.8515 and 1.1806 in the legend. | Grouped fold bars with CV means 1.8515 and 1.1806 in the legend |
| visuals/u06/f01_ols_fit.png | f01 (u06-c01) | Four points with the OLS line 0.3 plus 0.8x and four pink residual segments. | Four points with the OLS line 0.3 plus 0.8x and four pink residual segments |
| visuals/u06/f02_logistic_step.png | f02 (u06-c03) | Sigmoid curve with the w 0 dots and the w 0.25 dots sitting on the curve. | Sigmoid curve with the w 0 dots and the w 0.25 dots sitting on the curve |
| visuals/u06/f04_kernel_gram.png | f04 (u06-c07) | Kernel Gram heatmap with cell values and eigenvalues 0.3911, 0.9656, 1.6433 in the footer. | Kernel Gram heatmap with cell values and eigenvalues in the footer |
| visuals/u06/f03_margin.png | f03 (u06-c08) | Decision boundary with two margin lines, width 1.4142, and support vectors on one side only. | Decision boundary with two margin lines, width 1.4142, and support vectors on one side only |
| visuals/u07/f01_regions.png | f01 (u07-c01) | Three tree regions, six points, and two split fences. | Three tree regions, six points, and two split fences |
| visuals/u07/f02_impurity.png | f02 (u07-c02) | Three impurity curves over p with the node 4,1 values marked. | Three impurity curves over p with the node 4,1 values marked |
| visuals/u07/f03_adaboost_weights.png | f03 (u07-c08) | Weight bars before and after round 1, with the missed point at one half. | Weight bars before and after round 1, with the missed point at one half |
| visuals/u07/f04_bagging_mse.png | f04 (u07-c05/c06) | Bagged-stump validation MSE across B with the non-monotone curve and the single-stump line at 0.0421. | Bagged-stump validation MSE across B with the non-monotone curve and the single-stump line at 0.0421 |
| visuals/u08/f01_attention.png | f01 (u08-c09) | Two-by-three attention weight heatmap for the soft lookup. | Two-by-three attention weight heatmap for the soft lookup |
| visuals/u08/f02_grad_powers.png | f02 (u08-c07) | Log-scale curves of 0.5 to the t and 1.5 to the t across ten steps. | Log-scale curves of 0.5 to the t and 1.5 to the t across ten steps |
| visuals/u08/f03_mlp_forward.png | f03 (u08-c01) | Three boxes trace shapes from x with shape 2 to output 0.0. | Three boxes trace shapes from x with shape 2 to output 0.0 |
| visuals/u08/f04_adam_sgd.png | f04 (u08-c12) | One SGD arrow and one Adam arrow on the quadratic contours. | One SGD arrow and one Adam arrow on the quadratic contours |
| visuals/u09/f01_kmeans.png | f01 (u09-c01) | Six points with final centers and dashed paths from the start. | Six points with final centers and dashed paths from the start |
| visuals/u09/f02_pca.png | f02 (u09-c04) | Zoomed first point with the pink segment marking the dropped error. | Zoomed first point with the pink segment marking the dropped error |
| visuals/u09/f03_heldout.png | f03 (u09-c12) | Held-out inertia across k with the elbow at k 2. | Held-out inertia across k with the elbow at k 2 |
| visuals/u10/f01_elbo_gap.png | f01 (u10-c07) | Two bars at minus 0.7340 and minus 0.8779 with the gap arrow of 0.1438. | Two bars at minus 0.7340 and minus 0.8779 with the gap arrow of 0.1438 |
| visuals/u10/f02_mixture_samples.png | f02 (u10-c01) | True mixture density curve with ten sample dots drawn from it. | True mixture density curve with ten sample dots drawn from it |
| visuals/u10/f03_cdf.png | f03 (u10-c05) | Step CDF from ten samples against the smooth true CDF. | Step CDF from ten samples against the smooth true CDF |

## Logged visual decision, fix builder (F-02, chapter plates)

The spec asks for one chapter plate at the end of each concept. Only
U01 carries one (lesson-01, "## Chapter plate"). Building ten new
unit plates after the build would add drawings with no new checkable
state: every unit lesson already ends with its closing section (the
"Not yet understood" dependency list. U02 and U04a also carry an
explicit "Lesson close / Block close: the four-way link"), and every
rendered figure now has a captioned plate in the lesson. Decision:
exempt U02-U10 from new chapter plates. Logged 2026-10-06. The
exemption is explicit and does not relabel the gap as covered: if a
later run adds plates, this decision is superseded.

## Logged visual decision, fix builder (F-08, RUN 1 render provenance)

RUN 1 shipped visuals/u01/f09_honest_scatter.png with no preserved
render script (render scripts cover runs 2-5 and completion only).
Fix: render_run1.py added at the build root. It reproduces the exact
snippet from the lesson's C09 section with the course palette
(#F7F4EE) applied, and renders to /tmp for comparison. The original
PNG is RETAINED as the audited artifact (opened and read 2026-10-06:
six dots, axes at 0, labels carry units). This script is a provenance
record, not a byte-exact reproduction, because RUN 1's exact style
calls were not preserved. No figure was re-rendered over the
verified original.
