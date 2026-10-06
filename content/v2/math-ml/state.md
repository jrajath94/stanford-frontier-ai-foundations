# state.md, math-ml RUN 5 checkpoint

Date: 2026-10-06.
Run: RUN 5 of 6. Status: complete.
Worker: course builder (subagent 0e32d9f5, parent orchestrator).

## What RUN 5 delivered

- U06 "Linear models, kernels, and margins": full lesson with
  the 15-item contract on all 12 leaf concepts (C01-C12), each with
  toy, computed numbers, code, checks, costs, alternatives, failure
  case, ladder shells, and assessment items. Keys in lessons/u06/
  keys.md. 30 exercises E01-E30 + 10 deep ladders L01-L10.
- U09 remaining leaves (lesson-09b): k-means (C01),
  distance/scaling (C02), PCA (C03), reconstruction (C04), factor
  models bridge (C09), held-out evaluation (C12). 18 exercises +
  6 ladders. Keys in lessons/u09/keys-09b.md. C05-C08, C10, C11
  NOT re-taught (stay in lesson-04c). No figure repeats 04c
  content.
- Prerequisites: local P03/P07/P09 remediation R45-R54 added.
- Visuals: 7 PNGs rendered and read back (u06: f01-f04. u09:
  f01-f03). visual_audit.md extended with RUN 5 rows. Three
  label/title fixes caught and fixed (E-014, E-015, E-016).
- Labs continued: lab-06 (OLS audit, kernel expansion, C sweep),
  keys separate. T2/T3 numbers executed (f(q3)=0.348796,
  f(q4)=-0.662742. C totals 1.014/1.14/2.4/15.0).
- Interview bank: U06 installment per prompt quotas (6 breadth,
  2x5 ladders, 2 quantitative, 1 debug, 2 scenarios, 1 research
  critique), keys separate with scoring rubrics. Debug T1 premise
  executed before shipping (cond 3.227e6 vs 4.2, alphas ~1e6 vs
  sane). Transfer sets: 18 unfamiliar cross-unit items (A-F),
  keys separate. Oral defenses: 7 timed scripts (U01-U06, U09),
  rubrics separate.
- Capstones: R1 research on hinge vs logistic under label noise
  (hypothesis H1, EXECUTED single-seed run 0.7167 vs 0.7833,
  20-seed extension PROPOSED) and F1 applied/FDE (default-risk
  scorer: discovery, baseline, constraints, trust, acceptance,
  rollout, monitoring, rollback, handoff).
- Source gaps: G3 fourth attempt (curl 2026-10-06) still a loading
  shell. G2 attempt via subtitle fetch recorded. G1 closed. G4,
  G5, G6, G7 unchanged and open.
- Coverage: 84/120 rows TAUGHT+ASSESSED
  (U01 12 + U02 12 + U04a 6 + U03 12 + U04b 3 + U05 12 + U04c 3 +
  U09 6 + U06 12 + U09b 6). All rows keep SOURCE ATTRIBUTION
  PENDING.
- Style: ste_check.py run on every new/changed file. 0 hard fails.
- Watermark hygiene: wm_clean.py Layer A applied to every RUN 5
  file. No *.cleaned.md duplicates remained.

## Files added/changed in RUN 5

lessons/u06/lesson-06-linear-models-kernels-margins.md,
lessons/u06/keys.md, lessons/u09/lesson-09b-unsupervised-leaves.md,
lessons/u09/keys-09b.md, visuals/u06/f01_ols_fit.png,
visuals/u06/f02_logistic_step.png, visuals/u06/f03_margin.png,
visuals/u06/f04_kernel_gram.png, visuals/u09/f01_kmeans.png,
visuals/u09/f02_pca.png, visuals/u09/f03_heldout.png,
labs/lab-06-kernels-margins.md, labs/keys-lab-06.md,
interview/interview-u06-questions.md, interview/keys-u06.md,
interview/transfer-sets.md, interview/keys-transfer.md,
interview/oral-defenses.md, interview/keys-oral.md,
capstones/capstone-research-kernel-margins.md,
capstones/capstone-applied-fde.md,
compute_run5.py, render_run5.py.
Updated: prerequisites.md, notation_and_shapes.md, glossary.md,
coverage_matrix.md, visual_audit.md, source_gaps.md, errors.md,
state.md, mastery_ledger.md, currentness.md, course_map.md,
README.md.

## Unresolved gaps (see source_gaps.md)

G2 transcripts, G3 NPTEL preview JS page (fourth attempt failed),
G4 two offerings, G5 duration, G6 tutorials. G1 closed.
G7 partially located.

## Resume point for RUN 6

Next: RUN 6 adversarial audit (coverage, math, code, visual,
provenance, assessment), remediation, and packaging. Units left
untaught: U07 (trees/ensembles, 12), U08 (neural/sequence, 12),
U10 (generative bridge, 12). Remaining scope: 36/120 rows
PLANNED. The RUN 6 auditor is a different agent (builder !=
auditor). Suggested RUN 6 order: audit everything built, then
teach U07, U08, U10 or hand them to the next build run per the
coordinator's plan.

Continuation: 'Continue RUN 6. Read README.md, state.md,
source_manifest.md, coverage_matrix.md, notation_and_shapes.md,
visual_audit.md and errors.md. Resume the exact unfinished unit.
Preserve date, source, visual and integrity rules. Write complete
files, update ledgers, and stop at a clean checkpoint.'

---

# state.md, math-ml RUN 4 checkpoint

Date: 2026-10-06.
Run: RUN 4 of 6. Status: complete.
Worker: course builder (subagent 6297e4a7, parent orchestrator).

## What RUN 4 delivered

- U05 "Risk, regularization, and generalization": full lesson with
  the 15-item contract on all 12 leaf concepts (C01-C12), each with
  toy, computed numbers, code, checks, costs, alternatives, failure
  case, ladder shells, and assessment items. Keys in lessons/u05/
  keys.md. 30 exercises E01-E30 + 10 deep ladders L01-L10.
- Third source block (Lec 15-16, Lec 20-27, Tutorials 3-9):
  lesson-04c with 10 sections (SB17-SB26) covering U04-C03 (Bayes),
  U04-C09 (identifiability), U04-C12 (calibration), and the U09
  latent-model leaves C05 (mixture model), C06 (EM), C07 (Jensen),
  C08 (latent posterior), C10 (initialization), C11 (degeneracy),
  all with computed toys and assessment items. Keys in
  lessons/u04/keys-04c.md. 18 exercises + 5 ladders. Titles only.
  No transcript inspected (G2/G6 open). Source attribution PENDING.
  RUN 5 must not re-teach these U09 rows. It deepens the remaining
  U09 leaves (k-means, PCA, distance/scaling, reconstruction,
  factor models, held-out evaluation).
- Prerequisites: local P07/P09/P10 remediation R35-R44 added.
- Visuals: 4 PNGs rendered and read back (u05: f01-f04).
  visual_audit.md extended with RUN 4 rows. Two label collisions
  caught and fixed on f02 (E-013).
- Labs continued: lab-05 (risk/regularization, 3 tasks), keys
  separate. Task 3(c) executed in compute_run4.py Task 3(c) block
  (truth 2x+5, f(1) = 7.0, fresh seed-7 stream): bias^2 =
  7.042053, var = 0.035717, MSE = 7.077770, and bias^2 + var = MSE to 6 digits. (RUN 4 originally keyed bias^2 = 5.505186 against
  the stale truth 2.0, corrected by the fix builder 2026-10-06.)
- Interview bank: U05 installment per prompt quotas (6 breadth,
  2x5 ladders, 2 quantitative, 1 debug, 2 scenarios, 1 research
  critique), keys separate with scoring rubrics. Debug T1 premise
  executed before shipping (aligned 1.0 vs shuffled-X 0.2).
- Role bridges: role_gap_map.md populated with labeled bridges per
  role gate and stated residual gaps.
- Source gaps: G3 third attempt (curl 2026-10-06) still a loading
  shell. G1 closed. G7 partially located. G2, G4, G5, G6
  unchanged and open.
- Coverage: 66/120 rows TAUGHT+ASSESSED
  (U01 12 + U02 12 + U04a 6 + U03 12 + U04b 3 + U05 12 + U04c 3 +
  U09 6). U04 now complete (12/12). All rows keep SOURCE
  ATTRIBUTION PENDING.
- Style: ste_check.py run on every new/changed file. 0 hard fails.
- Watermark hygiene: wm_clean.py Layer A applied to every RUN 4
  file. No *.cleaned.md duplicates remained.

## Files added/changed in RUN 4

lessons/u05/lesson-05-risk-regularization-generalization.md,
lessons/u05/keys.md, lessons/u04/lesson-04c-third-source-block.md,
lessons/u04/keys-04c.md, visuals/u05/f01_bias_variance.png,
visuals/u05/f02_capacity_gap.png,
visuals/u05/f03_learning_curves.png, visuals/u05/f04_cv_folds.png,
labs/lab-05-risk-regularization.md, labs/keys-lab-05.md,
interview/interview-u05-questions.md, interview/keys-u05.md,
compute_run4.py, render_run4.py.
Updated: prerequisites.md, notation_and_shapes.md, glossary.md,
coverage_matrix.md, visual_audit.md, source_gaps.md, errors.md,
state.md, mastery_ledger.md, currentness.md, course_map.md,
role_gap_map.md, README.md.

## Unresolved gaps (see source_gaps.md)

G2 transcripts, G3 NPTEL preview JS page (third attempt failed),
G4 two offerings, G5 duration, G6 tutorials. G1 closed.
G7 partially located.

## Resume point for RUN 5

Next unit: math-ml-U06 "Linear models, kernels, and margins"
(prerequisites P03, P07, P09), mapped to Lec 28-30 and Lec 36-40.
The U05 interview installment is delivered. U06 interview
installment planned for RUN 5. The U09 latent-model rows taught in
the 04c source block (C05-C08, C10, C11) must not be re-taught.
RUN 5 deepens U09's remaining leaves only.

Continuation: 'Continue RUN 5. Read README.md, state.md,
source_manifest.md, coverage_matrix.md, notation_and_shapes.md,
visual_audit.md and errors.md. Resume the exact unfinished unit.
Preserve date, source, visual and integrity rules. Write complete
files, update ledgers, and stop at a clean checkpoint.'

---

# state.md, math-ml RUN 3 checkpoint (archived)

Date: 2026-10-06.
Run: RUN 3 of 6. Status: complete.
Worker: course builder (subagent 1f2cd64f, parent coordinator tree).

## What RUN 3 delivered

- U03 "Calculus and optimization": full lesson with the 15-item
  contract on all 12 leaf concepts (C01-C12), each with toy, computed
  numbers, code, checks, costs, alternatives, failure case, ladder
  shells, and assessment items. Keys in lessons/u03/keys.md.
  30 exercises E01-E30 + 10 deep ladders L01-L10.
- Second source block (Lec 11-Lec 19 + Tutorials 2, 7A, 7B):
  lesson-04b with 8 sections (SB09-SB16) covering U04-C05
  (multivariate Gaussian), C07 (MLE/MAP), C08 (entropy/KL), all with
  computed toys and assessment items. Keys in lessons/u04/keys-04b.md.
  18 exercises + 5 ladders. Titles only. No transcript inspected
  (G2/G6 open). Source attribution PENDING.
- Prerequisites: local P05/P09 remediation R23-R34 added.
- Visuals: 6 PNGs rendered and read back (u03: f01-f04. u04:
  f03, f04). visual_audit.md extended with RUN 3 rows and honest
  medium decisions. One label misread caught and fixed (E-009).
- Labs continued: lab-03 (calculus/optimization, 3 tasks), lab-04
  (entropy/KL/MLE, 3 tasks), keys separate.
- Interview banks: U03 installment and U04a installment per prompt
  quotas (6 breadth, 2x5 ladders, 2 quantitative, 1 debug, 2
  scenarios, 1 research critique), keys separate with scoring rubrics.
  Both debug-task premises executed before shipping (E-011).
- Source gaps: G3 second attempt (browser.open 2026-10-06) still a
  loading shell. G1 closed. G7 partially located. G2, G4, G5, G6
  unchanged and open.
- Coverage: 45/120 rows TAUGHT+ASSESSED
  (U01 12 + U02 12 + U04a 6 + U03 12 + U04b 3).
  All rows keep SOURCE ATTRIBUTION PENDING.
- Style: ste_check.py run on every new/changed file. 0 hard fails.
- Watermark hygiene: wm_clean.py Layer A applied to every RUN 3 file.

## Files added/changed in RUN 3

lessons/u03/lesson-03-calculus-and-optimization.md,
lessons/u03/keys.md, lessons/u04/lesson-04b-second-source-block.md,
lessons/u04/keys-04b.md, visuals/u03/f01_gd_path.png,
visuals/u03/f02_convex.png, visuals/u03/f03_newton.png,
visuals/u03/f04_learning_rates.png, visuals/u04/f03_entropy_kl.png,
visuals/u04/f04_mle_gaussian.png, labs/lab-03-calculus-optimization.md,
labs/lab-04-entropy-mle.md, labs/keys-lab-03.md,
labs/keys-lab-04.md, interview/interview-u03-questions.md,
interview/keys-u03.md, interview/interview-u04a-questions.md,
interview/keys-u04a.md, compute_run3.py, render_run3.py.
Updated: prerequisites.md, notation_and_shapes.md, glossary.md,
coverage_matrix.md, visual_audit.md, source_gaps.md, errors.md,
state.md, mastery_ledger.md, currentness.md, course_map.md,
README.md.

## Unresolved gaps (see source_gaps.md)

G2 transcripts, G3 NPTEL preview JS page (second attempt failed),
G4 two offerings, G5 duration, G6 tutorials. G1 closed.
G7 partially located.

## Resume point for RUN 4

Next units: math-ml-U05 "Risk, regularization, and generalization"
(prerequisites P07, P09, P10), and the third source block
(Lec 15-16 Bayes classifier + risk minimization, Lec 20-27 latent
variables/EM/MAP/Parzen/nearest-neighbor, Tutorials 3-9).
U03/U04b interview installments delivered in RUN 3. U05 interview
installment planned for RUN 4.

Continuation: 'Continue RUN 4. Read README.md, state.md,
source_manifest.md, coverage_matrix.md, notation_and_shapes.md,
visual_audit.md and errors.md. Resume the exact unfinished unit.
Preserve date, source, visual and integrity rules. Write complete
files, update ledgers, and stop at a clean checkpoint.'

## What RUN 2 delivered (archived)

- U02 "Linear geometry and decompositions": full lesson with the
  15-item contract on all 12 leaf concepts (C01-C12), each with toy,
  computed numbers, code, checks, costs, alternatives, failure case,
  ladder shells, and assessment items. Keys in lessons/u02/keys.md.
  30 exercises E01-E30 + 10 deep ladders L01-L10.
- First source block (Lec 02-Lec 10): lesson-04a with 8 sections
  (SB01-SB08) covering U04-C01, C02, C04, C06, C10, C11, all with
  computed toys and assessment items. Keys in lessons/u04/keys-04a.md.
  20 exercises + 4 ladders.
- Prerequisites: local P03/P04 remediation R11-R22 added.
- Visuals: 6 PNGs rendered and read back (u02: f04, f05, f08, f10.
  u04: f01, f02). visual_audit.md extended with RUN 2 rows and honest
  medium decisions.
- Labs started: lab-01 (linear geometry, 3 tasks), lab-02 (sampling
  and estimation, 3 tasks), keys separate.
- Interview bank started: U02 installment per prompt quotas (6
  breadth, 2x5 ladders, 2 quantitative, 1 debug task, 2 scenarios,
  1 research critique), keys separate.
- Source gaps: G1 closed as SOURCE-UNREACHABLE (second attempt).
  SRC-09 located (third-party weekly assignment objectives. G7 now
  partially located). G2-G6 unchanged and open.
- Coverage: 30/120 rows TAUGHT+ASSESSED (U01 12, U02 12, U04a 6).
  All rows keep SOURCE ATTRIBUTION PENDING.
- Style: ste_check.py run on every new/changed file. 0 hard fails.
- Watermark hygiene: wm_clean.py Layer A applied to every RUN 2 file.

## Files added/changed in RUN 2

lessons/u02/lesson-02-linear-geometry-and-decompositions.md,
lessons/u02/keys.md, lessons/u04/lesson-04a-first-source-block.md,
lessons/u04/keys-04a.md, visuals/u02/f04_projection.png,
visuals/u02/f05_transform.png, visuals/u02/f08_svd_values.png,
visuals/u02/f10_covariance.png, visuals/u04/f01_iid_sample.png,
visuals/u04/f02_histogram_density.png, labs/lab-01-linear-geometry.md,
labs/lab-02-sampling-estimation.md, labs/keys-lab-01.md,
labs/keys-lab-02.md, interview/interview-u02-questions.md,
interview/keys-u02.md, compute_run2.py, render_u02.py, render_u04a.py.
Updated: prerequisites.md, notation_and_shapes.md, glossary.md,
coverage_matrix.md, visual_audit.md, source_manifest.md,
source_gaps.md, errors.md, state.md, mastery_ledger.md,
currentness.md, course_map.md, README.md.

## Unresolved gaps (see source_gaps.md)

G2 transcripts, G3 NPTEL preview JS page, G4 two offerings, G5
duration, G6 tutorials. G1 closed. G7 partially located.

## Resume point for RUN 3

Next unit: math-ml-U03 "Calculus and optimization" (prerequisites
P05, P09), then the second source block (Lec 11-Lec 19: entropy, KL
divergence, KL minimization, ML estimate example, MLE for Gaussian,
MLE for discrete, density estimation for mixed distributions.
tutorials 2, 7A, 7B). U04a interview installment planned.

Continuation: 'Continue RUN 3. Read README.md, state.md,
source_manifest.md, coverage_matrix.md, notation_and_shapes.md,
visual_audit.md and errors.md. Resume the exact unfinished unit.
Preserve date, source, visual and integrity rules. Write complete
files, update ledgers, and stop at a clean checkpoint.'

## What RUN 1 delivered (archived summary)

- Course identity verified against four independent sources.
- Artifact audit: 89-entry YouTube playlist enumerated title by title.
  Transcripts not inspected. Inspection boundaries recorded per artifact.
- Prerequisite graph: 24 modules (P01-P24) with unit-level dependencies.
  Local self-contained remediation R1-R10 for P01/P02 essentials.
- Course map: 10 parent units mapped to playlist lectures and
  prerequisites.
- Diagnostics: diagnostic-01 with rubric. Keys separate.
- U01 lesson (12 leaf concepts) with keys. 13 figure ids in
  visual_audit.md. One PNG rendered and verified.
- 120 coverage rows classified. 12 TAUGHT+ASSESSED (U01).
- Style: 0 hard fails. Wm_clean.py Layer A applied.

RUN 1 files: README.md, state.md, source_manifest.md, source_gaps.md,
coverage_matrix.md, course_map.md, prerequisites.md,
notation_and_shapes.md, glossary.md, visual_audit.md, errors.md,
mastery_ledger.md, role_gap_map.md, currentness.md,
lessons/u01/lesson-01-mathematical-language.md, lessons/u01/keys.md,
diagnostics/diagnostic-01-math-language.md, diagnostics/keys-01.md,
visuals/u01/f09_honest_scatter.png.

---

# state.md, math-ml COMPLETION RUN checkpoint

Date: 2026-10-06.
Run: completion run (builds U07, U08, U10). Status: complete.
Worker: course builder (subagent 1f13b419, parent orchestrator).

## What the completion run delivered

- U07 "Trees and ensembles": full lesson with the 15-item
  contract on all 12 leaf concepts (C01-C12), each with toy,
  computed numbers, code, checks, costs, alternatives,
  failure case, ladder shells, and assessment items. Keys in
  lessons/u07/keys.md. 30 exercises E01-E30 + 10 deep
  ladders L01-L10.
- U08 "Neural and sequence architectures": full lesson with
  the 15-item contract on all 12 leaf concepts, same
  structure. Keys in lessons/u08/keys.md. 30 exercises +
  10 ladders.
- U10 "Generative bridge and mathematical synthesis": full
  lesson with the 15-item contract on all 12 leaf concepts
  (C07-C12 as synthesis: proof audit, counterexamples,
  implementation evidence, research critique, oral defense,
  source-gap audit). Keys in lessons/u10/keys.md. 30
  exercises + 10 ladders.
- Prerequisites: local remediation R55-R78 added (P06/P07/
  P10 for U07. P11-P14 for U08. P08/P18/P22 for U10).
- Visuals: 11 PNGs rendered and read back (u07: f01-f04.
  u08: f01-f04. u10: f01-f03). visual_audit.md extended
  with completion-run rows and honest-medium decisions.
  Two fixes during render (footer clip, tick overlap).
- Labs continued: lab-07 (split scan, AdaBoost, OOB vs
  in-bag), lab-08 (tanh backprop, attention, Adam vs SGD),
  lab-10 (ELBO gap, seed-12 rerun, beta sweep), keys
  separate. Lab reference numbers corrected twice by
  execution before shipping (lab-07 Task 1 best t, lab-08
  Task 1 gradients, lab-10 Task 1 ELBO and Task 2 ratio).
- Interview banks: U07, U08, U10 installments per prompt
  quotas (6 breadth, 2x5 ladders, 2 quantitative, 1 debug,
  2 scenarios, 1 research critique), keys separate with
  scoring rubrics. All three debug-task premises executed
  before shipping (E-019, E-020, E-021).
- Source gaps: G2 fifth attempt (yt-dlp not present on
  this box. installs out of scope). G1 closed. G3, G4,
  G5, G6, G7 unchanged and open.
- Coverage: 120/120 rows TAUGHT+ASSESSED
  (84 prior + U07 12 + U08 12 + U10 12). All rows keep
  SOURCE ATTRIBUTION PENDING.
- Style: ste_check.py run on every new/changed file. Hard
  fails fixed to 0 before shipping.
- Watermark hygiene: wm_clean.py Layer A applied to every
  completion-run file. No *.cleaned.md duplicates remained.

## Files added/changed in the completion run

lessons/u07/lesson-07-trees-ensembles.md, lessons/u07/keys.md,
lessons/u08/lesson-08-neural-sequence-architectures.md,
lessons/u08/keys.md,
lessons/u10/lesson-10-generative-bridge-synthesis.md,
lessons/u10/keys.md, visuals/u07/f01_regions.png,
visuals/u07/f02_impurity.png,
visuals/u07/f03_adaboost_weights.png,
visuals/u07/f04_bagging_mse.png, visuals/u08/f01_attention.png,
visuals/u08/f02_grad_powers.png, visuals/u08/f03_mlp_forward.png,
visuals/u08/f04_adam_sgd.png, visuals/u10/f01_elbo_gap.png,
visuals/u10/f02_mixture_samples.png, visuals/u10/f03_cdf.png,
labs/lab-07-trees-ensembles.md, labs/keys-lab-07.md,
labs/lab-08-neural-sequence.md, labs/keys-lab-08.md,
labs/lab-10-generative-synthesis.md, labs/keys-lab-10.md,
interview/interview-u07-questions.md, interview/keys-u07.md,
interview/interview-u08-questions.md, interview/keys-u08.md,
interview/interview-u10-questions.md, interview/keys-u10.md,
compute_completion.py, compute_completion2.py,
render_completion.py.
Updated: prerequisites.md, notation_and_shapes.md, glossary.md,
coverage_matrix.md, visual_audit.md, source_gaps.md, errors.md,
state.md, mastery_ledger.md, currentness.md, course_map.md,
README.md. (PACK-STATUS.md removed from this list by the fix
builder 2026-10-06: the pack-level PACK-STATUS.md at v2-pack/
is coordinator-owned. No build-root copy exists.)

## Unresolved gaps (see source_gaps.md)

G2 transcripts (5 attempts), G3 NPTEL preview JS page
(4 attempts), G4 two offerings, G5 duration, G6 tutorials,
G7 assessments partial. G1 closed (SOURCE-UNREACHABLE).

## Resume point for RUN 6

Next: RUN 6 adversarial audit (coverage, math, code, visual,
provenance, assessment), remediation, and packaging. All 120
rows are TAUGHT+ASSESSED with SOURCE ATTRIBUTION PENDING.
The RUN 6 auditor is a different agent (builder != auditor).

Continuation: 'Continue RUN 6. Read README.md, state.md,
source_manifest.md, coverage_matrix.md, notation_and_shapes.md,
visual_audit.md and errors.md. Audit the full build, remediate,
and package. Preserve date, source, visual and integrity rules.
Write complete files, update ledgers, and stop at a clean
checkpoint.'
