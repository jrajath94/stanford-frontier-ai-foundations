# math-ml: Mathematical Foundations of Machine Learning

## Course identity (verified 2026-10-06)

- NPTEL / IISc Bangalore, course ID 106108841, code noc26-cs02.
- Instructor: Prof. Prathosh A P (IISc 2021-2026. IIT Delhi 2017-2021).
- 12 weeks. First offering ran Jan 19, 2026 to Apr 10, 2026, exam Apr 25, 2026.
- Second offering began August 2026 and continues as of the baseline date.
- Lecture recordings: YouTube playlist PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu.
- Playlist enumerated directly on 2026-10-06: 89 entries
  (1 intro + 69 lectures Lec 01-Lec 69 + 19 tutorial videos).
- Full audit: see source_manifest.md and source_gaps.md.

## What this tree is

Standalone deep crash-course build from the v2 prompt pack.
Build root: ~/workspace/stanford-frontier-ai/v2-pack/math-ml/
Prompt: ~/workspace/prompt-pack-v2/frontier_ai_prompt_pack_v2/prompts/04_math-ml_crash_course_prompt.md
Parent units: 10. Coverage ledger rows: 120.

RUN 1 scope only: identity, artifact audit, prerequisite graph, course map,
diagnostics, and the first foundation lesson (U01). Do not begin RUN 2 here.

## Course map (parent units)

- math-ml-U01: Mathematical language and computational setup, RUN 1 lesson delivered.
- math-ml-U02: Linear geometry and decompositions, planned.
- math-ml-U03: Calculus and optimization, planned.
- math-ml-U04: Probability, density, and estimation, planned.
- math-ml-U05: Risk, regularization, and generalization, planned.
- math-ml-U06: Linear models, kernels, and margins, planned.
- math-ml-U07: Trees and ensembles, planned.
- math-ml-U08: Neural and sequence architectures, planned.
- math-ml-U09: Unsupervised and latent models, planned.
- math-ml-U10: Generative bridge and mathematical synthesis, planned.

Detailed unit-to-lecture and unit-to-prerequisite maps: course_map.md.

- U07, U08, U10: completion run 2026-10-06. U07 "Trees and
  ensembles": lesson-07 (12 leaf concepts, 15-item contract),
  keys, 4 PNGs, lab-07, interview bank U07. U08 "Neural and
  sequence architectures": lesson-08, keys, 4 PNGs, lab-08,
  interview bank U08. U10 "Generative bridge and
  mathematical synthesis": lesson-10, keys, 3 PNGs, lab-10,
  interview bank U10. All 120 rows TAUGHT+ASSESSED, SOURCE
  ATTRIBUTION PENDING. Ready for RUN 6 adversarial audit.

## File index

| File | Content |
|---|---|
| README.md | this file |
| state.md | RUN 1 checkpoint and resume point |
| source_manifest.md | source register with inspection boundaries |
| source_gaps.md | unresolved source gaps |
| coverage_matrix.md | 120 coverage rows with statuses |
| course_map.md | unit map: lectures, prerequisites, figure/assessment status |
| prerequisites.md | prerequisite graph and local P01/P02 remediation |
| notation_and_shapes.md | symbol registry for U01 |
| glossary.md | U01 term definitions |
| visual_audit.md | unit-to-figure audit for RUN 1 lesson |
| errors.md | error and uncertainty ledger |
| mastery_ledger.md | mastery tracking rules (empty at RUN 1) |
| role_gap_map.md | role bridge stub (RUN 4) |
| currentness.md | date-bound claims |
| lessons/u01/lesson-01-mathematical-language.md | RUN 1 lesson (U01, 12 leaf concepts) |
| lessons/u01/keys.md | answer keys for the U01 lesson (separate file) |
| lessons/u02/lesson-02-linear-geometry-and-decompositions.md | RUN 2 lesson (U02, 12 leaf concepts) |
| lessons/u02/keys.md | answer keys for the U02 lesson (separate file) |
| lessons/u04/lesson-04a-first-source-block.md | RUN 2 source block (Lec 02-10, U04 leaves C01/C02/C04/C06/C10/C11) |
| lessons/u04/keys-04a.md | answer keys for the source block (separate file) |
| lessons/u03/lesson-03-calculus-and-optimization.md | RUN 3 lesson (U03, 12 leaf concepts) |
| lessons/u03/keys.md | answer keys for the U03 lesson (separate file) |
| lessons/u04/lesson-04b-second-source-block.md | RUN 3 source block (Lec 11-19, U04 leaves C05/C07/C08) |
| lessons/u04/keys-04b.md | answer keys for the source block (separate file) |
| lessons/u05/lesson-05-risk-regularization-generalization.md | RUN 4 lesson (U05, 12 leaf concepts) |
| lessons/u05/keys.md | answer keys for the U05 lesson (separate file) |
| lessons/u04/lesson-04c-third-source-block.md | RUN 4 source block (Lec 15-16, 20-27, Tut 3-9. U04 leaves C03/C09/C12, U09 leaves C05-C08/C10/C11) |
| lessons/u04/keys-04c.md | answer keys for the source block (separate file) |
| diagnostics/diagnostic-01-math-language.md | placement diagnostic with scoring rubric |
| diagnostics/keys-01.md | diagnostic answer key (separate file) |
| labs/lab-01-linear-geometry.md | RUN 2 lab: linear geometry computations |
| labs/keys-lab-01.md | lab 01 keys (separate file) |
| labs/lab-02-sampling-estimation.md | RUN 2 lab: sampling and estimation |
| labs/keys-lab-02.md | lab 02 keys (separate file) |
| labs/lab-03-calculus-optimization.md | RUN 3 lab: calculus and optimization |
| labs/keys-lab-03.md | lab 03 keys (separate file) |
| labs/lab-04-entropy-mle.md | RUN 3 lab: entropy, KL, and MLE |
| labs/keys-lab-04.md | lab 04 keys (separate file) |
| labs/lab-05-risk-regularization.md | RUN 4 lab: risk, regularization, generalization |
| labs/lab-06-kernels-margins.md | RUN 5 lab: kernels and margins |
| labs/keys-lab-05.md | lab 05 keys (separate file) |
| labs/keys-lab-06.md | lab 06 keys (separate file) |
| interview/interview-u02-questions.md | RUN 2 interview bank: U02 questions |
| interview/keys-u02.md | U02 interview keys (separate file) |
| interview/interview-u03-questions.md | RUN 3 interview bank: U03 questions |
| interview/keys-u03.md | U03 interview keys (separate file) |
| interview/interview-u04a-questions.md | RUN 3 interview bank: U04a questions |
| interview/keys-u04a.md | U04a interview keys (separate file) |
| interview/interview-u05-questions.md | RUN 4 interview bank: U05 questions |
| interview/keys-u05.md | U05 interview keys (separate file) |
| interview/interview-u06-questions.md | RUN 5 interview bank: U06 questions |
| interview/keys-u06.md | U06 interview keys (separate file) |
| lessons/u10/lesson-10-generative-bridge-synthesis.md | completion-run lesson (U10, 12 leaf concepts) |
| lessons/u10/keys.md | answer keys for the U10 lesson (separate file) |
| lessons/u07/lesson-07-trees-ensembles.md | completion-run lesson (U07, 12 leaf concepts) |
| lessons/u07/keys.md | answer keys for the U07 lesson (separate file) |
| lessons/u08/lesson-08-neural-sequence-architectures.md | completion-run lesson (U08, 12 leaf concepts) |
| lessons/u08/keys.md | answer keys for the U08 lesson (separate file) |
| labs/lab-07-trees-ensembles.md | completion-run lab: trees and ensembles |
| labs/keys-lab-07.md | lab 07 keys (separate file) |
| labs/lab-08-neural-sequence.md | completion-run lab: neural and sequence |
| labs/keys-lab-08.md | lab 08 keys (separate file) |
| labs/lab-10-generative-synthesis.md | completion-run lab: generative bridge |
| labs/keys-lab-10.md | lab 10 keys (separate file) |
| interview/interview-u07-questions.md | completion-run interview bank: U07 questions |
| interview/keys-u07.md | U07 interview keys (separate file) |
| interview/interview-u08-questions.md | completion-run interview bank: U08 questions |
| interview/keys-u08.md | U08 interview keys (separate file) |
| interview/interview-u10-questions.md | completion-run interview bank: U10 questions |
| interview/keys-u10.md | U10 interview keys (separate file) |
| interview/transfer-sets.md | RUN 5 unfamiliar transfer sets A-F (18 items) |
| interview/keys-transfer.md | transfer set keys (separate file) |
| interview/oral-defenses.md | RUN 5 oral defense scripts (7 units) |
| interview/keys-oral.md | oral defense rubrics (separate file) |
| capstones/capstone-research-kernel-margins.md | RUN 5 research capstone R1 |
| capstones/capstone-applied-fde.md | RUN 5 applied/FDE capstone F1 |
| lessons/u06/lesson-06-linear-models-kernels-margins.md | RUN 5 lesson (U06, 12 leaf concepts) |
| lessons/u06/keys.md | answer keys for the U06 lesson (separate file) |
| lessons/u09/lesson-09b-unsupervised-leaves.md | RUN 5 lesson (U09 remaining leaves) |
| lessons/u09/keys-09b.md | answer keys for the 09b lesson (separate file) |
| visuals/u01/ | figures for the U01 lesson |
| visuals/u02/ | figures for the U02 lesson (4 PNGs) |
| visuals/u03/ | figures for the U03 lesson (4 PNGs) |
| visuals/u04/ | figures for the source block (4 PNGs) |
| visuals/u05/ | figures for the U05 lesson (4 PNGs) |
| visuals/u06/ | figures for the U06 lesson (4 PNGs) |
| visuals/u09/ | figures for the U09 leaves lesson (3 PNGs) |
| visuals/u07/ | figures for the U07 lesson (4 PNGs) |
| visuals/u08/ | figures for the U08 lesson (4 PNGs) |
| visuals/u10/ | figures for the U10 lesson (3 PNGs) |

## Continuation

Resume with the template in state.md. Read README.md, state.md,
source_manifest.md, coverage_matrix.md, notation_and_shapes.md,
visual_audit.md, and errors.md first.
