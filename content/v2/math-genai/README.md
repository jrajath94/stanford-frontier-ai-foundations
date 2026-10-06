# math-genai: Mathematical Foundations of Generative AI

## Course identity (verified 2026-10-06)

- Instructor: Prof. Prathosh A P. Assistant Professor, Division of
  Electrical, Electronics, and Computer Science (EECS), IISc Bangalore.
- Institution: IIT Madras, BS in Data Science and Applications.
- Course ID: BSDA5002. 4 credits. Degree-level course.
- Official syllabus: https://study.iitm.ac.in/ds/course_pages/BSDA5002.html
  (fetched 2026-10-06, full text). Course description covers the
  probabilistic foundations and learning algorithms of deep generative
  models: variational autoencoders, generative adversarial networks,
  autoregressive models, diffusion models, and large language models.
  Prescribed book: Generative Deep Learning (O'Reilly, 2023),
  plus recent papers and surveys.
- Instructor teaching page: https://prathosh.in/teaching.html (fetched
  2026-10-06). States a 4-credit, 12-week course on the mathematical
  principles of generative AI: diffusion models, VAEs, normalising
  flows, and score-based methods, designed for the IIT Madras BS Data
  Science programme.
- Supplied seed link: https://lnkd.in/gmF_VxRZ. Resolved on 2026-10-06
  from the interstitial page HTML to the official course playlist:
  youtube.com/playlist?list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu.
  The same playlist is linked from the instructor's own teaching page.
- Official lecture playlist: 73 videos enumerated directly on
  2026-10-06 (Week 1-12 lecture and tutorial titles, see
  source_gaps.md attachment A). Transcripts not inspected.
- NPTEL candidate overview: https://onlinecourses.nptel.ac.in/noc26_cs97/preview
  returns a JavaScript loading shell only. No overview text extracted.
  Exact equivalence between that NPTEL entry and this IITM BS course is
  unproven. A third-party notes repo claims noc26-cs97 maps to a
  separate 36-video playlist. It is treated as unconfirmed.
- Teaching repository: https://github.com/Chandan-IISc/IITM_GenAI.
  35 paths enumerated (16 notebooks, 9 note PDFs, 6 training GIFs).
  File inventory only. Contents not inspected or executed.
- Full audit: see source_manifest.md and source_gaps.md.

## What this tree is

Standalone deep crash-course build from the v2 prompt pack.
Build root: ~/workspace/stanford-frontier-ai/v2-pack/math-genai/
Prompt: ~/workspace/prompt-pack-v2/frontier_ai_prompt_pack_v2/prompts/05_math-genai_crash_course_prompt.md
Parent units: 10. Coverage ledger rows: 120.

RUN 1 scope only: identity, artifact audit, prerequisite graph, course
map, diagnostics, and the first foundation lesson (U01). Do not begin
RUN 2 here.

## Course map (parent units)

- math-genai-U01: Probabilistic generative modelling, RUN 1 lesson
  delivered (C01-C12).
- math-genai-U02: Variational divergence minimization, planned.
- math-genai-U03: GAN foundations and applications, planned.
- math-genai-U04: Wasserstein and improved adversarial training,
  planned.
- math-genai-U05: Variational autoencoders, planned.
- math-genai-U06: Discrete latent modelling and VQ-VAE, planned.
- math-genai-U07: DDPM derivation and parameterizations, planned.
- math-genai-U08: Diffusion variants and implementation, planned.
- math-genai-U09: Score-based models and autoregressive LMs, planned.
- math-genai-U10: LLM inference, quantization, and alignment, planned.

Detailed unit-to-week and unit-to-prerequisite maps: course_map.md.

## File index

| File | Content |
|---|---|
| README.md | this file |
| state.md | RUN 1 checkpoint and resume point |
| source_manifest.md | source register with inspection boundaries |
| source_gaps.md | unresolved source gaps, playlist titles attachment |
| coverage_matrix.md | 120 coverage rows with statuses |
| course_map.md | unit map: weeks, prerequisites, figure/assessment status |
| prerequisites.md | prerequisite graph and local P03/P06/P07/P08 remediation |
| notation_and_shapes.md | symbol registry for U01 |
| glossary.md | U01 term definitions |
| visual_audit.md | unit-to-figure audit for RUN 1 lesson |
| errors.md | error and uncertainty ledger |
| mastery_ledger.md | mastery tracking rules (empty at RUN 1) |
| role_gap_map.md | role bridge stub (RUN 4) |
| currentness.md | date-bound claims |
| lessons/u01/lesson-01-probabilistic-generative-modelling.md | RUN 1 lesson (U01, 12 leaf concepts) |
| lessons/u01/keys.md | answer keys for the U01 lesson (separate file) |
| lessons/u02/lesson-02-variational-divergence-minimization.md | RUN 2 lesson (U02, 12 leaf concepts) |
| lessons/u02/keys.md | answer keys for the U02 lesson (separate file) |
| lessons/u02/lesson-02b-first-source-block.md | RUN 2 source block: W1L3, W1L4, W5T10 (title-level) |
| lessons/u02/keys-source-block.md | answer keys for the source block (separate file) |
| diagnostics/diagnostic-01-genai-foundations.md | placement diagnostic with scoring rubric |
| diagnostics/keys-01.md | diagnostic answer key (separate file) |
| labs/lab-01-divergence-estimation.md | RUN 2 lab: divergence estimation (5 tasks) |
| labs/keys-lab-01.md | lab answer key (separate file) |
| interview/interview-u02-questions.md | RUN 2 interview bank U02 (questions only) |
| interview/keys-u02.md | interview answer key (separate file) |
| visuals/u01/f01_data_vs_model.png | computed figure, read back 2026-10-06 |
| visuals/u02/f01_jensen_gap.png ... f04_mc_convergence.png | computed figures, read back 2026-10-06 |
| compute_run2.py | RUN 2 number audit: reproduces every lesson number |
| render_u02.py | RUN 2 figure render script (matplotlib) |
| lessons/u03 .. u10, capstones/ | skeleton, RUN 3 onward |

## Continuation

Resume with the template in state.md. Read README.md, state.md,
source_manifest.md, coverage_matrix.md, notation_and_shapes.md,
visual_audit.md, and errors.md first.
