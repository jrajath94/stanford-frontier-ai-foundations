# index.md: cs229s course index (U01-U10)

Date: 2026-10-06.

## Start here

1. `README.md`, honesty note, layout, build law.
2. `prerequisites.md`, bridges plus the diagnostic
   (`diagnostics/diagnostic-cs229s.md`).
3. `notation_and_shapes.md`, symbols used across all lessons.
4. `glossary.md`, term definitions.
5. `course_map.md`, unit dependencies and scoping decisions.
6. `crash-course.md`, the condensed 10-unit course.
7. `cheatsheet.md`, formulas and tables.

## Lessons (U01-U10)

- `lessons/u01/lesson-01-sequence-transformer-workload.md`
- `lessons/u02/lesson-02-hardware-compilers.md`
- `lessons/u03/lesson-03-accounting-speculative.md`
- `lessons/u04/lesson-04-cuda-efficient-attention.md`
- `lessons/u05/lesson-05-quantization-sparsity.md`
- `lessons/u06/lesson-06-adaptation-data-pipelines.md`
- `lessons/u07/lesson-07-linear-attention-ssm-fft.md`
- `lessons/u08/lesson-08-serving-moe.md`
- `lessons/u09/lesson-09-parallelism-clusters-scheduling.md`
- `lessons/u10/lesson-10-retrieval-guests-synthesis.md`

Each lesson directory holds `keys.md` with exercise answer keys,
kept separate from the lesson.

## Labs

- `labs/lab-01-sequence-workload.md` / `labs/keys-lab-01.md`
- `labs/lab-02-roofline.md` / `labs/keys-lab-02.md`
- `labs/lab-03-kvcache-speculative.md` / `labs/keys-lab-03.md`
- `labs/lab-04-tiling-softmax.md` / `labs/keys-lab-04.md`
- `labs/lab-05-quant-prune.md` / `labs/keys-lab-05.md`
- `labs/lab-06-adaptation-pipelines.md` / `labs/keys-lab-06.md`
- `labs/lab-07-ssm-fft.md` / `labs/keys-lab-07.md`
- `labs/lab-08-serving-moe.md` / `labs/keys-lab-08.md`
- `labs/lab-09-parallelism-scheduling.md` / `labs/keys-lab-09.md`
- `labs/lab-10-retrieval-synthesis.md` / `labs/keys-lab-10.md`

All labs ran on this machine. Keys record observed outputs.
Each `verify_lab0X.py` is the executed check script.

## Interview banks

- `interview/interview-u01-questions.md` / `interview/keys-u01.md`
- `interview/interview-u02-questions.md` / `interview/keys-u02.md`
- `interview/interview-u03-questions.md` / `interview/keys-u03.md`
- `interview/interview-u04-questions.md` / `interview/keys-u04.md`
- `interview/interview-u05-questions.md` / `interview/keys-u05.md`
- `interview/interview-u06-questions.md` / `interview/keys-u06.md`
- `interview/interview-u07-questions.md` / `interview/keys-u07.md`
- `interview/interview-u08-questions.md` / `interview/keys-u08.md`
- `interview/interview-u09-questions.md` / `interview/keys-u09.md`
- `interview/interview-u10-questions.md` / `interview/keys-u10.md`
- `interview/transfer-sets.md` / `interview/keys-transfer.md`
  (10 changed-scenario sets, course-level)
- `interview/oral-defenses.md` / `interview/keys-oral.md`
  (10 deep ladders x 8 follow-ups, course-level)

## Figures

- `visuals/u01/render_u01.py` through `visuals/u10/render_u10.py`
  generate the PNG figures in the same directories. Every script
  strips PNG metadata inside `main()`.

## Capstones

- `capstones/capstone-a-ssm-replication.md` +
  `capstones/replicate_ssm.py` (executed, negative result
  reported honestly) + `capstones/ssm_identity.png`,
  `capstones/ssm_timing.png`.
- `capstones/capstone-b-rag-serving.md` +
  `capstones/rag_serving_model.py` (executed, hypothetical
  numbers labeled) + `capstones/rag_cost_latency.png`.

## Tracking

- `coverage_matrix.md`: per-row status with denominators
  (120/120).
- `visual_audit.md`: figure-to-concept mapping.
- `mastery_ledger.md`: assessment evidence (learner evidence:
  none yet, zero learner attempts on these materials).
- `errors.md`: build errors and fixes.
- `role_gap_map.md`: role rubric coverage and gaps.
- `currentness.md`: date-sensitive claims.
