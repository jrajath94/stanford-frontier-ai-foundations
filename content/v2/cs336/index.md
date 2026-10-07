# CS336 index , learner navigation

## Start here

1. Read `README.md` for the source-grounding note and honesty rules.
2. Take the diagnostic in `prerequisites.md`. Follow the remediation links
   for any item you cannot answer.
3. Read `notation_and_shapes.md` once. Keep it open while you study.
4. Work units in order: U01 through U18.
   Each unit lists its prerequisites and its local remediation block.
5. Consolidate: the two capstones in `capstones/`, the transfer sets and
   oral defenses in `interview/`, then `crash-course.md` and
   `cheatsheet.md` for review.

## Units (first half)

| Unit | Lesson | Lab | Interview bank | Figures |
|------|--------|-----|----------------|---------|
| U01 Text representation and tokenization | `lessons/u01_tokenization.md` | `labs/u01_lab.md` | `interview/u01_questions.md` | `visuals/u01_*.png` |
| U02 Executable tensor programming and resource accounting | `lessons/u02_tensor_programming.md` | `labs/u02_lab.md` | `interview/u02_questions.md` | `visuals/u02_*.png` |
| U03 Transformer block mechanics | `lessons/u03_transformer_block.md` | `labs/u03_lab.md` | `interview/u03_questions.md` | `visuals/u03_*.png` |
| U04 Attention alternatives and expert architectures | `lessons/u04_attention_moe.md` | `labs/u04_lab.md` | `interview/u04_questions.md` | `visuals/u04_*.png` |
| U05 Optimization and training correctness | `lessons/u05_optimization.md` | `labs/u05_lab.md` | `interview/u05_questions.md` | `visuals/u05_*.png` |
| U06 GPU/TPU architecture and profiling | `lessons/u06_gpu_tpu.md` | `labs/u06_lab.md` | `interview/u06_questions.md` | `visuals/u06_*.png` |
| U07 Kernels and IO-aware attention | `lessons/u07_kernels.md` | `labs/u07_lab.md` | `interview/u07_questions.md` | `visuals/u07_*.png` |
| U08 Data and optimizer sharding | `lessons/u08_data_sharding.md` | `labs/u08_lab.md` | `interview/u08_questions.md` | `visuals/u08_*.png` |
| U09 Model and hybrid parallelism | `lessons/u09_model_parallel.md` | `labs/u09_lab.md` | `interview/u09_questions.md` | `visuals/u09_*.png` |

## Units (second half)

| Unit | Lesson | Lab | Interview bank | Figures |
|------|--------|-----|----------------|---------|
| U10 Scaling laws and extrapolation | `lessons/u10_scaling_laws.md` | `labs/u10_lab.md` | `interview/u10_questions.md` | `visuals/u10_*.png` |
| U11 Inference algorithms and serving | `lessons/u11_inference.md` | `labs/u11_lab.md` | `interview/u11_questions.md` | `visuals/u11_*.png` |
| U12 Evaluation and measurement validity | `lessons/u12_evaluation.md` | `labs/u12_lab.md` | `interview/u12_questions.md` | `visuals/u12_*.png` |
| U13 Data sourcing and transformation | `lessons/u13_data_sourcing.md` | `labs/u13_lab.md` | `interview/u13_questions.md` | `visuals/u13_*.png` |
| U14 Filtering, deduplication, mixing, and synthesis | `lessons/u14_data_mixing.md` | `labs/u14_lab.md` | `interview/u14_questions.md` | `visuals/u14_*.png` |
| U15 Midtraining and supervised adaptation | `lessons/u15_midtraining.md` | `labs/u15_lab.md` | `interview/u15_questions.md` | `visuals/u15_*.png` |
| U16 Preference alignment and reasoning RL | `lessons/u16_alignment_rl.md` | `labs/u16_lab.md` | `interview/u16_questions.md` | `visuals/u16_*.png` |
| U17 Multimodality and training-system integration | `lessons/u17_multimodal_systems.md` | `labs/u17_lab.md` | `interview/u17_questions.md` | `visuals/u17_*.png` |
| U18 Guest material and final research defense | `lessons/u18_guests_defense.md` | `labs/u18_lab.md` | `interview/u18_questions.md` | `visuals/u18_*.png` |

## Consolidation

| Item | Path |
|------|------|
| Capstone A: research replication + falsifiable extension | `capstones/capstone_research.md` (run `capstones/capstone_research_run.py`) |
| Capstone B: applied/FDE deployment plan | `capstones/capstone_applied.md` (run `capstones/capstone_applied_run.py`) |
| Transfer sets (10 changed-scenario sets) | `interview/transfer-sets.md`, key in `interview/keys-transfer.md` |
| Oral defenses (10 deep ladders x 8 follow-ups) | `interview/oral-defenses.md`, key in `interview/keys-oral.md` |
| Crash course (all 18 units) | `crash-course.md` |
| Cheatsheet (formulas, decision rules) | `cheatsheet.md` |

Answer keys live in `keys/` (lesson assessments) and `labs/` (lab keys).
Interview answer keys live in `interview/` beside the question files.
Keys are separate files by design, do not open a key before attempting
the questions closed-book.

## Reference files

- `notation_and_shapes.md` , symbols, shapes, units.
- `glossary.md` , one-meaning definitions.
- `prerequisites.md` , bridge links, remediation, diagnostic.
- `course_map.md` , nineteen sessions and their unit mapping.
- `coverage_matrix.md` , what is taught and assessed, with denominators.
- `mastery_ledger.md` , your per-concept mastery record (starts untested).
- `errors.md` , common traps and the build error log.
- `role_gap_map.md` , how units serve each target role.
- `currentness.md` , dates and update policy.
- `visual_audit.md` , figure audit rows.
- `source_manifest.md`, `source_gaps.md` , provenance and open gaps.

## Study protocol

For each concept: read the lesson contract items 1 through 12, attempt
item 13 closed-book, check the key, then do the lab tasks. Log confident
mistakes in `mastery_ledger.md`. Revisit on a spaced schedule.
