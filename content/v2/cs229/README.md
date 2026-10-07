# CS229 Machine Learning, Stanford, crash course

Build root: `v2-pack/cs229/`. Date: 2026-10-06. Baseline: October 6, 2026.

This is an original learning course built from public course artifacts for
CS229 Machine Learning at Stanford. It does not reproduce instructor notes
verbatim. It teaches the mechanisms, the math, and the engineering that sit
behind the official syllabus.

## How to use this course

1. Read `diagnostics/diagnostic-01-ml-foundations.md`. Attempt it
   closed-book. Check your answers in `diagnostics/keys-01.md`.
2. Read `prerequisites.md`. Use the local remediation inside each unit
   lesson when a dependency list flags a gap. The shared prerequisite
   bridges live at `../shared/prerequisites/` and are linked, not copied.
3. Work units U01 to U17 in order. Each unit lesson lists its own
   not-yet-understood dependencies at the top.
4. Do the lab for each unit. Run the code. Check the keys after.
5. Use `interview/` for breadth recall, deep ladders, and transfer sets.
   Keys live in separate files. Keep them closed until you attempt.
6. Finish with `capstones/`.

## Directory map

- `lessons/uNN/` one lesson file per unit, answer keys beside it
- `visuals/uNN/` computed figures with render scripts
- `labs/` runnable labs with separate keys
- `interview/` question files and separate key files
- `capstones/` two capstone projects
- `diagnostics/` entry diagnostics
- root: `index.md`, `course_map.md`, `glossary.md`, `prerequisites.md`,
  `notation_and_shapes.md`, `crash-course.md`, `cheatsheet.md`

## Course status and honesty rules

- Every requested leaf concept starts PLANNED / SOURCE ATTRIBUTION
  PENDING. A row flips to TAUGHT+ASSESSED only when a lesson teaches
  it and an assessment item tests it.
- `state.md` carries the RUN checkpoint. `coverage_matrix.md` carries
  the per-row status. `source_manifest.md` carries the artifact
  inspection boundaries. `source_gaps.md` carries what is unreachable.
- No claim of coverage without a denominator and evidence.

### Claim classification

- Fact: a source-anchored statement (SRC-01 section/page cited) or a
  computed value with seed, script, and asserted output.
- Inference: a conclusion derived in the lesson from facts, labeled
  where it goes beyond the notes (e.g. "the notes do not develop
  this, taught as standard background").
- Guess: a labeled hypothesis or scenario, always marked
  HYPOTHETICAL (capstone B inputs, research-extension claims,
  "illustrative" toy numbers whose pattern, not value, is the
  claim).

## Laws

- ASD-STE100 on all prose. Every file passes `ste_check.py` with zero
  hard fails.
- Watermark remover Layer A on every deliverable.
- Figures: matplotlib-computed, with committed render scripts and
  audited values, in the shared visual system. No AI-generated images
  unless required, and then only Muse native tools.
- Never invent numbers, benchmarks, or test results.
