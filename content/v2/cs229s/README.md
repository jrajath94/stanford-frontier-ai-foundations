# CS229S: Systems for Machine Learning (v2-pack course tree)

Stanford CS229S, Fall 2024. Systems for Machine Learning.
Build scope of this tree: all 10 units, built by two course
builders (U01-U05 first builder, U06-U10 second builder) plus
full consolidation. A different agent audits the complete
course later.

Baseline: 2026-10-06. Course root: `v2-pack/cs229s/`.

## Honesty note

Official Fall 2024 calendar anchors all core families. Leaf
expansion is a required teaching plan, not proof each subtopic
appears verbatim in lectures. Every leaf concept carries claim
classification with dates. Leaves stay marked PLANNED / SOURCE
ATTRIBUTION PENDING until artifact verification. Never treat a leaf
as lecture-verified without artifact evidence.

## What lives here

- Root identity set: this README, `state.md`, `source_manifest.md`,
  `source_gaps.md`, `course_map.md`, `index.md`, `prerequisites.md`,
  `notation_and_shapes.md`, `glossary.md`, `currentness.md`,
  `coverage_matrix.md`, `visual_audit.md`, `mastery_ledger.md`,
  `errors.md`, `role_gap_map.md`.
- `lessons/u01` .. `lessons/u10`: lesson files with the full
  15-item contract per concept, plus `keys.md` answer keys kept
  separate.
- `labs/`: one lab per unit plus `keys-lab-0X.md` keys. Every lab
  ran on this machine. Expected outputs are recorded.
- `interview/`: per-unit interview banks plus course-level
  `transfer-sets.md` (with `keys-transfer.md`) and
  `oral-defenses.md` (with `keys-oral.md`). Questions and keys
  are separate files.
- `visuals/u01` .. `visuals/u10`: committed render scripts
  (`render_u0X.py`) and matplotlib-computed PNG figures. Every PNG
  passes a metadata strip inside the script. No AI-generated
  images.
- `diagnostics/`: prerequisite diagnostic plus keys.
- `capstones/`: unit labs end with unit-scope replication
  proposals, labeled PROPOSED and not executed. Course-level
  `capstones/`: (a) executed SSM replication + falsifiable
  extension with computed figures. (b) applied/FDE RAG serving
  design with labeled hypothetical numbers and computed figures.
- `crash-course.md` and `cheatsheet.md`: course-root summaries
  synthesized from all 10 units.

## Prerequisite bridges

Shared bridges P01-P24 live at
`v2-pack/shared/prerequisites/`. This tree links them, never
rebuilds them. Each unit carries local remediation in its lesson
file. See `prerequisites.md`.

## Claim classes

- `CALENDAR-ANCHORED`: family appears on the official Fall 2024
  calendar with a date. See `source_manifest.md`.
- `PLANNED / SOURCE ATTRIBUTION PENDING`: required teaching leaf,
  not yet artifact-verified. This is the default class for every
  concept row until a second pass verifies it.
- `SOURCE-UNREACHABLE`: a concept whose sources cannot be
  reached. None claimed yet. Gaps are listed in `source_gaps.md`.

## Build law

1. ASD-STE100 on every file, including `.py`. Zero hard fails.
   Checked with `~/workspace/skills/ste-lint/bin/ste_check.py`.
2. `wm_clean.py` Layer A on every deliverable.
3. Visual system from the course prompt binds every figure.
   Matplotlib-computed figures preferred. PNG metadata stripped
   in every render script.
4. Never invent numbers, timestamps, benchmarks, or test results.
5. Expansion first, then densify. No token limits.

## Status

See `state.md` for the build checkpoint. See `coverage_matrix.md`
for per-row status with denominators.
