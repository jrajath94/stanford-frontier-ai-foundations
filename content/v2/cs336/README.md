# CS336 , Language Modeling from Scratch (v2 pack)

Course builder scope: U01 through U09 plus the root identity set (first
builder). Second builder scope: U10 through U18 plus consolidation
(capstones, transfer sets, oral defenses, crash course, cheatsheet, root
doc updates). Coverage: 216/216 rows TAUGHT+ASSESSED.
Independent audit: later, by a different agent. This build is not audited.

## Source-grounding note

This material is source-grounded on the Stanford CS336 Spring 2026 offering
("Language Modeling from Scratch", instructors Percy Liang and Tatsunori
Hashimoto, March 30 to June 3, 2026) plus requested deep implementation
branches. The build baseline is October 6, 2026.

Honesty rules in force:

- The nineteen scheduled sessions (17 core lectures plus 2 guest lectures)
  map without omissions in `course_map.md`. Session titles and dates come
  from a secondary study repository, not from an inspected official artifact.
  Every leaf therefore carries claim class REQUESTED-BRANCH or
  OFFICIAL-SOURCE and status PLANNED / SOURCE ATTRIBUTION PENDING until an
  inspected artifact verifies it.
- No leaf is called "covered" because it appears in a prompt or inventory.
  A leaf closes only when it is taught in a lesson, assessed in an exercise,
  lab, or interview bank, and recorded in `coverage_matrix.md` with evidence.
- Numbers in lessons are computed by committed scripts on this machine
  (numpy, matplotlib) or marked "Not in source". No benchmark, speedup, or
  test result is invented.
- Assignment learning objectives are extracted from reported summaries.
  All practice is original equivalent work. No assessed work is solved.

## Directory layout

- `README.md` , this file.
- `state.md` , build state and RUN checkpoints.
- `source_manifest.md` , source register with inspection boundaries.
- `source_gaps.md` , unresolved sources and missing artifacts.
- `course_map.md` , nineteen sessions mapped to units, with claim classes.
- `index.md` , learner navigation.
- `prerequisites.md` , unit prerequisites, local remediation, diagnostic.
- `notation_and_shapes.md` , shared symbols, shapes, units.
- `glossary.md` , terms with one-meaning definitions.
- `currentness.md` , baseline dates and update policy.
- `coverage_matrix.md` , 108 concept rows (U01-U09) with status and evidence.
- `visual_audit.md` , figure audit rows per unit.
- `mastery_ledger.md` , per-concept learner mastery state (initial: untested).
- `errors.md` , error log for the build and for learner traps.
- `role_gap_map.md` , role rubrics versus unit coverage.
- `lessons/` , one lesson file per unit, 12 concepts each, 15-item contract.
- `keys/` , answer keys for lesson assessments, kept separate from lessons.
- `labs/` , one lab per unit plus a lab key with execution-verified outputs.
- `interview/` , interview banks per unit, questions and keys separate.
- `visuals/` , matplotlib render scripts and computed PNG figures.
- `capstones/` , reserved for the consolidation pass (second builder).

## Shared prerequisite bridges

Prerequisite modules P01 through P24 live once at
`../shared/prerequisites/`. This course links them and never rebuilds them.
Each unit lesson carries its own local remediation block.

## Binding laws

1. ASD-STE100: `~/workspace/skills/ste-lint/bin/ste_check.py` must report zero
   hard fails on every deliverable, including `.py` render and lab scripts.
2. `~/workspace/skills/watermarks-remover/bin/wm_clean.py` Layer A runs on
   every deliverable before ship.
3. Visual system `visual_system_generic.md` binds every figure. Figures are
   matplotlib-computed from committed scripts. Every PNG passes a PIL check
   (IHDR, IDAT, IEND chunks only) and a metadata re-save step.
4. No invented numbers, timestamps, benchmarks, or test results. No exam
   dumps. No authentication bypass.
