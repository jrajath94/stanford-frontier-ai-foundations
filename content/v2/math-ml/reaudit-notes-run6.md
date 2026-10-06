# Re-audit notes, math-ml RUN 6 fix attempt 1 (2026-10-06)

## Scope
Re-audit of touched files for F-01..F-09. Never wrote math-ml content.
Build root: ~/workspace/stanford-frontier-ai/v2-pack/math-ml/

## Evidence gathered (independent execution)
- compute_run4.py re-run exit 0. Task 3(c) line:
  E[w]=4.346313 bias^2=7.042053 var=0.035717 mse=7.077770 sum=7.077770.
  Matches keys-lab-05.md to 6 digits. Script uses f_true_c = 7.0,
  fresh default_rng(7) stream.
- compute_run2_labs.py re-run exit 0. All keys-lab-01/02 numbers
  reproduce. kappa = 4002.000750124839 (both script and key).
- F-10 vector check: max abs diff vs [-1,2,-1]/sqrt(6) =
  9.277261181495078e-10; vs [1,-2,1]/sqrt(6) = 1.6329931609277262.
- PNG sweep script: 35 on disk; all 35 embedded with alt text; all 35
  have Caption lines with "Source: original" + "Shell".
- ste_check.py on 11 touched files: all exit 0.

## Key judgments
- F-10 was a genuine false positive; original auditor's proposed
  "fix" would have inverted the sign. Recorded as rejected.
- 35 vs 38: definitive list of the 35 recorded in the addendum; 38 was
  a miscount by the original auditor.
- All F-01..F-09 PASS. No remaining items. Recommend swap/packaging.
