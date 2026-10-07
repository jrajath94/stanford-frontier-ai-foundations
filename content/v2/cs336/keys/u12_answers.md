# U12 answer key

All numeric claims from `../visuals/compute_u12.py` (synthetic toys,
executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

SE = sqrt(0.8*0.2/40) = 0.063, half-width 1.96*0.063 = 0.124.

## A1

(a) exp(mean NLL), effective branching factor. (b) 9.97, 5.47,
2.72. (c) No: different tokenizers need bits per byte.

## A2

(a) n-gram overlap and canaries. (b) 100.0% overlap, canary recall
1.00. (c) 2% on 13-grams: investigate, it exceeds chance but may be
boilerplate: check the matching items.

## A3

(a) loss*tokens/bytes/ln2, common denominator across tokenizers.
(b) A 0.687, B 0.605: B wins. (c) Ask for bits per byte on the same
corpus, or the token counts to compute it.

## A4

(a) Internal steers, external judges, never tune on the external.
(b) 1-0.95^20 = 64%. (c) Freeze v1 as the reported number, start
v2 fresh, never reuse a tuned-against set as external.

## A5

(a) Macro average over tasks, slices carry CIs. (b) 0.80+-0.124,
0.65+-0.121, 0.72+-0.039. (c) 1 point on n=500: SE ~0.02 per model,
the gap is ~1 SE: not significant, demand CIs or more items.

## A6

(a) The prompt is part of the instrument. (b) 0.717, 0.723, 0.657:
6.6-point format effect. (c) The comparison is invalid, rerun both
models on one fixed format.

## A7

(a) Report model, items, and full decoding settings with every
score. (b) 0.865 vs 0.455 top-1: the setting moves the reading.
(c) Misconfigured: pass@1 wants greedy or low temperature.
temperature 1.0 measures the distribution, not the mode.

## A8

(a) (agree-expected)/(1-expected), chance-corrected agreement.
(b) 0.83 raw, 0.63 kappa: substantial, not near-perfect. (c) No:
0.4 is weak, use humans for ranking, the judge only for
iteration with calibration.

## A9

(a) Differences inside the seed band are not differences. (b) Range
0.034, a 3-point move is noise. (c) n=200, one seed each: SE ~0.03
per model. 1 point is noise: no verdict.

## A10

(a) Per-slice (n, p, CI), rank only on significant gaps. (b) Gap
0.15 vs combined noise ~0.17: not significant. (c) n=30: CI
half-width ~0.17. 0.90 is [0.73, 1.0]: do not ship on this alone.

## A11

(a) Input, format, grading gaps. (b) Length 40 vs 400 tokens, the
score's meaning is taxed. (c) Low validity: convert to free-form
grading or report the multiple-choice score as a proxy with the
gap stated.

## A12

(a) (tokens/1000*price)/accuracy. (b) $0.0022 vs $0.0056: A wins.
(c) C beats A when its accuracy gain outweighs 9x the cost: never
at these numbers unless errors are costly, add the error cost.
