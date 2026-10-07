# U12 interview key

## B1-B6

B1. e^2.3 = 9.97, effective branching factor.
B2. n-gram overlap and canaries.
B3. Bytes are tokenizer-independent, tokens are not.
B4. Internal steers iteration, external judges, never tune on
the external.
B5. Chance agreement.
B6. (tokens/1000 * price) / accuracy.

## L1

L1.1 sqrt(p(1-p)/n).
L1.2 SE 0.020, CI [0.681, 0.759].
L1.3 1.96 is the normal 97.5th percentile.
L1.4 The CI function. 5-seed range 0.034 inside the band.
L1.5 Bootstrap needs fewer assumptions, debug: correlated seeds.
critique: normal approx fails at small n, experiment: 20 seeds.

## L2

L2.1 Benchmark text in training, canary = planted unique string.
L2.2 100.0% overlap, recall 1.00.
L2.3 Exact n-gram hits prove verbatim presence.
L2.4 The overlap counter, all planted items found.
L2.5 Fuzzy catches paraphrase at compute cost, debug: paraphrase
evades exact, critique: threshold, experiment: paraphrase test.

## E1

(a) A: 0.124, B: 0.121. (b) Gap 0.15, combined SE
sqrt(0.063^2+0.062^2)=0.088. 0.15 < 1.96*0.088: not significant.
(c) 4x items per slice (160/240). Rubric: (a) 1 pt, (b) 2 pts,
(c) 1 pt. Red flag: ranking on the point estimates.

## E2

(a) X: $0.0022, Y: $0.0056. (b) Expected cost incl. errors: X:
0.0016+0.28*0.01=$0.0044. Y: 0.0048+0.15*0.01=$0.0063: X still
wins. (c) Costly errors punish low accuracy, the metric must
include error cost. Rubric: (a) 1 pt, (b) 2 pts, (c) 1 pt.

## D1

Bug: returns raw agreement without subtracting chance agreement.
Fix: kappa = (agree - expected)/(1 - expected) with expected from
marginals. Rubric: find 2 pts, fix 2 pts.

## S1

(1) Freeze the run. (2) Audit overlap of the new eval against
training. (3) Drop or quarantine overlapping docs. (4) Document.
never train on post-release evals.

## S2

No: kappa 0.4 is weak, the judge cannot rank. Use it for
iteration only, with human calibration, rank with humans or a
better judge.

## R1

Gaps: (1) no CIs: significance unknown, add them. (2) no seeds:
noise unknown, run 3-5. (3) no decoding settings: not
reproducible, publish the config.
