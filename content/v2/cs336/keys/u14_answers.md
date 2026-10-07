# U14 answer key

All numeric claims from `../visuals/compute_u14.py` (synthetic toys,
executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

Wasted = 0.416 * 1e9 tokens = 416M tokens per epoch, at 4
bytes/token that is the planning arithmetic (the lesson states the
method).

## A1

(a) The threshold dials keep-good vs keep-bad. (b) t=0.5:
0.94/0.06. (c) t=0.5 keeps 94% of good at 6% bad: the balanced
point on the toy, t=0.7 if purity matters more.

## A2

(a) Blocklist, classifiers, human review, recall-first posture.
(b) 0.33 bad kept is unacceptable for safety classes. (c) Separate
safety classifiers with recall > 0.99 on probes, plus human review
of the news slice, accept the coverage cost.

## A3

(a) Exact: hash equality. Near: shingle Jaccard. (b) 41.6% dup
rate, 1,011 triple-plus. (c) 1T * (1-0.416) = 584B unique tokens.

## A4

(a) (1-e^{-kn/m})^k, k near (m/n)*ln2. (b) 0.0216 at k=6, minimum
of the sweep. (c) m = 8e9 bits, k = 6, FPR = 0.0216 (same ratio).

## A5

(a) Shingles -> MinHash signatures -> LSH bands. (b) 0.113 vs
0.125. S-curve points as listed. (c) Solve: need b=25, r=6 type
parameters, verify 1-(1-0.9^6)^25 >= 0.999 and 1-(1-0.5^6)^25 <=
0.1: compute and adjust.

## A6

(a) n-gram index of eval sets, drop training docs over threshold.
(b) 300 items, 13-grams, 80% rule on the toy. (c) Quarantine the
run: audit what the new eval overlaps, drop or restart affected
portions, never train on after release.

## A7

(a) Per-doc multiplier tgt/src, sampled in the loader. (b) 1.67
for code. (c) Update the loader weights live, the change applies
to future batches, logged with a step number.

## A8

(a) Search the simplex with proxy runs. (b) w=1.0, L=2.500 vs
2.850. (c) Fractional factorial or Bayesian optimization over the
5-simplex. 20 runs: screen with 8, refine around the best 3.

## A9

(a) Paraphrase, distillation, synthetic problems, validate every
mode. (b) 800k at 10% after the 80% check pass. (c) Do not use
synthetic: the generator cannot teach what it does not know.
collect real data or drop the domain.

## A10

(a) Classifier data, thresholds, source mix. (b) Audit keep rates
per domain at fixed t. (c) Fix the classifier or its threshold
for the dialect, ship only with the audit clean.

## A11

(a) Drop one stage at a time vs the full pipeline. (b) Dedup
+0.06, filter +0.03, reweight +0.01. (c) Ablate the two most
expensive stages first, use the remaining budget for the top
interaction.

## A12

(a) Recall rises logistically in repetitions. (b) 0.18/0.34/0.61/
0.78. (c) Cap repeats at ~3x for the bulk, allow deliberate
upsampling only for vetted high-value data.
