# U15 answer key

All numeric claims from `../visuals/compute_u15.py` (synthetic toys,
executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

2M * 0.49 = 980,000 supervised tokens (toy fractions).

## A1

(a) Pretrain: base LM objective. Midtrain: continued pretraining,
new mix. Post-train: behavior (SFT/RL). (b) 100x/10x/1x toy FLOP
ratios. (c) Midtraining: the objective defines the stage, not the
data's look.

## A2

(a) Raise RoPE theta, train on long documents. (b) 6.28e4 ->
3.14e6, ratio 50. (c) Theta x125, plus a long-document mix with
real long-range dependencies, verify with needle-in-haystack
probes.

## A3

(a) Fixed byte-exact format both sides. (b) 0.17/0.34/0.49 splits.
(c) Add modality boundary tokens and roles, pin the version.

## A4

(a) 1 on assistant tokens, 0 elsewhere. (b) 0.49, 492,356 per 1M.
(c) Mask 1 on the call JSON (assistant behavior), 0 on results.

## A5

(a) Correctness, difficulty, diversity, cleanliness. (b) 1M ->
250k through the funnel. (c) 10k pairs: filter hard on correctness
first, then balance tasks, small data must be clean.

## A6

(a) Trace with calls and results, mask 1 on calls, 0 on results.
(b) 120 supervised of 200. (c) Check the mask boundaries per call.
a 5-call trace has 5 mask segments to verify.

## A7

(a) Exposure = kept x epochs, the bar rises with exposure. (b)
7,500 bad lessons at 1%. (c) Near-human-grade bar: 5k x 10 = 50k
exposures, review a large sample, filter hard.

## A8

(a) Pack with boundary masks vs pad. (b) 47.6% vs 22.0%, 344 bins.
(c) Isolate the giant document (its own batch or truncated
windows), do not let it poison the packing.

## A9

(a) 2dr per pair, frozen base. (b) 131,072 vs 16,777,216. 8.39M vs
1074M. (c) LoRA: 10 adapters share one base, full tuning would
need 10 full copies.

## A10

(a) Track base-suite loss per checkpoint. (b) Mean +0.15 nats.
(c) Fix: 0.5 nats is large, add replay, lower LR, or shorten the
run before shipping.

## A11

(a) One suite per stage, flag regressions outside CIs. (b)
Instruction +0.10, base -0.02 (noise), context flat: clean SFT.
(c) Diagnose: check if the drop is real (CI), then add base-data
replay to the midtrain stage.

## A12

(a) Weights, optimizer moments, RNG, data position. (b) 84.0 GB
full, 14.0 GB weights-only. (c) No: the adapter's base changed.
retrain or verify compatibility on evals first.
