# Oral defenses key

## D1

1. L = E + A*(C/C0)^{-a}. E the data floor.
2. Fit recovers 0.340, RMS 0.0031.
3. log(L-E) = logA - a log(C/C0), least squares on [1, log C].
4. lstsq as in the lab.
5. Nonlinear when E unknown, linear when E known.
6. a = coef[1] without negation, or E set above the data.
7. iid lognormal, real runs correlate.
8. Joint fit over (E, a, A), wider bands, residual check.

## D2

1. K/V tensors appended per step, per layer.
2. 131,072 bytes/token. 8.59 GB at batch 8 x 8192.
3. Two tensors, L layers, h_kv heads, d_h dims, byte width.
4. `kv_bytes` as in the lab.
5. Recompute is O(T^2) per step, cache is O(T) memory.
6. MHA not GQA (4x), or float32 not bf16 (2x).
7. Real lengths vary, the toy is worst-case uniform.
8. Budget = percentile length * per-token * batch, shed policy.

## D3

1. (agree - expected)/(1 - expected), raw agreement ignores
   chance.
2. Table with kappa 0.4, e.g. agree 0.7, expected 0.5.
3. Expected from the row/column marginals.
4. The `kappa` function as in the lab.
5. Human-judge is the calibration, judge-judge is consistency.
6. Agreement below chance: systematic disagreement or label
   flip.
7. One prompt set cannot generalize, judge prompts overfit.
8. Multiple judges, multiple sets, kappa per slice, adjudicated
   disagreements.

## D4

1. Bit array, k hashes, no false negatives.
2. 8 bits/item, k=6, FPR 0.0216.
3. Bit-survival e^{-kn/m}, all k set: the power.
4. add/contains as in the lab.
5. Hash set: exact, 8x memory.
6. Non-uniform hash or correlated inputs, check the hash.
7. Real hashes approximate uniform, adversarial inputs break it.
8. Feed known-duplicate/known-unique sets, compare measured vs
   formula.

## D5

1. Loss on assistant tokens, teaches behavior, not data.
2. 3 pairs, toy token counts from the lab.
3. System text is a condition on the output, not a target to emit.
4. The mask builder as in the lab.
5. Mask-all teaches prompt parroting.
6. System tokens carried loss (mask bug).
7. Real traces have messy roles and tool outputs.
8. Mask: user 0, assistant 1, tool 0, redact tool text first.

## D6

1. r = pi_new/pi_old, min(rA, clip(r)A).
2. Terms [0.7, 0.9, -1.0, 1.2, 1.2].
3. The min trims the upside, keeps the downside: pessimistic.
4. `ppo_obj` as in the lab.
5. Vanilla PG: unbiased, unstable. PPO: biased, stable.
6. eps too small or advantages all negative with r > 1+eps.
7. Noisy advantages + clipping = dropped signal, the bias is
   real.
8. Sweep eps in {0.1, 0.2, 0.3}, report KL and reward per eps.

## D7

1. Per-stage ms, speedup bounded by the optimized fraction.
2. 143 ms, backward 55 ms = 38.5%. 2x -> 116 ms, 19.2%.
3. New total = total - share + share/speedup.
4. `amdahl` as in the lab.
5. Profile-driven wins, gut-feel optimizes the wrong stage.
6. The bottleneck moved: re-profile after every fix.
7. Real stages overlap and vary, the toy is lockstep.
8. Speed up each stage in turn, measure the real delta.

## D8

1. Full 2^n, fractional 2^{n-k}, OAT 1+n.
2. 16/8/5.
3. Counting formulas.
4. `design_runs` as in the lab.
5. Full: interactions, costly. Fractional: cheap, aliases.
   OAT: cheap, no interactions.
6. The aliased interaction confounds a main effect: the design
   resolution is too low.
7. Real outcomes are noisy, the toy is deterministic.
8. Inject an interaction, run all three designs, compare the
   conclusions.

## D9

1. Verbatim: exact text, paraphrase: same meaning, new words.
2. 20 planted, 20 found: recall 1.00, overlap 100%.
3. Exact n-gram hits are near-impossible by chance.
4. The overlap counter as in the lab.
5. n-gram: cheap, exact only. Embedding: catches paraphrase,
   costs compute.
6. Paraphrase evades exact matching: add embedding screening.
7. The threshold trades recall against false positives, justify
   it on labeled data.
8. Paraphrase eval items, measure the detection rate.

## D10

1. BA adapter, rank r, count 2*d*r per pair.
2. 8.39M trainable vs 1074M full.
3. A: d*r, B: r*d.
4. The count function as in the lab.
5. LoRA: cheap, limited shift. Full: costly, full shift.
6. Rank too low, LR too small, or the adapter not attached.
7. Everything else frozen: some tasks need base changes.
8. Sweep r in {8, 16, 64, 256} on the new language, report the
   quality curve.
