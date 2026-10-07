# Oral defenses , 10 deep ladders x 8 follow-ups

Each ladder: define, toy, derive, implement, compare, debug,
critique, design. Keys in `keys-oral.md`.

## D1 , power-law fitting (U10-C01)

1. Define the scaling law and the residual term.
2. Toy: a=0.34 -> 0.340, RMS 0.0031.
3. Derive the log-space regression.
4. Implement the fit function.
5. Compare nonlinear least squares with log-space linear fit.
6. Debug: the fit reports a negative exponent. What happened?
7. Critique the toy noise model.
8. Design: fit with E unknown. What changes?

## D2 , KV cache (U11-C02)

1. Define the per-token cache and where it lives.
2. Toy: 128 KB/token, 8.59 GB.
3. Derive 2*L*h_kv*d_h*bytes.
4. Implement `kv_bytes`.
5. Compare recompute-all with cache-all.
6. Debug: memory doubles vs the formula. What happened?
7. Critique the toy's uniform lengths.
8. Design: a cache budget for variable lengths.

## D3 , judge agreement (U12-C08)

1. Define kappa and why raw agreement lies.
2. Toy: build a 0.4-kappa confusion table.
3. Derive kappa from the marginals.
4. Implement `kappa`.
5. Compare human-judge with judge-judge agreement.
6. Debug: kappa is negative. What happened?
7. Critique one judge on one prompt set.
8. Design: a judge calibration study.

## D4 , Bloom dedup (U14-C02)

1. Define the structure and its guarantee.
2. Toy: k=6, FPR 0.0216.
3. Derive (1-e^{-kn/m})^k.
4. Implement add/contains.
5. Compare with an exact hash set.
6. Debug: the measured FPR is 10x the formula. What happened?
7. Critique the uniform-hash assumption.
8. Design: the empirical FPR test.

## D5 , SFT masking (U15-C07)

1. Define the mask rule and its purpose.
2. Toy: 3 pairs, count the masked tokens.
3. Derive why system tokens must be masked.
4. Implement the mask builder.
5. Compare mask-assistant-only with mask-all.
6. Debug: the model parrots the system prompt. What happened?
7. Critique the toy's clean roles.
8. Design: masking for multi-turn tool traces.

## D6 , PPO clipping (U16-C06)

1. Define the ratio and the clipped objective.
2. Toy: ratios [0.7..1.6], compute the terms.
3. Derive the pessimistic bound.
4. Implement `ppo_obj`.
5. Compare with vanilla policy gradient.
6. Debug: the objective ignores improvements. What happened?
7. Critique clipping with noisy advantages.
8. Design: an eps sweep experiment.

## D7 , Amdahl bottlenecks (U17-C11)

1. Define the stage table and the bound.
2. Toy: 143 ms, backward 38.5%, 2x -> 19.2%.
3. Derive the new-total formula.
4. Implement `amdahl`.
5. Compare profile-driven with gut-feel optimization.
6. Debug: the speedup moved the bottleneck. What happened?
7. Critique the toy's fixed stages.
8. Design: a sequential-speedup study.

## D8 , ablation designs (U18-C07)

1. Define full, fractional, OAT.
2. Toy: 16/8/5 runs for 4 factors.
3. Derive the run counts.
4. Implement `design_runs`.
5. Compare the three designs.
6. Debug: the fractional design aliases an interaction. What
   happened?
7. Critique the toy's noiseless outcomes.
8. Design: an interaction-effect simulation.

## D9 , contamination checks (U12-C03)

1. Define verbatim vs paraphrase leakage.
2. Toy: 100% overlap, recall 1.00.
3. Derive why exact matching proves presence.
4. Implement the overlap counter.
5. Compare n-gram with embedding screening.
6. Debug: paraphrased eval items pass the check. What happened?
7. Critique the threshold choice.
8. Design: a paraphrase-evasion test.

## D10 , LoRA sizing (U15-C10)

1. Define the adapter and its parameter count.
2. Toy: r=16, d=4096, 8.39M trainable.
3. Derive 2*d*r per adapted pair.
4. Implement the count function.
5. Compare LoRA with full fine-tuning.
6. Debug: the adapter changes nothing. What happened?
7. Critique the toy's frozen-everything-else.
8. Design: a rank sweep for a new language.
