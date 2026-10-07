# U14 interview key

## B1-B6

B1. Scores computed once per unique doc, dupes waste filter
compute.
B2. (1-e^{-kn/m})^k, toy optimum k=6, FPR 0.0216.
B3. Shingles, MinHash signatures, LSH bands.
B4. Eval items hide inside long docs, need n-gram level.
B5. Multiplier = target weight / source weight.
B6. Repeats raise recall logistically and burn compute on
memorization.

## L1

L1.1 Bit array, k hashes. "not seen" certain, "seen" probable.
L1.2 0.0216 at k=6.
L1.3 Bit-survival e^{-kn/m}, all k set.
L1.4 add/contains, no false negatives on the toy.
L1.5 8x less memory than a hash set, debug: adversarial
collisions, critique: uniformity, experiment: empirical FPR.

## L2

L2.1 n-gram sets, agreement-estimating signatures, candidate
bands.
L2.2 0.113 vs 0.125. S-curve points listed.
L2.3 Band match s^r, any-of-b 1-(1-s^r)^b.
L2.4 The pipeline, midpoint near s=0.55 on the toy.
L2.5 Exact pairs impossible at scale, debug: boilerplate.
critique: synthetic sets, experiment: sweep signature length.

## E1

(a) 8e9/1e9 = 8 bits/item. (b) k=6, FPR 0.0216. (c) Raise m
(more memory) or accept. FPR floor at 8 bits/item is ~0.02.
Rubric: (a) 1 pt, (b) 2 pts, (c) 1 pt. Red flag: claiming k=20
helps.

## E2

(a) dL/dw = -0.5+0.4(1-2w) = 0 -> w = 1.125 -> boundary w=1.0,
L=2.500. (b) dL/dw = -0.5+1.0(1-2w) = 0 -> w=0.75, L=2.5625.
(c) Stronger interaction pulls the optimum interior: mixing
beats purity. Rubric: (a) 2 pts, (b) 1 pt, (c) 1 pt.

## D1

Bug: the second `seen[h] = doc` overwrites unconditionally, so
the last copy wins, not the first. Fix: delete the overwrite
line. Rubric: find 2 pts, fix 1 pt, state first-vs-last 1 pt.

## S1

Policy: cap at ~3x effective repetition via downweighting.
monitor recall probes and eval deltas on the domain, alert on
memorization signatures (verbatim completion).

## S2

Add embedding-based near-dup screening for the eval items.
quarantine hits, the n-gram check is necessary but not
sufficient.

## R1

Gaps: (1) no token counts: the gain is unmeasured, report
before/after. (2) no method: not reproducible, name it. (3) no
leak check: the win may be contamination, run train-test dedup.
