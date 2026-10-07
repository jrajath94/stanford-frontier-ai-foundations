# U17 interview key

## B1-B6

B1. 16x16 patches, linear projection, 256 tokens.
B2. One causal token stream mixing modalities with boundary
tokens.
B3. Model (tok/s/GPU), data (GB/s), interconnect (bytes/step).
B4. Different optimal setups for generation vs training.
B5. id_n = sha256(id_{n-1} + stage_n).
B6. Speedup is bounded by the optimized fraction.

## L1

L1.1 util = min/max, backlog = (r-c)*t.
L1.2 83.3%, 1200/hr.
L1.3 Queue grows linearly in the rate gap.
L1.4 `backlog`, r=c gives 0.
L1.5 Lockstep idles both fleets in turn, debug: trainer stall.
critique: constant rates, experiment: stall simulation.

## L2

L2.1 Per-stage times of one step.
L2.2 143 ms, 38.5%, 116 ms, 19.2%.
L2.3 Subtract the share, add back share/speedup.
L2.4 `amdahl`, infinite speedup gives the share.
L2.5 Gut-feel usually wrong, debug: bottleneck moves, critique:
toy, experiment: sequential speedups.

## E1

(a) Backward, 55/143 = 38.5%. (b) 143 - 12.5 = 130.5 ms. (c)
Backward is still first (55 ms), then forward (40 ms). Rubric:
(a) 1 pt, (b) 2 pts, (c) 1 pt. Red flag: optimizing the
checkpoint stage.

## E2

(a) 8*256+512 = 2560 tokens, 80.0% image. (b) (2048-512)/256 = 6
images. (c) Perceiver-style compression, or fewer/lower-res
images. Rubric: (a) 1 pt, (b) 1 pt, (c) 2 pts.

## D1

Bug: the id ignores the parent chain and the run identity, so
two runs of one stage collide and tampering is invisible. Fix:
id = sha256(parent_id + run_id + inputs). Rubric: find 2 pts,
fix 2 pts.

## S1

The queue grows 1200/hr unbounded until memory dies, then fresh
rollouts drop and staleness explodes. Policy: bound the queue,
shed oldest first, alert on the stall.

## S2

Toys support: methods, arithmetic, interface math, protocol
design. They do not support: scale behavior, real hardware
numbers, transfer claims. The 70B decision needs staged
evidence, not toy extrapolation.

## R1

Gaps: (1) no stage breakdown: the win is unlocated, profile it.
(2) no hardware: not reproducible, name it. (3) no baseline
config: the delta is undefined, publish it.
