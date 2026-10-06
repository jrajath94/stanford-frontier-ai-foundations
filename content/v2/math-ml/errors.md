# errors.md, math-ml error and uncertainty ledger

Date: 2026-10-06. Records mistakes, near-misses, and open uncertainties
so later runs do not repeat them.

## E-001, short-link destination unresolved (closed: SOURCE-UNREACHABLE)

Seed link https://lnkd.in/gQe4kMkf. Second attempt 2026-10-06 (RUN 2):
HTTP 200 to a LinkedIn interstitial, no meta-refresh destination,
login/JS gated. Marked SOURCE-UNREACHABLE. No further attempts.

## E-002, no transcript inspected (open)

All source claims beyond identity rest on titles and secondary
summaries. Leaf rows stay PLANNED / SOURCE ATTRIBUTION PENDING.

## E-003, float64 sum luck (noted, not an error)

sum([0.1]*10) in float64 gives exactly 1.0 by rounding luck, not by
exactness. The lesson states this explicitly so the learner does not
conclude float64 is exact.

## E-004, embed fetch failed (closed)

The YouTube embed URL for the playlist returned no extractable content.
The plain playlist URL worked on the next attempt and gave all 89
titles. Lesson: use the canonical playlist URL, not the embed URL.

## E-005, two offerings (open)

Jan-Apr 2026 offering vs Aug 2026 ongoing offering not reconciled
against the 89-title playlist. RUN 2 task.

## E-006, duration claim unverified (open)

Third-party claim of ~46.6 hours not verified. Do not cite it as fact.

## E-007, numpy rejects float16 linalg (noted, measured 2026-10-06)

np.linalg.solve on float16 arrays raises TypeError: array type float16
is unsupported in linalg. Lesson E29 records this measured behavior
instead of inventing a float16 solve result. The float16-rounding
emulation of K (1.001 -> 1.00097656) solved in float64 still gives
[2, 0] exactly, so no precision-damage claim is made from that toy.

## E-008, visual id collision risk (closed by convention)

Lesson-internal figure ids (f01-f12 in U02, sb01-sb04 in U04a) repeat
across lessons. visual_audit.md namespaces them as u02-c01..c12 and
SB01..SB08, and PNG filenames live in separate visuals/u02 and
visuals/u04 folders. Convention holds for RUN 2. Revisit if a third
lesson reuses ids.

## E-009, SGD legend label misread (caught and fixed, 2026-10-06)

render_run3.py first labeled the SGD trace J=3.2143 in
visuals/u03/f01_gd_path.png. The measured value from compute_run3.py
is w = 1.210517318455124 after 20 steps, giving
J = (1.210517318455124 - 3)^2 = 3.2022. Fixed the label to J=3.2022,
re-rendered, re-read. Lesson: legend labels are claims too. Recompute
them from the trace, do not hand-copy.

## E-010, SB12 log-likelihood arithmetic slip (caught and fixed, 2026-10-06)

lesson-04b first draft stated the log-likelihood at theta = 0.75 as
-1.2457. Recomputation: 3*log(0.75) + log(0.25) = -2.249340578475233.
Fixed in the lesson. Lesson: hand arithmetic on logs is exactly the
place the C12 gradient-check habit applies. Recompute, do not trust.

## E-011, interview T1 premises verified before shipping (2026-10-06)

Two interview debug tasks had premises checked by execution before
finalizing: U03-T1 (forward-difference absolute error measured
6.03703385060328e-08, not the drafted 1.27e-7) and U04a-T1 (the first
PMF [0.1, 0.2, 0.3, 0.4] sums to exactly 1.0 in float64, so it would
print True. Replaced with [2/7, 3/7, 2/7], which sums to
0.9999999999999999 and prints False). Rule: every debug-task premise
must be executed, not reasoned about.

## E-012, G3 third attempt failed (2026-10-06, RUN 4)

curl fetch of https://onlinecourses.nptel.ac.in/noc26_cs02/preview
returned a 14,922-byte page containing "loading" and no syllabus,
schedule, or assignment text. Third failed attempt (RUN 1 fetch,
RUN 3 browser.open, RUN 4 curl). G3 stays open. No further attempts
planned this run. The page needs a JS-capable client on an
unfiltered network.

## E-013, f02 label collisions (caught and fixed, 2026-10-06)

render_run4.py first drew f02 on a linear y-axis: the train MSE
bars (0.031, 0.015, ~0) were invisible and the degree-3 test bar
(28.56) clipped off-chart with its label cut. Re-rendered with a
log y-axis. The "2.88" label then hid behind the legend in two
placements. Fixed by moving value labels inside the bars and the
legend to the empty lower-left. Final PNG read back and verified.
Lesson: check every label against every other element at final
size, not just the data range.

## E-014, f02 logistic dot misplacement (caught and fixed, 2026-10-06)

render_run5.py first plotted the w=0.25 dots at x-coordinates 0
and 1 while the axis was z = w x. The dots floated off the
sigmoid curve. Fixed by plotting at z = 0 and z = 0.25, where
they sit on the curve. Re-rendered and re-read. Lesson: the
axis quantity and the dot quantity must be the same variable.

## E-015, f01 k-means label clip (caught and fixed, 2026-10-06)

The "center 1" text label rendered half outside the axes.
Fixed by moving the label left of the diamond. Re-rendered and
re-read. Lesson: check every label against the axes box at
final size (repeats E-013).

## E-016, f03 held-out title overstated the data (caught and fixed, 2026-10-06)

First title claimed "held-out inertia picks k=2" with a U-shape
legend. Measured held-out inertia keeps falling past k=2
(0.4439, 0.3360, 0.2939). There is no U-turn, only an elbow.
Retitled to "the elbow is at k=2, extra centers buy little on
held-out data" and changed the legend from "U shape" to
"flattens". The lesson text already carried the honest note. 
the figure now matches it. Lesson: titles are claims too, and
they must survive the measured numbers.

## E-017, dual first draft dropped the balance constraint (caught and fixed, 2026-10-06)

compute_run5.py first used a = 0.5 per point, giving dual 0.0
against primal 1.0. The missing constraint was
sum alpha_i y_i = 0. Fixed to a* = 0.25 with the full pairwise
expansion: dual 0.25 = primal 0.25, y(w.x) = 1 both points.
The lesson records the error as a teaching case (C09, E27).
Lesson: strong duality is the check. A gap means a missing
constraint, not a wrong theorem.

## E-018, keys-09b E06 hand arithmetic slip (caught and fixed, 2026-10-06)

First draft gave the x-column population std as 0.9037 and
squared distances 2.69/2.72. Recomputation: std 0.8994,
distances 7.2363/7.4835. The verdict (nearest returns to B)
was right. The numbers were not. Fixed in keys-09b.md. Lesson:
hand arithmetic on standardized values is exactly where the
C12 gradient-check habit applies.

## E-019, interview debug premises verified before shipping (2026-10-06)

Three completion-run debug tasks had premises checked by
execution before finalizing: U07-T1 (weighted sum-of-
squares at the five C03 candidates: 0.6/0.75/1.0/0.75/0.6,
minimum at t=0.2/0.85, confirming the buggy pick),
U08-T1 (analytic subgradient -0.0 vs two-sided finite
difference -4.3499997782 at the relu kink z=0,
confirming the kink diagnosis), U10-T1 (beta=10 totals:
informative -5.0713 vs collapsed -1.6695, confirming
the collapse diagnosis). Rule (from E-011): every
debug-task premise must be executed, not reasoned about.

## E-020, lab reference numbers corrected by execution (2026-10-06)

Three lab references were wrong in draft and fixed by
running the computation: lab-07 Task 1 best threshold
(drafted t=1.6, measured t=0.55, gain 0.25). lab-08
Task 1 gradients (drafted dw2=-0.701414/dw1=1.392660,
measured -0.7011581396/1.3935265128). lab-10 Task 1
ELBO (drafted -0.8391010604/gap 0.0406, measured
-1.0750556815/gap 0.2765480) and Task 2 seed-12 ratio
(drafted 3.9065, measured 3.9594). Lesson: lab keys
are claims too. Execute, do not hand-compute.

## E-021, yt-dlp absent on this box (noted 2026-10-06)

The G2 fifth attempt checked /usr/local/bin/yt-dlp,
PATH, and the python module: yt-dlp is not installed
(the Oct 3 symlink noted in AGENTS.md did not survive).
Installs are out of scope per build laws, so no
caption fetch was attempted. G2 stays open. Lesson:
verify tool presence before planning an attempt
around it.

## E-022, u07 f01 footer clip and u10 f01 tick overlap (caught and fixed, 2026-10-06)

First read of visuals/u07/f01_regions.png showed the
footer text clipped at the left edge. Shortened the
footer and reduced to fontsize 9, re-rendered,
re-read. First read of visuals/u10/f01_elbo_gap.png
showed x-tick labels overlapping the footer. Moved
the values onto the bars as white labels, shortened
the footer, re-rendered, re-read. Lesson: read every
figure at final size before logging it verified
(repeats E-013, E-015).
