# currentness.md, math-ml date-bound claims

Date: 2026-10-06. Baseline: October 6, 2026. Separate newer updates if
executed later.

## Claims with dates

- Playlist enumeration: 89 titles, fetched 2026-10-06.
- Instructor page: fetched 2026-10-06. States second offering (Aug 2026)
  ongoing.
- NPTEL catalog row: Jan 19, 2026 to Apr 10, 2026, exam Apr 25, 2026.
- All toy computations: executed 2026-10-06, numpy 1.26.4,
  matplotlib 3.6.3, CPython float64.
- RNG sequences: numpy default_rng(7). PCG64 stream may differ across
  numpy major versions. Recorded under numpy 1.26.4.
- SRC-09 located 2026-10-06: third-party learner repo, listing fetched,
  snippets only. Not instructor evidence.
- Short link lnkd.in/gQe4kMkf: SOURCE-UNREACHABLE as of 2026-10-06.

## Review triggers

Re-verify the playlist if the second offering adds lectures. Re-run toy
computations if numpy changes major version.

- RUN 3 toy computations: executed 2026-10-06, numpy 1.26.4,
  matplotlib 3.6.3, CPython float64. Scripts: compute_run3.py,
  render_run3.py.
- SRC-02 second attempt 2026-10-06: still a loading shell. G3 open.
- SGD seed: numpy default_rng(7), 20 steps on the least-squares toy,
  recorded under numpy 1.26.4.

- RUN 4 toy computations: executed 2026-10-06, numpy 1.26.4,
  matplotlib 3.6.3, CPython float64. Scripts: compute_run4.py,
  render_run4.py. Seeds: numpy default_rng(7) throughout. Learning
  curves validation set uses default_rng(11).
- SRC-02 third attempt 2026-10-06 (curl): still a loading shell.
  G3 open.
- Interview U05 debug task T1 premise executed 2026-10-06:
  aligned split test accuracy 1.0, shuffled-X split 0.2, shuffled
  label agreement 0.35.

- RUN 5 toy computations: executed 2026-10-06, numpy 1.26.4,
  matplotlib 3.6.3, CPython float64. Scripts: compute_run5.py,
  render_run5.py. Seeds: numpy default_rng(7) throughout. 
  held-out k-means scan uses default_rng(7) with a second
  default_rng(7) for center init.
- Interview U06 debug task T1 premise executed 2026-10-06:
  gamma=1e-6 gives cond 3.227e6 and alphas near 1e6 in
  magnitude with residual 4.12e-11. Gamma=0.5 gives cond 4.2
  and sane alphas with residual 0.0.
- Capstone R1 executed run: seed 7 only, logistic 0.7167 vs
  SVM 0.7833. Not a general finding.

- Completion-run toy computations: executed 2026-10-06,
  numpy 1.26.4, matplotlib 3.6.3, CPython float64. Scripts:
  compute_completion.py, compute_completion2.py,
  render_completion.py. Seeds: numpy default_rng(7)
  throughout. the C09/U10 batch-noise trial loop uses
  default_rng(11), lab-10 Task 2 reruns it with seed 12.
- Interview debug premises executed 2026-10-06: U07-T1
  (weighted sum-of-squares 0.6/0.75/1.0/0.75/0.6), U08-T1
  (kink fd -4.3499997782 vs analytic -0.0), U10-T1
  (beta=10 totals -5.0713 vs -1.6695).
- Lab reference corrections 2026-10-06 (E-020): lab-07
  Task 1 (t=0.55), lab-08 Task 1 (dw2=-0.7011581396,
  dw1=1.3935265128), lab-10 Task 1 (ELBO -1.0750556815,
  gap 0.2765480) and Task 2 seed-12 ratio 3.9594.
- G2 fifth attempt 2026-10-06: yt-dlp not present on this
  box. installs out of scope. G2 open.
