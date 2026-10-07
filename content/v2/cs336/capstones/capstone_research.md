# Capstone A , research replication + falsifiable extension

## Question

Does the U10 power-law fitting procedure replicate on a new
synthetic law, and does the fitted exponent stay inside a
preregistered band when the noise quadruples?

## Falsifiable hypothesis

H1 (replication): on a new law (E=2.05, A=0.80, a=0.41, seeds
11/12), the log-space fit recovers a within 0.02 of truth.
H2 (extension, preregistered): at 4x noise (sigma 0.08, seed 13),
the fitted exponent lands in [0.30, 0.38].

## Literature

U10 lesson (this course): the fitting procedure, residuals, and
bootstrap. Chinchilla (Hoffmann et al., 2022): the empirical
context (secondary, not replicated here).

## Data

Synthetic: 24 log-spaced C values, 1e19 to 1e22, multiplicative
lognormal noise. Seeds fixed. No training runs, no external data.

## Baselines and controls

Baseline: the U10 toy fit (a=0.34 -> 0.340). Control: the same
fitting code runs on both laws, only constants and seeds change.

## Method

`capstone_research_run.py`: fit log(L-E) vs log C by least
squares. Matched budgets: identical n=24, identical C grid.

## Results

- Replication: true 0.41, fitted 0.412. PASS.
- Extension: fitted 0.405 against preregistered band [0.30,
  0.38]. FAILED.
- Half-range check (fit first half of C, predict second): max
  log-error 0.0075 < 0.02. PASS.

## Negative result, reported honestly

The extension failed, and the failure is experimenter error, not
noise: the band [0.30, 0.38] was anchored to the OLD law (a=0.34),
not the new one (a=0.41). The observed 0.405 sits 0.005 below the
truth 0.41: the fit was fine, the registration was wrong. A
corrected band [0.37, 0.45] holds the observation, but that
scoring is post-hoc diagnosis, not evidence: the preregistered
test stands as FAILED.

## Limitations

Synthetic laws only, iid lognormal noise. E assumed known, no
real training data. Conclusions do not transfer to empirical
scaling.

## Uncertainty

Seed variation: the extension used one seed (13). A multi-seed
repeat would widen the honest statement.

## Reproducibility

Run `python3 capstone_research_run.py`. Figures:
`capA_fig01.png` (replication fit), `capA_fig02.png` (the missed
band). Render: `python3 render_capstone_research.py`. All PNGs
PIL-verified (IHDR/IDAT/IEND only), metadata-stripped.

## Ethical considerations

None beyond honesty: the failed registration is reported as
failed, with the diagnosis labeled post-hoc.
