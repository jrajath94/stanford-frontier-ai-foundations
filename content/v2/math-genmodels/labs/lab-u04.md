# Lab U04: flows and exact likelihood

Unit: math-genmodels-U04. Date: 2026-10-06. Baseline: October 6, 2026.

Stack: Python 3, numpy 1.26.4, CPU only, float64. Seed every run and
print the seed. Answers in labs/keys/keys-u04.md. Do not open the keys
until the code runs.

## E1: invertibility audit

Write fwd/inv for x = 2z + 1 and for the coupling layer. Round-trip
1000 random points through fwd then inv. Report the max absolute
error. Then implement g(z) = z^2 with a "pseudo-inverse" sqrt(|x|)
and show it fails the round-trip on negative inputs.

## E2: density transform check

Implement p_x for the 1-D toy. Quadrature it over [-9, 11] and
confirm the area is 1 within 1e-4. Evaluate at x = 1 and x = 3.
Confirm 0.1995 and 0.1210. Then drop the |det| term and report the
new area.

## E3: coupling round-trip

Forward then inverse the point (0.5, -1.0) through the coupling
layer. Confirm (0.5, -0.75) forward and exact recovery backward.
Compute the Jacobian by finite differences and confirm it matches
[[1, 0], [0, 1.25]] within 1e-6. Report the log-det both ways.

## E4: exact likelihood

Score y = (0.5, -0.75) by hand: invert, score the base, subtract
the log-det. Confirm -2.6860. Then flip the sign of the log-det and
report the wrong number. Write the one-sentence lesson about the
sign.

## E5: asymmetry costing

For D in (2, 16, 256): count serial steps for MAF sampling and IAF
density eval. State which model you would train for a pure scoring
workload and which for a pure generation workload at D = 256.

## E6: log-det overflow demo

Compute prod of 100 copies of 1e6 in linear space and the
log-det in log space. Report inf versus 1381.55. Then build a
diag(1e6, 1e-6) layer, report its condition number, and write the
one-sentence numerical rule for a lab report.
