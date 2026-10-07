# Lab U03: autoencoders and the VAE bound

Unit: math-genmodels-U03. Date: 2026-10-06. Baseline: October 6, 2026.

Stack: Python 3, numpy 1.26.4, CPU only, float64. Seed every run and
print the seed. Answers in labs/keys/keys-u03.md. Do not open the keys
until the code runs.

## E1: bottleneck sweep

Rebuild the 8-point toy with seed 8. Compute reconstruction MSE for
bottleneck dims d in (1, 2): d = 1 uses projection onto w = (1,2),
d = 2 is the identity. Report both MSEs. Explain why d = 2 gets
zero and why that is not a win.

## E2: VAE ELBO by hand

On paper, then in code: for x = (1,2), mu = 0.9, sigma = 0.4,
decoder mean z*w, c = 0.5, compute the reconstruction term, the KL
term, and the ELBO. Confirm -2.1516, 0.9013, -3.0529. Then verify
the KL by Monte Carlo with 200000 samples: it must land within 0.01
of the closed form.

## E3: beta sweep

Evaluate the beta objective at beta in (0.1, 0.5, 1, 2, 4, 8) on
the toy numbers. Plot objective versus beta. Mark beta = 1 as the
only point where the number is a likelihood bound. State in one
sentence what the other points are.

## E4: codebook quantization

Place K = 4 codes at (0,0), (2,4), (-2,-4), (0,3). Assign the eight
C01 latent codes (as 2-D points z*w) to nearest codes. Report the
usage counts and the mean squared quantization error. Then move one
code to (100, 100) and report what happens to its usage.

## E5: straight-through bias measurement

Implement the threshold toy at mu = 0 and mu = 2.0. Compute the
true gradient by finite difference of Phi and the straight-through
estimate (1.0) at both. Report the bias at each. Confirm the bias
grows under saturation.

## E6: bits-per-dim audit

Compute BPD from the toy ELBO. Then recompute it as if the ELBO
were the exact log marginal: state which of the two numbers is
honest to report and why. Write the one-sentence reporting rule for
a lab report.
