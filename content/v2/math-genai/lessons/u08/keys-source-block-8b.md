# Answer keys, source block 08b (W9 titles, title-level)

Date: 2026-10-06. Ground truth: lesson-08 and compute_run5b.py.
No transcript inspected, all answers stay at the title boundary.

## Q1

W9L34 -> C07, U09-C01..C04. W9L35 -> C07, U09-C01/C02.
W9L36 -> C08, C09. W9L37 -> U06-C09/C12 + C11 (composition).
W9L38 -> C01, C02. W9L39 -> C02, C03, C12. W9T17 ->
U07-C10/C12, C07. W9T18 -> C01-C03, C12. W9T19 -> C09.
Ambiguous from the title alone: W9L34 (which
interpretations?), W9L37 (which autoencoder?), W9L36
(which guidance method first?).

## Q2

Candidates: (a) latent-variable/ELBO view -> U07-C07.
(b) score-matching view -> C07/U09-C02. (c)
SDE/probability-flow view -> U09-C04. Each is
consistent with "alternate interpretations", the
title does not say which are covered.

## Q3

Needs: U06 rate/distortion (compress without
destroying the signal) and U08 diffusion math
(unchanged in latent space). Seam risk: the
encode/decode round trip adds reconstruction error
that the diffusion never sees, the battery has no
assert for it (U08 not-yet-understood item 5).

## Q4

Hypotheses: (a) the tutorials were recorded later
and appended, (b) the playlist order is
upload/administrative, not pedagogical. E-009
supports (b): title order is a playlist fact, not a
course claim. The block pairs them logically with
U08 either way.

## Q5

(1) t drawn from 1..T inclusive (never 0). (2) The
noise level uses ab[t-1] for 1-indexed t (E-018).
(3) The loss target is the drawn eps, and the mean
is over batch and dims, not summed without
normalization.

## Q6

Contract: q(x_{t-1} | x_0) = N(sqrt(ab_{t-1}) x_0,
(1 - ab_{t-1}) I) for every sigma. Verifying
number: sigma^2 + (1 - ab_90 - sigma^2) =
0.15405473 + 0.40507252 = 0.55912725 = 1 - ab_90.

## R1

A title tells the topic and its week. It does not
tell definitions, derivations, examples, code, or
emphasis. Nine titles prove nine topics were
scheduled, nothing about depth.

## R2

Prediction: s_theta(x_t, t) = -eps_theta(x_t, t) /
sqrt(1 - ab_t), or equivalently the Tweedie form.

## R3

Yes: C07 derives and computes both forms
(2.05354466 both ways). Watching would add the
lecturer's motivation, possibly the denoising
score matching derivation, and worked examples,
none of that is in the lesson.

## R4

The lecture title promises the idea of guided
models, the tutorial title promises code. The
tutorial should add: the exact loop, the number
of forward passes per step under CFG, the
parameter g, and indexing conventions. The title
alone does not guarantee any of it is shown.

## R5

Attack: G2 says no transcript was inspected, so
"teaches deeply" is unmeasured. Nine titles show
scheduling, not depth. The lesson is authored
bridge content, not a transcript summary. The
claim confuses the map (titles) with the
territory (lectures).
