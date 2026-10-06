# Oral defense ladders, math-genai U01-U10

Date: 2026-10-06. Questions only. Keys in
interview/keys-oral.md. Closed-book, timed, spoken.
Each ladder runs define -> intuitive toy -> derive
-> implement and complexity -> compare -> debug ->
critique assumptions -> design experiment or
transfer. Interview provenance: role-derived
practice, not employer material.

## O-U01, the likelihood principle

O1.1. Define likelihood. How does it differ from
probability, in one sentence each?
O1.2. Toy: three flips HHT. Write the likelihood
of p = 0.5 and of p = 0.7. Which wins, by how
much?
O1.3. Derive the Bernoulli MLE from the
log-likelihood.
O1.4. Implement the MLE on n samples. State the
complexity. What breaks numerically at p = 0?
O1.5. Compare MLE with MAP under a Beta(2, 2)
prior on HHT. Give both numbers.
O1.6. Debug: your held-out log-likelihood is
-inf. Name two causes.
O1.7. Critique: on real support tickets, which
fails first, the IID assumption or the Bernoulli
model itself? Why?
O1.8. Design an experiment that tests whether
your MLE is calibrated on new flips.

## O-U02, Jensen to the ELBO

O2.1. Define a convex function with no jargon.
O2.2. Toy: X uniform on {0, 2}, f(x) = x^2.
Compute the Jensen gap.
O2.3. Derive the ELBO for log p(x) from Jensen
with a variational q(z | x).
O2.4. Implement a Monte Carlo ELBO estimate with
S samples. State the complexity in S.
O2.5. Compare the f-divergence dual estimate
with the ELBO approach. When is each the right
tool?
O2.6. Debug: your ELBO estimate is NaN. Name
the likeliest numerical cause.
O2.7. Critique: when is the ELBO tight? What
does a loose bound hide from you?
O2.8. Design an experiment measuring the ELBO
gap as the variational family grows richer.

## O-U03, the minimax game

O3.1. Define the GAN minimax game value.
O3.2. Toy: P = [0.5, 0.3, 0.2], Q uniform.
Compute D* and the game value V.
O3.3. Derive the optimal discriminator
p/(p + q).
O3.4. Implement one alternating update. State
the cost per step in batch size B and net cost
F.
O3.5. Compare the minimax and non-saturating
generator losses at D near 1.
O3.6. Debug: the generator loss is flat from
step 1. Diagnose.
O3.7. Critique: the D* derivation assumes
infinite capacity. What breaks with a finite
net?
O3.8. Design an experiment tracking a JS
estimate against human-judged sample quality.

## O-U04, the Wasserstein dual

O4.1. Define a transport coupling and the W1
distance, one sentence each.
O4.2. Toy: P is a point mass at 0, Q puts 0.9
at 0 and 0.1 at 10. Compute W1.
O4.3. State the Kantorovich dual. Why must the
critic be 1-Lipschitz?
O4.4. Implement the critic loop with 5 critic
steps per generator step. State the cost ratio.
O4.5. Compare weight clipping with the
gradient penalty: what each constrains, and the
price of each.
O4.6. Debug: the critic loss dives toward
-inf. Diagnose.
O4.7. Critique: W1 on pixels versus human
perceptual quality. Where does the metric
mislead?
O4.8. Design an experiment testing whether the
critic gap predicts sample quality.

## O-U05, the VAE

O5.1. Define the ELBO in one sentence.
O5.2. Toy: q = N(0.5, 0.5^2), prior N(0, 1).
Compute KL(q || prior) in nats.
O5.3. Derive the split of log p(x) into ELBO
plus KL(q || p(z | x)).
O5.4. Implement the reparameterized gradient.
Why does its variance beat the score-function
estimator?
O5.5. Compare analytic KL with Monte Carlo KL.
When is each valid?
O5.6. Debug: the KL term sits at 0.000 through
training. Diagnose.
O5.7. Critique: the diagonal Gaussian
posterior. What structure can it never
capture?
O5.8. Design a beta sweep experiment measuring
disentanglement against reconstruction.

## O-U06, vector quantization

O6.1. Define the codebook and the quantization
operation.
O6.2. Toy: codes (1, 0), (0, 1), (-1, 0),
z = (0.8, 0.6). Name the winner and both
losses.
O6.3. Derive the three VQ loss terms and state
what each one trains.
O6.4. Implement the straight-through backward
pass. State the lookup cost for K codes in d
dimensions.
O6.5. Compare gradient codebook updates with
EMA updates.
O6.6. Debug: half your codes are dead after
1000 steps. Diagnose and fix.
O6.7. Critique: the argmin is
non-differentiable. What does the STE pretend,
and when does the pretense fail?
O6.8. Design a codebook-size sweep measuring
rate against distortion.

## O-U07, the DDPM loss

O7.1. Define the forward Markov chain in one
sentence.
O7.2. Toy: linear betas, alpha_bar_50 =
0.77718. State what that number means.
O7.3. Derive how the ELBO term L_{t-1}
simplifies to epsilon matching.
O7.4. Implement the training loop. State the
cost per step. Name the 0-index label bug and
its fix.
O7.5. Compare eps, x0, and v prediction: which
coefficient multiplies the net output at t =
50?
O7.6. Debug: the loss trains but samples are
pure noise. Name two candidate causes.
O7.7. Critique: the reverse Gaussian is
diagonal. What data breaks this assumption?
O7.8. Design an experiment measuring the
simplified-vs-ELBO weighting effect on sample
quality.

## O-U08, DDIM sampling

O8.1. Define the DDIM marginal contract in one
sentence.
O8.2. Toy: the 100 -> 90 jump at eta = 0 gives
x_100 = 1.60480905, x0_hat = 2.13230847, x_90
= 1.71491474. State what each number is.
O8.3. Derive sigma(eta) and show eta = 1
recovers the jump posterior variance
0.15405473.
O8.4. Implement DDIM-10. State the serving
cost in MFLOP versus DDPM-100.
O8.5. Compare DDIM with DDPM sampling: what is
proved equal, and what is not?
O8.6. Debug: the round-trip assert fails
(eps_rec != eps_hat). Diagnose.
O8.7. Critique: state the capstone's
schedule-invariance result, its assumptions,
and why it fails for a learned net.
O8.8. Design the learned-net schedule
experiment from the capstone proposal.

## O-U09, score matching

O9.1. Define the score function.
O9.2. Toy: mixture 0.5 N(-1.5, 0.25) + 0.5
N(1.5, 0.25). s(1.0) = 1.999926. What does the
sign say?
O9.3. Derive the denoising score matching
objective and why the single-sample target is
valid.
O9.4. Implement one Langevin step. State the
cost per step and the step-size tradeoff the
lesson measured.
O9.5. Compare denoising score matching with
sliced score matching.
O9.6. Debug: the chain sits at one mode for
20000 steps. Diagnose.
O9.7. Critique: the score is undefined in
zero-density regions. Why does the noise
ladder fix this?
O9.8. Design an annealing-schedule experiment
measuring valley crossings against bias.

## O-U10, preference alignment

O10.1. Define the Bradley-Terry preference
model.
O10.2. Toy: rewards 0.9 and 0.2. Compute the
BT win probability. The lesson toy gives P =
0.7109 with loss 0.3412: state what the loss
is.
O10.3. Derive the DPO loss from the BT model
plus the KL-constrained RL objective.
O10.4. Implement the DPO gradient. State the
cost per pair. Why are no reward model and no
sampling loop needed?
O10.5. Compare PPO with DPO: which assumption
does each add?
O10.6. Debug: preference accuracy climbs but
human judges dislike the outputs. Diagnose.
O10.7. Critique: the BT transitivity
assumption. Where does it fail on real
preferences?
O10.8. Design a KL-coefficient sweep measuring
win-rate against drift.
