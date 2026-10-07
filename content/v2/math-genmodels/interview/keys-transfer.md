# Transfer set keys

Course: math-genmodels. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers, red flags, rubrics. Remediation pointers name the
unit concept.

## T1

Q1. The support breaks first. Uniform(0, 2 pi) has log-density
-log(2 pi) = -1.8379 everywhere inside, undefined outside.
The invertibility is untouched. Red flag: blaming the
coupling. Rubric: the support named with the number.
Remediation: U04-C07, U01-C02.
Q2. Take x = (6.0, 0.1): y2 = 0.1 x (1 + 36) + 6.0 = 9.7, far
outside [0, 2 pi). The base log-density there is -inf: the
evaluator returns -inf or NaN. Red flag: "the flow clips
it." Rubric: a concrete witness with the -inf verdict.
Q3. Wrap the coordinates on the circle: use circular
coupling (addition mod 2 pi) with a periodic base. Exact
likelihood survives. Lose: the Euclidean coupling family and
any layer that is not circle-equivariant. Rubric: the fix
plus the named loss.

## T2

Q1. The GAN is implicit (U05-C01): its loss is not a density
and cannot certify absence of copying. Run the
nearest-neighbor audit (U08-C06): min distance from each
synthetic row to the real set, histogram, decide on the pile
at zero. Red flag: trusting the loss. Rubric: the concept
plus the test.
Q2. Reading one: the laws are disjoint, JSD saturated, the
generator gets no gradient (U05-C05). Reading two: the
discriminator memorized the training rows. Both kill the
project: the first means no learning, the second means the
privacy bar is already broken. Red flag: "training
converged." Rubric: both readings with the verdict.
Q3. The VAE has an encoder: it maps real rows to latents,
which creates an inversion path from latents back toward
real data that the GAN never had. A VAE can also memorize
through a collapsed decoder. The privacy surface grows.
Rubric: the encoder inversion named.

## T3

Q1. DDIM with eta = 0 on strided timesteps. Assumption: the
trained eps net predicts noise well enough that the
marginal-preserving strided update stays close: the same net,
no retraining (U06-C07). Red flag: "DDIM needs its own
training." Rubric: the sampler plus the assumption.
Q2. Striding breaks the marginal match: each jump
approximates many small denoising decisions, and the error
compounds into washed-out detail. Cheapest fix: a
higher-order ODE solver (Heun) on the same 4 steps: 8 net
calls... over budget. Honest cheapest: redistribute the 4
steps (quadratic or late-dense spacing) and measure. Red
flag: "add more steps." Rubric: the mechanism plus a
no-retraining fix.
Q3. Gain: 1 call, real-time. Lose: the student inherits the
teacher's errors plus distillation error. Diversity drops.
no exact inversion. Experiment: matched-seed sample distance
(student vs 1000-step teacher) plus human sharpness ratings
at fixed prompts. Decide on the distance-versus-budget
curve. Rubric: gain, loss, and the experiment.

## T4

Q1. Not decided. Perplexity is per token: B has 64 tokens per
3 bases versus A's 4 per base. The scales differ. Compute
bits per base for both. The lower one wins. Red flag:
"B is better." Rubric: the scale argument with the
remedy.
Q2. Bits per base = (NLL in nats / ln 2) / (number of bases).
Equivalently: perplexity-per-token ^ (tokens per base),
then log2. Show both forms agree. Rubric: the formula with
both forms.
Q3. L multiplies by 3 (each base is a token) or stays put
for B. Attention is O(L^2 d): 9x the mults for the
byte-level model at the same base count. KV cache triples
too. Red flag: "vocabulary is the only cost." Rubric: the
quadratic scaling named.

## T5

Q1. The decoder learned to ignore the latents: the KL term
is dead, so q(z|x) equals the prior and z carries no
information. Posterior collapse. Red flag: "the model is
fine, reconstructions are sharp." Rubric: the one-sentence
diagnosis.
Q2. Reconstruction uses the encoder: with dead latents the
decoder is an unconditional model conditioned on nothing,
yet it reconstructs because... it cannot: sharp
reconstructions with zero KL mean the decoder memorized
through its own capacity or the KL reading is per-dim
averaged over dead and live dims. The prior samples are
garbage because the prior was never trained to generate:
the ELBO never asked it to. Red flag: confusing the two
paths. Rubric: the asymmetry explained.
Q3. Objective: beta < 1 (beta-VAE, U03-C06) or KL annealing:
pay less for rate, let the latents carry information.
Architecture: a weaker decoder (fewer layers) that cannot
memorize, forcing reliance on z. Both move up the
rate-distortion curve: more rate, less distortion, up to
the frontier. Rubric: two fixes with the tradeoff stated.

## T6

Q1. The score blows up near the boundaries (U06-C12): the
smoothed law has steep slopes at 0 and 100, so the learned
score points hard inward there. Mechanism: Gaussian
smoothing turns the hard edge into a slope of width sigma,
and the score measures that slope. Red flag: "the model
learns the bounds." Rubric: the blowup with the mechanism.
Q2. The noise floor sigma_min: the likelihood integrates
the smoothed law, and smaller sigma gives larger boundary
scores and different numbers. Two models compared at
different floors can flip ranking. Red flag: reporting the
number without the floor. Rubric: the parameter named with
the flip mechanism.
Q3. Dequantization-style: add small uniform noise to the
readings, or model in an unbounded transformed space
(logit of x/100). Cost: the likelihood now describes the
noisy/transformed variable, not the raw readings. Every
comparison must use the same transform. Rubric: the fix
plus the cost.

## T7

Q1. Lipschitz needs a metric on the input space. Token IDs
are categorical: neighboring IDs are not neighboring tokens,
so the slope is meaningless on raw IDs. The penalty needs
gradients, which need a continuous space: on embeddings the
gradient exists but "1-Lipschitz in embedding space" is not
the Kantorovich constraint on token distributions.
Conceptually awkward because the dual was derived for the
data space, not an embedding proxy. Red flag: "embeddings
fix it." Rubric: the metric problem stated.
Q2. They come from line segments between real and fake
embedding points: a distribution the model never samples at
test time. The penalty constrains the critic there, but the
generator's outputs (vertices, after straight-through) live
elsewhere: the constraint may not bind where the game is
played. This is the capstone's finding in miniature. Red
flag: "interpolations cover the space." Rubric: the
mismatch named.
Q3. An autoregressive model over tokens (U07): exact
likelihood, no game, no Lipschitz needed. Or a VAE with
discrete latents (U03-C07). Rubric: one alternative with
the reason.

## T8

Q1. Both score: MAF density eval is 1 parallel pass
(U04-C10). The AR transformer scores with 1 forward pass,
also parallel over positions (U07-C08). Serial costs: MAF
sampling is D = 128 serial steps (irrelevant for scoring).
AR sampling is L serial steps (irrelevant for scoring).
For pure scoring both are parallel. Red flag: confusing
scoring with sampling. Rubric: both directions with the
serial costs.
Q2. Per batch of 1024: MAF needs 1024 x 128-dim conditioner
calls in one batched pass. The transformer needs 1024 x L
positions in one pass. With D = 128, L = 64, d = 256: MAF
flops ~ 1024 x 128 x C_mlp, transformer ~ 1024 x 64^2 x
256. The transformer is roughly 64x256/128 = 128x the
per-batch flops of a small MAF: MAF wins the SLA unless
its conditioner is huge. The arithmetic must use the real
layer sizes. Red flag: "transformers are always faster."
Rubric: the flop comparison with real sizes.
Q3. Mode coverage (U08-C03): the likelihood averages over
points, so a missing fraud shape barely moves the mean
while the fraud team cares about exactly that shape.
Add: a held-out fraud-shape recall test with pre-registered
shapes, scored weekly. Rubric: the concept plus the test.

## T9

Q1. Null: the model samples from a law independent of the
training set (no memorization). Three near-duplicates at
0.5-pixel distance reject it: independent samples would
essentially never land that close. Red flag: "three is a
small number." Rubric: the null stated as independence.
Q2. The score is learned from the data: at low noise the
noisy score points at training points (Tweedie, U06-C03).
A model can memorize through its score field: the sampler
then walks back to exact training images. "Learning the
score" does not prevent copying. It is the mechanism of
copying at small t. Red flag: accepting the vendor's
theory. Rubric: the Tweedie mechanism.
Q3. Reference: the full training set (or the largest
obtainable subset). Samples: 50,000 generations. Metric:
nearest-neighbor distance distribution. Decision rule,
pre-registered: fewer than 1 in 50,000 below 1.0 pixel
distance, else reject. Report the histogram, not just the
count. Rubric: all four elements with the rule.

## T10

Q1. Cheapest: the unit's FID-0 construction: a model
matching their feature moments with the wrong shape scores
the same FID. Concretely: fit a Gaussian to their
Inception features' mean and covariance and sample it:
same FID, garbage images. The number survives. The claim
dies. Red flag: "but their FID is real." Rubric: the
construction.
Q2. The feature extractor and its domain fit, and the
uncertainty over seeds (error bars). Also the sampling
cost and the training budget. Red flag: accepting the
single number. Rubric: two omissions minimum.
Q3. Suite: (1) sample precision/recall against your
reference set with a fixed feature extractor. (2)
held-out likelihood if any candidate has a density, else
a memorization NN screen. (3) a distributional distance
with a characteristic kernel (MMD). Decision rule:
win at least two of three with non-overlapping error bars
over 5 seeds, else no decision. Rubric: three metrics plus
the rule.
