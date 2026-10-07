# Lab U05: the adversarial game

Unit: math-genmodels-U05. Date: 2026-10-06. Baseline: October 6, 2026.

Stack: Python 3, numpy 1.26.4, CPU only, float64. Seed every run and
print the seed. Answers in labs/keys/keys-u05.md. Do not open the keys
until the code runs.

## E1: implicit sampler audit

Implement g(z) = z^2 + 1 with z ~ Uniform(-1, 1). Draw 10,000
samples (seed 10). Report the min, max, and mean. Confirm every
sample lands in [1, 2]. Then try to write p(1.5) in closed form
and explain in one sentence why the attempt fails.

## E2: ratio and discriminator optimum

For p = N(0,1), q = N(1,1): compute r(x) and D*(x) on the grid
x in [-3, 4]. Report r(0), r(2), D*(0), D*(2). Find the crossing
point where r = 1 numerically and confirm it is x = 0.5. Verify
D* = r/(1+r) pointwise to 1e-12.

## E3: GAN value curve

Plot V(d) = log d + log(1 - d) for constant d in (0.01, 0.99).
Confirm the maximum is -1.3863 at d = 0.5. Then compute JSD(p, q)
by quadrature and confirm V* = -log 4 + 2 JSD = -1.1635. Write
the one-sentence meaning of the gap between -1.3863 and -1.1635.

## E4: saturation versus Wasserstein

For shifts s in (0, 1, 2, 3, 4, 5): compute JSD(N(0,1), N(s,1))
by quadrature and W1 = s. Report the table. Confirm JSD flattens
while W1 grows. Then repeat with N(0,0.01) vs N(5,0.01) and
confirm JSD = log 2 within 0.01.

## E5: bilinear spiral

Run min_x max_y x y with simultaneous GDA, lr = 0.1, from (1,1),
for 200 steps. Record the radius every 20 steps. Confirm it
grows monotonically. Then halve the learning rate and confirm
the growth slows but does not stop. Write the one-sentence
lesson for GAN training.

## E6: collapse diagnosis

Target: 0.5 N(-3,1) + 0.5 N(3,1), 2,000 draws (seed 10).
Collapsed model: all mass at 0. Compute precision (fraction of
model samples within 7 of 0) and recall (fraction within 1 of
-3). Report both. Then build a covering-but-blurry model
(N(0, 9)) and compute its precision and recall. State which model
each single metric would wrongly crown.
