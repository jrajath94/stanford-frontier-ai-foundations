# Audit: chapter plates for CS229 L07-L12
# Builder: chapter-plate builder, session c3753ee0, 2026-10-06.
# 24 plates, 4 per lesson. All SVGs rendered by build/gen-chap-plates-cs229-l07-12.py.

## Method
- Every number on every plate was recomputed in Python and asserted
  before rendering (38 asserts, all passing). Any recomputation that
  disagreed with the lesson was checked against the lesson text.
- Layout matches the bar (mse435 plate-l01-chap-*.svg): 960 wide,
  warm paper #F7F4EE, left card cream (cost without the rule),
  center card blue (the stored object), right card green (cost with
  the rule), tradeoff bar, one-connection footer.
- Reject list verified by grep: no gradients, glows, shadows,
  embedded images, robots, brains, glowing networks, clip art,
  watermarks, logos. All 24 SVGs parse as XML. Text widths checked
  analytically against card bounds after one trim pass.
- Style checked: no contractions, em dashes, semicolons, or banned
  filler in plate text or captions. Apostrophes are possessives only.
- Captions follow the bar format exactly:
  `Chapter plate L<nn>-C<n>. Left: ... Center: ... Right: ... Bottom: ....`
- Plates placed at the end of each concept's ## section, directly
  before the next section header. Task said do NOT run build/build.py
  and do NOT touch l01-l06 or l13-l17: both honored.

## L07 - Neural Networks 1 (4 plates)

1. assets/plate-l07-chap-neuron.svg - "the bend and the stack"
   Concept: One neuron / Where one neuron breaks / Stacking: the MLP.
   Left: linear models without bends, XOR impossible (w1+w2 > -2b yet
   < -b: impossible). Center: the neuron sigma(w^T x + b), walkability
   toy z = 2.8 -> 2.8 and z = -3.2 -> 0. Right: the MLP stack, the
   bump 0, 1, 2, 1, 0, 784-256-128-10 = 235,146 knobs, layer 1 owns
   85 percent. Bottom: bends buy learned features and any-shape
   approximation; representable does not mean learnable.

2. assets/plate-l07-chap-loss.svg - "the loss vocabulary"
   Concept: The loss vocabulary / The softmax-cross-entropy pair.
   Left: the wrong scoreboard, MSE on class labels. Center: loss
   (prediction, truth): MSE gap 20 -> 400, -log(0.7) = 0.357,
   -log(0.1) = 2.303. Right: softmax(2, 1, 0.5) = (0.63, 0.23, 0.14),
   gradient p - y = (-0.37, 0.23, 0.14), hinge s = 0.3 pays 0.7,
   (1000,999) log-sum-exp = 1000.31. Bottom: match the loss to the
   data type.

3. assets/plate-l07-chap-activation.svg - "activations and initialization"
   Concept: The activation zoo / Initialization: the symmetry problem.
   Left: sigmoid slope 0.00005 at z = 10, tanh slope 0.071 at z = 2,
   zero init clones 64 neurons into one. Center: match activation to
   layer job and init to activation; GELU(2) = 1.95, GELU(-2) =
   -0.046. Right: He Var = 2/100 = 0.02 (std 0.141), Xavier Var =
   1/100 = 0.01 (std 0.10). Bottom: the failure is silent, so match
   everything.

4. assets/plate-l07-chap-residual.svg - "depth and the skip"
   Concept: Where depth breaks / Residual connections.
   Left: 0.25^50 ~ 10^-30, plain nets stall past ~20 layers.
   Center: out = x + F(x), skip multiplies the gradient by 1.
   Right: ResNet-152, 3.57 percent top-5 error, ILSVRC 2015 winner.
   Bottom: depth became a dial; x and F(x) must match dimension.

## L08 - Backpropagation (4 plates)

1. assets/plate-l08-chap-theorem.svg - "the O(N) gradient"
   Concept: The job / Finite differences / The chain rule /
   The fundamental theorem. Left: finite differences need n+1 forward
   passes: 10,001 per step for 10,000 knobs, 1B for 1B knobs.
   Center: the toy backward walk (y_hat = -1.5, L = 3.125,
   dL/dW2 = [0, -3.75], dL/dW1 = [[0,0],[2.5,5]]). Right: ~12 ops
   forward vs ~10 backward; gradient O(N); 1B knobs cost 2-3 forward
   passes. Bottom: finite differences stay as the debugging ground
   truth (relative error below 1e-7); memory is the invoice.

2. assets/plate-l08-chap-module.svg - "modules, VJPs, and the graph"
   Concept: The module contract / Vector-Jacobian products /
   The computational graph. Left: hand-derived gradients; full
   Jacobian 1024x1024 = 1,048,576 stored entries. Center: forward
   plus backward, locally; the tape saves x. Right: matrix-free VJP
   v^T J; dL/dW rank 1 per example; fan-out adds 4 + 3 = 7.
   Bottom: frameworks compose contracts; differentiability is the
   price.

3. assets/plate-l08-chap-memory.svg - "the tape's price"
   Concept: The memory bill / Checkpointing / Reversible layers /
   Sparse gradients. Left: 12 layers x 100 MB = 1,200 MB tape;
   activations dominate. Center: saved intermediates the backward
   reads; sparse embedding backward touches 64 values, not 640,000.
   Right: checkpoint every 4th layer -> 300 MB for one extra forward
   (30-40 percent slower); reversible layers give O(1) tape in
   depth. Bottom: checkpoint when memory binds, buy GPUs when money
   binds.

4. assets/plate-l08-chap-curvature.svg - "rank one and curvature"
   Concept: The rank-1 gradient / Second order without the matrix /
   Double backward. Left: Newton's O(d^3); Hessian 10^18 entries at
   d = 1B. Center: dL/dW = signal outer input (toy: [0,-3.75] =
   (-2.5) x [0,1.5]); Hessian-vector product H*v. Right: H*v in O(N)
   via double backward; the matrix never exists; batch size caps the
   update rank. Bottom: curvature survives only for matrix-free
   methods; double backward doubles the tape.

## L09 - K-Means and GMM (4 plates)

1. assets/plate-l09-chap-kmeans.svg - "the ad-hoc algorithm"
   Concept: First attempt: k-means / Where it breaks. Left: a million
   unlabeled records, no y. Center: assign, update to means; toy
   {1,2,3,10,11,12} converges in 1 round to mu = 2 and 11.
   Right: distortion falls every round; k = 1: 125.5, k = 2: 4.0;
   bad seeds strand a cluster (distortion near 200). Bottom:
   NP-hardness forbids a global guarantee.

2. assets/plate-l09-chap-seeding.svg - "seed far apart, choose k"
   Concept: k-means++ / Choosing k: beyond the elbow. Left: the seed
   lottery; eyeball-only elbows. Center: seed proportional to squared
   distance (toy: 64, 81, 100 vs 1, 1); sklearn default plus 10
   restarts. Right: O(log k) expected approximation ratio; elbow
   125.5, 4.0, 2.7, 1.5 bending at k = 2; silhouette 0.85.
   Bottom: seeding is provably close; choosing k stays heuristic.

3. assets/plate-l09-chap-gmm.svg - "the soft grown-up"
   Concept: Softening: Gaussian mixture models. Left: hard
   all-or-nothing assignments force overlapping points. Center:
   responsibilities gamma_j(x); point 3 claims 1.0 vs ~0, midpoint
   6.5 splits 0.5/0.5. Right: 70/30 splits; k-means is the
   zero-variance limit of the GMM likelihood. Bottom: softness costs
   an EM loop, because the labels stay hidden.

4. assets/plate-l09-chap-shapes.svg - "shapes and covariance dials"
   Concept: The clustering zoo / GMM covariance types. Left: straight
   bisectors (centers 2 and 11, border at 6.5); the bisector splits
   both crescents. Center: choose by shape (DBSCAN, spectral,
   hierarchical). Right: d = 10, k = 5: full 275, diagonal 50,
   spherical 5, tied 55. Bottom: the covariance type is the GMM's
   bias-variance dial.

## L10 - EM and PCA (4 plates)

1. assets/plate-l10-chap-elbo.svg - "the bound that never falls"
   Concept: The key question / Jensen / The ELBO. Left: log of a sum
   over labelings, no closed form. Center: ELBO = E_Q[log p(x,z)] +
   H(Q); log(5) = 1.609 beats the chord's 1.099; tight at Q =
   posterior. Right: E-step touches, M-step climbs: L(new) >=
   B(new) >= B(old) = L(old). Bottom: monotonicity, not the global
   maximum, not speed.

2. assets/plate-l10-chap-em.svg - "EM: two easy steps"
   Concept: EM alternation / EM beyond mixtures / MAP-EM. Left: hard
   guess-fit-repeat forces the point at 5.0 into one cluster.
   Center: E-step responsibilities, M-step weighted MLE; coordinate
   ascent on (Q, theta). Right: toy converges to mu = 0.33 and 10.0;
   sigma_1^2 = 0.389; MAP with prior 0.433; collapse is sigma -> 0,
   density -> infinity. Bottom: local maxima, slow crawl, degenerate
   collapse; the prior is the guardrail.

3. assets/plate-l10-chap-pca.svg - "the axes of variation"
   Concept: PCA / PCA via SVD / Eckart-Young. Left: 200 redundant
   features; exact eigendecomposition O(d^3). Center: eigenvectors
   of the covariance; diagonal toy eigenvalues (2.5, 0), one axis
   keeps 100 percent; SUV PC1 = (1, 0.618) keeps 97.9 percent.
   Right: shipped as SVD(X), s_1 = 3.162; Eckart-Young proves
   truncated SVD is the optimal rank-k summary. Bottom: linear only.

4. assets/plate-l10-chap-scale.svg - "PCA at scale"
   Concept: Whitening / Randomized SVD. Left: exact SVD O(n d^2),
   dead at d = 100,000. Center: sketch Y = X Omega, orthogonalize,
   SVD the small core. Right: O(n d k); n = 1M, d = 100K, k = 50
   finishes; oversample by 10, tiny error. Bottom: approximate, with
   oversampling as the control.

## L11 - Diffusion Models (4 plates)

1. assets/plate-l11-chap-forward.svg - "the fixed destruction"
   Concept: The forward process / The noise schedule. Left: one-shot
   generation, GAN instability, mode collapse. Center: x_t from
   x_{t-1} with fixed betas; pixel toy 0.8 -> 0.846 (0.796 + 0.05);
   closed form jumps any t in O(1). Right: 100 steps give 0.48
   signal inside 0.80 noise; 1,000 steps pure static; at t = 500,
   linear keeps 0.28 signal vs cosine 0.71. Bottom: destruction
   needs no intelligence; the schedule is the curriculum.

2. assets/plate-l11-chap-reverse.svg - "the learned reversal"
   Concept: The reverse process / noise points uphill. Left: one
   giant reversal has an unlearnable multimodal posterior. Center:
   the network predicts each step's noise; ELBO becomes
   noise-prediction MSE per level. Right: stable regression, no
   adversary; noise prediction is score estimation; sampling is
   hill-climbing from static. Bottom: the method is correct and the
   meter is running.

3. assets/plate-l11-chap-bill.svg - "the sampling bill"
   Concept: The sampling bill / Why T is large / Consistency models.
   Left: GAN sampling at 1 network eval per image. Center: T tiny
   reversals, each near-Gaussian and learnable; 10,000 images at 100
   evals/s/GPU = 27.8 GPU-hours. Right: DDIM 50 steps (20x cheaper),
   distilled 4 steps (250x, 7 minutes). Bottom: price the quality
   tier, not the method.

4. assets/plate-l11-chap-steer.svg - "steer and compress"
   Concept: Classifier-free guidance / Latent diffusion. Left:
   unconditional samples ignore the prompt; pixel space is 786,432
   numbers per step. Center: guided = uncond + w(cond - uncond);
   w = 7.5 gives (0.1,0.2),(0.3,0.1) -> (1.6,-0.55); VAE encodes
   once, decodes once. Right: 64x64x4 = 16,384, 48x smaller (Stable
   Diffusion's design); past w ~ 15 artifacts. Bottom: guidance is a
   dial, not a switch; latent diffusion is the product.

## L12 - Foundation Models (4 plates)

1. assets/plate-l12-chap-paradigm.svg - "the 500-label problem"
   Concept: The job / First attempt / The paradigm. Left: 10,001
   knobs on 500 examples (20 params per example): train ~100
   percent, test 58. Center: representations from shared contexts
   across billions of sentences. Right: linear probe 88 vs 58
   percent; the 30-point gap is pre-training; probe ladder 62, 80,
   85. Bottom: the billions built the map, the 500 labels steer; the
   training set is everything, so benchmark numbers need
   contamination reports.

2. assets/plate-l12-chap-objectives.svg - "labels hiding in the data"
   Concept: The self-supervised objectives. Left: labels cost money;
   the text order is free. Center: next-token, masked, contrastive;
   a trillion words give a trillion examples. Right: -log(0.02) =
   3.91, -log(0.3) = 1.20, one positive pulled against 127
   negatives. Bottom: the objective picks the strength.

3. assets/plate-l12-chap-lora.svg - "adapt without moving billions"
   Concept: LoRA / The adaptation zoo. Left: full fine-tuning moves
   7B knobs on 500 labels; a 14 GB copy per task. Center: W_adapted =
   W + AB, W frozen; d = 1000, r = 10: 20,000 vs 1,000,000 (50x);
   d = 4096, r = 8: 256x. Right: 128 KB per matrix in fp16; deploy
   W' = W + AB once, zero extra latency. Bottom: rank r is the dial;
   large behavioral changes still need full fine-tuning.

4. assets/plate-l12-chap-scaling.svg - "the scaling bill"
   Concept: Scaling laws / Multitask and parallelism. Left: 70B in
   fp16 is 140 GB, does not fit one 80 GB GPU. Center: 6 x N x D
   FLOPs; 70B on 1.4T tokens = 5.9e23 FLOPs; 2,000 A100s, 34 days,
   about $3.3M. Right: Chinchilla's 20 tokens per parameter; 70B on
   1.4T beats 280B on 300B; 3D parallelism; adaptation costs tens of
   dollars. Bottom: concentrate pre-training, democratize adaptation.

## Open items / notes for the coordinator
- The generator lives at build/gen-chap-plates-cs229-l07-12.py and
  re-renders all 24 SVGs deterministically (asserts first).
- Render gate (R1: overflow at 375/768/1200) is a render-check job:
  SVGs are vector, 960 wide, text widths checked analytically. Not
  verified in a browser here.
- One lesson rounding note: l10's M-step squared deviations are
  0.0289/0.6889/0.4489 exactly; the lesson prints 0.028/0.694/0.444
  and averages to 0.389. The plate uses the lesson's stated
  rounding, and the generator asserts the exact squares and the
  0.389 average separately. Flagged, not changed: lesson text is the
  builder's source of truth.
- Captions carry the source per plate; sources are the lecture, the
  cited papers (He et al. 2015; Arthur and Vassilvitskii 2007;
  Ho et al. 2020; Hu et al. 2021; Hoffmann et al. 2022; Halko et
  al. 2009), or "original synthesis of the lesson/lecture" for
  original toys.
