# Cheatsheet: formulas, shapes, traps, decision rules

One dense page for the whole course. Formulas use the notation fixed in
[notation and shapes](notation_and_shapes.html). Section pointers name
the lesson that proves each line.

## U01, mathematical language

| Formula / shape | Trap | Decision rule |
|---|---|---|
| sum_i x_i: add over i | A bare sum with no index range hides the set | Write the index set on first use |
| ln: log base e | log base 10 in code, base e on paper | State the base once, then hold it |
| x in R^n: column vector | Row-vector code breaks matrix products | Keep vectors as columns in formulas |
| price = 65 / 150 = 0.4333 per sq ft | Unit mismatch silently scales answers | Check units before you check code |
| Round trip: 0.4333 * 150 = 65.0 | One-way arithmetic hides factor errors | Multiply back, the trip must close |

## U02, linear geometry

| Formula / shape | Trap | Decision rule |
|---|---|---|
| A: m x n maps R^n -> R^m | Transpose confusion flips domain and codomain | Read columns: they are images of basis vectors |
| rank(A): independent columns | Noise makes every singular value positive | Judge rank from the singular-value gap |
| Nullspace: {x: A x = 0} | Forgetting it loses solutions to A x = b | Check the nullspace before you invert |
| A v = lambda v | Not every matrix has a full eigenbasis | Symmetric matrices do, use them when you can |
| A = U S V^T, U m x r, S r x r, V n x r | Truncation rank picked by habit | Keep the singular values that carry the energy |
| Covariance: d x d, symmetric, PSD | Uncentered data corrupts it | Center first, then form X^T X / n |
| kappa(A): sensitivity of A x = b | Large kappa amplifies input error | Rescale or regularize before you solve |
| Projection: (b dot a)/(a dot a) * a | Forgetting to divide by (a dot a) | Normalize the direction, then project |

## U03, calculus and optimization

| Formula / shape | Trap | Decision rule |
|---|---|---|
| df/dx via chain rule: dy/dx = dy/dz * dz/dx | Dropped link in a long chain | Write every link, multiply in order |
| Gradient: vector of partials | Gradient is not the direction of steepest ascent in a warped metric | Pair the gradient with the right geometry |
| Hessian PSD everywhere => convex | One non-PSD point kills global guarantees | Test curvature, do not assume it |
| Taylor: f(x) ~= f(a) + grad f(a)^T (x-a) + 1/2 (x-a)^T H (x-a) | Trusting the quadratic far from a | Keep the trust region small |
| GD: w <- w - eta grad | eta too large diverges, too small crawls | Sweep eta on a log grid |
| Newton: w <- w - H^{-1} grad | O(d^3) per step, indefinite H misleads | Use for small d with PSD Hessian |
| KKT: stationarity, primal and dual feasibility, complementary slackness | Ignoring complementarity | At the optimum, each constraint is either tight or its multiplier is zero |
| Gradient check: analytic vs finite difference to 8 digits | Skipping it | U03-C01 matched to 6.99999999298484 vs 7: run the check first |

## U04, probability and estimation

| Formula / shape | Trap | Decision rule |
|---|---|---|
| E[X] = sum x p(x), Var = E[(X - E[X])^2] | Confusing sample and population moments | Name which distribution the average is under |
| p(x, y) = p(x | y) p(y) | Conditioning on the wrong event | Write the conditioning bar explicitly |
| H = -sum p log p | Entropy is not uncertainty about the world | It is the code length under p |
| KL(p || q) = sum p log(p/q) | KL is not symmetric | The first argument is the truth you code |
| min KL(p_data || p_theta) = max likelihood | Forgetting the entropy term is constant | Drop constants only after you prove them constant |
| MLE: argmax_theta prod_i p(x_i, theta) | Maximizing density at a point for continuous data | Work with the log-likelihood, sums beat products |
| Gaussian MLE: mu_hat = mean, sigma2_hat = variance of data | Biased variance at small n | Know the n vs n-1 story before you report |
| Bernoulli MLE: p_hat = k / n | n = 8 gives wobble (0.5 vs 0.3 in the lesson toy) | Report the estimator's spread, not just its value |
| EM: E-step fills latent posteriors, M-step maximizes | Local optima | Restart from several inits |

## U05, risk and generalization

| Formula / shape | Trap | Decision rule |
|---|---|---|
| R_hat(h) = (1/n) sum loss(h(x_i), y_i) | Treating it as the goal | It is the proxy, R is the goal |
| R(h) = E[loss(h(x), y)] | You never observe it | Bound it with validation |
| Bias: error from a too-simple class | Adding data to fix bias | Bias does not shrink with n, change the model |
| Variance: error from a too-sensitive fit | Adding capacity to fix variance | Variance shrinks with n, or regularize |
| L2: + lambda ||w||^2 | lambda picked by superstition | Tune on validation, never on test |
| L1: + lambda ||w||_1 | Sparsity is not feature selection with guarantees | Validate the selected set |
| k-fold CV | Leakage across folds | Split before any preprocessing that sees labels |
| Learning curves | Reading noise as signal | Plot mean and spread across seeds |
| Leakage | Test info in training features | Audit the feature pipeline, not just the score |

## U06, linear models, kernels, margins

| Formula / shape | Trap | Decision rule |
|---|---|---|
| OLS: w = (X^T X)^{-1} X^T y | X^T X singular | Use the pseudoinverse or add lambda I |
| Squared loss = Gaussian noise MLE | Assuming the noise story | Match the loss to the noise: Laplace buys absolute loss |
| logit(p) = log(p/(1-p)), logit(0.7) = 0.847298 | Forgetting the link | Model log-odds, report probabilities |
| Softmax: exp(z_i) / sum_j exp(z_j) | Overflow in exp | Subtract the row max first |
| Kernel: k(x, x') replaces dot products | Assuming the kernel equals a naive feature dot product (43 vs 49 in the lesson) | Trust the Gram matrix |
| Gram K: n x n, symmetric, PSD | Non-PSD K breaks the dual | Check eigenvalues before you optimize |
| SVM dual: solve in alpha, decide with sum alpha_i y_i k(x_i, x) + b | Tuning C on training accuracy | Tune C on validation, C trades margin for slack |
| kappa(X^T X) raw 3.350789e6 in the lesson | Solving the normal equations directly | Prefer QR or SVD for ill-conditioned X |

## U07, trees and ensembles

| Formula / shape | Trap | Decision rule |
|---|---|---|
| Gini = 1 - sum p_k^2, node [4,1]: 0.32 | Gini 0 with impure children elsewhere | Score the split, not the node |
| Entropy = -sum p_k log p_k, node [4,1]: 0.6730 | Log base changes the number | Fix the base before you compare |
| Split gain = impurity(parent) - weighted impurity(children) | Best split on training = best model | Validate the depth, not the gain |
| Pruning: trade depth for honesty | Pruning by gut | Prune on validation error |
| Bagging: average B bootstrapped trees | Correlated trees keep the covariance term | Subsample features to decorrelate |
| Boosting: fit residuals in sequence | Chasing noise | Shrink the step, stop on validation |
| Imbalance: class weights or resampling | Accuracy on skewed classes | Report per-class scores and the baseline |
| Feature importance from trees | Fragile under correlated features | Treat it as a hint, not a ranking |

## U08, neural and sequence

| Formula / shape | Trap | Decision rule |
|---|---|---|
| MLP: affine, nonlinearity, repeat | Dead ReLUs | Watch the activation histogram |
| Backprop: chain rule with a budget | Wrong gradient, wrong everything | Finite-difference check first (U03-C12 habit) |
| Conv: shared kernel across space | Padding changes the shape | Track shapes through every layer |
| RNN: same weights each step | Gradient product explodes or dies (1.5^10 = 57.665) | Gate it or shorten the horizon |
| LSTM/GRU gates | Gate saturation | Initialize to let signal through early |
| Attention: softmax(Q K^T / sqrt(d_k)) V | Forgetting the sqrt(d_k) scale | Scores grow with d_k, scale them |
| Attention weights sum to 1 per row | Reading weights as explanations | Weights show routing, not reasons |
| Transformer: stacked attention + MLP | Quadratic cost in sequence length | Budget n^2 before you scale n |
| Normalization (batch/layer) | Train/test statistic mismatch | Know which statistics each mode uses |
| Adam vs SGD | Adam's adaptivity hides bad conditioning | Compare on wall-clock, not steps |

## U09, unsupervised and latent

| Formula / shape | Trap | Decision rule |
|---|---|---|
| K-means: alternate assign and update | Distance decides, units decide distance | Scale features first |
| PCA: top k eigenvectors of covariance | Variance is not meaning | Inspect what the components keep |
| Reconstruction MSE at k = 1: 0.009172 (lesson toy) | Judging by train error only | Evaluate on held-out data |
| Eigenvalues: 6.541656 and below (lesson toy) | Keeping components by habit | Cut where the spectrum drops |
| GMM + EM | Degeneracy: a component collapses on one point | Constrain or regularize the covariances |
| Held-out inertia | Train inertia always flatters | Select k on data the fit never saw |

## U10, generative synthesis

| Formula / shape | Trap | Decision rule |
|---|---|---|
| ELBO = E_q[log p(x|z)] - KL(q||p) | Reporting the bound as the likelihood | It is a lower bound, the gap is the KL to the true posterior |
| VAE: encoder q, decoder p, prior p(z) | Posterior collapse | Check that the latent codes carry information |
| GAN: min_G max_D game | Mode collapse | Watch sample diversity, not just quality |
| Samples are not densities | Comparing GAN samples to VAE likelihoods | Compare like with like |
| Every proof rests on assumptions | Citing the theorem past its assumptions | List the assumptions, then test them |
| Toy counterexample beats a vague objection | Arguing from intuition | Build the smallest case that breaks the claim |

## Cross-unit decision rules

| Situation | Rule |
|---|---|
| Loss will not decrease | Check the gradient against finite differences (U03-C12), then the learning rate, then the data pipeline |
| Model choice under time pressure | Fit OLS/logistic first, kernels when n is small and features are weak, trees when features are mixed-type, nets when data is abundant and structured |
| A number looks too good | Check for leakage (U05-C11), then the split, then the metric |
| A proof is quoted | Ask for the assumptions (U10-C07), test them on a toy (U10-C08) |
| A figure is shown | Ask for the code and the axes (U01-C09), honest plots carry both |
