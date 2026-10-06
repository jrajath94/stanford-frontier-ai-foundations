# notation_and_shapes.md, math-ml symbol registry

Date: 2026-10-06. RUN 1 introduces the U01 symbols. New symbols are added
here before first use in later runs.

## Conventions

- Scalar: lowercase italic letter, for example x.
- Vector: bold lowercase, for example x = [x_1, x_2, x_3].
- Matrix: bold uppercase, for example X with shape (rows, columns).
- Index i runs over items. N counts items.
- Shapes are written as tuples: (4,) is a 4-vector. (2, 3) is 2 rows,
  3 columns.

## U01 symbol table

| Symbol | Meaning | First use |
|---|---|---|
| x | a quantity (scalar or vector, per context) | U01-C01 |
| f(x) | output of function f at input x | U01-C01 |
| {a, b, c} | set of items a, b, c | U01-C01 |
| \|S\| | count of items in set S | U01-C01 |
| in | membership: a in S means a is an item of S | U01-C01 |
| R | the real numbers | U01-C02 |
| sum x_i | sum over index i | U01-C04 |
| prod x_i | product over index i | U01-C04 |
| log_b(x) | log of x base b. Log means base e unless stated | U01-C05 |
| e | 2.718281828459045 | U01-C05 |
| [x_1, x_2] | vector with two items | U01-C06 |
| x dot y | dot product of vectors x and y | U01-C06 |
| \|\|x\|\| | length (norm) of vector x | U01-C06 |
| X[i, j] | item at row i, column j of matrix X | U01-C07 |
| axis=0 | the row axis of a table (runs down) | U01-C07 |
| axis=1 | the column axis of a table (runs across) | U01-C07 |

## Reuse promises

- The chip for a vector keeps the same color across plates.
- sum and prod symbols keep the same meaning in U03 (gradients of sums)
  and U04 (sums over events).
- axis=0/axis=1 keep the same meaning in U08 (tensor axes) and P12.

## U02 symbol table (added 2026-10-06, RUN 2)

| Symbol | Meaning | First use |
|---|---|---|
| span(v_1, ..., v_k) | set of all combinations of the listed vectors | U02-C01 |
| dim | count of independent directions of a subspace | U02-C01 |
| rank(A) | count of independent rows/columns of A | U02-C02 |
| null(A) | set of x with A x = 0 | U02-C02 |
| col(A) | set of all A x (column space) | U02-C02 |
| proj_a(b) | shadow of b on the line of a | U02-C04 |
| det(A) | signed area scale factor of square A | U02-C06 |
| A^{-1} | inverse of A. Undoes A | U02-C07 |
| A^+ | pseudoinverse of A | U02-C07 |
| A v = lambda v | eigen equation: v keeps its line, lambda stretches | U02-C08 |
| M = U Sigma V^T | SVD. Sigma diagonal with singular values sigma_i | U02-C09 |
| sigma_i | i-th singular value, non-negative, sorted descending | U02-C09 |
| x^T A x | quadratic form of A along x | U02-C10 |
| Cov(X) | covariance matrix of data X | U02-C11 |
| kappa(A) | sigma_max / sigma_min, condition number | U02-C12 |

## U04a symbol table (added 2026-10-06, RUN 2)

| Symbol | Meaning | First use |
|---|---|---|
| Omega | sample space, set of all outcomes | U04a-SB01 |
| X | random variable: outcome -> number | U04a-SB01 |
| p(x) | PMF or PDF value at x | U04a-SB02 |
| F(x) | CDF: P(X <= x) | U04a-SB02 |
| p(x, y) | joint distribution of X and Y | U04a-SB03 |
| p(x), p(y) | marginals | U04a-SB03 |
| p(y | x) | conditional: Y given X = x | U04a-SB03 |
| E[X], mu | expectation of X | U04a-SB04 |
| Var(X), sigma^2 | variance of X | U04a-SB04 |
| x_1..x_n | IID sample of size n | U04a-SB05 |
| L(theta) | likelihood of the data at parameter theta | U04a-SB07 |
| l(theta) | log-likelihood | U04a-SB07 |
| theta_hat | MLE: argmax of L | U04a-SB07 |

## U03 symbol table (added 2026-10-06, RUN 3)

| Symbol | Meaning | First use |
|---|---|---|
| df/dx | partial derivative of f in x, others frozen | U03-C01 |
| dy/dx = dy/du * du/dx | chain rule: link slopes multiply | U03-C02 |
| grad f | vector of partial derivatives, shape (n,) | U03-C03 |
| J | Jacobian, shape (m, n) for g: R^n -> R^m | U03-C03 |
| H | Hessian of second derivatives, shape (n, n) | U03-C03 |
| T_k(x) | degree-k Taylor portrait around a | U03-C04 |
| eta | learning rate (step size) | U03-C07 |
| w_new = w - eta * grad | gradient descent update | U03-C07 |
| w - f'/f'' | Newton step | U03-C08 |
| L(x, lam) | Lagrangian: objective minus lam times constraint | U03-C09 |
| lam | Lagrange multiplier (fence fine), lam >= 0 | U03-C09 |
| KKT | four checks: allowed, lam >= 0, stationary, slackness | U03-C10 |
| g(lam) | dual function: min over x of L | U03-C10 |

## U04b symbol table (added 2026-10-06, RUN 3)

| Symbol | Meaning | First use |
|---|---|---|
| H(p) | entropy: -sum p(x) log p(x) | U04b-SB09 |
| D_KL(p\|\|q) | KL divergence: sum p log(p/q), p truth, q model | U04b-SB10 |
| H(p, q) | cross-entropy: H(p) + D_KL(p\|\|q) | U04b-SB10 |
| theta_hat | ML estimate: argmax of the likelihood | U04b-SB12 |
| mu_hat, sigma2_hat | Gaussian MLE: mean and mean squared spread | U04b-SB13 |
| p_hat | discrete MLE: counts / total | U04b-SB14 |
| N(mu, Sigma) | multivariate Gaussian, mu shape (d,), Sigma (d, d) | U04b-SB15 |
| sum_k w_k N(mu_k, Sigma_k) | mixture density, weights sum to 1 | U04b-SB16 |

## U05 symbol table (added 2026-10-06, RUN 4)

| Symbol | Meaning | First use |
|---|---|---|
| R(h) | population risk: E[l(y, h(x))] | U05-C01 |
| R_hat(h) | empirical risk: (1/n) sum_i l(y_i, h(x_i)) | U05-C01 |
| l(y, y_hat) | loss: price of predicting y_hat for truth y | U05-C01 |
| H | hypothesis class: the allowed rules | U05-C01 |
| ERM | empirical risk minimization: argmin over H of R_hat | U05-C01 |
| lambda | penalty strength, lambda >= 0 | U05-C05 |
| J(w) | penalized objective: RSS + n lambda w^2 | U05-C05 |
| k | fold count in cross-validation | U05-C08 |
| TPR, FPR, FNR | true/false positive rates, false negative rate | U04c-SB20 |
| AUC | area under the ROC curve | U04c-SB20 |

## U04c/U09 symbol table (added 2026-10-06, RUN 4)

| Symbol | Meaning | First use |
|---|---|---|
| pi_k | prior probability of class or component k | U04c-SB18 |
| p(k | x) | posterior of class k at x | U04c-SB18 |
| z | latent label, unobserved | U04c-SB21 |
| r_i | responsibility: p(z_i = 1 | x_i) | U04c-SB22 |
| Q | expected complete log-likelihood (EM) | U04c-SB22 |
| h | Parzen bandwidth | U04c-SB25 |
| K(u) | kernel function, integrates to 1 | U04c-SB25 |

## U06 symbol table (added 2026-10-06, RUN 5)

| Symbol | Meaning | First use |
|---|---|---|
| w_hat | least-squares estimate: (X^T X)^{-1} X^T y | U06-C01 |
| RSS | residual sum of squares: sum (y_i - pred_i)^2 | U06-C01 |
| sigma2_hat | noise variance MLE: RSS / n | U06-C02 |
| sigmoid(z) | 1 / (1 + exp(-z)) | U06-C03 |
| l(w) | log-likelihood of the logistic model | U06-C03 |
| p_k | softmax probability of class k | U06-C04 |
| g(mu) | link function: maps mean to linear score | U06-C05 |
| phi(x) | feature map into the kernel space | U06-C06 |
| k(x, z) | kernel: dot product in feature space | U06-C06 |
| K | Gram matrix: K_ij = k(x_i, x_j), shape (n, n) | U06-C07 |
| 2/\|\|w\|\| | geometric margin width | U06-C08 |
| xi_i | slack: margin violation of point i | U06-C08 |
| C | violation price, C >= 0 | U06-C08 |
| alpha_i | dual variable: vote of point i | U06-C09 |
| eta | step size (reused from U03-C07) | U06-C10 |
| kappa(G) | condition number of G (reused from U02-C12) | U06-C11 |

## U09 leaf symbol table (added 2026-10-06, RUN 5)

| Symbol | Meaning | First use |
|---|---|---|
| a_i | k-means cluster assignment of point i | U09-C01 |
| mu_k | center of cluster k | U09-C01 |
| J | k-means objective: within-cluster sum of squares | U09-C01 |
| v_i | i-th principal direction (eigenvector) | U09-C03 |
| lambda_i | i-th eigenvalue: variance along v_i | U09-C03 |
| z | PCA scores: Xc v, shape (n, k) | U09-C04 |
| x_hat | reconstruction from k components | U09-C04 |
| W | factor loadings, shape (d, k) | U09-C09 |
| Psi | diagonal noise variances in factor models | U09-C09 |
| inertia | mean squared distance to nearest center | U09-C12 |


## U07 symbol table (added 2026-10-06, completion run)

| Symbol | Meaning | First use |
|---|---|---|
| R_j | j-th leaf region (rectangle) | U07-C01 |
| c_j | majority vote of region R_j | U07-C01 |
| G | Gini impurity: 1 - sum_k p_k^2 | U07-C02 |
| H | entropy in nats: -sum_k p_k log p_k | U07-C02 |
| t | split threshold | U07-C03 |
| gain | parent impurity minus weighted child impurity | U07-C03 |
| R_alpha(T) | cost-complexity score: R(T) + alpha |T| | U07-C04 |
| alpha | prune exchange rate (trees), also AdaBoost stump vote | U07-C04/C08 |
| B | number of ensemble members | U07-C05 |
| rho | pairwise tree correlation | U07-C06 |
| sigma^2 | single-model variance | U07-C06 |
| OOB | out-of-bag: points missing a replicate | U07-C05 |
| eps | weighted error of a stump | U07-C08 |
| w_i | AdaBoost weight of point i | U07-C08 |
| r_i | residual: y_i - F(x_i) | U07-C09 |
| nu | boosting learning rate | U07-C09 |
| F_m | ensemble after m rounds | U07-C09 |

## U08 symbol table (added 2026-10-06, completion run)

| Symbol | Meaning | First use |
|---|---|---|
| W, b | layer weights and biases | U08-C01 |
| z, a | pre-activation and activation | U08-C01 |
| relu | max(z, 0) per item | U08-C01 |
| do, dz, da | backward-pass adjoints | U08-C02 |
| k | convolution kernel | U08-C03 |
| n - k + 1 | valid convolution output length | U08-C03 |
| h_t | RNN hidden state at step t | U08-C06 |
| W_h, W_x | recurrent and input weights | U08-C06 |
| f, i, g, o | LSTM forget/input/candidate/output gates | U08-C08 |
| c_t | LSTM cell state | U08-C08 |
| Q, K, V | query, key, value matrices | U08-C09 |
| A | attention weight matrix, rows sum to 1 | U08-C09 |
| d_k | per-head key dim | U08-C09 |
| m, v | Adam first and second moment estimates | U08-C12 |
| m_hat, v_hat | bias-corrected moments | U08-C12 |
| eta | step size (reused from U03-C07) | U08-C12 |

## U10 symbol table (added 2026-10-06, completion run)

| Symbol | Meaning | First use |
|---|---|---|
| z | latent variable (generative) | U10-C01 |
| p(x, z) | joint of visible and latent | U10-C01 |
| q(z|x) | approximate posterior (encoder) | U10-C02 |
| ELBO | E_q[log p(x|z)] - KL(q||p) | U10-C02 |
| D, G | GAN discriminator and generator | U10-C03 |
| V | minimax value: E[log D(real)] + E[log(1-D(fake))] | U10-C03 |
| JSD | Jensen-Shannon divergence | U10-C04 |
| F_n | empirical CDF from n samples | U10-C05 |
| beta | KL weight in beta-VAE | U10-C06 |
