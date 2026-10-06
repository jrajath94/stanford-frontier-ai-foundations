# source_gaps.md, math-ml unresolved gaps

Date: 2026-10-06. Keep visible until closed. Do not hide gaps behind the
inventory.

## G1, short-link destination unresolved -> SOURCE-UNREACHABLE

Seed link https://lnkd.in/gQe4kMkf. Second resolution attempt 2026-10-06
(RUN 2): HTTP 200 to a LinkedIn interstitial page (title "LinkedIn"),
no meta-refresh destination, content gated behind LinkedIn login/JS.
The destination cannot be resolved without authentication. Marked
SOURCE-UNREACHABLE. Per policy, no further attempts.

## G2, no transcript inspected

All 89 playlist entries are mapped by title only. No transcript, slide deck,
or notebook went through inspection. Every requested leaf concept keeps the
status PLANNED / SOURCE ATTRIBUTION PENDING until an artifact confirms it.
Attempt 2026-10-06 (RUN 5): curl of the playlist page returned one video
ID (vbs9WGWjS9U). Timedtext endpoints returned HTTP 200 with 0 bytes. 
youtubei player endpoint returned HTTP 400. No caption track retrieved.
yt-dlp is not installed on this box and installs are out of scope. G2
stays open.
Attempt 2026-10-06 (completion run): yt-dlp checked directly
(/usr/local/bin/yt-dlp, PATH, python module): not present on this box.
Per build laws installs are out of scope, so no caption fetch was
possible. G2 stays open.

## G3, NPTEL preview page is JavaScript-rendered

SRC-02 returned a loading shell to the text fetcher. Overview text,
syllabus, schedule, and assignment set remain unextracted. Second
attempt 2026-10-06 (RUN 3) via browser.open: still "Loading..." only.
Third attempt 2026-10-06 (RUN 4) via curl: 14,922-byte page, still
"loading", no syllabus text. Fourth attempt 2026-10-06 (RUN 5) via
curl: HTTP 200, 14,922 bytes, identical loading shell, no syllabus
text. Stays open. The page needs a
JS-capable client on an unfiltered network.

## G4, two offerings not reconciled

First offering: Jan 19, 2026 to Apr 10, 2026 (SRC-05).
Second offering: began August 2026, ongoing (SRC-03).
The 89-title playlist may mix both offerings. Not reconciled.

## G5, playlist duration unverified

Third-party claim of ~46.6 hours (SRC-07) not verified. Durations were not
extracted. The count of 89 titles is verified by direct enumeration.

## G6, tutorial contents not inspected

19 tutorial videos are listed by title. Their spoken content, code, and
demos are not inspected.

## G7, assessments and papers partially located

Assignment objectives for the first offering located 2026-10-06 in a
third-party learner repo (SRC-09: weeks 1-12 objectives, no contents
copied). Official assignment artifacts, exams, required papers, and
guest material have not been located. The week-12 generative preview
and the Lec 68/69 LLM/RL intros still need artifact confirmation.
The second offering's assignment set may differ. Stays open.

## Attachment A, playlist titles as enumerated 2026-10-06

Intro. Lec 01 Overview of Function Approximation. Lec 02 Recap of
Probability Theory - 1 Part 1. Lec 03 Recap of Probability Theory - 1
Part 2. Lec 04 Recap of Probability Theory - 1 Part 3. Lec 05 Recap of
Probability Theory Part 2. Lec 06 Understanding a Chest X-Ray as Sample
from Distribution. Lec 07 IID Assumption. Lec 08 Distribution Estimation.
Lec 09 Density Function. Lec 10 Challenge With ML. Tutorial 1 Introduction
to Python Basics. Tutorial 2 Simple Problem solving in Probability Theory.
Lec 11 Entropy. Lec 12 Kullback-Leibler (KL) Divergence. Lec 13 Minimization
of KL Divergence. Lec 14 Example of ML Estimate. Lec 15 Risk Minimization
Framework. Lec 16 Bayes Classifier. Tutorial 3 Risk Minimization Framework.
Lec 17 MLE for Gaussian Distribution. Lec 18 MLE for Generalized Discrete
Random Variable. Lec 19 Density Estimation for Mixed Distribution.
Lec 20 Latent Variable Models. Lec 21 MLE for Latent Variable Models.
Lec 22 Expectation Maximization Algorithm. Tutorial 4 Minmax Classifier.
Tutorial 5 Neyman Pearson Classifier. Tutorial 6 Example of NP Classifier,
ROC Curve. Tutorial 7A MLE for Gaussian Distribution. Tutorial 7B MLE for
Generalized Discrete Distribution. Lec 23 Convergence of EM. Lec 24 EM for
GMMs. Lec 25 MAP Estimate. Lec 26 Parzen Window. Lec 27 Nearest Neighbor
Classifier. Tutorial 8 Computation of EM for GMMs. Tutorial 9 MAP Estimate.
Lec 28 Ordinary Least Squares (OLS). Lec 29 Generalized Least Squares (GLS).
Lec 30 Linear Models for Classification. Lec 31 Bias-Variance Decomposition
and Analysis. Lec 32 Bias and Variance in Practice. Tutorial 10 Part A
Numerical Example on Bayes Classifier. Tutorial 10 Part B Numerical Example
on MLE and MAP Estimate. Lec 33 Regularization. Lec 34 Regularized ERM and
MAP Estimate. Lec 35 Stochastic Gradient Descent as a Regularizer.
Lec 36 Max-Margin Classifier and SVM. Lec 37 SVM Formulation.
Lec 38 Dual Function in SVM. Lec 39 SVM for Non-Linear Separable Case.
Lec 40 SVM with Kernel Function. Lec 41 Neural Networks and Universal
Approximation Theorem. Lec 42 ERM on Neural Networks and Error
Backpropagation. Lec 43 Local Receptive Field and Parameter Sharing.
Lec 44 Convolutional Neural Networks (CNNs) as Regularized MLP.
Lec 45 Recurrent Neural Networks (RNNs). Lec 46 Backpropagation in RNNs and
Vanishing Gradients Problem. Lec 47 LSTMs and GRUs. Tutorial 11 PyTorch -
Tensors and Data Loaders. Tutorial 12 PyTorch - Building MLP and Auto Grad.
Tutorial 13 PyTorch - Training the Model. Lec 48 Attention Part 1.
Lec 49 Attention Part 2. Lec 50 Multi-Head Attention and Transformer
Architecture. Lec 51 Positional Embeddings. Lec 52 Transfer Learning and
Knowledge Distillation. Lec 53 SGD, RMS Prop, ADAM: Optimizers.
Tutorial 14 Part 1 CNNs. Tutorial 14 Part 2 Transfer Learning using CNNs.
Tutorial 15 Part 1 RNNs, LSTMs and GRUs. Tutorial 15 Part 2 Deep RNNs,
LSTMs and GRUs. Lec 54 Decision Trees and Impurity Measures. Lec 55
Regression Trees. Lec 56 Ensemble Methods, Bagging and Boosting.
Lec 57 Gradient Boosting Algorithm. Lec 58 Ada-Boosting. Lec 59 Cross
Validation. Lec 60 Un-Supervised Learning. Lec 61 K-Means Clustering.
Lec 62 PCA - Principal Component Analysis. Lec 63 NCE - Noise Contrastive
Estimation. Lec 64 NCE, Info-NCE, SimCLR, JEPA. Lec 65 Introduction to
Generative Models. Lec 66 GAN - Generative Adversarial Networks. Lec 67
Variational Auto Encoders: VAEs. Lec 68 Introduction to Large Language
Models: LLMs. Lec 69 Introduction to Reinforcement Learning: RL.
