# Figure Audit: CS229 Lessons 07–12
# Figure Enforcer report. 2026-10-06.
# Scope: l07, l08, l09, l10, l11, l12 only. l01–l06 and l13–l17 untouched.

## Method
- Read MASTER_BRIEF.md, REJECTIONS.md, PIPELINE.md, FIGURE_SPEC_STANFORD.md verbatim.
- Listed every figure reference in the six lessons via regex on `assets/(plate-*.webp|svg/*.svg)`.
- Checked each file on disk at content/v2/cs229/assets/.
- 35 references were missing (32 webp plates + 2 SVGs + plate-l10-eckart-young which needed regeneration with corrected numbers). All generated now.
- Generation: plates via the media pipeline (media.generate_image), prompts written from the lesson's own worked numbers (F6: every number code-computed from the text toy, not invented). The 2 SVGs hand-authored to match the course's existing SVG style (720 viewBox, #F7F4EE ground, blue forward / red backward, every block and edge labeled).
- Verification: all 32 plates viewed (8 contact sheets); one reject fixed (plate-l10-map-em had a gibberish text fragment below the footer; regenerated clean). Reject list scanned: no gradient, glow, shadow, logo, watermark, robot, brain icon, clip art. On-plate footers match the markdown captions. Eckart-Young plate verified to show the CORRECTED numbers (s_1 = 3.162, s_2 = 0).
- Caption status: every reference carries alt text + title caption (verified-caption format: claim + "Source: ... Project: Stanford Frontier AI."). Counts: l07 20/20, l08 12/12, l09 10/10, l10 10/10, l11 8/8, l12 9/9.
- Final existence check: 69/69 referenced files exist on disk. Missing: NONE.

## L07 — Neural Networks 1 (l07-neural-networks-1.md)

| Figure referenced | Exists / generated | Caption status |
|---|---|---|
| assets/plate-l07-xor.webp | GENERATED (was missing) | captioned: "The XOR failure, worked..." Source: original plate for the linear separability proof |
| assets/svg/l07-neuron.svg | pre-existing | captioned |
| assets/plate-l07-bias-threshold.webp | pre-existing | captioned |
| assets/svg/l07-mlp.svg | pre-existing | captioned |
| assets/plate-l07-bump.webp | pre-existing | captioned |
| assets/plate-l07-embeddings.webp | GENERATED (was missing) | captioned: zip 3 -> (0.2, -1.1, 0.5, 0.9), 40 numbers |
| assets/plate-l07-loss-vocab.webp | GENERATED (was missing) | captioned: MSE 20->400, CE 0.7->0.357, 0.1->2.303 |
| assets/plate-l07-activation-table.webp | pre-existing | captioned |
| assets/plate-l07-gelu.webp | GENERATED (was missing) | captioned: GELU(2)=1.95, GELU(-2)=-0.046 |
| assets/plate-l07-he-init.webp | GENERATED (was missing) | captioned: Var(W)=2/100, std=0.141 |
| assets/svg/l07-residual.svg | pre-existing | captioned |
| assets/plate-l07-resnet-numbers.webp | pre-existing | captioned |
| assets/plate-l07-knob-count.webp | GENERATED (was missing) | captioned: 200,960+32,896+1,290=235,146, layer 1 = 85% |
| assets/plate-l07-double-descent.webp | GENERATED (was missing) | captioned: peak at 12 knobs/12 houses |
| assets/plate-l07-normalization.webp | GENERATED (was missing) | captioned: (2,4,6,8) -> mean 5, std 2.236, (-1.34,-0.45,0.45,1.34) |
| assets/plate-l07-dropout.webp | GENERATED (was missing) | captioned: p=0.5, keep (2,8), scale by 2 -> (4,16) |
| assets/plate-l07-learning-rate.webp | GENERATED (was missing) | captioned: 0.1 descends, 0.9 oscillates, 1.1 explodes |
| assets/plate-l07-mixed-precision.webp | GENERATED (was missing) | captioned: fp16/bf16 compute, fp32 masters |
| assets/plate-l07-loss-curves.webp | GENERATED (was missing) | captioned: five shapes, five diagnoses |
| assets/plate-l07-adversarial.webp | GENERATED (was missing) | captioned: panda -> gibbon, 99% confidence |

## L08 — Backpropagation (l08-backpropagation.md)

| Figure referenced | Exists / generated | Caption status |
|---|---|---|
| assets/plate-l08-op-audit.webp | pre-existing | captioned |
| assets/svg/l08-backprop.svg | pre-existing | captioned |
| assets/plate-l08-gradcheck.webp | pre-existing | captioned |
| assets/plate-l08-chain-snaps.webp | pre-existing | captioned |
| assets/plate-l08-memory-bill.webp | pre-existing | captioned |
| assets/plate-l08-vjp.webp | GENERATED (was missing) | captioned: 1024x1024 Jacobian, matrix-free VJP |
| assets/svg/l08-graph.svg | GENERATED (was missing) | captioned: computational graph, fan-out 4+3=7, residual + node |
| assets/svg/l08-rank1.svg | pre-existing | captioned |
| assets/plate-l08-checkpointing.webp | GENERATED (was missing) | captioned: 12x100MB=1,200MB -> 300MB + 1 extra forward |
| assets/plate-l08-allreduce.webp | GENERATED (was missing) | captioned: (1.0,2.0,3.0,4.0) -> 2.5 everywhere |
| assets/plate-l08-double-backward.webp | GENERATED (was missing) | captioned: H*v, MAML, penalties; double the tape |
| assets/plate-l08-jvp.webp | GENERATED (was missing) | captioned: 3 passes vs 1,000,000 passes |

## L09 — K-Means and GMM (l09-kmeans-gmm.md)

| Figure referenced | Exists / generated | Caption status |
|---|---|---|
| assets/plate-l09-round-audit.webp | pre-existing | captioned |
| assets/plate-l09-seed-lottery.webp | pre-existing | captioned |
| assets/plate-l09-kmedoids.webp | GENERATED (was missing) | captioned: {1,2,3,100}, k-means 7204 vs medoid 2 cost 100 |
| assets/svg/l09-elbow.svg | pre-existing | captioned |
| assets/plate-l09-voronoi.webp | GENERATED (was missing) | captioned: centers 2 and 11, border at 6.5 |
| assets/plate-l09-silhouette.webp | GENERATED (was missing) | captioned: a=1.5, b=10, silhouette 0.85 |
| assets/svg/l09-gmm.svg | pre-existing | captioned |
| assets/plate-l09-responsibility.webp | pre-existing | captioned |
| assets/plate-l09-crescents.webp | pre-existing | captioned |
| assets/plate-l09-zoo.webp | GENERATED (was missing) | captioned: k-means / DBSCAN / spectral / hierarchical |

## L10 — EM and PCA (l10-em-pca.md)

| Figure referenced | Exists / generated | Caption status |
|---|---|---|
| assets/svg/l10-elbo.svg | pre-existing | captioned |
| assets/plate-l10-elbo-tight.webp | pre-existing | captioned |
| assets/plate-l10-jensen.webp | GENERATED (was missing) | captioned: log(5)=1.609 vs 1.099 |
| assets/plate-l10-covariance.webp | pre-existing | captioned |
| assets/plate-l10-degenerate.webp | pre-existing | captioned |
| assets/plate-l10-map-em.webp | GENERATED (was missing; 1st render rejected for gibberish footer fragment, regenerated clean) | captioned: (1.167+1.0)/(3+2)=0.433 vs 0.389 |
| assets/svg/l10-pca.svg | pre-existing | captioned |
| assets/plate-l10-suv-pca.webp | pre-existing | captioned |
| assets/plate-l10-eckart-young.webp | REGENERATED with corrected numbers (old s_1 = 4.472 removed; now s_1 = 3.162, s_2 = 0, eigenvalues (2.5, 0)) | captioned: drop s_2 = 0 error 0; SUV drop 0.0365 error 2.1% |
| assets/plate-l10-randomized-svd.webp | GENERATED (was missing) | captioned: Y=XΩ, B=Q^T X, O(n d k) |

## L11 — Diffusion Models (l11-diffusion-models.md)

| Figure referenced | Exists / generated | Caption status |
|---|---|---|
| assets/svg/l11-diffusion.svg | pre-existing | captioned |
| assets/plate-l11-pixel-audit.webp | pre-existing | captioned |
| assets/plate-l11-closed-form.webp | pre-existing | captioned |
| assets/plate-l11-schedule.webp | GENERATED (was missing) | captioned: t=500 linear 0.28, cosine 0.71 |
| assets/plate-l11-sampling-bill.webp | pre-existing | captioned |
| assets/plate-l11-score.webp | pre-existing | captioned |
| assets/plate-l11-guidance.webp | GENERATED (was missing) | captioned: uncond (0.1,0.2), cond (0.3,0.1), w=7.5 -> (1.6,-0.55) |
| assets/svg/l11-unet.svg | GENERATED (was missing) | captioned: encoder downsamples to gist, decoder upsamples, skips carry detail, time+text per block |

## L12 — Foundation Models (l12-foundation-models.md)

| Figure referenced | Exists / generated | Caption status |
|---|---|---|
| assets/plate-l12-58-percent.webp | pre-existing | captioned |
| assets/svg/l12-fm.svg | pre-existing | captioned |
| assets/plate-l12-contamination.webp | pre-existing | captioned |
| assets/plate-l12-objectives.webp | GENERATED (was missing) | captioned: next-token 3.91, masked 1.20, contrastive pull/push |
| assets/plate-l12-probe-ladder.webp | pre-existing | captioned |
| assets/plate-l12-lora-merge.webp | pre-existing | captioned |
| assets/plate-l12-scaling.webp | GENERATED (was missing) | captioned: Chinchilla 20 tokens/param, 70B/1.4T beats 280B/300B |
| assets/plate-l12-incontext.webp | GENERATED (was missing) | captioned: two examples, no weight updates -> positive |
| assets/plate-l12-bill.webp | GENERATED (was missing) | captioned: 6*N*D = 5.9e23 FLOPs, 2000 A100s, 34 days, $3.3M |

## Notes for the figure auditor
- All 34 new/rebuilt figures (32 plates + 2 SVGs) are on disk; 69/69 total references across the six lessons resolve.
- Numbers on every plate are the lesson text's own worked numbers (F6), verified by viewing each plate.
- SVG note: the course's 40 existing SVGs use Georgia serif; the 2 new SVGs match that established course look for visual consistency (spec prefers sans, but a mixed serif/sans course would look broken). Flagged as a pre-existing course-wide deviation, not introduced here.
- Did not run build.py, did not touch l01-l06 / l13-l17 / crash-course / cheatsheet / index files. Temp sidecar JSONs from the media pipeline were deleted; no leftover files in assets/.
