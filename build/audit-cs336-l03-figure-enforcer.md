# Figure Enforcer Audit — cs336 l03: Architecture

Date: 2026-10-06. Enforcer: FIGURE ENFORCER (subagent). Content gates G1–G9 PASS — prose untouched except figure captions.

## Fixes applied before audit

- l03-gqa.svg REDRAWN: MHA KV chips overflowed the 260px MHA panel and collided with the GQA panel (broken layout). New layout: three 280px panels, prose numbers (16KB/4KB/512B per token per layer; 41GB vs 10GB at 80 layers x 32K).
- Font stack fixed across all l03 SVGs: system-ui stack -> "Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif".

## Page audit: unit -> figure (zero blank cells)

| # | Unit (heading / equation / code / arch noun) | Unit id | Figure id | Medium | Status |
|---|---|---|---|---|---|
| 1 | The block: residual stream, x + F(x) | u-block | f-chap-block | SVG chapter plate | ADDED |
| 2 | Prenorm vs postnorm | u-prenorm | f-prenorm | SVG | kept, caption fixed |
| 3 | RMSNorm (0.17% FLOPs, 25% runtime) | u-rmsnorm | f-rmsnorm | SVG | kept, caption fixed |
| 4 | The activation ladder (ReLU/GELU/SwiGLU) | u-activation | f-activation-ladder | webp | kept, caption fixed |
| 5 | SwiGLU block (gate symbol) | u-glu | f-glu | SVG | kept, caption fixed |
| 6 | RoPE (rotated Q/K) | u-rope | f-rope | SVG | kept, caption fixed |
| 7 | The position zoo (4 answers) | u-position-zoo | f-position-zoo | webp | kept, caption fixed |
| 8 | Forgiving hyperparameters (Kaplan ratios) | u-hparams | f-hparams | SVG | kept, caption fixed |
| 9 | z-loss (penalize (log z)^2) | u-zloss | f-zloss | SVG | kept, caption fixed |
| 10 | QK norm (inputs at scale 1) | u-qknorm | f-qknorm | SVG | kept, caption fixed |
| 11 | Stability: guard both softmaxes | u-stability | f-chap-stability | SVG chapter plate | ADDED |
| 12 | MHA vs GQA vs MQA | u-gqa | f-gqa | SVG | REDRAWN (layout), caption fixed |
| 13 | MLA compresses the KV cache (1.1KB) | u-mla | f-mla | webp | kept, caption fixed |
| 14 | KV cache economics | u-kv | f-chap-kv | SVG chapter plate | ADDED |
| 15 | Sliding windows (Gemma 2 local-global) | u-sliding | f-chap-kv | SVG chapter plate | kept |
| 16 | Architecture census (survey table) | u-census | t-census (inline table) | markdown table | REPLACED (image-table webp) |
| 17 | DeepSeek-V3 full config worked | u-dsv3 | t-census | markdown table | kept |
| 18 | Recap | u-recap | — | text list | kept |

## Figures kept / fixed / added, by medium

- SVG lesson plates (kept, caption fixed): l03-prenorm.svg, l03-rmsnorm.svg, l03-glu.svg, l03-rope.svg, l03-zloss.svg, l03-qknorm.svg, l03-hparams.svg.
- SVG lesson plates (redrawn): l03-gqa.svg (broken panel layout).
- SVG chapter plates (new): l03-chap-block.svg, l03-chap-kv.svg, l03-chap-stability.svg.
- webp kept (caption fixed): media-generation-cs336-l03-activation-ladder-0-9d265600-01ca-42a7-a8de-e4dcba3bb219.webp, media-generation-cs336-l03-mla-0-e97e18ec-2e90-44e2-b614-df31f3910e2f.webp, media-generation-cs336-l03-position-zoo-0-2139d8cb-af73-418e-9c34-4036e0fcd0b4.webp.
- webp deleted: media-generation-cs336-l03-model-census-0-bb36c866-9d51-48b8-ab6e-c5d445e512f3.webp (+ sidecar) — image-table replaced by the existing inline survey table (7 rows incl. GPT-4 unknown row).

## Prose problems noticed, NOT touched

- Census prose table (line ~817) vs deleted webp rows differed: webp said "GQA-8"/"sliding window GQA-8"/"MLA"/"decoupled RoPE"/"MoE SwiGLU"/"local-global GQA"/"GeGLU"; prose table adds Norm column and GPT-4 unknown row. Prose table wins; noted for auditor.
- l03-position-zoo webp's four panels use glyph shapes, not the spec's chip symbols; flat and compliant, kept.
