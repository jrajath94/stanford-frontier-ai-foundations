# Figure Enforcer Audit — cs336 l02: Resource Accounting

Date: 2026-10-06. Enforcer: FIGURE ENFORCER (subagent). Content gates G1–G9 PASS — prose untouched except figure captions.

## Page audit: unit -> figure (zero blank cells)

| # | Unit (heading / equation / code / arch noun) | Unit id | Figure id | Medium | Status |
|---|---|---|---|---|---|
| 1 | FLOPs vs FLOP/s (H100 1979/989) | u-flops | f-flops-vs-flops | SVG | kept, caption fixed |
| 2 | Precision ladder (fp32/fp16/bf16/fp8/fp4) | u-precision | f-precision | SVG | kept, caption fixed |
| 3 | Memory movement: ReLU 1M bf16 worked | u-mem-move | f-mem-move | SVG | kept, caption fixed |
| 4 | Bytes per parameter: AdamW 2+2+4+4=12 | u-bytes-param | f-bytes-per-param | SVG | kept, caption fixed |
| 5 | Memory budget: 640/12 = 53B | u-budget | f-chap-memory | SVG chapter plate | ADDED |
| 6 | Roofline, knee at 295 | u-roofline | f-roofline | SVG | kept, caption fixed |
| 7 | Intensity table (ReLU 0.25, GELU 5, matmul n/3) | u-intensity | f-roofline | SVG | kept |
| 8 | Backward is two matmuls (einsum names) | u-backward | f-backward | SVG | kept, caption fixed |
| 9 | Attention breaks 6ND (crossover N=6 d_model) | u-attn-breaks | f-attention-breaks | webp | kept, caption fixed |
| 10 | 6ND on the three public runs | u-6nd-runs | t-runs (inline table) | markdown table | REPLACED (image-table webp) |
| 11 | 6ND priced: GPT-3 3.15e23, 8.8 days | u-6nd-price | f-chap-6nd | SVG chapter plate | ADDED |
| 12 | MoE: active params are the N (18x trap) | u-moe-n | f-chap-6nd | SVG chapter plate | ADDED |
| 13 | Gradient accumulation, worked (8 microbatches) | u-grad-accum | f-chap-memory | SVG chapter plate | kept |
| 14 | Activation checkpointing, sqrt(L) | u-checkpoint | f-checkpoint | SVG | kept, caption fixed |
| 15 | Recap | u-recap | — | text list | kept |

## Figures kept / fixed / added, by medium

- SVG lesson plates (kept, caption fixed): l02-flops-vs-flops.svg, l02-precision.svg, l02-mem-move.svg, l02-bytes-per-param.svg, l02-roofline.svg, l02-backward.svg, l02-checkpoint.svg.
- SVG chapter plates (new): l02-chap-6nd.svg, l02-chap-memory.svg, l02-chap-roofline.svg.
- webp kept (caption fixed): media-generation-cs336-l02-attention-breaks-6nd-0-fb070730-ab61-4432-9f79-36f3629b14dc.webp.
- webp deleted: media-generation-cs336-l02-real-runs-0-86d4b3c5-b8f9-4452-8f12-24d1aaf45f7c.webp (+ sidecar JSON) — image-table replaced by the existing inline "6ND on the three public runs" markdown table.
- Tables: real-runs table is the figure (5 cols, 3 rows + MoE note).
- mermaid: none in l02. ASCII: none. Equations: 6ND, bytes, roofline inline.

## Mechanical fixes

- None structural. All 7 SVGs already had in-SVG shell+source footers; md captions rewritten to match ("Shell N. <claim>. Source: <source>.").

## Prose problems noticed, NOT touched

- l02-roofline.svg shows matmul intensity ~340 vs prose "n/3 ~ 340": consistent. No action.
- The real-runs inline table carries a Source column the deleted webp lacked; the webp's "MoE sparsity is a multiplier" claim is preserved in prose.
