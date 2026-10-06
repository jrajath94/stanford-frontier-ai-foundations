# Figure Enforcer Audit — cs336 l05: GPUs

Date: 2026-10-06. Enforcer: FIGURE ENFORCER (subagent). Content gates G1–G9 PASS — prose untouched except figure captions.

## Fixes applied before audit

- Font stack fixed across all l05 SVGs.

## Page audit: unit -> figure (zero blank cells)

| # | Unit (heading / equation / code / arch noun) | Unit id | Figure id | Medium | Status |
|---|---|---|---|---|---|
| 1 | CPU vs GPU (108 SMs, SIMT) | u-cpu-gpu | f-cpu-vs-gpu | SVG | kept, caption fixed |
| 2 | Memory hierarchy (latencies) | u-mem-hier | f-mem-hierarchy | SVG | kept, caption fixed |
| 3 | SIMT divergence (warp branches) | u-divergence | f-simt-divergence | SVG | kept, caption fixed |
| 4 | Fusion (sin^2+cos^2, 5 kernels -> 1) | u-fusion | f-fusion | SVG | kept, caption fixed |
| 5 | Coalescing (burst rule) | u-coalesce | f-coalesce | SVG | kept, caption fixed |
| 6 | Tiling (N/T reads) | u-tiling | f-tiling | SVG | kept, caption fixed |
| 7 | Wave quantization (120 tiles, 96 idle) | u-wave | f-wave-quant | SVG | kept, caption fixed |
| 8 | The five tricks, one move | u-tricks | f-chap-tricks | SVG chapter plate | ADDED |
| 9 | FlashAttention (never materialize n x n) | u-flashattn | f-flashattn | SVG | kept, caption fixed |
| 10 | Online softmax worked (tiles [3,1],[5,2]) | u-online-softmax | f-flashattn | SVG | kept |
| 11 | FA1/FA2/FA3 generations | u-fa-versions | t-fa-versions (new table) | markdown table | REPLACED (webp) |
| 12 | FlashAttention chapter economics | u-fa-econ | f-chap-flashattn | SVG chapter plate | ADDED |
| 13 | TPU systolic array (different dataflow) | u-tpu | f-chap-tricks | SVG chapter plate | kept |
| 14 | Chip census, October 2026 | u-chip-census | t-chip-census (inline table) | markdown table | REPLACED (image-table webp) |
| 15 | Debug a 10%-of-peak kernel (ordered) | u-debug | f-chap-roofline-ref | — | kept (prose QA) |
| 16 | Recap | u-recap | — | text list | kept |

## Figures kept / fixed / added, by medium

- SVG lesson plates (kept, caption fixed): l05-cpu-vs-gpu.svg, l05-mem-hierarchy.svg, l05-simt-divergence.svg, l05-fusion.svg, l05-coalesce.svg, l05-tiling.svg, l05-wave-quant.svg, l05-flashattn.svg.
- SVG chapter plates (new): l05-chap-tricks.svg, l05-chap-flashattn.svg.
- webp deleted: media-generation-cs336-l05-fa-versions-0-ed34b967-3807-41e4-9ff0-0a9b6d36b53b.webp (+ sidecar) — replaced by markdown table (Version | the one idea | hardware payoff: FA1 IO-aware; FA2 sequence-dim ~2x; FA3 Hopper async 740 TFLOPS = 75% of 989 peak).
- webp deleted: media-generation-cs336-l05-chip-census-0-af52998b-d316-4495-bfab-688f33b155f1.webp (+ sidecar) — image-table replaced by the existing inline census table (H100 80GB/3.35/989; H200 141GB/4.8/989; B200 192GB/8/FP4; MI300X 192GB/5.3; TPU v5p with [uncertain] marks kept).
- Tables: FA versions table added; chip census table is the figure.

## Prose problems noticed, NOT touched

- Chip census prose table carries [uncertain] marks on MI300X dense peak and TPU v5p bandwidth/peak: kept verbatim, honest per MASTER_BRIEF.
- FA3 "no full Blackwell port as of early 2026" is prose; the new table does not repeat it (no duplication).
