# Figure Enforcer Audit — cs336 l06: Triton Kernels

Date: 2026-10-06. Enforcer: FIGURE ENFORCER (subagent). Content gates G1–G9 PASS — prose untouched except figure captions.

## Fixes applied before audit

- Font stack fixed across all l06 SVGs.

## Page audit: unit -> figure (zero blank cells)

| # | Unit (heading / equation / code / arch noun) | Unit id | Figure id | Medium | Status |
|---|---|---|---|---|---|
| 1 | Grid of blocks (programming model) | u-grid | f-grid-blocks | SVG | kept, caption fixed |
| 2 | Occupancy (128x160=3 blocks, 18%) | u-occupancy | f-occupancy | SVG | kept, caption fixed |
| 3 | Bank conflicts (32 banks, swizzle) | u-bank | f-bank-conflict | SVG | kept, caption fixed |
| 4 | Benchmark rules (measure loop) | u-bench | f-bench-rules | SVG | kept, caption fixed |
| 5 | GeLU race (naive vs fused) | u-gelu-race | f-gelu-race | SVG | kept, caption fixed |
| 6 | Triton kernel shape (pid/offsets/mask) | u-triton-shape | f-triton-gelu | SVG | kept, caption fixed |
| 7 | Softmax blocks (row fits / tiles) | u-softmax | f-softmax-block | SVG | kept, caption fixed |
| 8 | Tiled matmul (C tile per block) | u-matmul | f-matmul-tiling | SVG | kept, caption fixed |
| 9 | Think in blocks, not threads | u-blocks | f-chap-kernel | SVG chapter plate | ADDED |
| 10 | Tiling: reuse is the whole game | u-tiling | f-chap-tiling | SVG chapter plate | ADDED |
| 11 | Compiler vs Triton vs lower (decision) | u-decision | f-mer-decision | mermaid (new) | REPLACED (flowchart webp) |
| 12 | Kernel stack in production | u-stack | t-stack (inline table) | markdown table | REPLACED (image webp) |
| 13 | The stack: pay only where it pays | u-stack-econ | f-chap-stack | SVG chapter plate | ADDED |
| 14 | Mapping back: what each idea fixes | u-mapping | t-mapping (inline table) | markdown table | kept |
| 15 | Recap | u-recap | — | text list | kept |

## Figures kept / fixed / added, by medium

- SVG lesson plates (kept, caption fixed): l06-grid-blocks.svg, l06-occupancy.svg, l06-bank-conflict.svg, l06-bench-rules.svg, l06-gelu-race.svg, l06-triton-gelu.svg, l06-softmax-block.svg, l06-matmul-tiling.svg.
- SVG chapter plates (new): l06-chap-kernel.svg, l06-chap-tiling.svg, l06-chap-stack.svg.
- webp deleted: media-generation-cs336-l06-compile-vs-triton-0-8f645e34-8a8f-4fa3-bd75-ad1f39a82c62.webp (+ sidecar) — decision flowchart replaced by inline mermaid (5 nodes, compliant).
- webp deleted: media-generation-cs336-l06-kernel-stack-0-e06a6f3c-7e3b-4473-9730-13390b88be1b.webp (+ sidecar) — lowering-stack order replaced by pointer to the existing inline kernel-stack table (7 layers: Inductor, FA2/3, FlashInfer, paged-attention, Liger-Kernel, FlexAttention, ThunderKittens/CUTLASS).
- mermaid: decision flowchart added (compiler -> Triton -> ThunderKittens/CUTLASS).
- Tables: kernel-stack table is the figure.

## Prose problems noticed, NOT touched

- l06-matmul-tiling.svg footer says "Shell 5: tiling returns in FlashAttention" — cross-lesson reuse link to l05, correct per spec (Shell 5).
- The mermaid decision node "Need cross-block or hardware control?" compresses the prose's three hand-write cases into two branches; the prose cases remain the authority.
