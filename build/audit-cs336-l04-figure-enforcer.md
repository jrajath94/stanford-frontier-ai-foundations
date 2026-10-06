# Figure Enforcer Audit — cs336 l04: Linear Attention and MoE

Date: 2026-10-06. Enforcer: FIGURE ENFORCER (subagent). Content gates G1–G9 PASS — prose untouched except figure captions.

## Fixes applied before audit

- l04-moe.svg FIXED: router box (x=620,y=140) overlapped the "AFTER: four experts" title. Moved to x=500,y=336 inside the panel; "4P params, ~P FLOPs" moved to x=650,y=366.
- l04-load-balance.svg FIXED: footer lacked the Shell line. Added "Shell 4: the balance symbol."
- Font stack fixed across all l04 SVGs.

## Page audit: unit -> figure (zero blank cells)

| # | Unit (heading / equation / code / arch noun) | Unit id | Figure id | Medium | Status |
|---|---|---|---|---|---|
| 1 | The one idea: move the parentheses | u-assoc | f-associativity | SVG | kept, caption fixed |
| 2 | The kernel view (softmax as dot product) | u-kernel-view | f-kernel-view | webp | kept, caption fixed |
| 3 | RNN duality (causal-dense vs recurrent) | u-rnn-duality | f-rnn-duality | SVG | kept, caption fixed |
| 4 | State size in numbers (34MB vs 4GB) | u-state-size | f-chap-linear | SVG chapter plate | ADDED |
| 5 | Linear cost: n x d^2, wins when n>d | u-linear-cost | f-chap-linear | SVG chapter plate | ADDED |
| 6 | Mamba-2 gate (S_t = gamma S_{t-1} + k v^T) | u-mamba-gate | f-mamba-gate | SVG | kept, caption fixed |
| 7 | Gated delta net (projector) | u-deltanet | f-deltanet | SVG | kept, caption fixed |
| 8 | DSA bolt-on indexer | u-dsa | f-dsa | SVG | kept, caption fixed |
| 9 | MoE: 4P params, ~P FLOPs | u-moe | f-moe | SVG | FIXED (overlap), caption fixed |
| 10 | Sparsity arithmetic (806M/25M, 32x) | u-sparsity | f-chap-moe | SVG chapter plate | ADDED |
| 11 | TopK router (matmul+softmax+top-k) | u-topk | f-topk-router | SVG | kept, caption fixed |
| 12 | The routing zoo (token vs expert choice) | u-routing-zoo | f-routing-zoo | webp | kept, caption fixed |
| 13 | Load balancing (F x P loss) | u-load-balance | f-load-balance | SVG | FIXED (shell line), caption fixed |
| 14 | MLA attention upgrade | u-mla | f-chap-linear | SVG chapter plate | kept |
| 15 | MTP training upgrade | u-mtp | none (ladder: prose suffices) | — | noted |
| 16 | What is used where (production table) | u-used-where | t-used-where (inline table) | markdown table | REPLACED (image-table webp) |
| 17 | Mapping back: what each idea fixes | u-mapping | t-mapping (inline table) | markdown table | kept |
| 18 | Recap | u-recap | — | text list | kept |

## Figures kept / fixed / added, by medium

- SVG lesson plates (kept, caption fixed): l04-associativity.svg, l04-rnn-duality.svg, l04-mamba-gate.svg, l04-deltanet.svg, l04-dsa.svg, l04-topk-router.svg, l04-load-balance.svg.
- SVG lesson plates (fixed): l04-moe.svg (router overlap), l04-load-balance.svg (missing Shell line).
- SVG chapter plates (new): l04-chap-linear.svg, l04-chap-moe.svg.
- webp kept (caption fixed): media-generation-cs336-l04-kernel-view-0-07d8e057-fa20-40e5-8b60-90976dd3caad.webp, media-generation-cs336-l04-routing-zoo-0-f8431a92-aa78-4928-a9db-9c083adb3750.webp.
- webp deleted: media-generation-cs336-l04-used-where-0-479e8f2f-323a-41da-9bea-edcca99fe22d.webp (+ sidecar) — image-table replaced by the existing inline production table (5 rows: DeepSeek-V3 256 top-8, Kimi K2 384 top-8, Qwen3 235B 128 top-8 no-shared, Nemotron-3 Mamba-2 hybrid).
- Tables: used-where table is the figure; mapping-back table kept.

## Prose problems noticed, NOT touched

- l04-moe.svg footer claims "(Switch loss, OlMoE 2x)" — needs prose grounding check by the figure auditor (F6).
- l04-routing-zoo webp's expert-choice panel uses arrow glyphs, not the router chip symbol; flat and compliant, kept.
