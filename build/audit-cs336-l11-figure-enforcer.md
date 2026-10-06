# FIGURE ENFORCER audit — cs336/l11-scaling-advanced
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (Shell + Source added, F5) and added inline
# figure equations/tables (F1/F2).

## Figure inventory (12 plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f01 | assets/l11-lr-batch-laws.svg | SVG lesson plate (kept) | Batch size follows data only; LR falls with size, rises with data |
| f02 | assets/l11-wsd.svg | SVG lesson plate (kept) | Trapezoid: warmup, long stable phase, rapid decay |
| f03 | assets/l11-wsd-reuse.svg | SVG lesson plate (NEW, replaces webp) | Cosine: restart at 100%. WSD: rewind and re-decay at 10% |
| f04 | assets/l11-optimizer-caution.svg | SVG lesson plate (kept) | Two axes: compute and Chinchilla ratio. Marin: blowup |
| f05 | assets/l11-mup.svg | SVG lesson plate (kept) | The muP checklist; every size's minimum at 1e-2 |
| f06 | assets/l11-two-philosophies.svg | SVG lesson plate (kept) | Stabilize with muP, or fit with scaling laws; both ship |
| f07 | assets/l11-stabilize-vs-fit.svg | SVG lesson plate (NEW, replaces webp) | MiniCPM/Kimi K2 stabilize; DeepSeek/Qwen/StepFun fit; both ship |
| f08 | assets/l11-mup-derivation.svg | SVG lesson plate (kept) | A1/A2 invariants; out come the init and LR rules |
| f09 | assets/l11-mup-adam-rule.svg | SVG lesson plate (NEW, replaces webp) | Layer A fan-in 4096: 1/4096. Layer B fan-in 512: 8x larger |
| f10 | assets/l11-muon.svg | SVG lesson plate (kept) | Momentum plus orthogonalization; spectral Adam |
| f11 | assets/l11-muon-vs-adam.svg | SVG lesson plate (NEW, replaces webp) | Adam: coordinates. Muon: spectral directions |
| f12 | assets/l11-moe-scaling.svg | SVG lesson plate (kept) | Sparsity has a scaling law; architectures get bake-offs |
| c1 | assets/l11-chap-wsd.svg | SVG chapter plate (NEW) | Cosine 6 runs to WSD ~5.1; the decay does the annealing |
| c2 | assets/l11-chap-mup.svg | SVG chapter plate (NEW) | Invariants out, per-layer rules in; assumptions break it |
| c3 | assets/l11-chap-muon.svg | SVG chapter plate (NEW) | Orthogonalize the update; superiority unproven |
| f-eq-laws | inline equation (kept) | equation | D_crit, the critical batch limit |
| f-eq-width | inline equation (NEW) | equation | eta x n; double the width, halve the LR |
| f-eq-depth | inline equation (NEW) | equation | L x v; 1/sqrt(L) per branch |
| f-tab-init | inline table (NEW) | table | Xavier, He, muP: variance, derived for, controls, blind spot |
| f-tab-optimizers | inline table (NEW) | table | SGD to SOAP/Shampoo: what each normalizes, cost, scale story |
| f-tab-moebake | inline table (kept) | table | 4B, 11B, 24B, 32B models to papers |
| f-tab-recipes | inline table (kept) | table | Four scaling recipes: what, cost, source |
| f-ascii-eft | inline ascii (kept) | ASCII | EFT vs PFT: all four vs one |

## Fixes applied to existing figures
- Font stack reordered to spec (Anthropic Sans first) in all l11 SVGs.
- Captions on all 8 kept plates: added "Shell N." and "Source:" per F5.
- Four generated webp stills DELETED (wsd-reuse, stabilize-vs-fit, mup-adam-rule, muon-vs-adam) and redrawn as flat SVG lesson plates per the medium ladder (F2). No generated stills remain in l11.

## Medium-ladder justification for new figures
- f03 WSD reuse: table no (the claim is the reuse mechanism: rewind at 10%, not restart at 100%); SVG passes first.
- f07 stabilize vs fit: table no (the kept prose already maps labs to philosophies; the plate is the before/after of the field's two bets); SVG passes first.
- f09 muP Adam rule: table no (the claim is the per-layer difference: same network, two LRs; ASCII no (position matters); SVG passes first.
- f11 Muon vs Adam: table no (the kept f-tab-optimizers now carries the value comparison; the plate is the coordinate vs spectral distinction); SVG passes first.
- f-eq-width / f-eq-depth: prose already derives these in words; the equation is the first medium that pins the numbers.
- f-tab-init / f-tab-optimizers: comparisons of values and ideas; table is the first medium that passes.
- c1-c3: chapter plates mandated by the spec.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-breaks | Three breaks: LR/batch moves, decay lies, Adam updates drift | scheduling assumed | the frontier's three lessons | f01, f04 | SVG | lecture |
| u-h-wsd | Cosine wastes restarts; WSD reuses stable checkpoints | restarts assumed | the trapezoid | f02, f03, c1 | SVG | lecture |
| u-h-wsdlaws | Extrapolate LR/batch with laws; StepFun's rules | tuning assumed | the laws | f01 | SVG | StepFun |
| u-h-decaylies | Partial decays lie; judge only after the decay | interim curves trusted | the decay does the annealing | f02, c1 | SVG | lecture |
| u-h-break2 | Beautiful scaling, then sudden blowup | curves trusted | two axes: compute and ratio | f04 | SVG | lecture |
| u-h-adamdrift | Update-to-weight ratio = eta x n | one LR fits all | the drift | f-eq-width | equation | original |
| u-h-mup | The muP checklist; every minimum at 1e-2 | LR drifts | the reparameterization | f05, c2 | SVG | MiniCPM |
| u-h-phil | Stabilize or fit: both ship models | one philosophy | two bets, no winner yet | f06, f07 | SVG | company reports |
| u-h-deriv | A1/A2; out come the rules | rules asserted | derived from invariants | f08, c2 | SVG | lecture |
| u-h-init | Xavier, He, muP: variance bookkeeping | init assumed | three generations | f-tab-init | table | lecture |
| u-h-depth | 1/sqrt(L): variances add | depth free | the root rule | f-eq-depth | equation | original |
| u-h-wdecay | Weight decay breaks muP; re-derive it | decay copied | scale-sensitive term | c2 | chapter plate | lecture |
| u-h-adamrule | 1/fan-in per layer: A at 1/4096, B 8x larger | one global LR | per-layer LRs | f09, c2 | SVG | lecture |
| u-h-embed | Embeddings at half LR; residuals 1/sqrt(l) | uniform treatment | special rules | f05, c2 | SVG | lecture |
| u-h-muverify | Verify with 2-5% short runs; scale attention and LR together | transfer assumed | the verification protocol | c2 | chapter plate | lecture |
| u-h-muon | Newton-Schultz x5; spectral Adam; Kimi K2 at scale | optimizer assumed | orthogonalize the update | f10, f11, c3 | SVG | lecture |
| u-h-soap | SOAP/Shampoo: curvature; research only at scale | second-order ignored | the honest caution | f-tab-optimizers | table | lecture |
| u-h-moe | Sparsity has a scaling law; architectures get bake-offs | architecture assumed | the bake-off evidence | f12 | SVG | lecture |
| u-h-recipes | Four scaling recipes: what, cost, source | recipes scattered | one-screen map | f-tab-recipes | table | original synthesis |
| u-h-qa | Interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: nocookie embed + 4 links | n/a (media law unit) | video + links | n/a - media unit | video/links | papers |
| u-h-official | Official sources and further reading | n/a (sourcing unit) | sources + caveats | n/a - sourcing unit | links | lecture/book |
| u-h-conn | Connections to other courses | n/a (sourcing unit) | cross-links | n/a - sourcing unit | prose | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. The width-scaling derivation in prose says "update-to-weight ratio = eta x n" while earlier subchapters use eta/sqrt(n) for the update magnitude; the added equation plate follows the prose's convention. The figure auditor should confirm the convention is consistent across the derivation.
2. The prose claims "Muon's 5 Newton-Schultz iterations cost ~1% of a step" in one place and the plate says "5 extra matmuls per step, GPU-friendly"; both are in prose, not figures. Consistent.
3. "Kimi K2: works at scale" appears in plate f11 and chapter plate C3; the prose is careful to say no ablation was published. The plates repeat the caution.
