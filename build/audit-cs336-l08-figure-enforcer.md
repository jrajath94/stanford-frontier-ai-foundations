# FIGURE ENFORCER audit — cs336/l08-4d-parallelism
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (Shell + Source added, F5) and added inline
# figure tables (F1/F2).

## Figure inventory (12 plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f01 | assets/l08-zero-ladder.svg | SVG lesson plate (kept) | Shard more, pay nothing: stages 1-2 at 2P, stage 3 at 3P |
| f02 | assets/l08-zero-stages-cost.svg | SVG lesson plate (NEW, replaces webp) | 7B on 8 A100s: DDP 112GB dead; ZeRO-3 fits 50B at 3P |
| f03 | assets/l08-bubble.svg | SVG lesson plate (kept) | Naive: one GPU at a time. Micro-batches fill the pipe. B/W split kills the drain |
| f04 | assets/l08-zero-bubble.svg | SVG lesson plate (NEW, replaces webp) | B is the critical path, W is a leaf; B chains, W fills the gaps |
| f05 | assets/l08-tp-cuts.svg | SVG lesson plate (kept) | Column cuts up, row cuts down; f/g duality flips in backward |
| f06 | assets/l08-topology-philosophy.svg | SVG lesson plate (kept) | TPU mesh vs GPU tree; MoE pushed TPUs toward trees |
| f07 | assets/l08-activation-memory.svg | SVG lesson plate (kept, s^2 fixed) | 34sbh + 5as^2/h; TP divides the big terms, sequence parallel the rest |
| f08 | assets/l08-activation-floor.svg | SVG lesson plate (NEW, replaces webp) | 34 = 24 + 10; TP divides the 24, sequence parallel the 10; floor 34sbh/t |
| f09 | assets/l08-ep-vs-tp.svg | SVG lesson plate (kept) | Route tokens to whole experts; decouple TP for attention |
| f10 | assets/l08-4d-prescription.svg | SVG lesson plate (kept) | Fit with TP/EP in-node, PP/FSDP across, DP for the rest |
| f11 | assets/l08-training-runs.svg | SVG lesson plate (kept) | OLMo to Qwen 3: same recipe, different numbers; TP stays at 8 |
| f12 | assets/l08-recipe-cards.svg | SVG lesson plate (NEW, replaces webp) | DeepSeek-V3 verified (report 3.2); Llama 3 405B 4D, degrees [uncertain] |
| c1 | assets/l08-chap-zero.svg | SVG chapter plate (NEW) | DDP 112GB dead; ZeRO-3 holds one slice; 50B where 7B did not fit |
| c2 | assets/l08-chap-bubble.svg | SVG chapter plate (NEW) | 75% idle to near-zero; every schedule trades memory for utilization |
| c3 | assets/l08-chap-activation.svg | SVG chapter plate (NEW) | 36.5GB to 4.6GB; no single tool moves all three parts |
| c4 | assets/l08-chap-4d.svg | SVG chapter plate (NEW) | One cut fails; the blend ships; 148 failures in 54 days |
| f-tab-flat | inline table (NEW) | table | Flat parameter: 440MB vector, 55MB shards, 385MB per all-gather |
| f-tab-zb | inline table (NEW) | table | ZB-H1/H2/V vs DualPipe: bubble, memory, schedule, co-design |
| f-tab-cp | inline table (NEW) | table | Context parallel at s=128K: 67MB slices, 469MB moved per device |
| f-tab-failures | inline table (NEW) | table | 148 failures in 54 days: ~3/day; T ~ sqrt(2 x 1440 x C / F) |
| f-tab-mapping | inline table (kept) | table | Pain, tool, how |

## Fixes applied to existing figures
- Font stack reordered to spec (Anthropic Sans first) in all l08 SVGs.
- l08-activation-memory.svg: "5 x a x s / h" and "5 as/h" corrected to "5 x a x s^2 / h" and "5 a s^2/h" (F6: the prose says quadratic 5as^2/h; the plate had dropped the square).
- Captions on all 8 kept plates: added "Shell N." and "Source:" per F5. Training-runs caption carries the [uncertain] mark for exact degrees.
- Four generated webp stills DELETED (zero-stages-cost, zero-bubble, activation-floor, recipe-cards) and redrawn as flat SVG lesson plates per the medium ladder (F2). No generated stills remain in l08.

## Medium-ladder justification for new figures
- f02 zero stages: table no (the claim is the ladder's fall plus the 3P jump, a state change across stages); ASCII no (four stages exceed a clean trace); SVG passes first.
- f04 zero-bubble: table no (B vs W is a schedule move, not a value comparison); SVG passes first.
- f08 activation floor: table no (the claim is the decomposition into divided parts, positional); equation alone cannot show which tool divides which part; SVG passes first.
- f12 recipe cards: the claim is two verified architectural choices; the [uncertain] marks must stay visible on the recipes; SVG passes first.
- f-tab-flat / f-tab-zb / f-tab-cp / f-tab-failures: all comparisons of values: table is the first medium that passes.
- c1-c4: chapter plates mandated by the spec.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | DDP replicates everything; 16 bytes/param, ~5 copies | memory uncounted | 7B needs 112GB per GPU, A100 holds 80 | c1 | chapter plate | lecture |
| u-h-7b-fail | 7B at 16 bytes = 112GB > 80GB: baseline could not fit 7B | fit assumed | the failure worked | c1 | chapter plate | lecture |
| u-h-dp-caps | DDP capped by batch and critical batch size | batch unbounded | 8 examples feed at most 8 GPUs | c1 | chapter plate | lecture |
| u-h-keyq | What if no GPU held the whole model? | replication assumed | the ZeRO ladder | f01, c1 | SVG | lecture |
| u-h-zero1 | ZeRO-1: shard optimizer state, 2P, free | cost assumed to rise | same comm, memory / N | f01, f02 | SVG | lecture |
| u-h-zero2 | ZeRO-2: shard gradients in the backward sweep | full grads materialized | never materialize the full gradient | f01, f02 | SVG | lecture |
| u-h-zero3 | ZeRO-3/FSDP: shard params, 2 all-gathers + 1 reduce-scatter | params replicated | 3P; sweep-and-free; overlap hides it | f01, f02, c1 | SVG | lecture |
| u-h-overlap | Overlap: all-gather layer n+1 during layer n's compute | comm on the critical path | hides when compute dominates | f02, c1 | SVG | lecture |
| u-h-payoff | Stage 3 fits 50B where baseline could not fit 7B | payoff unworked | the headline number | f02, c1 | SVG | lecture |
| u-h-stagecosts | 7B on 8 GPUs: 112/56/40GB; 2P, 2P, 2P, 3P | stages uncompared | the ladder's fall and the 3P jump | f02 | SVG | lecture |
| u-h-flat | Flat parameter: 440MB vector, 55MB shards, 385MB moved | sharding unnamed | one all-gather per unit | f-tab-flat | table | original toy |
| u-h-unit | One transformer layer per unit: bandwidth regime, low peak | unit size arbitrary | the standard answer | f-tab-flat | table | lecture |
| u-h-fsdp2 | FSDP2/DTensor: per-parameter sharding, modern default | flat param assumed | no padding waste, composes with TP | c1 | chapter plate | lecture |
| u-h-usedwhere-fsdp | FSDP2 default; Llama 3 FSDP lineage; OLMo proof | framework unnamed | reach for FSDP2 first | c1 | chapter plate | lecture, Oct 2026 |
| u-h-pipe | Cut by layers: naive 75% idle tax | layers as one unit | one GPU works at a time | f03, c2 | SVG | lecture |
| u-h-micro | Micro-batches shrink the bubble as 1/m | tax fixed | 4 stages 8 micros: bubble falls | f03, c2 | SVG | lecture |
| u-h-pipe-comm | Point-to-point bsh: lives on the slowest links | comm assumed big | far less than param matrices | c2 | chapter plate | lecture |
| u-h-zerobubble | Zero-bubble: B critical, W leaf | backward as one block | B first, W in the gaps | f04, c2 | SVG | lecture |
| u-h-bw-why | W needs activations: holding W raises memory | split free | the memory price | f04, c2 | SVG | lecture |
| u-h-zb1 | ZB-H1: backward bubble ~0, forward fill remains | one schedule | baseline zero-bubble | f-tab-zb, c2 | table | Qi et al. |
| u-h-zb2 | ZB-H2: less bubble, more memory | one dial assumed | bubble vs memory, one dial | f-tab-zb, c2 | table | Qi et al. |
| u-h-zbv | ZB-V: V-shaped, near zero at 1F1B-like memory | complexity unpriced | hardest schedule | f-tab-zb, c2 | table | Qi et al. |
| u-h-zb-worked | p=8, m=32: 1F1B 22%, ZB-H1 halves, ZB-V near zero | family uncompared | the comparison worked | f-tab-zb | table | original toy |
| u-h-dualpipe | DualPipe: waves from both ends, 4 chunks, MoE co-design | one schedule | comm hides behind compute both ways | f-tab-zb, c2 | table | DeepSeek-V3 |
| u-h-tp | TP: column-up, row-down, f/g duality | width as one unit | all-reduce per matmul, TP-8 max | f05 | SVG | lecture |
| u-h-tp-limit | TP stops at 8 on GPUs; TPUs push further on the mesh | limit unexplained | the fast domain decides | f05, f06 | SVG | lecture |
| u-h-topoph | TPU mesh vs GPU tree; MoE pushed TPUs to trees | network fixed | workloads define the wire | f06 | SVG | lecture |
| u-eq-act | 34sbh + 5as^2/h; recompute deletes the quadratic | memory uncounted | the parts list | f07, c3 | SVG | lecture |
| u-h-34 | The 34: attention ~10, MLP ~24, by saved tensors | 34 as magic number | walk the parts | f08, c3 | SVG | lecture |
| u-h-tp24 | TP divides the 24; the 10 is stubborn (elementwise) | TP divides all | layernorms/dropout/residuals remain | f07, f08, c3 | SVG | lecture |
| u-h-seqpar | Sequence parallel divides the 10 along s | 10sbh stuck | rides TP's existing sync | f07, f08, c3 | SVG | lecture |
| u-h-floor | Floor 34sbh/t: 4.6GB at s=8192,b=8,h=8192,t=8 | floor unworked | fits with room to spare | f08, c3 | SVG | original toy |
| u-h-ep | EP: route tokens to whole experts, keep matmuls big | slice everything | all-to-all dispatch, latency is everything | f09 | SVG | lecture |
| u-h-ep-decouple | Decouple: high TP for attention, high EP for MoE layers | one degree assumed | EP covers MLPs only | f09 | SVG | lecture |
| u-h-cp | Context parallel: circulate K/V around a ring, exact via online softmax | long seq undivided | 67MB slices, 469MB moved per device | f-tab-cp | table | Liu et al. |
| u-h-cp-zigzag | Zigzag schedule: respect causality or waste bandwidth | rotation naive | masked blocks are pure waste | f-tab-cp | table | lecture |
| u-h-4d | 4D prescription: TP/EP in-node, PP/FSDP across, DP for rest | one strategy | minimize model parallel, maximize data | f10, c4 | SVG | lecture |
| u-h-accum | Gradient accumulation restores small-batch utilization | small batch starves | K micro-steps, one sync | c4 | chapter plate | lecture |
| u-h-roofline | Big batch: FSDP compute-bound; shrinking batch: add TP | dimension arbitrary | the roofline decides | c4 | chapter plate | lecture |
| u-h-recompute | Recomputation buys batch size buys utilization | compute sacred | counterintuitive and true | c4 | chapter plate | lecture |
| u-h-study | NVIDIA/Stanford study: DP maxed, TP to 8, PP grows | pattern unshown | utilization flat throughout | c4 | chapter plate | study |
| u-h-wild | OLMo, V1/V3, Yi, Llama3, Gemma2, Mixtral, Qwen3 | one recipe | same recipe, different numbers | f11 | SVG | lecture |
| u-h-failures | 148 failures in 54 days: ~3/day; T ~ sqrt(2x1440xC/F) | failures uncounted | reliability is half the job | f-tab-failures, c4 | table | lecture |
| u-h-v3-verified | DeepSeek-V3: EP64, PP16 DualPipe, ZeRO-1, no training TP | recipe [uncertain] | verified from report 3.2 | f12 | SVG | DeepSeek-V3 report |
| u-h-llama-verified | Llama 3 405B: 4D, exact degrees unpublished [uncertain] | degrees assumed | reported, not published | f12 | SVG | Llama 3 paper |
| u-tab-mapping | Pain to tool mapping | scattered claims | one-screen mapping | f-tab-mapping | table | original synthesis |
| u-h-price | Red in every row; prescription is a starting point | strategy proven | honest price stated | c1, c2, c3, c4 | chapter plates | original synthesis |
| u-h-coverage | Coverage map | claims scattered | traceability table | n/a - coverage unit | table | original |
| u-h-recap | 8-step recap | lesson as sequence | one-screen consolidation | c1, c2, c3, c4 | chapter plates | original synthesis |
| u-h-qa | Interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: 2 nocookie embeds + 5 links | n/a (media law unit) | video + links | n/a - media unit | video/links | papers |
| u-h-official | Official sources and further reading | n/a (sourcing unit) | sources + caveats | n/a - sourcing unit | links | lecture/docs |
| u-h-conn | Connections to other courses | n/a (sourcing unit) | cross-links | n/a - sourcing unit | prose | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. The lesson uses 16 bytes/param (Adam, L08) while L07 used 12 bytes/param (mixed-precision Adam, L02 accounting). Both are in the lessons as stated; the plates repeat each lesson's own numbers. A cross-lesson note reconciling 12 vs 16 would help the reader but is a content call.
2. l08-training-runs.svg's "TP8 PP16 DP128" for Llama 3 is presented without the [uncertain] mark inside the plate; the md caption now carries it. The figure auditor may want the mark inside the SVG.
3. "148 GPU failures" is from the lecture; the plate and table repeat it. Fine.
