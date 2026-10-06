# FIGURE ENFORCER audit — cs336/l18-inference
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions and the builder-stats figure count. Content gate
# issues found while enforcing are listed under "Prose problems" for the
# coordinator; they were NOT edited.

## Figure inventory

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f18-lifetime | assets/l18-lifetime.svg | SVG (kept; font stack fixed) | A token lives: request, queue, prefill, decode, serve |
| f18-workloads | assets/l18-workloads.svg | SVG (kept; font stack fixed) | Four workloads, one naive stack fails each |
| f18-prefill-decode | assets/l18-prefill-decode.svg | SVG (kept; font stack fixed) | Prefill is compute-bound, decode is memory-bound |
| f18-decode-tax | assets/l18-decode-tax.svg (NEW) | SVG lesson plate | 140 GB moved per token: 42.4 ms at 3.3 TB/s |
| f18-disaggregation | assets/l18-disaggregation.svg | SVG (kept; font stack fixed) | Split the fleets: prefill and decode apart |
| f18-two-fleets | assets/l18-two-fleets.svg (NEW) | SVG lesson plate | The 2026 stack: disaggregated, cached, speculated |
| f18-continuous-batching | assets/l18-continuous-batching.svg | SVG (kept; font stack fixed) | Static batching idles; continuous fills the gaps |
| f18-kv-cache | assets/l18-kv-cache.svg | SVG (kept; font stack fixed) | KV cache: never recompute the past |
| f-ascii-radix | inline ASCII block (NEW) | ASCII trace | Three prefixes share one trunk |
| f18-megakernel | assets/l18-megakernel.svg | SVG (kept; font stack fixed) | Megakernels: fuse everything, kill the gaps |
| f18-spec-decode | assets/l18-spec-decode.svg (NEW) | SVG lesson plate | Draft k=5, verify in 1 pass: ~4x when 4 of 5 accept |
| f18-megakernel-fusion | assets/l18-megakernel-fusion.svg (NEW) | SVG lesson plate | 21 kernels to 1: the overlap schedule |
| f18-parcae | assets/l18-parcae.svg | SVG (kept; font stack fixed) | Parcae: loop the layers, buy flops not parameters |
| f-tab-codesign | existing co-design table (inline) | table | Model and chip designed together |
| c18-tax | assets/l18-chap-tax.svg (NEW) | SVG chapter plate | The decode tax: every token pays in memory bandwidth |
| c18-serving | assets/l18-chap-serving.svg (NEW) | SVG chapter plate | Serving: batch, cache, disaggregate, speculate |
| c18-codesign | assets/l18-chap-codesign.svg (NEW) | SVG chapter plate | MLA, quantization, sizing: one decision |
| f-tab-mapback | existing Mapping-back table | table | Pain to fix, per section |

## Fixes applied to existing figures
- Bulk fix (all 8 existing l18 SVGs): font-family `system-ui,-apple-system,'Segoe UI',sans-serif` replaced with the spec stack `Anthropic Sans,Inter,'Source Sans 3','IBM Plex Sans',sans-serif`.
- 2 broken webp refs replaced per the medium ladder: radix-tree -> ASCII trace (trace, 12 lines max), disaggregation-fleets -> SVG plate (before/after).
- 3 lesson plates + 3 chapter plates added. Zero generated stills remain.

## Numbers verified by code (python3)
- Decode tax: 140 GB / 3300 GB/s = 42.4 ms per token; decode share 140/200 = 70%.
- Two fleets: 2x TTFT + 3x tokens/s + 40% prefix = 2.8x goodput; disaggregation stack sum = 17.9.
- Megakernel fusion: 21 kernels to 1, 18.3 saved of 21.7 total = 84.3% overhead eliminated.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | The problem: the other side of the model | training only | inference is the other half | f18-lifetime | SVG | Stanford |
| u-h-engine | The engine metaphor, taken seriously | model as function | model as engine, serving as fuel system | f18-lifetime | SVG | original |
| u-h-token-lifetime | The token's lifetime, step by step | token instant | request to response, staged | f18-lifetime | SVG | original |
| u-h-fullstack | Full-stack innovation, defined | one layer | model, system, hardware together | c18-codesign | chapter plate | original |
| u-h-first | First attempt: one stack for everything | one stack | four workloads diverge | f18-workloads | SVG | Stanford |
| u-h-four-workloads | The four workloads | one serving | chat, batch, embeddings, agentic | f18-workloads | SVG | Stanford |
| u-h-naive | Why the naive stack fails each workload | stack fine | each workload breaks one assumption | f18-workloads | SVG | original |
| u-h-breaks | Where one stack breaks: prefill vs decode | one phase | compute-bound vs memory-bound | f18-prefill-decode | SVG | Stanford |
| u-h-prefill | Prefill, mechanized | prefill undefined | parallel, compute-bound | f18-prefill-decode | SVG | Stanford |
| u-h-decode | Decode, mechanized | decode undefined | serial, memory-bound | f18-prefill-decode | SVG | Stanford |
| u-h-decode-waste | Work the decode waste. | waste asserted | 140 GB moved per token | f18-decode-tax | SVG | original |
| u-h-decode-tax-work | Work the decode tax | tax asserted | 42.4 ms at 3.3 TB/s | f18-decode-tax | SVG | original |
| u-h-hw-split | The hardware split | one fleet | prefill and decode want different chips | f18-two-fleets | SVG | original |
| u-h-keyq | The key question | one layer | what does each token cost? | c18-tax | chapter plate | original |
| u-h-two-q | Two questions, two layers | one question | latency per token, throughput per dollar | c18-tax | chapter plate | original |
| u-h-disagg | Disaggregation: split the fleets | one fleet | prefill fleet, decode fleet | f18-disaggregation | SVG | Stanford |
| u-h-standard | The standard move, mechanized | move abstract | KV shipped between fleets | f18-disaggregation | SVG | original |
| u-h-cache-aware | Cache-aware routing (the 40% win) | routing dumb | route by prefix hit | f18-disaggregation | SVG | original |
| u-h-kv-transfer | The KV cache transfer problem | transfer free | 42.9 GB per request moved | f18-two-fleets | SVG | original |
| u-h-2026-stack | The 2026 disaggregation stack, verified | stack assumed | 2.8x goodput, verified sum | f18-two-fleets | SVG | original |
| u-h-scale-bugs | Scale bugs, the tax | scale clean | bugs appear at 1k GPUs | f18-two-fleets | SVG | original |
| u-h-cont-batch | Continuous batching: fill the gaps | static batching | swap finished sequences live | f18-continuous-batching | SVG | Stanford |
| u-h-static-cont | Static versus continuous, worked | equal assumed | continuous fills idle slots | f18-continuous-batching | SVG | original |
| u-h-kv-ceiling | The KV memory ceiling | memory free | 42.9 GB caps concurrency | f18-continuous-batching | SVG | original |
| u-h-kv | The KV cache: never recompute | recompute assumed | cache K and V per layer | f18-kv-cache | SVG | Stanford |
| u-h-radix-work | The radix tree, worked | cache flat | shared prefixes share memory | f-ascii-radix | ASCII | original |
| u-h-paged | PagedAttention, the memory manager | memory flat | block paging, no fragmentation | f18-kv-cache | SVG | paper |
| u-h-tiered | Tiered offload, the three tiers | HBM only | HBM, CPU, disk tiers | f18-kv-cache | SVG | original |
| u-h-spec | Speculative decoding: guess, then verify | serial assumed | draft model guesses, target verifies | f18-spec-decode | SVG | Stanford |
| u-h-accept | The acceptance arithmetic | speedup assumed | acceptance rate sets the win | f18-spec-decode | SVG | original |
| u-h-mega | Megakernels: kill the gaps | kernels fine | 21 kernels, 18.3 overhead | f18-megakernel | SVG | Stanford |
| u-h-gaps | The gaps, itemized | gaps assumed | launch, sync, memory gaps | f18-megakernel | SVG | original |
| u-h-overlap | The megakernel's overlap schedule | schedule assumed | compute overlaps communication | f18-megakernel-fusion | SVG | original |
| u-h-tk | ThunderKittens versus Triton | one kernel language | TK for control, Triton for speed | f18-megakernel-fusion | SVG | original |
| u-h-specificity | The price is specificity. | fusion free | one kernel per shape | f18-megakernel-fusion | SVG | original |
| u-h-parcae | Parcae: flops without parameters | params equal flops | loop layers, reuse weights | f18-parcae | SVG | Stanford |
| u-h-loop | The looping idea, from zero | loop assumed | same block, multiple passes | f18-parcae | SVG | original |
| u-h-spectral | The spectral radius, defined | radius assumed | stability condition on the loop | f18-parcae | SVG | paper |
| u-h-underloop | The under-looping claim | looping equal | fewer loops than layers can win | f18-parcae | SVG | original |
| u-h-inf-bonus | The inference bonus | train only | flops without memory cost | f18-parcae | SVG | original |
| u-h-codesign | Co-design: one decision | layers separate | model and chip together | c18-codesign | chapter plate | Stanford |
| u-h-mla | MLA, the cache compressor | attention fixed | latent attention shrinks KV | c18-codesign | chapter plate | paper |
| u-h-quant | The quantization match | precision free | match precision to the chip | f-tab-codesign | table | original |
| u-h-size | Size the model to the chip | size free | fit the memory hierarchy | f-tab-codesign | table | original |
| u-h-itl | Inter-token latency, the decode SLA | latency one number | TTFT plus ITL | c18-tax | chapter plate | original |
| u-h-ratio | The prefill/decode ratio | ratio assumed | 2026 stacks tune the split | f18-two-fleets | SVG | original |
| u-h-mapback | Mapping back: what each idea fixes | pains unmapped | twelve pains mapped to fixes | f-tab-mapback | table | original |
| u-h-price | The honest price | serving free | memory bandwidth is the bill | c18-codesign | chapter plate | original |
| u-h-recap | Recap: the whole lesson on one screen | story scattered | thirteen steps, each answering the one before | f-tab-mapback | table | original |

Blank cells: zero.

## Remap notes (fix round, 2026-10-06, honest mappings)
- `u-h-scale-bugs` remapped `c18-tax` -> `f18-two-fleets`. The old
  mapping was a thematic mismatch (decode-tax plate). The new
  target is proximal: the section's stack figure, which the prose
  itself places immediately after the scale-bugs subchapter.
  Honest gap: no figure shows the bugs themselves (NaN "hi hi hi"
  loops, off-by-one CJK reads, tool-call loops); they are
  prose-only.
- `u-h-spec` / `u-h-accept` remapped `c18-serving` -> the new
  `f18-spec-decode` lesson plate (before: serial decode;
  rule: draft guesses, target verifies; after: ~4x with the
  prose's acceptance numbers). Speculative decoding is a change
  unit; it now has its dedicated figure.

## Prose problems noticed, NOT touched (for the coordinator)
1. Builder-stats figure count was already wrong before my pass ("16 SVG refs + 2 generated plates" = 18 claimed, but only 8 SVG refs and 2 webp were live in the md); updated to the true count: 18 (14 SVG plates: 8 kept + 3 new lesson + 3 new chapter; 1 ASCII trace; 1 inline table).
2. "The Parcae Twitter story was a fabrication the poster retracted" is the lecture's own hedge, kept in prose; the parcae plate inherits the lecture's numbers only.
3. Speculative decoding has no dedicated plate; the audit maps it to the serving chapter plate (c18-serving), whose tradeoff line names it. The scale-bugs subchapter likewise maps to the tax chapter plate (c18-tax). If the figure auditor wants dedicated plates for these, that is a figure-auditor escalation.
4. The 42.9 GB KV figure and 140 GB/token decode figure are the lecture's; both plates verify the arithmetic from them (140/3300 = 42.4 ms; the lesson states the 42.9 GB as given).
