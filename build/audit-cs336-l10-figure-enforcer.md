# FIGURE ENFORCER audit — cs336/l10-inference
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (Shell + Source added, F5) and added inline
# figure tables (F1/F2).

## Figure inventory (12 plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f01 | assets/l10-metrics.svg | SVG lesson plate (kept) | TTFT, latency, throughput: three questions, three answers |
| f02 | assets/l10-why-different.svg | SVG lesson plate (kept) | Training: all tokens at once. Inference: one token at a time |
| f03 | assets/l10-kv-cache.svg | SVG lesson plate (kept) | Naive: T^3. KV cache: prefill once, decode in linear time |
| f04 | assets/l10-decode-bytes.svg | SVG lesson plate (NEW, replaces webp) | 70B bf16: 140GB in, one token out, 42ms; batch 64: 955 tok/s |
| f05 | assets/l10-intensity.svg | SVG lesson plate (kept) | MLP scales with batch; generation attention sits at intensity 1 |
| f06 | assets/l10-prefill-decode.svg | SVG lesson plate (NEW, replaces webp) | Prefill: fat, compute-bound, TTFT. Decode: thin, memory-bound, latency |
| f07 | assets/l10-latency-throughput.svg | SVG lesson plate (kept) | Batch 1: 124 tok/s. Batch 256: the bus. Memory caps the route |
| f08 | assets/l10-kv-shrink.svg | SVG lesson plate (kept) | GQA, MLA, CLA, sliding window: four axes of shrinking |
| f09 | assets/l10-spec-decode.svg | SVG lesson plate (kept) | Draft cheap and sequential; verify big and parallel; accept min(1,q/p) |
| f10 | assets/l10-spec-math.svg | SVG lesson plate (NEW, replaces webp) | (K x r)/(1 + K x c): 3.3x, 1.7x, 0.83x |
| f11 | assets/l10-serving.svg | SVG lesson plate (kept) | Continuous batching, selective batching, PagedAttention |
| f12 | assets/l10-serving-stack.svg | SVG lesson plate (NEW, replaces webp) | vLLM, SGLang, TensorRT-LLM, DeepSeek: the same ideas, hardened |
| c1 | assets/l10-chap-kvcache.svg | SVG chapter plate (NEW) | O(T^3) to O(T); caching is exact for causal models |
| c2 | assets/l10-chap-decode.svg | SVG chapter plate (NEW) | Intensity ~1; shrink the cache; check beats generate |
| c3 | assets/l10-chap-serving.svg | SVG chapter plate (NEW) | Jagged traffic to steady pipe; do not mix chatbot and batch |
| f-ascii-scale | inline ascii block (kept) | ASCII | 8.6T tokens/day; agents removed the speed limit |
| f-tab-prefill | inline table (kept) | table | Prefill vs decode: work, shape, intensity, bound, batching, metric |
| f-tab-quant | inline table (kept) | table | QAT, PTQ, GPTQ, AWQ: idea and cost |
| f-tab-mla | inline table (NEW) | table | MHA 64KB to MLA 1.1KB per token per layer; 480GB to 8.4GB at 128K |
| f-tab-cachebytes | inline table (NEW) | table | KIVI 8x (3.36GB to 0.42GB); FP8 2x, Hopper-native |
| f-tab-drafts | inline table (NEW) | table | Medusa 2.0x, EAGLE 2.8x, distilled 2.7x, MTP 3.4x |
| f-tab-disagg | inline table (NEW) | table | 1.3GB at 4K (disaggregate) vs 42GB at 128K (the link decides) |
| f-tab-mapping | inline table (kept) | table | Pain, technique, how |

## Fixes applied to existing figures
- Font stack reordered to spec (Anthropic Sans first) in all l10 SVGs.
- Captions on all 8 kept plates: added "Shell N." and "Source:" per F5.
- Four generated webp stills DELETED (decode-bytes, prefill-decode, spec-math, serving-stack) and redrawn as flat SVG lesson plates per the medium ladder (F2). No generated stills remain in l10.

## Medium-ladder justification for new figures
- f04 decode bytes: table no (the claim is the per-step byte flow into one token, a state change); ASCII no (the byte accounting needs position); SVG passes first.
- f06 prefill/decode: table no (the kept f-tab-prefill already covers the value comparison; the plate is the workload-shape asymmetry, a state change); SVG passes first.
- f10 spec math: table no (three speedup outcomes of one formula is a before/after of acceptance); SVG passes first.
- f12 serving stack: table could list the systems, but the claim is the mapping of lecture ideas to production systems; SVG passes first.
- f-tab-mla / f-tab-cachebytes / f-tab-drafts / f-tab-disagg: comparisons of values: table is the first medium that passes.
- c1-c3: chapter plates mandated by the spec.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | Inference has no ceiling: 8.6T tokens/day; agents removed the limit | training as the cost | inference is the daily cost | f-ascii-scale | ASCII | lecture |
| u-h-v4 | DeepSeek-V4's 32T training tokens = under 4 days of OpenAI inference | scale unframed | the comparison that frames the lecture | f-ascii-scale | ASCII | lecture |
| u-h-metrics | TTFT, latency, throughput: three questions, three answers | speed as one number | name the metric first | f01 | SVG | lecture |
| u-h-batch-trade | Batch pits latency against throughput | metrics independent | the tradeoff previewed | f01, f07 | SVG | lecture |
| u-h-naive | Naive generation: O(T^3), 1e9 ops for a paragraph | training code reused | pure waste: the past never changes | f02, f03, c1 | SVG | lecture |
| u-h-intensity1 | Autoregressive: intensity near 1, 300x below H100's 295 | generation assumed fast | the chip waits on memory | f02, f05 | SVG | lecture |
| u-h-keyq | Never recompute the past; then: less memory, more speed? | recompute assumed | the KV cache; the second question | f03, c1 | SVG | lecture |
| u-h-kvcache | Prefill in parallel (compute-bound); decode one token (memory-bound) | cache unnamed | the two phases | f03, c1 | SVG | lecture |
| u-eq-cachesize | Cache = B x S x layers x KV-heads x head-dim x 2 x 2 bytes | size uncounted | 3.36GB for Llama2-13B-ish | f03, c1 | SVG | lecture |
| u-h-decode | 70B bf16: 140GB read, 42ms, 24 tok/s; batch 64: 955 tok/s | step unworked | a memory copy with a little math | f04 | SVG | original toy |
| u-h-amort | Batching amortizes the parameter read: the entire throughput game | batch unexplained | 15 tok/s per seq, 955 total | f04 | SVG | lecture |
| u-h-crossover | Past B=108 the cache dominates the parameters | cache 1% always | the crossover batch | f04 | SVG | original toy |
| u-eq-intensity | MLP: B x T. Attention: S x T/(S+T) | intensity unnamed | the formulas | f05, c2 | SVG | lecture |
| u-h-attnwall | Generation attention sits at ~1; batching cannot fix it | batch fixes all | per-sequence wall | f05, c2 | SVG | lecture |
| u-tab-prefill | Prefill vs decode: one table | asymmetry scattered | one-screen comparison | f-tab-prefill, f06 | table | original synthesis |
| u-h-bus | Llama2-13B: batch 1 at 124 tok/s; memory caps the batch | batch free | the bus analogy | f07 | SVG | lecture |
| u-h-shrink | GQA, MLA, CLA, sliding window: four axes, one tradeoff each | cache fixed | less memory is more speed | f08, c2 | SVG | lecture |
| u-h-gqa | GQA: fewer KV heads, cache / (N/K); K=8 keeps accuracy | heads fixed | the default shrink | f08 | SVG | lecture |
| u-h-mla | MLA: 64KB to 1.1KB per token per layer, 57x | compression unnamed | 480GB to 8.4GB at 128K | f-tab-mla, c2 | table | DeepSeek-V2 |
| u-h-rope | RoPE needs raw keys: decoupled RoPE carries position | wrinkle unhandled | 64 extra dims | f-tab-mla | table | DeepSeek-V2 |
| u-h-cla | CLA: share KV across layers | layers independent | another sharing axis | f08 | SVG | lecture |
| u-h-sliding | Sliding window: cache independent of length; hybrid with global | context unbounded | the bound | f08 | SVG | lecture |
| u-h-linear | Linear attention/Mamba: fixed state; hybrids combine all three | history uncompressed | more expressive than windows | f08 | SVG | lecture |
| u-h-kivi | KIVI: keys per-channel, values per-token, 2-bit; 3.36GB to 0.42GB | cache unquantized | match granularity to outliers | f-tab-cachebytes | table | Liu et al. |
| u-h-fp8 | FP8 cache: 2x, Hopper-native, nearly free | precision fixed | the coarse alternative | f-tab-cachebytes | table | lecture |
| u-h-quantfam | QAT, PTQ, GPTQ, AWQ: idea and cost | precision fixed | the family | f-tab-quant | table | lecture |
| u-h-prune | Pruning 15B to 8B; distillation heals | params fixed | the Frankenstein repair | f-tab-quant | table | NVIDIA |
| u-h-specdec | Draft K cheap, verify in parallel, accept min(1,q/p); exact samples | generation sequential | checking beats generating | f09, c2 | SVG | lecture |
| u-eq-spec | Speedup = (K x r)/(1 + K x c); K=3-4 | speedup unformula'd | 3.3x, 1.7x, 0.83x | f10, c2 | SVG | lecture |
| u-h-drafts | Medusa 2.0x, EAGLE 2.8x, distilled 2.7x, MTP 3.4x | one draft assumed | the family's ranking | f-tab-drafts | table | original toys |
| u-h-contbatch | Continuous batching (Orca): evict finished, admit new | traffic jagged | the pipe never stalls | f11, c3 | SVG | lecture |
| u-h-selbatch | Selective batching: MLP mega-seq, attention per-seq | batch uniform | flatten what flattens | f11 | SVG | lecture |
| u-h-paged | PagedAttention: blocks like OS pages; prefix sharing | cache fragmented | defragged memory | f11, c3 | SVG | Kwon et al. |
| u-h-stack | vLLM, SGLang, TensorRT-LLM, DeepSeek: hardened ideas | stack unnamed | the production map | f12 | SVG | project docs |
| u-h-disagg | Disaggregation: prefill and decode fleets; 1.3GB vs 42GB | phases colocated | the transfer decides | f-tab-disagg, c3 | table | Splitwise/DistServe |
| u-h-chunked | Chunked prefill: 256 chunks, decode latency protected | long prompt blocks all | chop the big unit | c3 | chapter plate | Sarathi |
| u-tab-mapping | Pain to technique mapping | scattered claims | one-screen mapping | f-tab-mapping | table | original synthesis |
| u-h-price | Estimates, worked examples, tradeoffs; serving cannot fix bad training | optimization free | honest price stated | c1, c2, c3 | chapter plates | original synthesis |
| u-h-coverage | Coverage map | claims scattered | traceability table | n/a - coverage unit | table | original |
| u-h-recap | 8-step recap | lesson as sequence | one-screen consolidation | c1, c2, c3 | chapter plates | original synthesis |
| u-h-qa | Interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: nocookie embed + 4 links | n/a (media law unit) | video + links | n/a - media unit | video/links | papers |
| u-h-official | Official sources and further reading | n/a (sourcing unit) | sources + caveats | n/a - sourcing unit | links | lecture/book |
| u-h-conn | Connections to other courses | n/a (sourcing unit) | cross-links | n/a - sourcing unit | prose | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. The prose says "At 3.35 TB/s of HBM bandwidth, 140GB takes 42ms: 24 tokens per second per GPU" and the Q&A says batch 64 gives 955 tok/s. The plate repeats both. 64/0.067 = 955: consistent.
2. "Llama2-13B-ish" is a worked example, not a benchmark; the plate's latency-throughput caption inherits the lecture's framing. The honest-price section covers this.
3. The MLA 57x: 64KB/1.1KB = 58x; the prose says 57x. The plate and table repeat the prose's 57x. Rounding, not an error, but the figure auditor may want the exact 58.2 noted.
