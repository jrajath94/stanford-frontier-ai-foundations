# FIGURE ENFORCER audit — cs336/l09-scaling-laws
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (Shell + Source added, F5) and added inline
# figure tables/equations (F1/F2).

## Figure inventory (11 plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f01 | assets/l09-history.svg | SVG lesson plate (kept) | 1993 to 2022: the same power law, rediscovered each decade |
| f02 | assets/l09-loglog.svg | SVG lesson plate (kept) | Slope is the exponent: -1 for mean estimation, ~-0.1 for neural nets |
| f03 | assets/l09-slope-exponent.svg | SVG lesson plate (NEW, replaces webp) | Double the data: halve the error (steep) vs 7% (shallow) |
| f04 | assets/l09-data-uses.svg | SVG lesson plate (kept) | Mixtures shift intercepts, repetition survives 4 epochs, filters loosen with scale |
| f05 | assets/l09-critical-batch.svg | SVG lesson plate (kept) | Noise-limited: perfect returns. Bias-limited: diminishing. B_crit grows as loss drops |
| f06 | assets/l09-bcrit-worked.svg | SVG lesson plate (NEW, replaces webp) | Batch 1k: doubling halves steps. Batch 1M: doubling saves 5% |
| f07 | assets/l09-upstream-downstream.svg | SVG lesson plate (kept) | Perplexity linear and clean; downstream noisy; NL12 vs NL32XL |
| f08 | assets/l09-kaplan-vs-chinchilla.svg | SVG lesson plate (kept) | Kaplan N^0.27 train giants; Chinchilla N^0.5 20 tok/param; details decided |
| f09 | assets/l09-three-methods.svg | SVG lesson plate (kept) | Envelope, IsoFLOP, parametric fit; IsoFLOP is the default |
| f10 | assets/l09-isoflop-worked.svg | SVG lesson plate (NEW, replaces webp) | Fix FLOPs, sweep N vs D, join the minima: that line is the law |
| f11 | assets/l09-overtrain.svg | SVG lesson plate (kept) | GPT-3: 3, Chinchilla: 20, modern: 100+ tok/param, deliberately overtrained |
| f12 | assets/l09-tokens-per-param.svg | SVG lesson plate (NEW, replaces webp) | Production ratios vs Chinchilla: V3 22, Llama3 39, Kimi K2 16, Qwen3 153 |
| c1 | assets/l09-chap-scaling.svg | SVG chapter plate (NEW) | Tune small, extrapolate big; the line is a promise the regime will not change |
| c2 | assets/l09-chap-chinchilla.svg | SVG chapter plate (NEW) | Three misses sank Kaplan; 20:1 is training-optimal, not serving-optimal |
| c3 | assets/l09-chap-overtrain.svg | SVG chapter plate (NEW) | Serve small, train long; test-time compute is the third dial |
| f-ascii-naive | inline ascii block (kept) | ASCII | Naive vs scaling approach: tune big vs optimize small and extrapolate |
| f-tab-fit | inline table (NEW) | table | 3-point fit: slope -0.0235, 10x data = 1.056x error cut |
| f-tab-emergence | inline table (NEW) | table | p^3 vs p: the jump was in the ruler, not the thing measured |
| f-tab-datawall | inline table (NEW) | table | 3 epochs on 1T = 1.6T fresh-equivalent; shrinking usually wins |
| f-tab-bnsl | inline table (NEW) | table | Single line promised 1.58x; the break delivered 1.41x |
| f-eq-lr | inline equation block (NEW) | equation | eta ~ 1/width vs muP fixed: pick one and commit |
| f-tab-testtime | inline table (NEW) | table | Training-heavy vs inference-heavy: queries decide the split |
| f-tab-tokparam | inline table (kept) | table | Qwen3 153, Llama3 39, V3 22, Kimi K2 16: verified Oct 2026 |
| f-tab-mapping | inline table (kept) | table | Pain, tool, how |

## Fixes applied to existing figures
- Font stack reordered to spec (Anthropic Sans first) in all l09 SVGs.
- Captions on all 8 kept plates: added "Shell N." and "Source:" per F5.
- Four generated webp stills DELETED (slope-exponent, bcrit-worked, isoflop-worked, tokens-per-param) and redrawn as flat SVG lesson plates per the medium ladder (F2). No generated stills remain in l09.

## Medium-ladder justification for new figures
- f03 slope exponent: table no (two lines' steepness is positional); ASCII sparkline too crude for the labeled -1 vs -0.1 contrast; SVG passes first.
- f06 B_crit: table no (the claim is the crossover of two regimes, a state change); SVG passes first.
- f10 IsoFLOP: table no (U-curves and the minima line are positional); SVG passes first.
- f12 tokens-per-param: the claim is a bar comparison of values, which a table could carry (and the kept f-tab-tokparam does); the plate adds the visual pattern (research vs production) the lesson leans on. Both kept; the plate is the state-change view, the table the values.
- f-tab-fit / f-tab-emergence / f-tab-datawall / f-tab-bnsl / f-tab-testtime: comparisons of values: table is the first medium that passes.
- f-eq-lr: a definition: equation is the first medium that passes.
- c1-c3: chapter plates mandated by the spec.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | 10,000 B200s for a month: optimize small, extrapolate | tune on the big run | scaling approach | f-ascii-naive, c1 | ASCII | lecture |
| u-h-regular | The small-to-large connection must be made regular | connection assumed | right x-axis, right hyperparams | c1 | chapter plate | lecture |
| u-h-history | 1993-2022: Cortes/Vapnik, Banko/Brill, Hestness, Kaplan, Chinchilla | ideas new | ideas old, scale new | f01 | SVG | lecture |
| u-h-genbounds | Generalization bounds: the theorists' scaling laws | theory separate | error bounds decay with sample size | f01 | SVG | lecture |
| u-h-emergence-note | Hestness 2017: accuracy discontinuous where loss smooth | emergence new | noted 2017 | f01 | SVG | lecture |
| u-h-firstattempt | Fix the model, grow the data: error ~ 1/n^alpha | scaling unnamed | log-log line = power law | f02, c1 | SVG | lecture |
| u-h-stats | Gaussian mean slope -1; neural nets -0.1 to -0.3 | rates uncompared | parametric vs flexibility price | f02, f03 | SVG | lecture |
| u-h-10dims | Nets learn like nonparametric regression in ~10 dims | exponent unexplained | the slope's meaning | f02, f03 | SVG | lecture |
| u-h-readloglog | Read a log-log plot: slope = exponent, intercept = constant | plot unreadable | two readings separated | f03 | SVG | lecture |
| u-h-fitline | Fit: least squares on logs; slope -0.0235 on 3 points | mechanics unnamed | 10x data = 1.056x cut | f-tab-fit | table | original toy |
| u-h-fit-judgments | Three judgments: regime points, error bars, extrapolation distance | fit naive | fit the straight middle only | f-tab-fit | table | lecture |
| u-h-mixtures | Mixtures shift intercepts, not slopes | composition untested | best small mix = best large mix | f04 | SVG | lecture |
| u-h-repetition | ~4 epochs safe, then below fresh-data projection | repetition free | 4 epochs, then ensemble | f04 | SVG | lecture |
| u-h-filtering | Filtering: hard at small compute, loose at big | filter fixed | scale-dependent rule | f04 | SVG | lecture |
| u-h-arch | Transformers vs LSTMs: the plot justifies the transformer | arch untested | intercepts differ, maybe slopes | f04 | SVG | lecture |
| u-h-tay | Tay et al.: GLU scales, Performer does not, Switch does | arch unranked | scaling law as the filter | f04 | SVG | Tay et al. |
| u-h-sgdadam | SGD vs Adam: different intercepts, same slopes | optimizer untested | even Adam leaves the slope alone | f04 | SVG | Hestness |
| u-h-kaplan-emb | Kaplan excluded embeddings; MoE needs total vs active axes | params uncounted | count all the parameters | f04, f08 | SVG | lecture |
| u-h-emergence | Emergence: real as user phenomenon, suspect as internal claim | emergence binary | both, at different levels | f-tab-emergence | table | Wei/Schaeffer |
| u-h-schaeffer | Thresholded metrics manufacture jumps: p^3 vs p | jumps real | the ruler, not the thing | f-tab-emergence | table | Schaeffer et al. |
| u-h-datawall | 1T fresh + compute for 3T: 1.6T fresh-equivalent; shrink usually wins | repeats free | x-axis is fresh-equivalent | f-tab-datawall | table | Muennighoff et al. |
| u-h-keyq | Fixed FLOPs: what N/D split minimizes loss? | bigger model or more data | FLOPs ~ N x D, two dials | f10, c2 | SVG | lecture |
| u-h-bcrit | B_crit: noise-limited vs bias-limited | batch unbounded | the crossover | f05, f06 | SVG | lecture |
| u-h-bcrit-worked | Batch 1k halves steps; batch 1M saves 5%; B_crit = E_min/S_min | crossover unworked | the arithmetic | f06 | SVG | lecture |
| u-h-bcrit-grows | B_crit grows as a power law of target loss | batch fixed | ramp the batch as loss falls | f05, f06 | SVG | lecture |
| u-h-lr | LR ~ 1/width vs muP fixed: pick one and commit | LR retuned at scale | the two rules | f-eq-lr | equation | lecture |
| u-h-updown | NL12 won perplexity, NL32XL won tasks | perplexity = quality | fit on perplexity, verify on tasks | f07 | SVG | lecture |
| u-h-kaplan | Kaplan: N^0.27, train giants; GPT-3 era listened | law as truth | the old canon | f08, c2 | SVG | Kaplan et al. |
| u-h-chinchilla | Chinchilla: N^0.5, 20 tok/param; three methods agreed | law as truth | the new canon | f08, c2 | SVG | Hoffmann et al. |
| u-h-kaplan-lost | Unembedding, warmup, batch size: minor details, big shifts | details irrelevant | details decide | f08, c2 | SVG | lecture |
| u-h-epochai | Epoch AI: method 3 was underfit; refit agrees | methods disagree | robustify the fit | f09 | SVG | Epoch AI |
| u-h-isolflop | IsoFLOP: sweep, minima, join them; fewest assumptions | procedure unnamed | the reliable default | f10 | SVG | lecture |
| u-h-bnsl | Broken laws: 1.58x promised, 1.41x delivered | line continues | fit the bends | f-tab-bnsl | table | Caballero et al. |
| u-h-overtrain | GPT-3: 3, Chinchilla: 20, modern: 100+ tok/param | 20 as the answer | serve small, train long | f11, f12, c3 | SVG | lecture |
| u-tab-tokparam | Qwen3 153, Llama3 39, V3 22, Kimi K2 16, verified Oct 2026 | ratio unmeasured | the production pattern | f-tab-tokparam, f12 | table | company reports |
| u-h-unknowns | GPT-4/5, Gemini, Claude ratios: unknown, never published | estimates as facts | mark unknown | f12 | SVG | honesty note |
| u-h-testtime | 7B + 100x thinking vs 70B 1x; queries decide | training only | the third dial | f-tab-testtime, c3 | table | Snell et al. |
| u-tab-mapping | Pain to tool mapping | scattered claims | one-screen mapping | f-tab-mapping | table | original synthesis |
| u-h-price | Exponents fit not derived; 20:1 training-optimal; emergence outside frame | laws as physics | honest price stated | c1, c2, c3 | chapter plates | original synthesis |
| u-h-coverage | Coverage map | claims scattered | traceability table | n/a - coverage unit | table | original |
| u-h-recap | 8-step recap | lesson as sequence | one-screen consolidation | c1, c2, c3 | chapter plates | original synthesis |
| u-h-qa | Interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: nocookie embed + 4 links | n/a (media law unit) | video + links | n/a - media unit | video/links | papers |
| u-h-official | Official sources and further reading | n/a (sourcing unit) | sources + caveats | n/a - sourcing unit | links | lecture/papers |
| u-h-conn | Connections to other courses | n/a (sourcing unit) | cross-links | n/a - sourcing unit | prose | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. The lesson says neural-net slopes are "-0.1 to -0.3" but the plate and much of the prose use "about -0.1". The range is preserved in the prose; the plate picks the headline value. Consistent with the lesson's emphasis, but the figure auditor should confirm.
2. "10^0.0235 = 1.056x" is a worked toy with a shallow slope "even by neural standards" per the prose; the plate does not repeat this toy (it lives in the table f-tab-fit). Fine.
3. The tokens-per-param table and plate carry the same four verified rows; GPT-4/Gemini/Claude unknowns live in prose only, noted in the plate footer.
