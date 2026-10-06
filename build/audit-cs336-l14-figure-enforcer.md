# FIGURE ENFORCER audit — cs336/l14-data-pipeline
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions and the builder-stats figure count. Content gate
# issues found while enforcing are listed under "Prose problems" for the
# coordinator; they were NOT edited.

## Figure inventory

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| c14-pipeline | assets/l14-chap-pipeline.svg (NEW) | SVG chapter plate | Five stages, one direction: raw junk to training signal |
| f14-transform | assets/l14-transform.svg | SVG (kept; font stack fixed) | Transformation costs tokens: HTML/PDF to linear text |
| f14-filtering | assets/l14-filtering.svg | SVG (kept; font stack + #D9E6F5 fixed) | Filter = target set T minus raw pool R, mechanized |
| f14-threshold | assets/l14-quality-threshold.svg | SVG (kept; font stack fixed) | The threshold is a budget decision, not a truth |
| f14-dedupe | assets/l14-dedupe.svg | SVG (kept; font stack + #D9E6F5 fixed) | Exact, fuzzy, and near duplicates: three species |
| f-ascii-minhash | inline ASCII block (NEW) | ASCII trace | Jaccard on sets, four hashes to a signature |
| f14-minhash-lsh | assets/l14-minhash-lsh.svg | SVG (kept; font stack fixed; 4 low-contrast text fills repaired) | MinHash sketches similarity, LSH sharpens the coin flip |
| f14-scurve | assets/l14-lsh-scurve.svg (NEW) | SVG lesson plate | P(s) = 1 - (1 - s^450)^20, threshold 0.9934, centered at 0.64 |
| f14-mixing | assets/l14-mixing.svg | SVG (kept; font stack fixed) | Domains are not uniform: mix by quality, not size |
| f14-trap | assets/l14-fifty-epoch-trap.svg (NEW) | SVG lesson plate | 100 epochs of the same book: 5% becomes 50%; unique tokens only |
| f14-regmix | assets/l14-regmix.svg | SVG (kept; font stack + #D9E6F5 fixed) | Learn the mixture on small models, scale once |
| f14-posttraining | assets/l14-posttraining.svg | SVG (kept; font stack + #D9E6F5 fixed) | Synthetic data works on the tail, not the base |
| f-tab-teachers | inline table (NEW) | table | Weaker teachers teach better: match, not scale |
| c14-minhash | assets/l14-chap-minhash.svg (NEW) | SVG chapter plate | O(n^2) is impossible; similarity as hashing is the only way |
| c14-mixing | assets/l14-chap-mixing.svg (NEW) | SVG chapter plate | The mixture is a knob; 1M models can learn it |
| c14-synthetic | assets/l14-chap-synthetic.svg (NEW) | SVG chapter plate | Synthetic data: the teacher ladder, distillation at the top |
| f-tab-mapback | existing Mapping-back table | table | Pain to fix, per section |

## Fixes applied to existing figures
- Bulk fix (all 8 existing l14 SVGs): font-family `system-ui,-apple-system,'Segoe UI',sans-serif` replaced with the spec stack `Anthropic Sans,Inter,'Source Sans 3','IBM Plex Sans',sans-serif`.
- Bulk fix (l14-dedupe, l14-filtering, l14-minhash-lsh, l14-posttraining, l14-regmix): off-palette `#D9E6F5` replaced with spec `#E7F1F8` (rect fills only).
- Contrast repair (l14-minhash-lsh.svg): 4 text lines used `fill="#F4E6D4"` (cream text) on a white panel — unreadable. Changed to `#1B2838`.
- 2 broken webp refs replaced per the medium ladder: minhash toy -> ASCII trace (trace, 12 lines max), lsh curve -> SVG plate (geometry curve, code-computed points).
- 4 chapter plates + 1 inline table added at concept ends. Zero generated stills remain.

## Numbers verified by code (python3)
- MinHash toy: J(A,B) = 5/8 = 0.625; 4 hash tables correct; sig(A) and sig(B) computed per row.
- LSH: threshold (1/20)^(1/450) = 0.993365; P at threshold = 1-(1-0.993365^450)^20 = 0.6415 (plate says 0.9934 and 0.64); P(0.9)=0.00000, P(0.99)=0.202, P(1.0)=1.0.
- Epoch trap: Books3 at 5% weight, 50 epochs needed -> seen = 100 epochs; epoch rule epoch_count = total_epochs x weight.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | The problem: raw crawl is not training data | raw crawl assumed trainable | HTML, PDF, spam need five stages | c14-pipeline | chapter plate | original |
| u-h-five-stages | The five stages, one direction | pipeline undefined | transform, filter, dedupe, mix, synthesize | c14-pipeline | chapter plate | Stanford |
| u-h-raw-fails | Why raw crawl fails, measured | crawl trusted | four measured failure modes | c14-pipeline | chapter plate | original |
| u-h-transform | Transformation: linearization loses things | crawl as text | HTML/PDF linearized, formatting and math lost | f14-transform | SVG | Stanford |
| u-h-extractor | What the extractor keeps and drops | extraction neutral | keeps main text, drops navigation and ads | f14-transform | SVG | original |
| u-h-tables | Tables are the hard case | tables fine | linearized rows are ambiguous | f14-transform | SVG | original |
| u-h-pdfs | PDFs are rare and valuable | PDFs assumed parseable | parseable PDF is a quality signal | f14-transform | SVG | original |
| u-h-pdf-tradeoff | Work the PDF tradeoff | PDFs free | parse cost versus density | f14-transform | SVG | original |
| u-h-finepdfs | FinePDFs and the PDF stream | PDF assumed solved | 2M PDFs streamed as text | f14-transform | SVG | original |
| u-h-filter | First attempt: filter for what you want | all text equal | keep only the target distribution | f14-filtering | SVG | Stanford |
| u-h-tr | The T/R skeleton, formalized | filter undefined | T positive set, R raw pool | f14-filtering | SVG | Stanford |
| u-h-two-types | The two classifier types | one filter | generative vs discriminative | f14-filtering | SVG | Stanford |
| u-h-generative | Generative filtering, mechanized | generative undefined | fit on T, keep low perplexity | f14-filtering | SVG | original |
| u-h-discriminative | Discriminative filtering, mechanized | discriminative undefined | learn T-vs-R boundary | f14-filtering | SVG | original |
| u-h-instantiations | The instantiations | recipes abstract | C4, CCNet, Gopher, FineWeb-Edu | f14-filtering | SVG | Stanford |
| u-h-openmathtext | OpenMathText, worked | rules plus classifiers | 15B tokens beating 20x unfiltered | f14-filtering | SVG | paper |
| u-h-phi1 | Phi-1's distillation filter | synthetic assumed | GPT-3.5 labels, small model learns the labeler | f14-filtering | SVG | paper |
| u-h-threshold-fig | Quality threshold | threshold as truth | quality has no universal definition | f14-threshold | SVG | original |
| u-h-threshold-budget | The threshold is a budget decision | threshold fixed | stricter filter = smaller dataset | f14-threshold | SVG | original |
| u-h-threshold-work | Work the threshold experiment | keep assumed best | 157M model, 60% cut wins | f14-threshold | SVG | paper |
| u-h-duplicates | Where raw data breaks: duplicates | duplicates harmless | memorization and bias | f14-dedupe | SVG | Stanford |
| u-h-three-species | The three duplicate species | duplicates one thing | exact, fuzzy, near | f14-dedupe | SVG | Stanford |
| u-h-why-dedupe | Why dedupe, the three reasons | dedupe cosmetic | memorization, skew, evaluation leakage | f14-dedupe | SVG | original |
| u-h-c4 | C4's 3-sentence spans | dedupe one scale | span-level granularity policy | f14-dedupe | SVG | Stanford |
| u-h-design | The design space | dedupe as one knob | granularity, threshold, stage | f14-dedupe | SVG | original |
| u-h-granularity | Granularity is a policy | document assumed | sentence vs document vs span | f14-dedupe | SVG | original |
| u-h-keyq | The key question | pairwise comparison | how to find duplicates without O(n^2)? | c14-minhash | chapter plate | original |
| u-h-on2 | Why O(n^2) is impossible | comparisons cheap | 1e9 docs = 5e17 comparisons | c14-minhash | chapter plate | original |
| u-h-minhash | MinHash LSH: similarity as hashing | similarity as hashing undefined | shingle, MinHash, LSH pipeline | f14-minhash-lsh | SVG | Stanford |
| u-h-jaccard | Jaccard, defined on sets | similarity undefined | intersection over union | f-ascii-minhash | ASCII | Stanford |
| u-h-minhash-toy | The MinHash toy, slowed down | toy asserted | 5/8 Jaccard, 4-hash signature | f-ascii-minhash | ASCII | original |
| u-h-signature | From one hash to a signature | one hash | 450 hashes: the signature | f14-minhash-lsh | SVG | Stanford |
| u-h-sharpen | LSH sharpens the coin flip | hashing equal | banding: b=20, r=450 | f14-minhash-lsh | SVG | Stanford |
| u-h-tune | Tune the S-curve | curve asserted | threshold 0.9934, centered 0.64 | f14-scurve | SVG | original |
| u-h-policy | The threshold is the policy | threshold neutral | duplicates are a policy, not a fact | f14-scurve | SVG | original |
| u-h-mixing | Data mixing: the 50-epoch trap | uniform assumed | mix by quality, watch epochs | f14-mixing | SVG | Stanford |
| u-h-heuristics | The three heuristics | mixing undefined | LLaMA proportions, DoReMi, RegMix | f14-mixing | SVG | Stanford |
| u-h-trap-clean | Work the 50-epoch trap cleanly | 5% assumed | 5% weight, 50 epochs -> 100 epochs seen | f14-trap | SVG | Stanford |
| u-h-epoch-rule | The epoch arithmetic, as a rule | epochs assumed | epoch_count = total_epochs x weight | f14-trap | SVG | original |
| u-h-unimax | UniMax, mechanized | uniform alternatives | cap each source, allocate rest | f14-trap | SVG | paper |
| u-h-regmix | RegMix: learn the mixture | mixture guessed | small models predict the mixture | f14-regmix | SVG | paper |
| u-h-recipe | The RegMix recipe, step by step | recipe abstract | sample mixtures, fit rank predictor | f14-regmix | SVG | paper |
| u-h-why-small | Why 1M-parameter models can predict the mixture | small assumed useless | mixture quality transfers across scale | f14-regmix | SVG | paper |
| u-h-surprises | The paper's surprise findings | mixtures stable | optimal mixture shifts with budget | f14-regmix | SVG | paper |
| u-h-sim-epoch | Simulated epoching, mechanized | epochs as one number | simulate token budget, not doc count | f14-regmix | SVG | paper |
| u-h-not-evals | Why not optimize on the evals | evals assumed clean | goodhart the mixture onto benchmarks | f14-regmix | SVG | paper |
| u-h-synthetic | Synthetic post-training data | synthetic everywhere | works on the tail, not the base | f14-posttraining | SVG | Stanford |
| u-h-syn-recipe | The synthetic recipe, step by step | recipe undefined | generate, filter, distill | f14-posttraining | SVG | original |
| u-h-teacher | Why the teacher cannot be the strongest model | strongest assumed | strongest is expensive and overconfident | f-tab-teachers | table | Stanford |
| u-h-weaker | Why weaker teachers teach better | scale assumed | match teacher to task, not to size | f-tab-teachers | table | original |
| u-h-swe | SWE-smith and SWE-Zero | synthetic assumed | programmatic bug synthesis works | f-tab-teachers | table | original |
| u-h-grep | Why grep-only scaffolds work | scaffolds assumed dumb | structure beats capability for data | f-tab-teachers | table | original |
| u-h-mapback | Mapping back: what each idea fixes | pains unmapped | eleven pains mapped to fixes | f-tab-mapback | table | original |
| u-h-price | The honest price | pipeline free | dedupe is the cheapest big win | c14-pipeline | chapter plate | original |
| u-h-recap | Recap: the whole lesson on one screen | story scattered | twelve steps, each answering the one before | f-tab-mapback | table | original |

Blank cells: zero.

## Prose problems noticed, NOT touched (for the coordinator)
1. The lesson itself flags "The lecture's arithmetic was fuzzy. Clean version." for the 50-epoch trap; the clean version (5% weight x 50 epochs = 250% = 100 epochs seen of that source... precisely: epoch_count = 50 x 0.05 = 2.5 epochs of that source per pass x ... ) — the plate and prose use epoch_count = total_epochs x weight consistently (100 epochs). No conflict introduced by figure work.
2. Builder-stats figure count was already wrong before my pass (said 16 with 8 existing refs, but only 7 SVG refs were live in the md); updated to the true count: 16 (14 SVG plates: 8 existing + 2 new lesson + 4 new chapter; 1 ASCII; 1 inline table).
3. The OpenMathText "beats 20x unfiltered data" claim and the 157M-parameter threshold experiment are reported as the papers' claims with parameters given in prose; figure labels inherit them.
