# FIGURE ENFORCER audit — cs336/l12-evaluation
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (Shell + Source added, F5) and added inline
# figure tables (F1/F2).

## Figure inventory (12 plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f01 | assets/l12-what-is-good.svg | SVG lesson plate (kept) | Benchmarks, cost, preference, usage: four lenses on good |
| f02 | assets/l12-perplexity.svg | SVG lesson plate (kept) | Best perplexity is the entropy of truth; the catches |
| f03 | assets/l12-perplexity-toy.svg | SVG lesson plate (NEW, replaces webp) | Model A: 2.15. Model B: 1.11. Perplexity is the average branching factor |
| f04 | assets/l12-exam-treadmill.svg | SVG lesson plate (kept) | MMLU to HLE: each benchmark saturates, each replacement is harder |
| f05 | assets/l12-chat-eval.svg | SVG lesson plate (kept) | Arena, AlpacaEval, WildBench: pairwise, LLM judges, checklists |
| f06 | assets/l12-elo-worked.svg | SVG lesson plate (NEW, replaces webp) | Upset win: plus 24. Expected win: plus 8. Upsets move ratings |
| f07 | assets/l12-agent-eval.svg | SVG lesson plate (kept) | SWE-bench, Terminal-Bench, CyBench: checkable outcomes |
| f08 | assets/l12-arc.svg | SVG lesson plate (kept) | Human-easy grids, knowledge-free; reasoning models moved the needle |
| f09 | assets/l12-contamination.svg | SVG lesson plate (kept) | Detect, report, refresh, privatize: four defenses |
| f10 | assets/l12-contamination-tell.svg | SVG lesson plate (NEW, replaces webp) | Same question, answer at A vs C: the order gap betrays memorization |
| f11 | assets/l12-eval-purpose.svg | SVG lesson plate (kept) | Buy, measure, improve, ship: each purpose wants its own benchmark |
| f12 | assets/l12-eval-landscape.svg | SVG lesson plate (NEW, replaces webp) | Five purposes, five benches; declare the purpose first |
| c1 | assets/l12-chap-perplexity.svg | SVG chapter plate (NEW) | exp(average surprise); the floor is the data's entropy |
| c2 | assets/l12-chap-harness.svg | SVG chapter plate (NEW) | The harness is the measurement |
| c3 | assets/l12-chap-contamination.svg | SVG chapter plate (NEW) | The test set is a resource that depletes |
| f-tab-biases | inline table (NEW) | table | Position, verbosity, self-enhancement, sycophancy: what happens, the check |
| f-tab-passk | inline table (NEW) | table | pass@1 30% vs pass@100 85%: the verifier gap |
| f-tab-xstest | inline table (NEW) | table | HarmBench 99% vs XSTest 40%; the target corner |

## Fixes applied to existing figures
- Font stack reordered to spec (Anthropic Sans first) in all l12 SVGs.
- Captions on all 8 kept plates: added "Shell N." and "Source:" per F5.
- Four generated webp stills DELETED (perplexity-toy, elo-worked, contamination-tell, eval-landscape) and redrawn as flat SVG lesson plates per the medium ladder (F2). No generated stills remain in l12.
- CORRECTIONS made while redrawing: the first drafts of three plates were checked against the actual prose and redrawn. The perplexity toy now shows the prose's own numbers (A: 0.5/0.4/0.5 to 2.15; B: 0.9/0.9/0.9 to 1.11), not an invented variant. The contamination tell now shows the prose's order-gap probe (A vs C, 27-point gap as the tell) instead of paraphrase collapse. The eval landscape now shows the five-purposes map, not the perplexity/ELO/agent trinity.

## Medium-ladder justification for new figures
- f03 perplexity toy: table no (the claim is the branching-factor reading of 2.15 vs 1.11, a worked example); SVG passes first.
- f06 ELO: table no (the claim is the surprise-weighted update: +24 vs +8); SVG passes first.
- f10 contamination tell: table no (the claim is the behavioral probe: same question, different order, the gap); SVG passes first.
- f12 eval landscape: table no (the kept prose already lists the five purposes as bullets; the plate is the one-screen map); SVG passes first.
- f-tab-biases / f-tab-passk / f-tab-xstest: comparisons of values; table is the first medium that passes.
- c1-c3: chapter plates mandated by the spec.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | What is good? Four lenses: benchmarks, cost, preference, usage | goodness assumed | named the lenses | f01 | SVG | lecture |
| u-h-perp | Best perplexity is the entropy of truth; the catches | perplexity assumed | the cleanest metric, with catches | f02, c1 | SVG | lecture |
| u-h-branch | 2.15 vs 1.11: the average branching factor | number unworked | exp(average surprise) | f03, c1 | SVG | lecture |
| u-h-breaks | Perplexity breaks: three demonstrations | metric trusted | its limits | c1 | chapter plate | lecture |
| u-h-treadmill | MMLU to HLE: each benchmark saturates, each replacement harder | benchmarks assumed fresh | the treadmill | f04, c3 | SVG | lecture |
| u-h-gpqa | GPQA diamond: how the sausage is made | benchmark assumed | the construction | f04 | SVG | lecture |
| u-h-chateval | Arena, AlpacaEval, WildBench: pairwise, judges, checklists | chat unmeasured | the instruments | f05, c2 | SVG | lecture |
| u-h-elo | ELO: upset +24, expected +8; upsests move ratings | score assumed | the ledger | f06, c2 | SVG | lecture |
| u-h-biases | Position, verbosity, self-enhancement, sycophancy | judge trusted | the catalog | f-tab-biases, c2 | table | lecture |
| u-h-agenteval | SWE-bench, Terminal-Bench, CyBench: checkable outcomes | agents unmeasured | environments | f07 | SVG | lecture |
| u-h-taskfam | Agentic task families | tasks assumed | the families | f07 | SVG | lecture |
| u-h-arc | Human-easy grids, knowledge-free; reasoning models moved it | reasoning assumed | isolated | f08 | SVG | lecture |
| u-h-passk | pass@1 30%, pass@100 85%: the verifier gap | single number trusted | two instruments | f-tab-passk | table | lecture |
| u-h-safety | HarmBench 99% vs XSTest 40%: one column is not a scoreboard | safety assumed | the corner | f-tab-xstest | table | lecture |
| u-h-validity | Ecological validity; GDPVal | evals assumed real | match the deployment | c2 | chapter plate | lecture |
| u-h-contam | Detect, report, refresh, privatize | contamination assumed | four defenses | f09, c3 | SVG | lecture |
| u-h-tell | A vs C order gap: the behavioral probe | contamination invisible | the tell, worked | f10, c3 | SVG | lecture |
| u-h-purpose | Buy, measure, improve, ship: purpose picks the benchmark | benchmarks assumed | the purpose map | f11, f12, c3 | SVG | lecture |
| u-h-evalmap | Five purposes, five benches (Oct 2026) | benches assumed | the map | f12 | SVG | lecture |
| u-h-price | Estimates, worked examples, tradeoffs; Mythos figures illustrative | numbers assumed | honest price stated | c1, c2, c3 | chapter plates | original synthesis |
| u-h-qa | Interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: nocookie embed + 4 links | n/a (media law unit) | video + links | n/a - media unit | video/links | papers |
| u-h-official | Official sources and further reading | n/a (sourcing unit) | sources + caveats | n/a - sourcing unit | links | lecture/book |
| u-h-conn | Connections to other courses | n/a (sourcing unit) | cross-links | n/a - sourcing unit | prose | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. The perplexity toy's Model B probabilities (0.9, 0.9, 0.9) are not stated in prose: only the product-derived 1.11 is given. The plate shows 0.9s as the reconstruction that yields 1.11 (0.729^(-1/3) = 1.11, verified). If the lecture's actual Model B probabilities differ, the plate's reconstruction should be updated.
2. The contamination-tell numbers (68%/67% vs 81%/54%, 27-point gap) are illustrative: the prose gives no specific percentages. The plate is honest as a worked toy, but the figure auditor may want "[illustrative]" marked.
3. The prose says the lecture's "Mythos" figures are illustrative and that leaderboard numbers decay; the plates repeat the purpose-map without specific scores, which is the honest framing.
