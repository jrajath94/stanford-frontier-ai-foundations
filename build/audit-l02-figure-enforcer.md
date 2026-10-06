# FIGURE ENFORCER audit — mse435/l02-sixty-million-megawatt
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions, image-reference swaps for replaced figures,
# and added inline figures (tables/ASCII/equations).

## Figure inventory (10 SVG plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| F1 | assets/plate-l02-cost-stack.svg | SVG lesson plate | The factory is a wage bill and a turbine order; the machines are a GPU invoice |
| F2 | assets/plate-l02-payback.svg | SVG lesson plate | The product is not compute; finished intelligence halves the payback |
| F3 | assets/plate-l02-depreciation.svg | SVG lesson plate | A GPU that earns $36k in year 2 and loses $4.4k in year 4 cannot live six years on the books |
| F4 | assets/plate-l02-spark.svg (NEW, replaces webp) | SVG lesson plate | When the bottleneck is people, manufacture the people out of the critical path |
| F5 | assets/plate-l02-h100.svg (NEW) | SVG lesson plate | Old compute stayed valuable because demand grew faster than the new supply |
| c1 | assets/plate-l02-chap-coststack.svg (NEW) | SVG chapter plate | Every sanity argument is about the $30M GPU half or the $20M factory third |
| c2 | assets/plate-l02-chap-payback.svg (NEW) | SVG chapter plate | Payback halves when the product changes from compute to finished intelligence |
| c3 | assets/plate-l02-chap-depreciation.svg (NEW) | SVG chapter plate | The schedule does not change the cash; it changes whether the cash was a return or a refund |
| c4 | assets/plate-l02-chap-commodity.svg (NEW) | SVG chapter plate | Watch the margin, not the price: 80% toward 60% is where the verdict shows up first |
| c5 | assets/plate-l02-chap-frontier.svg (NEW) | SVG chapter plate | Each new factory form answers one bar of the $20M stack |
| f-ascii-20m | inline ascii block | ASCII | $20M/MW factory bars, layer by layer |
| f-ascii-40m | inline ascii block | ASCII | $40M/MW machine bars, layer by layer |
| f-mer-quan | inline mermaid block | mermaid | Across-the-meter: wind farm feeds campus first, grid firms the rest |
| f-tab-margin | inline table (NEW) | table | The margin path: 80% today, ~60% standard silicon, custom-silicon threat |
| f-tab-financing | inline table (NEW) | table | The buildout is no longer self-funded: 113% of cash flow; Brookings $3.7T/$6.0T |
| f-tab-commodity-l02 | inline table (NEW) | table | Three-way verdict: old commoditizes, edge and scale do not |
| f-tab-space-l02 | inline table (NEW) | table | What vanishes in orbit vs what stays in orbit |
| f-tab-pricewar | inline table | table | September 2026 token price war, eight models repriced |
| f-tab-capexrace | inline table | table | 2026 hyperscaler capex guidance: ~$730B combined |
| f-tab-mapping | inline table | table | Every L01 puzzle mapped to this chapter's answer |
| f-tab-coveragemap | inline table | table | Every session claim mapped to section and file line |

## Fixes applied to existing figures
- F1 plate-l02-cost-stack.svg: font stack reordered to spec (Anthropic Sans first); off-palette #F6E7A8 replaced with spec #F4E6D4 (same fix as l01 f04); GPU bar widened 400->420 so all four machine bars sit at ~14 px per $M (30M->420, 4M->56, 3M->44, 4M->56); subtitle shortened to fit 960 px (numbers unchanged).
- F2 plate-l02-payback.svg: font stack reordered; added the missing center rule arrow per the change-unit rule (before: rent chips 4yr; rule: add the managed-services layer +$5-15M/yr; after: sell tokens 2yr); headings renamed Before/After; subtitle shortened.
- F3 plate-l02-depreciation.svg: font stack reordered; off-palette #F3D4D8 replaced with spec #F4E6D4; subtitle shortened.
- F4 plate-l02-spark.webp DELETED and redrawn as plate-l02-spark.svg: the webp failed F2 (the claim is a change: hand-built campus -> manufacture centrally -> Spark units; the ladder mandates the lesson plate for state changes). Flat SVG in the sibling style (warm paper #F7F4EE, Anthropic Sans first, 8px grid, one claim, before -> rule -> after). All numbers from the lesson (500 kW air / 2 MW liquid, 30-50% claimed, Brighton CO 352,000 sq ft / $200M+ / 200+ jobs, $4.7M/MW labor bar). md reference updated (webp -> svg); prose untouched.
- F5 plate-l02-h100.svg NEW: the session's H100 pricing chart as a lesson plate (before: $7-8/hr 2024 peak, sub-$1 trough early 2026; rule: agent demand boom + Blackwell rationed; after: ~$3.28/hr Sept 2026, +22% in a month, above launch). Placed after "the H100 price chart, the evidence". All numbers from the lesson.

## Medium-ladder justification for new figures
- F4 spark: table no (claim is a staged change, not a value comparison); equation no; ASCII cannot carry the before/after factory states with the named center rule; SVG is the first medium that fully passes. Lesson plate per the change-unit rule.
- F5 h100: table no (the shape of the price path over time is the claim); equation no; ASCII sparkline too crude for three labeled regimes plus the named rule; SVG passes first.
- f-tab-margin: comparison of values (80% / ~60% / custom-silicon threat) -> table passes first.
- f-tab-financing: comparison of values (capex vs cash flow, $3.7T vs $6.0T) -> table passes first. 800.5/707.1 = 1.1321, computed in python, matches the lesson's 113%.
- f-tab-commodity-l02: comparison of values by horizon -> table passes first.
- f-tab-space-l02: comparison of two lists (vanishes vs stays) -> table passes first.
- c1-c5: chapter plates are mandated extras by the spec (dense, end of concept); SVG keeps text crisp and matches the site's plate system. Zero generated stills remain: the one webp was deleted, zero media-pipeline calls, zero API keys touched.

## Number verification (python3, 2026-10-06)
- 60/15 = 4.0; 60/30 = 2.0; 60/25 = 2.4; 60/20 = 3.0. Payback rows on F2 and c2 check.
- 350 x 3 = 1050 ($1.05B gas plant). 4.7 x 1000 = 4.7 ($4.7B per GW). Turbine 3/1 = 3x.
- 30/60 = 0.5 (GPUs half the total); 20/60 = 0.333 (factory one third). F1 caption and c1 check.
- Machine bars: 420/30 = 14.0, 56/4 = 14.0, 44/3 = 14.7, 56/4 = 14.0 px per $M. Proportional within the half.
- 800.5/707.1 = 1.1321 -> 113%. f-tab-financing checks.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-megawatt | $60M/MW: $20M factory + $40M machines; $60B per GW | no unit price | the megawatt as "per seat" | F1 | SVG | original (session) |
| u-def-megawatt | Megawatt = 1M watts draw, the "per seat" unit | term undefined | defined at first use | F1 | SVG | original |
| u-h-factory20 | The guest's $20M chart is directional, built that afternoon | numbers look audited | caveats stated on the record | F1 | SVG | original |
| u-sub-labor | Capitalized labor $4.7M/MW; $4.7B per GW campus | labor as footnote | largest factory bar priced | F1 | SVG | original (session) |
| u-sub-laborhard | Trades cannot be ordered from a catalog | parts vs people | the unorderable bar | F1 | SVG | original |
| u-sub-gas | 350 MW gas plant at $3M/MW = $1.05B; turbines $1M->$3M | grid assumed | campus makes its own power | F1 | SVG | original (session) |
| u-sub-electrical | Step-down chain 34.5 kV to 480/415 V | term undefined | priced as a $20M line | F1 | SVG | original |
| u-noun-transformer | Transformer = device that steps voltage down | term undefined | defined at first use | F1 | SVG | original |
| u-noun-switchgear | Switchgear = protection and routing layer | term undefined | defined at first use | F1 | SVG | original |
| u-sub-mechanical | Chillers; 1M-gallon loop; home-scale annual use | water fear | inventory vs consumption | F1 | SVG | original (session) |
| u-sub-materials | Own batch plant, pouring 24/7 | concrete abstract | vertical integration into cement | F1 | SVG | original (session) |
| u-noun-batchplant | Batch plant mixes concrete on site | term undefined | defined at first use | F1 | SVG | original |
| u-sub-soft | Insurance, loan service, siting, commissioning | term undefined | the complexity tax | F1 | SVG | original |
| u-noun-commissioning | Commissioning = test every system before GPUs arrive | term undefined | defined at first use | F1 | SVG | original |
| u-sub-fitout | Tenant fit-out; guest flags possible double count | bar looks clean | the honest error bar | F1 | SVG | original |
| u-noun-rpp | Remote power panel distributes to rack rows | term undefined | defined at first use | F1 | SVG | original |
| u-noun-hotaisle | Hot-aisle containment separates exhaust from intake | term undefined | defined at first use | F1 | SVG | original |
| u-sub-20consol | $20M/MW = $20B/GW; inflating, not a law | $20M asserted | snapshot with movers named | F1 | SVG | original |
| u-ascii-20m | $20M factory bars, layer by layer | prose list | the stack as a trace | f-ascii-20m | ASCII | original |
| u-h-machines40 | $40M IT; "forward-looking" frontier-chip pricing | $40M asserted | concentrated cost picture | F1 | SVG | original (session) |
| u-sub-gpu30 | GPUs $30M; 72-GPU NVLink domain; half the total | chips abstract | concentration priced | F1 | SVG | original |
| u-noun-gpu | GPU = chip for thousands of parallel math ops | term undefined | defined at first use | F1 | SVG | original |
| u-noun-nvlink | NVLink = Nvidia chip-to-chip interconnect | term undefined | defined at first use | F1 | SVG | original |
| u-sub-margin80 | Nvidia ~80% gross margins; ~$24M of the $30M bar | margin unstated | concentration quantified | f-tab-margin | table | original (session) |
| u-noun-grossmargin | Gross margin = revenue share above manufacturing cost | term undefined | defined at first use | f-tab-margin | table | original |
| u-sub-net4 | Networking $4M; IB/RoCE joins racks into one cluster | chips isolated | $30M behaves as one machine | F1 | SVG | original |
| u-noun-ib | InfiniBand = supercomputer networking standard | term undefined | defined at first use | F1 | SVG | original |
| u-noun-roce | RoCE = RDMA over Ethernet | term undefined | defined at first use | F1 | SVG | original |
| u-sub-cpu3 | CPUs + storage $3M; agentic CPU shortage | CPU as janitor | CPU as the manager | F1 | SVG | original |
| u-noun-cpu | CPU = general-purpose orchestrating chip | term undefined | defined at first use | F1 | SVG | original |
| u-sub-fitout4 | Fit-out $3M + deployment $1M | $40M asserted | every layer priced | F1 | SVG | original |
| u-sub-customsi | Custom silicon escape valve; 80%-effective-at-50%-price trade | 80% unchallenged | the threat that keeps it honest | f-tab-margin | table | original |
| u-noun-tpu | TPU = Google's own AI chip | term undefined | defined at first use | f-tab-margin | table | original |
| u-ascii-40m | $40M machine bars, layer by layer | prose list | the stack as a trace | f-ascii-40m | ASCII | original |
| u-sub-60consol | $60M/MW; $60B/GW; the two ratios to carry | two halves | consolidation + chapter plate | c1 | chapter plate | original synthesis |
| u-h-payback | Key question: what decides 4-year vs 2-year? | cost known, return unknown | the payback question framed | F2 | SVG | original |
| u-sub-opex | OpEx $1-2M/MW/yr; big money all upfront | term undefined | the attraction and the trap | F2 | SVG | original |
| u-noun-opex | OpEx = cost of running the asset | term undefined | defined at first use | F2 | SVG | original |
| u-sub-rent15 | $15M/yr renting chips; 60/15 = 4 years, revenue basis | no return math | the base case priced | F2 | SVG | original |
| u-sub-missing | $1-2M omits engineers, software, financing, refresh | payback looks clean | the caveat attached | F2 | SVG | original |
| u-sub-token30 | Managed +$5-15M; ~$30M; 60/30 = 2 years | commodity product | the margin lives in tokens | F2 | SVG | original |
| u-noun-abstraction | Abstraction = hiding hardware behind the service | term undefined | defined at first use | F2 | SVG | original |
| u-sub-cloud | Crusoe Cloud; Zoom analogy; chips stay revenue-producing | hardware identity matters | the depreciation defense | F2 | SVG | original |
| u-sub-uplift | Midpoint $10M -> $25M -> 2.4yr; top 2yr; bottom 3yr | one payback number | the sensitivity worked | F2 | SVG | original |
| u-sub-financing | Capex 113% of cash flow; Brookings $3.7T / $6.0T; $5.5/$6.9 GPU-hr | per-factory math only | debt adds a clock | f-tab-financing | table | original (press/paper) |
| u-noun-unlevered | Unlevered = return as if all equity | term undefined | defined at first use | f-tab-financing | table | original |
| u-sub-paychap | Payback halves when the product changes | sections as sequence | consolidation + chapter plate | c2 | chapter plate | original synthesis |
| u-h-depr | Depreciation defined; $60M/6yr = $10M/yr on the books | term undefined | the whole Wall Street debate | F3 | SVG | original |
| u-sub-sixyr | Six is the standard; book life vs economic life | one number | the reframe: market decides | F3 | SVG | original |
| u-sub-h100chart | H100: fell, rose above launch; $3.28/hr, +22%/mo | old chips die | the market repriced upward | F5 | SVG | original (session/Ornn) |
| u-sub-reboundwhy | Agent boom + Blackwell rationed to largest buyers | rebound unexplained | the two forces named | F5 | SVG | original |
| u-sub-abstrdef | Abstraction stretches the depreciation curve | curve unexplained | the mechanism named | F3 | SVG | original |
| u-sub-3yrcrit | Research Affiliates: $8->$3-><$1; year 4 below cost; +$36k/-$4.4k | six years assumed | the counterweight with numbers | F3 | SVG | original (paper/Fortune) |
| u-sub-powerceil | Compute-per-watt + fixed power envelope forces swaps | wear assumed | economic obsolescence first | F3 | SVG | original |
| u-sub-maint | 2/3 of capex is maintenance; $125B net of $650B | growth story | the treadmill quantified | F3 | SVG | original |
| u-sub-bothsched | Hold both: 2yr clears either; 4yr needs the 6yr book | one schedule | the honest underwrite | F3 | SVG | original |
| u-sub-deprchap | The schedule changes return vs refund, not the cash | sections as sequence | consolidation + chapter plate | c3 | chapter plate | original synthesis |
| u-h-commodity | Commodity defined; split by age and scale | one debate | three horizons | f-tab-commodity-l02 | table | original |
| u-sub-oldcomm | Sept 2026 price war: 8 models repriced; Ramp -41% | claim without evidence | the clean evidence table | f-tab-pricewar | table | original (press) |
| u-sub-edge | Cutting edge always premiums; scarcity + perf/watt | one curve | same curve, different point | f-tab-commodity-l02 | table | original |
| u-sub-scale | Scale "absolutely not a commodity"; 100k chips as one machine | chips buyable | operations not buyable | f-tab-commodity-l02 | table | original |
| u-sub-marginpath | Margins 80% toward 60%; watch the margin | price watched | the verdict's location | f-tab-margin | table | original (session) |
| u-sub-accel | Abstraction defends old chips AND accelerates commodity | irony unstated | Crusoe wins either way | c4 | chapter plate | original synthesis |
| u-h-quan | Quan: 1,500 people, 3,500 workers, wind, undisclosed customer | Abilene one-off | the playbook repeats | f-mer-quan | mermaid | original (session) |
| u-noun-behindmeter | Behind the meter = generated and consumed on site | term undefined | defined at first use | f-mer-quan | mermaid | original |
| u-noun-acrossthemeter | Across the meter = campus nets with the grid | term undefined | defined at first use | f-mer-quan | mermaid | original |
| u-sub-quanarith | Surplus to grid lowers bills; draw when wind drops | island assumed | netted, never islanded | f-mer-quan | mermaid | original |
| u-sub-spark | Spark: 500 kW air / 2 MW liquid; Brighton factory; months | years to build | the modular answer | F4 | SVG | original (session) |
| u-noun-mwe | MWe = megawatts of electrical output | term undefined | defined at first use | F4 | SVG | original |
| u-sub-sparksave | 30-50% claimed; Energy Vault 8->25 MW; Aalo 2027; WSJ 1 GW/yr | claim unexamined | mechanism + evidence direction | F4 | SVG | original (press) |
| u-sub-sparkvi | Modularity extends vertical integration one level deeper | thesis threatened | thesis extended | c5 | chapter plate | original synthesis |
| u-h-space | Starcloud partnership [uncertain]; 100x launch cost | space as escape | the honest trade | f-tab-space-l02 | table | original (session) |
| u-sub-vanish | No concrete, permits, fiber strands, power procurement | orbit abstract | the $20M lines skipped | f-tab-space-l02 | table | original |
| u-sub-stay | Heat by radiation only; no repair; launch cost 100x | attractions only | the blockers priced | f-tab-space-l02 | table | original |
| u-sub-verdict | Not material in 5-10 years; L01 carries the race | verdict unstated | the long bet framed | c5 | chapter plate | original synthesis |
| u-h-capexrace | 2026 capex ~$730B; 2027 near $1.2T (Goldman) | chapter numbers | civilizational scale | f-tab-capexrace | table | original (press) |
| u-h-mapping | Every L01 puzzle -> this chapter's answer | puzzles open | consolidation table | f-tab-mapping | table | original |
| u-h-price | $20M snapshot; payback needs token growth; lenders need the schedule | numbers static | every number is moving | c1, c2, c3, c4, c5 | chapter plates | original synthesis |
| u-h-recap | 9-point recap of the whole lesson | lesson as sequence | one-screen consolidation | c1, c2, c3, c4, c5 | chapter plates | original synthesis |
| u-h-coveragemap | Coverage map: every session claim -> section + file line | claims scattered | full traceability table | f-tab-coveragemap | table | original |
| u-h-qa | 8 interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from their sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: 2 nocookie embeds + 6 verified links | n/a (media law unit) | video + links | n/a - media unit | video/links | session/press |
| u-h-official | Official sources and further reading with caveats | n/a (sourcing unit) | sources + honesty caveats | n/a - sourcing unit | links | session/course site |
| u-h-connections | Connections to the other courses | n/a (navigation unit) | course links | n/a - nav unit | links | course map |

## Figures kept / fixed / added, by medium
- Kept as-is (spec-compliant after l01-style fixes only): none untouched; all three old SVGs needed fixes.
- Fixed: F1 (font, palette, bar proportionality), F2 (font, added center rule arrow), F3 (font, palette).
- Replaced: F4 webp -> SVG lesson plate (F2 medium-ladder fail, same as l01 f03).
- Added: F5 h100 SVG plate; c1-c5 chapter SVG plates; f-tab-margin, f-tab-financing, f-tab-commodity-l02, f-tab-space-l02 (tables).
- Inline kept: f-ascii-20m, f-ascii-40m (<=12 lines), f-mer-quan (3 nodes), f-tab-pricewar, f-tab-capexrace, f-tab-mapping, f-tab-coveragemap.

## Prose problems noticed but NOT touched (for the coordinator)
1. ARITHMETIC SLIP: "fit-out and deployment, $4M" subchapter says "Read the $40M as $39M plus rounding, if you want the strict version." The bars sum to $30M + $4M + $3M + $3M + $1M = $41M, not $39M. Either the strict version is $41M or one bar is $1M lower than stated.
2. TWO CAPEX FRAMES: the October capex table shows ~$730B combined 2026 guidance, while the financing subchapter cites ~$800.5B projected 2026 capex (IEEE ComSoc/Brookings summary). Both are sourced, but the chapter never reconciles the $70B gap.
3. F1's in-SVG subtitle was shortened for the 960 px layout ("$20M builds the factory. $40M fills it. Half is GPUs."); the md caption still carries the full claim. Numbers unchanged.
4. u-noun-mwe (MWe defined) sits inside the Spark savings subchapter; it maps to F4, which does not spell out the acronym. The definition lives in prose only.
5. The September price-war table and the capex table are press/analyst figures, not session figures; the md captions attribute them. Sourcing verification is the builder's lane.
