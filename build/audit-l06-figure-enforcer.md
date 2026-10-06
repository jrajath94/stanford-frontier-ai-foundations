# FIGURE ENFORCER audit — mse435/l06-where-value-accrues
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (plate numbering), image-reference swaps for
# replaced figures, and added inline figures (tables/ASCII/equations/mermaid).

## Figure inventory (4 SVG lesson plates + 6 SVG chapter plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f1 | assets/plate-l06-seats-tokens.svg | SVG lesson plate | The meter moved from the user to the unit of cognition; pricing follows the meter |
| f2 | assets/plate-l06-gateway.svg (NEW, replaces webp) | SVG lesson plate | Semantic caching stands down 300 GPUs for a trivial request; routing is the business |
| f3 | assets/plate-l06-sandbox.svg (NEW, replaces webp) | SVG lesson plate | Models plus computers beat models alone, like hires plus laptops |
| f4 | assets/plate-l06-accrual.svg | SVG lesson plate | In 2026 value sits below the model; the durable skill is the method, not the map |
| c1 | assets/plate-l06-chap-accrual.svg (NEW) | SVG chapter plate | The durable skill is the method, not the 2026 answer |
| c2 | assets/plate-l06-chap-tokens.svg (NEW) | SVG chapter plate | Pricing follows the meter; the meter is now the token |
| c3 | assets/plate-l06-chap-gateway.svg (NEW) | SVG chapter plate | The gateway measures the commoditization in real time |
| c4 | assets/plate-l06-chap-sandbox.svg (NEW) | SVG chapter plate | The security must live where the computer is created |
| c5 | assets/plate-l06-chap-blocks.svg (NEW) | SVG chapter plate | In the block economy, composability is money |
| c6 | assets/plate-l06-chap-bets.svg (NEW) | SVG chapter plate | Bet on layers where demand grows faster than supply can respond |
| f-tab-forces | inline table (NEW) | table | The three forces with their numbers |
| f-tab-capex | inline table (NEW) | table | The financing wall, 2026 figures |
| f-tab-demand | inline table (NEW) | table | Hyperscaler guidance: about $730B across the four plus Oracle |
| f-tab-pricewar | inline table (NEW) | table | The September 22 price war: before, after, size |
| f-eq-5x | inline ascii equation block (NEW) | equation | Output tokens cost 5x input tokens |
| f-tab-discounts | inline table (NEW) | table | Batch, cache, off-peak: the discount stack worked |
| f-tab-cache | inline table (NEW) | table | Exact caching versus semantic caching |
| f-tab-gatewayidx | inline table (NEW) | table | The August 2026 index: token share versus spend share |
| f-ascii-docker | inline ascii block (NEW) | ASCII | Hand the model a computer; it learns by doing |
| f-ascii-sandboxsec | inline ascii block (NEW) | ASCII | Every new sandbox is an attack surface |
| f-ascii-local | inline ascii block (NEW) | ASCII | 200 tokens to modify versus 50,000 |
| f-ascii-86 | inline ascii block (NEW) | ASCII | 86 out of 86: agents vote for the blocks they know |
| f-tab-spark | inline table (NEW) | table | The Muse Spark pivot: releases and dates |
| f-tab-bets | inline table (NEW) | table | The long/short positions with their numbers |
| f-mer-method | inline mermaid block (NEW) | mermaid | Find the bottleneck, price the unit, follow the margin |
| f-tab-method | inline table (NEW) | table | The framework worked on sandbox security |
| f-tab-ladder | prose markdown table (pre-existing) | table | The October 2026 price ladder: $0.50 to $50 per million output |

## Fixes applied to existing figures
- f1 plate-l06-seats-tokens.svg: font stack reordered to spec (Anthropic Sans first). One claim; before/after (the seat / the unit). Caption already names Shell 3 and the source.
- f2 plate-l06-gateway.webp DELETED and redrawn as plate-l06-gateway.svg: the webp failed F2 (the claim is a routing change, naive -> semantic cache -> small model; the ladder mandates the SVG lesson plate for state changes, per the l01/l02 webp precedent). Flat SVG in the sibling style (warm paper #F7F4EE, Anthropic Sans first, 8px grid, one claim, before -> rule -> after). The $990/day margin arithmetic is kept verbatim from the lesson. md reference updated (webp -> svg); prose untouched.
- f3 plate-l06-sandbox.webp DELETED and redrawn as plate-l06-sandbox.svg: same F2 failure (before: EC2, a computer per credit card; rule: the unit changes from page to agent; after: ephemeral computer per task). The dashed "provisioned, used, destroyed" box is kept: dashed means ephemeral per the spec (hatch = absent, released, or not yet created). md reference updated (webp -> svg); prose untouched.
- f4 plate-l06-accrual.svg: font stack reordered to spec; off-palette #F6E7A8 (model-labs box) and #F3D4D8 (short box) replaced with spec #F4E6D4 (same fix as l01/l02). One claim; the right panel names what moves the value.

## Medium-ladder justification for new figures
- f2 gateway: table no (the claim is a routing change, not a value comparison); equation no; ASCII cannot carry the two-path diagram with the margin arithmetic; mermaid shows order, not the before/after cost states; SVG is the first medium that fully passes. Lesson plate per the change-unit rule.
- f3 sandbox: table no (the claim is a unit change); SVG passes first. Lesson plate per the change-unit rule.
- f-tab-forces: comparison of three forces by number -> table passes first.
- f-tab-capex: comparison of analyst figures -> table.
- f-tab-demand: comparison of players by guidance -> table.
- f-tab-pricewar: comparison of cuts by size -> table.
- f-eq-5x: the claim is a rule about prices (output = 5 x input) -> equation passes first.
- f-tab-discounts: comparison of three discounts by worked example -> table.
- f-tab-cache: comparison of two caching levels -> table.
- f-tab-gatewayidx: comparison of labs by token share versus spend share -> table.
- f-ascii-docker: before/rule/after trace, 3 lines -> ASCII.
- f-ascii-sandboxsec: era comparison trace, 3 lines -> ASCII.
- f-ascii-local: token-count comparison trace, 3 lines -> ASCII.
- f-ascii-86: result trace, 2 lines -> ASCII.
- f-tab-spark: comparison of releases by date -> table.
- f-tab-bets: comparison of four positions -> table.
- f-mer-method: the claim is the ordered three-step procedure -> mermaid passes (4 nodes, each 4 words or fewer).
- f-tab-method: the three steps with a number each -> table.
- c1-c6: chapter plates are mandated extras by the spec (dense, end of concept); SVG keeps text crisp and matches the site's plate system. Zero generated stills: the two webps were deleted, zero media-pipeline calls, zero API keys touched.

## Number verification (python3, 2026-10-06)
- Hyperscaler sum: 210 + 200 + 137.5 + 175 = 722.5, about $730B across the four plus Oracle's ~$70B. f-tab-demand checks.
- Opus 5.5 $4/$20 vs Opus 5.0 $5/$25: 20 percent cut; cache reads $4 -> $0.20 = 20x cut. f-tab-pricewar checks.
- GPT-6 Sol $1.06 vs $1.99: 46.7 percent less, "about 50 percent." Luna $0.07 vs $0.18: 61 percent less. f-tab-pricewar checks.
- Batch Astra $10/$50 -> $5/$25. f-tab-discounts checks.
- Thanks margin: 20M tokens x $50/M = $1,000; x $0.50/M = $10; the gateway keeps $990. f2 checks.
- 5x rule: 1->5, 2->10, 4->20, 10->50, 0.10->0.50, 0.30->1.20 all check. Grok 4.7 $2/$6 is the one ladder row that breaks the rule; see prose problem 1.
- Local reasoning: 50,000 / 200 = 250x. f-ascii-local checks.
- Support arithmetic: $20 x 1,000 seats = $20,000; 50M tokens x $2/M = $100. f1 and f-tab-method contexts check.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-coreq | From chips to agents: where will value accrue? | no question | the course's core question stated | c1 | chapter plate | original |
| u-h-below | In 2026, right this second, value concentrates below the model | no dating | chips, data centers, power, cooling, energy | c1 | chapter plate | session |
| u-h-capexmoat | $60M per MW, $60B per gigawatt; the financing wall | moat as slogan | $785B spending, 94% of cash flow, $240B debt | f-tab-capex | table | session/analysts |
| u-h-demand | Demand outruns supply: about $730B guidance, $514B backlog | scarcity asserted | player-by-player numbers | f-tab-demand | table | earnings |
| u-h-commod | Models commoditize faster than concrete | no mechanism | -41% in six months; open source erodes | f-tab-forces | table | session/index |
| u-def-token | A token is a chunk of text, roughly a word or part of one | term undefined | defined at first use | c2 | chapter plate | original |
| u-h-seattotoken | $20 per seat per month versus $2 per million input tokens | no arithmetic | $20,000 a month versus $100 of input | f1 | SVG | original toy |
| u-h-ladder | The October 2026 price ladder, $0.50 to $50 per million output | no ladder | a hundredfold spread; the workhorse tier converged at $2 | f-tab-ladder | table | vendor pricing |
| u-h-pricewar | September 22, 2026: the ninety-minute price war | no event | OpenAI and Anthropic cut within ninety minutes | f-tab-pricewar | table | press |
| u-h-5x | Output is priced at exactly five times input | no rule | the price of sequentiality; shrink the output first | f-eq-5x | equation | original |
| u-h-discounts | Batch halves, cache reads cost 20x less, DeepSeek halves off-peak | list prices assumed | the realized price is a fraction of the list | f-tab-discounts | table | vendor pricing |
| u-def-cdn | CDNs scaled and secured the delivery of pixels | term undefined | Akamai and Fastly, 1998 | c3 | chapter plate | original |
| u-def-gateway | The AI gateway is a CDN for tokens | term undefined | observes, fails over, caches, balances | f2 | SVG | original |
| u-h-analogy | Tokens need the same treatment pixels got | no analogy | the token CDN does for cognition what Akamai did for images | f2 | SVG | original |
| u-def-semcache | Semantic caching: caching by meaning, not by exact text | term undefined | "thanks" = "thank you" | f2 | SVG | original |
| u-h-semtoy | Saying "thanks" to a flagship activates something like 300 GPUs | no arithmetic | the gateway routes to a small model instead | f2 | SVG | session |
| u-h-cachelist | Exact caching versus semantic caching, with the price list | no levels | cache-read rates and the $990-a-day margin | f-tab-cache | table | original |
| u-h-reuse | The gateway reuses about 95 percent of the CDN machinery | no reuse | the same rocket engine, new fuel | c3 | chapter plate | session |
| u-h-gwindex | The August 2026 index: 56 percent of tokens, 14 percent of spend | no measurement | the barbell; the price fell 23.2 percent in August | f-tab-gatewayidx | table | Vercel index |
| u-def-ec2 | EC2 taught the world elastic compute: a computer per credit card | term undefined | designed for human-written code | c4 | chapter plate | original |
| u-def-sandbox | The sandbox is EC2 for agents: an ephemeral computer per task | term undefined | defined at first use | f3 | SVG | original |
| u-h-docker | In post-training, models learn inside Docker containers | no precedent | like a new hire with a laptop | f-ascii-docker | ASCII | original |
| u-h-sandboxecon | Marginal cost of one more sandbox: near zero | no economics | Vercel tripled in months on existing machinery | c4 | chapter plate | original |
| u-h-sandboxsec | Agents bring exfiltration and leakage, like PCs brought viruses | no corollary | isolation, permissions, monitoring: a category born in real time | f-ascii-sandboxsec | ASCII | original |
| u-def-blockecon | The block economy: the market in composable building blocks | term undefined | Mitchell Hashimoto's term | c5 | chapter plate | original |
| u-h-localreason | Local reasoning: 200 tokens versus the whole 50,000-token file | no mechanism | the agent picks the block it can reason about | f-ascii-local | ASCII | original |
| u-def-ctxwindow | The context window holds about a million tokens | term undefined | not nearly enough for all of humanity's code | f-ascii-local | ASCII | original |
| u-h-86 | Claude chose Vercel 86 out of 86 runs; 90.1 percent near monopoly | no statistic | agents vote for the blocks they know | f-ascii-86 | ASCII | cited report |
| u-h-ergonomics | Minimize the tokens an agent needs to use your block correctly | no rule | open infrastructure wins | c5 | chapter plate | original |
| u-def-mcp | MCP: the open standard for connecting agents to tools and data | term undefined | defined at first use | c6 | chapter plate | original |
| u-h-rauchlong | Rauch's long: anyone moving at the speed of tokens | no position | consumption pricing, instant signup, agent-ready interfaces | f-tab-bets | table | session |
| u-h-ratelimit | "No more rate limits": the guesses keep being wrong | no provocation | a YC company's volumes surprise Vercel itself | f-tab-bets | table | session |
| u-h-rauchshort | Rauch's short: static content, code-is-scarce builders, closed firms | no position | models answer for free | f-tab-bets | table | session |
| u-h-lochlong | Lochmiller's long: long the buildout | no position | the $60B per gigawatt moat | f-tab-bets | table | session |
| u-h-lochshort | Lochmiller's short: open source taking share from closed models | no position | old models commoditize | f-tab-bets | table | session |
| u-h-octbets | Anthropic's lead, the price war, and Meta's pivot vindicate the bets | bets undated | three October 2026 data points | f-tab-bets | table | press |
| u-h-spark | Muse Spark: Meta went closed at the frontier, April 2026 | no pivot | 1.1, 1.2, 1.3; Glimmer open; Muse Code; Contributor Endpoint | f-tab-spark | table | press |
| u-h-accrualmap | The value accrual map, dated 2026 | no map | where value sits and what moves it | f4 | SVG | session |
| u-h-method | Find the bottleneck, price the unit, follow the margin | no method | the whole course in one line | f-mer-method | mermaid | session |
| u-h-framework | The three-step interview framework | no procedure | bottleneck with a number, unit in dollars, margin | f-mer-method | mermaid | original |
| u-def-batchapi | Batch APIs: non-urgent requests processed in bulk, 50 percent off | term undefined | defined at first use | f-tab-discounts | table | original |
| u-def-promptcache | Prompt caching: repeated context billed at cache-read rates, 10x cut | term undefined | defined at first use | f-tab-discounts | table | original |
| u-h-sandboxfw | The framework worked on sandbox security | no worked example | three steps, one number each | f-tab-method | table | original |
| u-h-qa | 7 interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from their sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: 1 nocookie embed + 4 verified links | n/a (media law unit) | video + links | n/a - media unit | video/links | session/press |
| u-h-official | Official sources and further reading with caveats | n/a (sourcing unit) | sources + honesty caveats | n/a - sourcing unit | links | session/course site |
| u-h-coverage | Coverage and sourcing note (no transcript; Oct 2026 updates marked) | n/a (sourcing unit) | sourcing ground rules | n/a - sourcing unit | prose | session |
| u-h-recap | 10-point recap of the whole lesson | lesson as sequence | one-screen consolidation | c1-c6 | chapter plates | original synthesis |
| u-h-coveragemap | Coverage map: every session claim -> section + file line | claims scattered | full traceability table | n/a - sourcing unit | table | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. The five-times rule asserts "output is priced at exactly five times input on every model," but the lesson's own ladder lists xAI Grok 4.7 at $2/$6, which is 3x, not 5x. The rule contradicts the ladder's Grok row. The equation figure states the rule as the prose claims it (with the ladder rows that do fit); the prose needs a qualifier or the Grok row needs re-checking. For the figure auditor: f-eq-5x is prose-faithful but the prose is internally inconsistent on this point.
2. The f4 accrual plate's "data centers, GPUs" row says "2-4yr payback." No "2-4 year" payback figure appears in the L06 body text (the $30M figure comes from L02). The plate predates this pass; the number is not sourced in this lesson. Flagged for the figure auditor (F6: every number code-computed or from the lesson).
3. The f2 gateway plate and its caption present the 300-GPU "thanks" as a metered number; the lesson's own caveats call it "his illustrative meme, not a metered measurement." The plate is prose-faithful but the caveat lives only in the "Caveats from these sources" section.
4. The f-tab-ladder prose table marks Gemini 4 Argon $2/$10 as [uncertain]; kept verbatim with the marker (honesty law). No change.
