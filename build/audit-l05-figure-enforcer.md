# FIGURE ENFORCER audit — mse435/l05-software-is-dead
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (plate numbering), image-reference swaps for
# replaced figures, and added inline figures (tables/ASCII/mermaid).

## Figure inventory (3 SVG lesson plates + 7 SVG chapter plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f1 | assets/plate-l05-tam.svg | SVG lesson plate | The biggest TAM expansion in software history is a demand shock for compute |
| f2 | assets/plate-l05-triangle.svg (NEW, replaces webp) | SVG lesson plate | The cloud is being rebuilt around the agent as the unit |
| f3 | assets/plate-l05-split.svg | SVG lesson plate | SaaS was a compromise, and the compromise was arithmetic; the arithmetic flipped |
| c1 | assets/plate-l05-chap-tam.svg (NEW) | SVG chapter plate | The programmer-headcount cap on cloud revenue is gone |
| c2 | assets/plate-l05-chap-deploy.svg (NEW) | SVG chapter plate | The learning only starts when a user confronts the running version |
| c3 | assets/plate-l05-chap-ladder.svg (NEW) | SVG chapter plate | The agent is the entity that ships, so the cloud must be rebuilt around it |
| c4 | assets/plate-l05-chap-triangle.svg (NEW) | SVG chapter plate | The agent-era buyer is a token budget with a goal; sell to it or be invisible to it |
| c5 | assets/plate-l05-chap-saas.svg (NEW) | SVG chapter plate | The honest version of "software is dead" is "undifferentiated software is dead" |
| c6 | assets/plate-l05-chap-pricing.svg (NEW) | SVG chapter plate | Agents buy with token budgets, not seats; price for the token budget |
| c7 | assets/plate-l05-chap-retention.svg (NEW) | SVG chapter plate | "Software is basically now free"; free drives engagement, the demand engine |
| f-tab-generations | inline table (NEW) | table | Vercel's three generations: what it unified and the proof |
| f-tab-agentshare | inline table (NEW) | table | Under 3 percent to over half; 2T to 20T tokens a month |
| f-tab-rungs | inline table (NEW) | table | The four rungs: unit of work, integrator, economics |
| f-tab-v0 | inline table (NEW) | table | v0 pricing tiers: $5 free credits, $30 team, $100 business |
| f-tab-eve | inline table (NEW) | table | The Eve fleet: six agents with measured returns |
| f-mer-agentbuy | inline mermaid block (NEW) | mermaid | Agents buy from agents: discover, negotiate, transact |
| f-ascii-compromise | inline ascii block (NEW) | ASCII | The compromise is arithmetic; the cost flips to prompts |
| f-tab-pricing | inline table (NEW) | table | Seat, usage, outcome: what each meter prices |
| f-ascii-retention | inline ascii block (NEW) | ASCII | Measure Wednesday-morning retention, not Saturday-morning generation |
| f-tab-market | inline table (NEW) | table | The coding-agent market, October 2026 |

## Fixes applied to existing figures
- f1 plate-l05-tam.svg: font stack reordered to spec (Anthropic Sans first). Four peer boxes in a row at equal height; the 216px "agents" box is the highlighted after-state. One claim. Caption already names Shell 2 and the source.
- f2 plate-l05-triangle.webp DELETED and redrawn as plate-l05-triangle.svg: the webp failed F2 (the claim is the named three-sided architecture; the ladder mandates the SVG lesson plate, per the l01/l02 webp precedent). Flat SVG in the sibling style (warm paper #F7F4EE, Anthropic Sans first, 8px grid, one claim); the triangle outline is kept because the triangle IS the named structure from the session. md references updated in l05, crash-course.md, and cheatsheet.md (webp -> svg); prose untouched.
- f3 plate-l05-split.svg: font stack reordered to spec. One claim; before/after (goes plastic / stays hard). Caption already names Shell 3 and the source.

## Medium-ladder justification for new figures
- f2 triangle: table no (the claim is a named three-sided architecture, not a value comparison); equation no; ASCII cannot carry the three labeled sides; SVG is the first medium that fully passes.
- f-tab-generations: comparison of three generations by what they unified -> table passes first.
- f-tab-agentshare: comparison of two dates by metric -> table.
- f-tab-rungs: comparison of four rungs by unit of work and economics -> table.
- f-tab-v0: comparison of pricing tiers -> table.
- f-tab-eve: comparison of six agents by measured return -> table.
- f-mer-agentbuy: the claim is the order of a transaction between agents -> mermaid passes (5 nodes, each 4 words or fewer).
- f-ascii-compromise: the cost arithmetic trace, 3 lines -> ASCII.
- f-tab-pricing: comparison of three meters by worked number -> table.
- f-ascii-retention: the survival trace, 3 lines -> ASCII.
- f-tab-market: comparison of five players by position and number -> table.
- c1-c7: chapter plates are mandated extras by the spec (dense, end of concept); SVG keeps text crisp and matches the site's plate system. Zero generated stills: the one webp was deleted, zero media-pipeline calls, zero API keys touched.

## Number verification (python3, 2026-10-06)
- 20T / 2T = 10x gateway token growth. f-tab-agentshare and c1 check.
- Rung 3: 400,000 x $2/1M = $0.80 input; 100,000 x $10/1M = $1.00 output; $1.80 total against 10 minutes of review. f-tab-rungs checks.
- Rung 2: 40 answers x $0.01 = $0.40; 40 x 30 seconds = 1,200 seconds = 20 minutes. f-tab-rungs checks.
- Seat: $20 x 1,000 = $20,000 a month. Usage: 10,000 x 5,000 = 50M tokens x $2/1M = $100. f-tab-pricing checks.
- SaaS compromise: $2M / 1,000 = $2,000 per customer. f-ascii-compromise checks.
- Lead Agent: $5,000 a year returning about 32x = about $160k in pipeline. f-tab-eve checks.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-question | If every company can generate its own tools, what happens to SaaS? | L04's specialization | the SaaS reckoning | c5 | chapter plate | original |
| u-def-saas | SaaS: software as a service | term undefined | defined at first use | c5 | chapter plate | original |
| u-h-generations | Vercel reorganized around the agent era in 2026 | one company | three generations of unification | f-tab-generations | table | original |
| u-def-devx | Developer experience was Vercel's founding obsession | term undefined | deploying took a seasoned engineer weeks | f-tab-generations | table | original |
| u-h-agentshare | Agent deployments: under 3 percent to over half of commits | no forcing number | the reorganization explained | f-tab-agentshare | table | company disclosures |
| u-def-codingagent | A coding agent writes software toward a goal using tools | term undefined | defined at first use | c3 | chapter plate | original |
| u-h-raise | August 2025 raise at about $9B, triple the previous | no valuation | the market priced the agent thesis | f-tab-generations | table | press |
| u-def-tam | TAM: everyone who could ever buy your product | term undefined | defined at first use | f1 | SVG | original |
| u-h-accessladder | Expanding access: mainframes to bootcamps | no history | each wave multiplied creators | f1 | SVG | original |
| u-h-agentsbreakcap | Agents break the cap: anyone who can describe it can create | 20M developer bet | deployments surged around October 2025 | f1 | SVG | session |
| u-def-elastic | Elastic compute: a computer per credit card, for human-written code | term undefined | defined at first use | c1 | chapter plate | original |
| u-h-elcap | Cloud demand was capped by programmer headcount; the cap is gone | bounded demand | agents write around the clock | c1 | chapter plate | original |
| u-h-deploysharp | Writing code does not make you special; deploying does | value in the artifact | value in running software in front of customers | c2 | chapter plate | session |
| u-h-deadpile | Mountains of code that never runs anywhere | no evidence | code is inventory, running software is revenue | c2 | chapter plate | original |
| u-h-agentsdeploy | Agents deploy compulsively, without the human's fear | human bias assumed | write, test, ship, iterate | c2 | chapter plate | original |
| u-h-valuemoved | "Amazon Agent Services": the shippable entity is the agent | EC2 for human code | the primitives get rebuilt | c2 | chapter plate | original |
| u-h-rung1 | Autocomplete: the model completes the line | no ladder | assistance, not agency | f-tab-rungs | table | original |
| u-h-rung2 | Chat: one answer at a time; the human integrates | no rung | $0.01 per answer; 40 answers cost $0.40 and 20 minutes | f-tab-rungs | table | original toy |
| u-h-rung3 | The agent: tools, goals, adaptation; Claude Code and Codex | no rung | one finished task: $1.80 against 10 minutes of review | f-tab-rungs | table | original toy |
| u-h-rung4 | The fleet: many agents, one project; orchestration is the constraint | no rung | the marginal programmer is an agent with a token budget | f-tab-rungs | table | original |
| u-h-triangle | Agentic infrastructure has three sides | no map | Rauch's product map | f2 | SVG | session |
| u-h-side1 | Side 1: infrastructure for coding agents; every agent-written app is a deployment customer | no side | deploy somewhere with domains, CDN, scaling | f2 | SVG | original |
| u-def-v0 | v0: Vercel's prompt-to-app product | term undefined | defined at first use | f-tab-v0 | table | original |
| u-h-v0loop | The loop closes: prompt, generate, deploy, observe, iterate | generation alone | the moat is the shortest path from prompt to a URL | f-tab-v0 | table | original |
| u-h-side2 | Side 2: ship your own agents; the school example; the support agent at 93 percent | no side | an agent can be the product, not the demo | f2 | SVG | session |
| u-h-side3 | Side 3: the self-driving cloud; the ducks anecdote | no side | configure, monitor, optimize, report back | f2 | SVG | original |
| u-h-eve | The Eve fleet: 100+ agents in production with measured returns | vision deck assumed | d0, Lead Agent 32x, Vertex 92%, Athena, V | f-tab-eve | table | press |
| u-h-agentbuy | Agents buy from agents: the buyer is a token budget with a goal | human buyers | instant signup, consumption pricing, agent-ready interfaces | f-mer-agentbuy | mermaid | original |
| u-h-compromise | One interface for the most customers: $2M, $2,000 each | no mechanism | the compromise is arithmetic | f-ascii-compromise | ASCII | original toy |
| u-h-parking | A CEO vibe-coded the parking software in v0 | no exhibit | an entire category existed only because building was not worth it | c5 | chapter plate | session |
| u-h-salesforce | Two people rebuilt Salesforce's surface for internal use | no exhibit | the data layer stayed; only the pixels went plastic | c5 | chapter plate | original |
| u-def-sor | The system of record: database, ACLs, workflows | term undefined | defined at first use | c5 | chapter plate | original |
| u-h-sorholds | The system of record stays; the presentation layer goes plastic | split assumed | SaaS keeps the record, grows an agent surface | c5 | chapter plate | original |
| u-h-seat | The seat: $20 per user per month, $20,000 for 1,000 users | no meter | the seat was insurance | f-tab-pricing | table | original toy |
| u-h-usage | Usage: $2 per million input tokens; $100 for 50M tokens | no meter | the token is a utility bill | f-tab-pricing | table | original toy |
| u-h-outcome | Outcome: the Lead Agent at $5,000 a year returning 32x | no meter | the vendor guarantees the result | f-tab-pricing | table | original |
| u-def-reflex | Reflexivity: once you know you are one prompt away, you never go back | term undefined | defined at first use; Tobi Lutke's term | c7 | chapter plate | original |
| u-h-throwaway | Custom demos per sales call, dead by Wednesday | no behavior | living software beats a slide deck | c7 | chapter plate | original |
| u-h-retention | 100 apps, 90 die in a week, 10 become workflows | generation counted | the 10 compound; at prompt prices they are worth it | f-ascii-retention | ASCII | original toy |
| u-h-reflex | "Software is basically now free"; free drives engagement | no engine | reflexivity is the demand-side engine | c7 | chapter plate | session |
| u-h-quorum | Three agents plus smart humans stare at a single line of code | prompting assumed | the plumbing stays genuinely hard | f3 | SVG | original |
| u-h-agentsupport | Customers whose agent brought them there; debug via transcript | human support | support is becoming agent-to-agent | f-tab-eve | table | original |
| u-h-market | The coding-agent market, October 2026 | no market map | Claude Code leads, Codex challenges, Cursor distributes | f-tab-market | table | press/vendor |
| u-h-mapping | Mapping back: five SaaS fears answered | fears scattered | one-screen consolidation | f-tab-market | table | original |
| u-h-price | Retention is the price: Wednesday morning, not Saturday | no price | undifferentiated software is dead | c7 | chapter plate | original |
| u-h-qa | 7 interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from their sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: 1 nocookie embed + 5 verified links | n/a (media law unit) | video + links | n/a - media unit | video/links | session/press |
| u-h-official | Official sources and further reading with caveats | n/a (sourcing unit) | sources + honesty caveats | n/a - sourcing unit | links | session/course site |
| u-h-coverage | Coverage and sourcing note (no transcript; Oct 2026 updates marked) | n/a (sourcing unit) | sourcing ground rules | n/a - sourcing unit | prose | session |
| u-h-recap | 12-point recap of the whole lesson | lesson as sequence | one-screen consolidation | c1-c7 | chapter plates | original synthesis |
| u-h-coveragemap | Coverage map: every session claim -> section + file line | claims scattered | full traceability table | n/a - sourcing unit | table | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. 93 percent versus 92 percent: the session's support agent "answers 93 percent of user inquiries"; June 2026 press on the Eve fleet reports "92 percent for the Vertex agent." Both figures are kept as stated (different agents, different sources), but the lesson never explicitly disambiguates the two. A reader could read them as one metric moving.
2. The Q&A follow-up claims "daily deployments doubled since January" as the TAM-expansion evidence; the lesson body gives "more than half of all commits" and the 2T-to-20T gateway figures. "Daily deployments doubled" appears nowhere in the body; the QA introduces a new number. Noted for the examiner (E1 risk).
3. f1's footer says "Oct 2025, strong coding models arrive" while the caveats note Opus 4.5 launched November 24, 2025 (auditor-verified) and the session said "since October last year." The plate follows the session phrasing; the caveat section carries the correction. No change made.
