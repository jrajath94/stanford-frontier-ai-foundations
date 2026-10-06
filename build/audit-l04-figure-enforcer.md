# FIGURE ENFORCER audit — mse435/l04-evals-rlvr-enterprise
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (plate numbering), image-reference swaps for
# replaced figures, and added inline figures (tables/ASCII/equations/mermaid).

## Figure inventory (4 SVG lesson plates + 8 SVG chapter plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f1 | assets/plate-l04-eval-loop.svg | SVG lesson plate | Evals define the hill; RL is the eval-maxing machine |
| f2 | assets/plate-l04-tiers.svg (NEW, replaces webp) | SVG lesson plate | Good is not universal; whose eval you optimize decides whose product you become |
| f3 | assets/plate-l04-budget.svg | SVG lesson plate | Post-training is 5 percent of the compute and buys the reasoning |
| f4 | assets/plate-l04-pareto.svg | SVG lesson plate | The frontier is not one model; the specialists are cheap |
| c1 | assets/plate-l04-chap-posttraining.svg (NEW) | SVG chapter plate | Where the reward comes from decides what the model becomes |
| c2 | assets/plate-l04-chap-evals.svg (NEW) | SVG chapter plate | The eval is the steering wheel, so getting it right is the highest-impact work |
| c3 | assets/plate-l04-chap-tiers.svg (NEW) | SVG chapter plate | The specialization layer is defensible because tier-2 evals cannot be copied |
| c4 | assets/plate-l04-chap-fivepct.svg (NEW) | SVG chapter plate | Cheap post-training prices the specialization layer; the floor is built, buy the ceiling |
| c5 | assets/plate-l04-chap-doordash.svg (NEW) | SVG chapter plate | The eval loop from this chapter's opening, made concrete on one enterprise |
| c6 | assets/plate-l04-chap-pareto.svg (NEW) | SVG chapter plate | Windsurf 2.0 is the pattern as a company, with a price tag |
| c7 | assets/plate-l04-chap-continual.svg (NEW) | SVG chapter plate | The next gains go to whoever solves the permission problem |
| c8 | assets/plate-l04-chap-datamarket.svg (NEW) | SVG chapter plate | The smarter the models get, the better the pipelines run; the flywheel eats the vendor |
| f-ascii-rl | inline ascii block (NEW) | ASCII | The policy is a hill-climber and the reward is the hill |
| f-ascii-boat | inline ascii block (NEW) | ASCII | Whatever the reward measures, the model becomes |
| f-ascii-sft | inline ascii block (NEW) | ASCII | SFT buys obedience to format and style, not new reasoning |
| f-mer-rlhf | inline mermaid block (NEW) | mermaid | RLHF: preference pairs to reward model to RL |
| f-ascii-grpo | inline ascii block (NEW) | ASCII | The update is proportional to the advantage; the baseline is free |
| f-tab-stack | inline table (NEW) | table | The post-training family: what each step teaches and what it costs |
| f-eq-eval | inline ascii equation block (NEW) | equation | eval score = tasks passed / tasks attempted |
| f-tab-variants | inline table (NEW) | table | Four eval kinds with cost and failure mode |
| f-ascii-contam | inline ascii block (NEW) | ASCII | The 20-point gap is memorization wearing a capability costume |
| f-ascii-goodhart | inline ascii block (NEW) | ASCII | The measure becomes the target; the model edits the tests |
| f-ascii-tierstack | inline ascii block | ASCII | Tier 1 is lab evals; tier 2 is enterprise evals |
| f-ascii-door | inline ascii block (NEW) | ASCII | 12 percent to 6 percent to 3 percent error; no prompt engineering |
| f-tab-opus | inline table (NEW) | table | Opus for the agent, the cheap model for the chat |
| f-ascii-composer | inline ascii block (NEW) | ASCII | Accept is 1, revert is 0; online training on huge batches |
| f-ascii-tab | inline ascii block (NEW) | ASCII | 400M requests a day; retrain every 90 minutes |
| f-tab-privacy | inline table (NEW) | table | The three bricks of the privacy wall |
| f-ascii-squeeze | inline ascii block (NEW) | ASCII | The vendor is paid to obsolete its own product line |
| f-ascii-gap | inline ascii block (NEW) | ASCII | 12 verified examples for about $1.10; the ratio is the point |
| f-tab-longshort-l04 | inline table (NEW) | table | The Nvidia long, the data-market caution, and the in-house risk |
| f-tab-enterprise | inline table (NEW) | table | Enterprise AI winners, October 2026 |

## Fixes applied to existing figures
- f1 plate-l04-eval-loop.svg: font stack reordered to spec (Anthropic Sans first); off-palette #F3D4D8 (the trap box) replaced with spec #F4E6D4 (same fix as l01/l02). One claim; before -> rule -> after with named arrows. Caption already names Shell 3 and the source.
- f2 plate-l04-tiers.webp DELETED and redrawn as plate-l04-tiers.svg: the webp failed F2 (the claim is a layering change, tier 1 -> specialization -> tier 2; the ladder mandates the SVG lesson plate for state changes, per the l01/l02 webp precedent). Flat SVG in the sibling style (warm paper #F7F4EE, Anthropic Sans first, 8px grid, one claim, before -> rule -> after). All content from the lesson (JPMorgan versus Goldman, floor versus ceiling, proprietary by construction). md reference updated (webp -> svg); prose untouched.
- f3 plate-l04-budget.svg: font stack reordered to spec. Bar widths verified: 820/864 = 94.9% pre-training, 44/864 = 5.1% post-training. Caption names Shell 2 and the DeepSeek source.
- f4 plate-l04-pareto.svg: font stack reordered to spec; off-palette #F6E7A8 (proprietary-data box) replaced with spec #F4E6D4. "100x cheaper, 5x faster" verified: 0.02/0.0002 = 100; 8/1.5 = 5.33, about 5x.

## Medium-ladder justification for new figures
- f2 tiers: table no (the claim is a layering change, not a value comparison); equation no; ASCII cannot carry the tier panels with the named center rule; mermaid shows order, not the layering; SVG is the first medium that fully passes. Lesson plate per the change-unit rule.
- f-ascii-rl: one trace, before/after lines, 4 lines -> ASCII passes first.
- f-ascii-boat: one trace of the intended/reward/result failure, 4 lines -> ASCII.
- f-ascii-sft: one trace of the worked toy, 4 lines -> ASCII.
- f-mer-rlhf: the claim is the order of the three steps -> mermaid passes (5 nodes, each 4 words or fewer).
- f-ascii-grpo: the arithmetic of one group of 8 attempts, 4 lines -> ASCII.
- f-tab-stack: comparison of values (what each step teaches / costs) -> table passes first.
- f-eq-eval: the claim is a definition (score = passed/attempted) -> equation passes first.
- f-tab-variants: comparison of four kinds by cost and failure mode -> table.
- f-ascii-contam: the toy trace with the reported/true/gap numbers, 4 lines -> ASCII.
- f-ascii-goodhart: before/rule/after trace, 3 lines -> ASCII.
- f-ascii-door: error-rate trace, 3 lines -> ASCII.
- f-tab-opus: comparison of two routing decisions -> table.
- f-ascii-composer: accept/revert trace, 3 lines -> ASCII.
- f-ascii-tab: throughput trace, 3 lines -> ASCII.
- f-tab-privacy: comparison of three obstacles -> table.
- f-ascii-squeeze: cost trace, 3 lines -> ASCII.
- f-ascii-gap: cost arithmetic trace, 4 lines -> ASCII.
- f-tab-longshort-l04: comparison of positions with numbers -> table.
- f-tab-enterprise: comparison of vendors with numbers -> table.
- c1-c8: chapter plates are mandated extras by the spec (dense, end of concept); SVG keeps text crisp and matches the site's plate system. Zero generated stills: the one webp was deleted, zero media-pipeline calls, zero API keys touched.

## Number verification (python3, 2026-10-06)
- 150,000 / 2,664,000 = 0.0563 -> about 5 to 6 percent; the guest rounds to 5 percent. f3 and c4 check.
- GRPO: scores sum to 2/8 = 0.25 average; advantage +0.75 / -0.25. f-ascii-grpo checks.
- 100,000 / 365 = 273.97 -> about 274 merchants a day. c5 checks.
- 400,000,000 / 86,400 = 4,629.6 -> about 4,600 requests a second. f-ascii-tab checks.
- Synthetic: 100 x $0.01 = $1.00; 100 x $0.001 = $0.10; total about $1.10 for 12 examples vs $5 to $20 each human-made: roughly 60x to 220x cheaper. f-ascii-gap checks.
- Pareto: 0.02/0.0002 = 100x cheaper; 8/1.5 = 5.33x faster. f4 checks.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | Models are brilliant generalists that know nothing about your business | frontier generalist | specialization layer per enterprise | c3 | chapter plate | original synthesis |
| u-def-pretraining | Pre-training reads trillions of tokens and learns next-token prediction | term undefined | defined; builds the representations | f3 | SVG | original |
| u-def-posttraining | Post-training is everything after, turning the predictor into a product | term undefined | defined; the guest's economics | f3 | SVG | original |
| u-def-rl | RL means learning by trial and reward | term undefined | agent, action, reward, policy defined | f-ascii-rl | ASCII | original |
| u-def-agent | The agent is the learner, here the model | term undefined | defined at first use | f-ascii-rl | ASCII | original |
| u-def-action | An action is what the agent does, here writing an answer | term undefined | defined at first use | f-ascii-rl | ASCII | original |
| u-def-reward | A reward is a number the world returns, 1 or 0 | term undefined | defined; the reward is the hill | f-ascii-boat | ASCII | original |
| u-def-policy | The policy is the agent's strategy for picking the next token | term undefined | defined at first use | f-ascii-rl | ASCII | original |
| u-h-rltoy | 100 math questions: 40 right, next round 55 | no mechanism | the nudge toward the 40, away from the 60 | f-ascii-rl | ASCII | original toy |
| u-h-boathack | Reward the targets, get endless circles | intended goal: finish fast | tens of thousands of points, never finishes | f-ascii-boat | ASCII | original toy |
| u-h-rewardeval | The reward is the eval score; the model is the boat | reward abstract | eval score = the hill the model climbs | f1 | SVG | original |
| u-def-sft | SFT teaches by example, 10,000 to 100,000 pairs | term undefined | defined; imitate token by token | f-ascii-sft | ASCII | original |
| u-def-loss | The loss is the gap between the model's words and the example's | term undefined | defined at first use | f-ascii-sft | ASCII | original |
| u-h-sfttoy | Two-line summary toy across 50,000 examples | no mechanism | the loss counts the extra two lines | f-ascii-sft | ASCII | original toy |
| u-def-rlhf | RLHF learns from human feedback, three steps | term undefined | preference pairs -> reward model -> RL | f-mer-rlhf | mermaid | original |
| u-def-prefpair | A preference pair is two answers with one judged better | term undefined | defined at first use | f-mer-rlhf | mermaid | original |
| u-def-chosen | Chosen is the better answer; rejected is the other | term undefined | defined at first use | f-mer-rlhf | mermaid | original |
| u-def-rewardmodel | The reward model predicts which answer a human would prefer | term undefined | learns the taste | f-mer-rlhf | mermaid | original |
| u-h-rlhfsteps | The three-step mechanism | SFT's limit stated | steps named in order | f-mer-rlhf | mermaid | original |
| u-h-rlhfprice | The reward model is a proxy; agrees 70 to 75 percent | RLHF praised | the 25 to 30 percent where the policy games the proxy | c1 | chapter plate | original |
| u-def-rlvr | RLVR uses verifiable rewards: a rule decides pass or fail | term undefined | defined; the DeepSeek R1 recipe | c1 | chapter plate | original |
| u-h-rlvrtoy | 1,000 problems, 8 attempts, 8,000 graded automatically | no mechanism | the policy climbs the verifier; scales with compute | c1 | chapter plate | original |
| u-def-grpo | GRPO needs no separate value network; the group average is the baseline | term undefined | defined; from the DeepSeek R1 paper | f-ascii-grpo | ASCII | original |
| u-h-grpoarith | Eight attempts: average 0.25, advantages +0.75 / -0.25 | no arithmetic | the comparison inside the group | f-ascii-grpo | ASCII | original toy |
| u-def-distillation | Distillation copies the reasoning into a smaller model | term undefined | defined; most capability at a fraction of serving cost | f-tab-stack | table | original |
| u-h-stack | The family in order, each answering a named pain | four bare names | what each teaches and costs | f-tab-stack | table | original |
| u-h-5pctstack | The guest's 5 percent is mostly the RLVR step | step unnamed | RLVR and distillation are the startup steps | f3 | SVG | original |
| u-def-eval | An eval is a benchmark: fixed tasks with right answers | term undefined | defined; score = passed / attempted | f-eq-eval | equation | original |
| u-h-evalclaim | Evals are the most protected asset; evals set the roadmap | no strategy | whoever writes the eval writes everyone's roadmap | f1 | SVG | session |
| u-h-swebench | SWE-bench: real GitHub issues, pass/fail by repo tests | term undefined | defined; started the code-model race | f-eq-eval | equation | original |
| u-h-evalloop | Define the hill with an eval; RL is the eval-maxing machine | no loop | define, climb on different data, pick the next hill | f1 | SVG | session |
| u-h-guards | A public eval lets competitors aim at the same hill | no defense | publish the hills you want competitors climbing | c2 | chapter plate | original |
| u-h-goodhart | The model games the proxy: edits the tests | no failure mode | score climbs, product rots | f-ascii-goodhart | ASCII | original toy |
| u-def-static | Static benchmarks: fixed questions with fixed answers | term undefined | MMLU: 15,908 questions, 57 subjects | f-tab-variants | table | original |
| u-def-mmlu | MMLU is the canonical static benchmark | term undefined | defined at first use | f-tab-variants | table | original |
| u-def-agentic | Agentic evals: the model acts in an environment | term undefined | SWE-bench: the environment is the test | f-tab-variants | table | original |
| u-def-humaneval | Human evals: people rank outputs, the gold standard for taste | term undefined | $1 to $5 per comparison | f-tab-variants | table | original |
| u-def-llmjudge | LLM-as-judge: a strong model scores, fast and cheap, biased | term undefined | a few cents per judgment | f-tab-variants | table | original |
| u-h-evalkindclaim | The kind of eval decides the kind of model you get | four kinds look equal | memorizers, tool users, pleasantness, judge-pleasers | f-tab-variants | table | original |
| u-def-contamination | Contamination means the test leaked into the training data | term undefined | defined; the internet is the training set | f-ascii-contam | ASCII | original |
| u-h-contamtoy | 1,000 questions, 200 leaked: reported 85 percent, true 65 | no failure math | the 20-point gap is memorization | f-ascii-contam | ASCII | original toy |
| u-h-contamdefenses | Private held-out sets, dynamic evals, canary strings | no defense | none free; all cheaper than shipping a lying 85 percent | f-ascii-contam | ASCII | original |
| u-def-tier | Evals stack in tiers: lab evals versus enterprise evals | term undefined | defined at first use | f-ascii-tierstack | ASCII | original |
| u-h-tierstack | Tier 1 shared, tier 2 private; JPMorgan's good is not Goldman's | one good assumed | same base model, two products | f2 | SVG | original |
| u-h-tiercecon | Applied Compute lives at tier 2; the ceiling is proprietary | no economics | the general model sets the floor | c3 | chapter plate | original |
| u-def-h800 | H800 is Nvidia's export-compliant GPU for China | term undefined | defined at first use | f3 | SVG | original |
| u-h-5pct | 150,000 / 2,664,000 = about 5 percent | no arithmetic | $95 builds representations, $5 buys reasoning | f3 | SVG | session/report |
| u-h-read1 | Post-training is shockingly cheap for what it buys | the 5 percent unexplained | single-digit percent for the defining behavior | c4 | chapter plate | session |
| u-h-read2 | The share is rising: data-center-wide RL, its own scaling laws | the 5 percent static | a snapshot of a growing share | c4 | chapter plate | session |
| u-h-95floor | The 5 percent only works on top of the 95 | cheap looks free | pre-training is the fixed cost of the floor | c4 | chapter plate | original |
| u-h-doordash | DoorDash: 100,000+ merchants a year, strict style guide | no case study | the chapter's central case | c5 | chapter plate | original |
| u-h-ddquant | 100,000 merchants a year is about 274 a day | no volume math | a human-labeled pipeline at that volume is a small factory | f-ascii-door | ASCII | original |
| u-def-vlm | VLM: a transformer that reads images and text together | term undefined | defined at first use | c5 | chapter plate | original |
| u-h-ddloop | Humans correct menus; the error rate is the reward; RL climbs | general models failed | 12 percent to 6 percent to 3 percent | f-ascii-door | ASCII | original |
| u-h-ddwait | Why not wait for GPT-17: ROI now beats waiting years | waiting looks free | no general model will ever know the style guide | c5 | chapter plate | original |
| u-h-pareto | General 90%/$0.02/8s versus specialist 91%/$0.0002/1.5s | one model assumed | 100x cheaper, 5x faster, a point more accurate | f4 | SVG | original toy |
| u-h-ensemble | Orchestrators plus specialists plus proprietary data | one model assumed | the production pattern | f4 | SVG | original |
| u-h-ramp | Ramp Labs: RL for fast search inside spreadsheets | term undefined | the same pattern, a mention | f4 | SVG | original |
| u-h-windsurf | Windsurf 2.0: acquisition, SWE-1.5, $492M revenue, $26B valuation | no verified history | the Pareto argument as a company | c6 | chapter plate | press |
| u-h-opus54 | 54 percent of Claude Code sessions on Opus versus 10 percent of chat | no routing number | spend the expensive intelligence where the task is agentic | f-tab-opus | table | press analysis |
| u-def-continual | Continual learning: a deployed model that improves from real-world use | term undefined | the hot-stove problem | c7 | chapter plate | original |
| u-h-composer | Cursor's Composer: accept/revert as implicit rewards, hours per step | no mechanism | collect data, take a step, repeat | f-ascii-composer | ASCII | original |
| u-h-contextbases | Context bases: agents extract learnings at the same token budget | term undefined | weight updates plus context plus harness | c7 | chapter plate | original |
| u-h-tabloop | The Tab model retrains every 90 minutes on 400M requests a day | no fast loop | about 4,600 requests a second; the reward is a firehose | f-ascii-tab | ASCII | engineering analysis |
| u-h-privacy | The privacy wall: consent, privacy mode, scrubbing cost | no blocker | the permission problem, not the gradient problem | f-tab-privacy | table | original |
| u-def-squeeze | The data-market squeeze: success obsoletes the product line | term undefined | task 1 $10k/week, task 5 $100k/month | f-ascii-squeeze | ASCII | original toy |
| u-def-synthetic | Synthetic data: training data generated by models | term undefined | defined at first use | f-ascii-gap | ASCII | original |
| u-def-gap | The generator-verifier gap: generating is hard, checking is cheap | term undefined | defined at first use | f-ascii-gap | ASCII | original |
| u-h-gapcode | Code worked: 12 verified examples for about $1.10 | no arithmetic | roughly 60x to 220x cheaper per example | f-ascii-gap | ASCII | original toy |
| u-h-gapmath | Math worked: 1,000 problems, 8 attempts, 8,000 graded | no arithmetic | no human reads a single solution | c1 | chapter plate | original toy |
| u-h-flywheel | The synthetic-data flywheel compounds without new labels | no accelerator | smarter models make better synthetic data | c8 | chapter plate | original |
| u-h-nvidia | Long Nvidia at 75 percent margins; the in-house-silicon risk | no position | 80 percent as effective per chip, but far more chips | f-tab-longshort-l04 | table | session |
| u-h-usedwhere | Enterprise AI, October 2026: Anthropic leads models and spend | no market map | the model layer and the platform layer have different winners | f-tab-enterprise | table | press/vendor |
| u-h-mapping | Mapping back: six enterprise puzzles answered | answers scattered | one-screen consolidation | f-tab-enterprise | table | original |
| u-h-price | The squeeze is structural: specialization is a treadmill | no price | the ceiling is proprietary | c8 | chapter plate | original |
| u-h-qa | 7 interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from their sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: 1 nocookie embed + 6 verified links | n/a (media law unit) | video + links | n/a - media unit | video/links | session/press |
| u-h-official | Official sources and further reading with caveats | n/a (sourcing unit) | sources + honesty caveats | n/a - sourcing unit | links | session/course site |
| u-h-coverage | Coverage and sourcing note (no transcript; Oct 2026 updates marked) | n/a (sourcing unit) | sourcing ground rules | n/a - sourcing unit | prose | session |
| u-h-recap | 10-point recap of the whole lesson | lesson as sequence | one-screen consolidation | c1-c8 | chapter plates | original synthesis |
| u-h-coveragemap | Coverage map: every session claim -> section + file line | claims scattered | full traceability table | n/a - sourcing unit | table | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. Contamination toy arithmetic: 1,000 questions, 200 leaked verbatim, "reported 85 percent, true capability 65 percent, the 20-point gap." Worked strictly, 200 memorized at 100% plus 800 unseen at 65% gives 72%, not 85%. The toy's numbers do not reconcile; the 85/65/20 figures are kept as stated. The figure quotes the prose, it does not re-derive it.
2. The Q&A follow-up cites "about 31 percent of tracked agent deployments" for Copilot Studio; the lesson body gives absolute counts (400,000 agents, 160,000 organizations). Different metrics appear in the two places; noted for the examiner.
3. The inline ascii tier block (tier 1 / tier 2) predates this pass and duplicates f2's claim at a lower fidelity; kept as-is (prose untouched), the plate is the audit's figure of record for the unit.
