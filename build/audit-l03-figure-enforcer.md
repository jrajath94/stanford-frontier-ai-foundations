# FIGURE ENFORCER audit — mse435/l03-how-models-get-smarter
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions, the webp->table/svg swaps, and added inline
# figures (tables/ASCII/equations).

## Figure inventory (7 SVG plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| F1 | assets/plate-l03-eras.svg | SVG lesson plate | Progress is a moving bottleneck; each era's constraint becomes the next era's solved problem |
| F2 | inline table f-tab-axes (replaces webp) | table | Three dials buy intelligence: train bigger, steer longer, think harder per answer |
| F2b | assets/plate-l03-axes.svg (NEW) | SVG lesson plate | Same three dials, as a plate for the crash-course/cheatsheet image slots |
| F3 | assets/plate-l03-bottleneck.svg | SVG lesson plate | Whoever cracks learning from sparse real-world reward owns the next era |
| c1 | assets/plate-l03-chap-scaling.svg (NEW) | SVG chapter plate | Science gave investors a curve; finance is now climbing it |
| c2 | assets/plate-l03-chap-steering.svg (NEW) | SVG chapter plate | Pre-training built the brain; RLHF built the assistant |
| c3 | assets/plate-l03-chap-reasoning.svg (NEW) | SVG chapter plate | The new axis buys intelligence without more pre-training data |
| c4 | assets/plate-l03-chap-datawall.svg (NEW) | SVG chapter plate | Whoever owns the verifier owns the domain |
| f-ascii-attn | inline ascii block | ASCII | Self-attention: frame pulls counselor + helped in one step |
| f-ascii-backprop | inline ascii block (NEW) | ASCII | Backprop: blame flows backward; a trillion steps is knowledge |
| f-ascii-compress | inline ascii block (NEW) | ASCII | Pre-training compresses trillions of tokens into billions of weights, ~1000:1 |
| f-ascii-rlhf | inline ascii block (NEW) | ASCII | RLHF: human rankings -> reward model -> RL maximizes the score |
| f-ascii-testtime | inline ascii block (NEW) | ASCII | Test-time compute: 100 tokens wrong, 10,000 tokens right |
| f-ascii-aha | inline ascii block (NEW) | ASCII | R1-Zero: pure RL selects backtracking; the aha moment emerges |
| f-ascii-grpo | inline ascii block (NEW) | ASCII | GRPO: advantages from grouped rollouts, no critic network |
| f-mer-agent | inline mermaid block | mermaid | The agent loop: plan -> act -> observe -> correct |
| f-eq-attncost | inline equation block (NEW) | equation | Attention work grows as the square of the sequence length |
| f-eq-agentcost | inline equation block (NEW) | equation | Agent price = test-time tokens per step x number of steps |
| f-eq-deepseek | inline equation block (NEW) | equation | R1 RL ~$294k is ~5% of the $5.576M pre-training run |
| f-tab-attnvariants | inline table (NEW) | table | The three attention flavors and their rules |
| f-tab-chinchilla | inline table (NEW) | table | Chinchilla's 20-tokens-per-parameter rule, worked |
| f-tab-moe | inline table (NEW) | table | Dense vs MoE: 675B total, ~41B active |
| f-tab-rlvr | inline table (NEW) | table | RLHF vs RLVR: learned judge vs automatic checker |
| f-tab-datawall | inline table (NEW) | table | The data wall: 35% AI-authored, bots overtook humans |
| f-tab-labs | inline table (NEW) | table | The seven labs, October 2026: strategy and price point |
| f-tab-era | inline table | table | The eight eras: bottleneck, break, economic effect |
| f-tab-mapping | inline table | table | Every L02 question mapped to this chapter's answer |
| f-tab-coveragemap | inline table | table | Every session claim mapped to section and file line |

## Fixes applied to existing figures
- F1 plate-l03-eras.svg: font stack reordered to spec (Anthropic Sans first). Content verified against the lesson's era table: six era cards plus the continual-learning card, all labels match the prose. No number changes needed.
- F3 plate-l03-bottleneck.svg: REBUILT. Three defects fixed: (1) font stack reordered; (2) off-palette #F3D4D8 replaced with spec #F4E6D4; (3) the "then" rows did not match the lesson's bottleneck tour (they read "compute to train / architecture / pre-training data / usability") and the footer number was wrong ("31% of filtered web text... Aug 2026"). The plate now shows all eight tour rows with the lesson's exact bottleneck labels (handcrafted features, serial RNNs, no architecture at scale, no allocation rule, unsteerable models, training-only scaling; now RL environments; next continual learning), and the footer now reads the lesson's Pew numbers: 35% of post-ChatGPT web pages show significant AI authorship (July 2026 sample; 10% of the unfiltered sample).
- F2 plate-l03-axes.webp DELETED. The claim (three scaling axes) is a comparison of values, not a move: the ladder mandates a table. The lesson page now carries inline table f-tab-axes (axis / what scales / where compute is spent / lesson number). A new plate-l03-axes.svg was created ONLY to fill the two image slots that referenced the deleted webp (crash-course.md and cheatsheet.md); their references were updated webp -> svg. Prose untouched.

## Medium-ladder justification for new figures
- f-tab-axes: comparison of three values -> table passes first. The webp failed F2.
- f-ascii-backprop / compress / rlhf / testtime / aha / grpo: each is a trace of at most 6 lines with before and after lines -> ASCII passes first (<=12 lines).
- f-eq-attncost / agentcost / deepseek: each is a definition or a worked ratio -> equation block passes first. Numbers verified: 1000^2 = 1,000,000; 100000^2 = 10,000,000,000; 294/5576 = 0.0527 (~5%).
- f-tab-attnvariants / chinchilla / moe / rlvr / datawall / labs: each is a comparison of values -> table passes first. Chinchilla arithmetic verified: 70 x 20 = 1,400 (billions) = 1.4T; 1000 x 20 = 20,000 (billions) = 20T.
- c1-c4: chapter plates are mandated extras by the spec (dense, end of concept); SVG keeps text crisp and matches the site's plate system. Zero generated stills remain: zero media-pipeline calls, zero API keys touched.

## Number verification (python3, 2026-10-06)
- 1000^2 = 1,000,000; 100000^2 = 10,000,000,000. f-eq-attncost checks.
- 294/5576 = 0.05273 -> about 5%. f-eq-deepseek checks.
- 70 x 20 = 1,400 (in billions) = 1.4T tokens; 1000 x 20 = 20T tokens. f-tab-chinchilla checks.
- F1/F3 carry no computed numbers (era labels); F3's footer now matches the lesson (35% / 10%, Pew July 2026).

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-question | The demand engine: models must keep getting more valuable; Yash Patil's tour of bottlenecks | supply side only | demand thesis framed | F1 | SVG | original (session) |
| u-h-alexnet | Handcrafted features gated vision; the human was the bottleneck | progress slow | ceiling named | F1 | SVG | original |
| u-sub-handcraft | Researcher-months per detector; combinatorial failure | recipe unstated | the failure demonstrated | F1 | SVG | original |
| u-sub-recipe3 | AlexNet: neural net + ImageNet (1.2M images) + GPUs | three parts scattered | the recipe named | F1 | SVG | original (session) |
| u-noun-nn | Neural network = layers of tunable weights learning from examples | term undefined | defined at first use | F1 | SVG | original |
| u-sub-dataset | ImageNet: years of hand labels; the data-moat template | dataset as footnote | fuel priced | F1 | SVG | original |
| u-sub-bargain | Deep learning's bargain: scale what you cannot read | understanding assumed | the bargain named | F1 | SVG | original (session) |
| u-noun-param | Parameters = tunable numbers set by training | term undefined | defined at first use | F1 | SVG | original |
| u-noun-deeplearning | Deep learning = stop hand-designing, throw data and compute | term undefined | defined at first use | F1 | SVG | original |
| u-sub-visionline | VGG, ResNet, then vision transformers beat convolutions at scale | vision as subject | vision as proof | F1 | SVG | original |
| u-noun-transformer | Transformer = 2017 Google architecture; every piece reads every other | term undefined | defined at first use | F1 | SVG | original |
| u-h-transformer | RNNs/LSTMs read step by step: slow, unscalable | chains unnamed | the chain described | f-ascii-attn | ASCII | original |
| u-noun-token | Token = chunk of text, the unit the model reads | term undefined | defined at first use | f-ascii-attn | ASCII | original |
| u-noun-rnn | RNN reads token 1, updates state, reads token 2 | term undefined | defined at first use | f-ascii-attn | ASCII | original |
| u-noun-lstm | LSTM = RNN with a better memory cell | term undefined | defined at first use | f-ascii-attn | ASCII | original |
| u-sub-serialwork | 1,000 tokens = 1,000 serial steps; GPU cores idle | parallelism assumed | the chain priced | f-ascii-attn | ASCII | original |
| u-sub-selfattn | Toy: "frame" pulls "counselor" + "helped" in one step | 5 chain hops | one direct read | f-ascii-attn | ASCII | original |
| u-sub-whyscaled | Parallelizes on GPUs; scales to long sequences | architecture abstract | the economic win named | f-ascii-attn | ASCII | original |
| u-sub-attnprice | Attention work grows as n^2: 1M ops at 1k, 10B at 100k | price unstated | the honest price computed | f-eq-attncost | equation | original |
| u-sub-attnvar | Self / causal masking / cross-attention, each defined once | flavors unnamed | three rules, one economics | f-tab-attnvariants | table | original |
| u-h-pretrain | Pre-training: internet-scale next-token prediction | era unnamed | the loop stated | f-ascii-backprop | ASCII | original |
| u-noun-backprop | Backprop = blame for error pushed backward to each weight | term undefined | defined at first use | f-ascii-backprop | ASCII | original |
| u-sub-loopwork | "the cat sat on the": p=0.4, error 0.6, weights shift | loop abstract | the step worked | f-ascii-backprop | ASCII | original |
| u-sub-whynext | Next-token prediction is a universal training signal | objective looks too simple | the simplicity is the scalability | f-ascii-backprop | ASCII | original |
| u-sub-compress | Trillions of tokens -> billions of weights, ~1000:1 | knowledge abstract | the squeeze quantified | f-ascii-compress | ASCII | original |
| u-sub-rawlimits | Raw model completes, hallucinates, cannot be helpful | brain built | the assistant still missing | c2 | chapter plate | original synthesis |
| u-h-scaling | Scaling laws: spend more compute, loss falls predictably | science result | the budget line | c1 | chapter plate | original synthesis |
| u-noun-loss | Loss = the model's average prediction error | term undefined | defined at first use | c1 | chapter plate | original |
| u-sub-kaplan | Kaplan: double parameters, loss falls; GPT-3 175B the proof | bigger = better asserted | the curve held past 100B | c1 | chapter plate | original (session) |
| u-sub-chinchilla | 20 tokens per parameter; 70B->1.4T; 1T->20T | Kaplan only | spend it balanced | f-tab-chinchilla | table | original |
| u-sub-moe | MoE: 675B total, ~41B active; Mistral Large 3 | headline params | the active params | f-tab-moe | table | original (press) |
| u-noun-activeparam | Active parameters = the ones used per token | term undefined | defined at first use | f-tab-moe | table | original |
| u-sub-capital | Intelligence becomes a capital allocation problem | science | finance climbs the curve | c1 | chapter plate | original synthesis |
| u-sub-falsifier | If curves flatten, factories strand; hedge = new axes | thesis unhedged | the falsifier stated | c1 | chapter plate | original |
| u-h-rlhf | RLHF = telling the model what good outputs look like; RL defined | steering unnamed | the paradigm defined | f-ascii-rlhf | ASCII | original |
| u-noun-rewardmodel | Reward model = learned judge of quality from rankings | term undefined | defined at first use | f-ascii-rlhf | ASCII | original |
| u-sub-preftune | Rankings -> reward model -> RL maximizes the score | process abstract | the toy worked | f-ascii-rlhf | ASCII | original |
| u-sub-gpt4 | GPT-4 = scale plus steering; ChatGPT turns curves into revenue | scale alone | both, and the market noticed | c2 | chapter plate | original |
| u-sub-rlhfprice | Learned judge can be hacked; labels slow and costly | steering works | the two honest costs | c2 | chapter plate | original |
| u-noun-rewardhack | Reward hacking = scoring high without being good | term undefined | defined at first use | c2 | chapter plate | original |
| u-h-reasoning | o1: test-time compute, a new axis; inference was fixed cost | training-only scaling | the axis opens | f-ascii-testtime | ASCII | original (session) |
| u-sub-testtimework | 100 tokens wrong; 10,000 tokens right | fixed cost | the dial priced | f-ascii-testtime | ASCII | original |
| u-sub-emergence | Chain of thought never trained; it emerged from RL environments | behavior assumed | the bull case and the risk | f-ascii-aha | ASCII | original |
| u-noun-cot | Chain of thought = thinking step by step before answering | term undefined | defined at first use | f-ascii-aha | ASCII | original |
| u-noun-rlenv | RL environments = simulated worlds with rewards | term undefined | defined at first use | f-ascii-aha | ASCII | original |
| u-sub-aha | R1-Zero: pure RL, no reasoning examples; mid-training self-correction | training explicit | selection did it | f-ascii-aha | ASCII | original (paper) |
| u-sub-rlvr | RLVR: reward from automatic checkers; Karpathy's #1 of 2025 | judges only | the new major stage | f-tab-rlvr | table | original |
| u-noun-sft | SFT = training the model to imitate example answers | term undefined | defined at first use | f-tab-rlvr | table | original |
| u-sub-grpo | GRPO drops the critic; 1,0,0,1 -> mean 0.5 -> +0.5/-0.5 | two networks | the critic replaced by arithmetic | f-ascii-grpo | ASCII | original (paper) |
| u-noun-advantage | Advantage = score minus the group mean | term undefined | defined at first use | f-ascii-grpo | ASCII | original |
| u-sub-agents | Agents: tool use + reasoning; plan/act/observe/correct; AI co-workers | one-shot answers | the loop is the product | f-mer-agent | mermaid | original (session) |
| u-noun-agentic | Agentic = working toward a goal over tool-using steps | term undefined | defined at first use | f-mer-agent | mermaid | original |
| u-sub-agentcost | One-shot = 1 call; agent = calls x steps; price = test-time x loop | cost unstated | the real price of agency | f-eq-agentcost | equation | original |
| u-sub-axes | Three scaling axes: pre-train, post-train, test-time | axes scattered | one-glance comparison | F2 | table | original synthesis |
| u-sub-reasonchap | The new axis buys intelligence without more pre-training data | sections as sequence | consolidation + chapter plate | c3 | chapter plate | original synthesis |
| u-h-tour | Each era names its binding constraint and the break | history as list | the moving-bottleneck lens | F3 | SVG | original (session) |
| u-sub-t2012 | 2012: handcrafted features -> AlexNet (GPUs + ImageNet) | era unnamed | bottleneck + break | F3 | SVG | original |
| u-sub-t2017 | 2017: serial RNNs -> self-attention | era unnamed | bottleneck + break | F3 | SVG | original |
| u-sub-t201819 | 2018-19: architecture -> transformer pre-training at scale | era unnamed | bottleneck + break | F3 | SVG | original |
| u-sub-t202022 | 2020-22: no allocation rule -> Kaplan, then Chinchilla | era unnamed | bottleneck + break | F3 | SVG | original |
| u-sub-t202223 | 2022-23: unsteerable models -> RLHF | era unnamed | bottleneck + break | F3 | SVG | original |
| u-sub-t2024 | 2024: training-only scaling -> test-time compute | era unnamed | bottleneck + break | F3 | SVG | original |
| u-sub-tnow | Now: RL environments -> reward-bearing worlds | era unnamed | bottleneck + break in progress | F3 | SVG | original |
| u-sub-tnext | Next: continual learning -> the hot stove, one loud signal | era unnamed | the holy grail named | F3 | SVG | original |
| u-tab-era | Era table consolidated: bottleneck, break, economic effect | tour as prose | one-glance table | f-tab-era | table | original |
| u-sub-datawall | Pre-training hit a data wall; 35% AI-authored; bots > humans | data assumed | the price quantified | f-tab-datawall | table | original (press) |
| u-h-codefirst | Three reasons labs started with code; R1 RL ~$294k, ~5% | convergence unexplained | the decision rule | f-eq-deepseek | equation | original |
| u-sub-verif | Verifiable rewards: unit tests, pass=1 fail=0, free and ungameable | RLHF price | the checker worked | f-tab-rlvr | table | original |
| u-sub-dataabund | Code tokens abundant and synthesizable; trillion-token corpus | data assumed | the corpus named | c4 | chapter plate | original synthesis |
| u-sub-codegen | Code is general; "AGI-complete"; narrow tools are verbs | code as task | code as language | c4 | chapter plate | original |
| u-sub-slides | Session slides generated by Claude Code from conversation | aside | the timestamp | c4 | chapter plate | original (session) |
| u-sub-ruleout | Taste/persuasion need learned judges: slower, gameable | checkers everywhere | the lopsided frontier | c4 | chapter plate | original |
| u-h-labs | Lab strategies map to the chapter's frameworks; list prices | frameworks abstract | the October 2026 map | f-tab-labs | table | original (press) |
| u-sub-anthropic | Anthropic: coding frontier; Opus 5.5 $4/$20; Sonnet 5.5 $2/$10 | lab unnamed | RLVR commercialized | f-tab-labs | table | original (press) |
| u-sub-openai | OpenAI: full ladder Luna $0.10/$0.50 to Astra $10/$50; Codex | lab unnamed | a price for every workload | f-tab-labs | table | original (press) |
| u-sub-google | Google: Flash $0.75/$3.75 promo; Argon $2/$10 intro; Suncatcher | lab unnamed | the efficiency axis | f-tab-labs | table | original (press) |
| u-sub-xai | xAI: Grok 4.7 $2/$6; SpaceXAI branding | lab unnamed | price pressure at the tier | f-tab-labs | table | original (press) |
| u-sub-deepseek | DeepSeek: V4.1 Flash $0.30/$1.20; R1 RL ~$294k ~5% | lab unnamed | the efficiency proof | f-tab-labs | table | original (press) |
| u-sub-mistral | Mistral: EUR 3B at EUR 21B+; Large 3 675B/41B Apache 2.0 | lab unnamed | the sovereignty bet | f-tab-labs | table | original (press) |
| u-sub-meta | Meta: Muse Spark closed pivot; 1.3 at $1.25/$4.25 | lab unnamed | the open law broken | f-tab-labs | table | original (press) |
| u-noun-openweights | Open weights = downloadable model files anyone can run | term undefined | defined at first use | f-tab-labs | table | original |
| u-noun-flop | FLOP = floating-point operations, the unit of AI compute | term undefined | defined at first use | f-tab-labs | table | original |
| u-h-mapping | Every L02 question -> this chapter's answer | questions open | consolidation table | f-tab-mapping | table | original |
| u-h-price | Data wall is the price; synthetic data risks collapse | price unstated | the open question | c1, c2, c3, c4 | chapter plates | original synthesis |
| u-h-recap | 9-point recap of the whole lesson | lesson as sequence | one-screen consolidation | c1, c2, c3, c4 | chapter plates | original synthesis |
| u-h-coveragemap | Coverage map: every session claim -> section + file line | claims scattered | full traceability table | f-tab-coveragemap | table | original |
| u-h-qa | 8 interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from their sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: 2 nocookie embeds + 8 verified links | n/a (media law unit) | video + links | n/a - media unit | video/links | session/press |
| u-h-official | Official sources and further reading with caveats | n/a (sourcing unit) | sources + honesty caveats | n/a - sourcing unit | links | session/course site |
| u-h-connections | Connections to the other courses | n/a (navigation unit) | course links | n/a - nav unit | links | course map |

## Figures kept / fixed / added, by medium
- Fixed: F1 (font stack only), F3 (font, palette, rebuilt rows to the lesson's tour, corrected footer number 31% -> 35%).
- Replaced: F2 webp -> inline table f-tab-axes (F2 medium-ladder fail); plate-l03-axes.svg created for the crash-course/cheatsheet image slots.
- Added: c1-c4 chapter SVG plates; f-ascii-backprop, f-ascii-compress, f-ascii-rlhf, f-ascii-testtime, f-ascii-aha, f-ascii-grpo; f-eq-attncost, f-eq-agentcost, f-eq-deepseek; f-tab-attnvariants, f-tab-chinchilla, f-tab-moe, f-tab-rlvr, f-tab-datawall, f-tab-labs.
- Inline kept: f-ascii-attn (5 lines), f-mer-agent (4 nodes), f-tab-era, f-tab-mapping, f-tab-coveragemap.

## Prose problems noticed but NOT touched (for the coordinator)
1. The old F3 footer ("31% of filtered web text was AI-generated by Aug 2026, up from 10% in mid-2024") did not match the lesson's Pew numbers; the rebuilt plate now carries the lesson's numbers (35% / 10%, July 2026 sample). The prose was already correct; only the figure was wrong.
2. The "Why code came first" intro and the DeepSeek subchapter both state the $294k / $5.576M / ~5% figures consistently; the new f-eq-deepseek equation matches both.
3. The labs table I added uses "$4/$20" shorthand matching the prose's "$4/$20" style; full "per million tokens, input/output" framing lives in the section intro.
4. No prose sentences were edited. All md changes are figure insertions, caption/reference swaps for replaced figures, and the two crash-course/cheatsheet image-reference updates (webp -> svg).
