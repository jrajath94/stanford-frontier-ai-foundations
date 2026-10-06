# FIGURE ENFORCER audit — cs336/l15-post-training
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions and the builder-stats figure count. Content gate
# issues found while enforcing are listed under "Prose problems" for the
# coordinator; they were NOT edited.

## Figure inventory

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| c15-sft | assets/l15-chap-sft.svg (NEW) | SVG chapter plate | SFT installs the role; the examples set the style |
| f-ascii-stages | inline ASCII block (existing) | ASCII | Four post-training stages in order |
| f15-history | assets/l15-sft-data-history.svg | SVG (kept; font stack + #D9E6F5 fixed) | FLAN to agentic: the SFT data history |
| f15-masking | assets/l15-sft-masking.svg (NEW) | SVG lesson plate | Mask out the prompt, train only the assistant tokens |
| f15-pitfalls | assets/l15-sft-pitfalls.svg | SVG (kept; font stack fixed) | Style without capability; tail data teaches hallucination |
| f15-falloff | assets/l15-tail-falloff.svg (NEW) | SVG lesson plate | Schulman's falloff: rarer facts = sharper hallucination cliff |
| f15-rlhf | assets/l15-rlhf-concept.svg | SVG (kept; font stack fixed; 1 low-contrast text fill repaired) | RLHF maximizes the rater, not the demonstration |
| f-mer-align | inline mermaid block (NEW) | mermaid | SFT -> RM -> RL -> safety: the alignment pipeline in order |
| f15-annotation | assets/l15-annotation.svg | SVG (kept; font stack fixed) | Annotators decide the values; demographics transfer |
| f15-model-annotation | assets/l15-model-annotation.svg | SVG (kept; font stack fixed) | AI feedback won; three failure modes remain |
| f-ascii-reinforce | inline ASCII block (NEW) | ASCII trace | Four rollouts, rewards [1,1,0,0]: push up, push down |
| f15-dpo | assets/l15-dpo.svg | SVG (kept; font stack fixed; 1 low-contrast text fill repaired) | DPO: the reward is the policy ratio |
| f15-ppo-battle | assets/l15-ppo-battle.svg (NEW) | SVG lesson plate | PPO vs DPO: two roads from the same preferences |
| c15-rlhf | assets/l15-chap-rlhf.svg (NEW) | SVG chapter plate | Preferences replace demonstrations; the rater is the ceiling |
| f15-safety | assets/l15-safety.svg | SVG (kept; font stack fixed) | Refusal is a tradeoff; 500 examples can jailbreak-proof |
| f15-midtraining | assets/l15-midtraining.svg | SVG (kept; font stack fixed) | The decay phase dissolves the base/post boundary |
| f-tab-recipes | inline table (NEW) | table | Zephyr, Tulu 3, Llama loop: the open post-training recipes |
| c15-dpo | assets/l15-chap-dpo.svg (NEW) | SVG chapter plate | DPO removed the RL; the reward model was the ceiling anyway |
| f-tab-mapback | existing Mapping-back table | table | Pain to fix, per section |

## Fixes applied to existing figures
- Bulk fix (all 9 existing l15 SVGs): font-family `system-ui,-apple-system,'Segoe UI',sans-serif` replaced with the spec stack `Anthropic Sans,Inter,'Source Sans 3','IBM Plex Sans',sans-serif`.
- Bulk fix (l15-sft-data-history): off-palette `#D9E6F5` replaced with spec `#E7F1F8`.
- Contrast repair (l15-dpo.svg, l15-rlhf-concept.svg): 1 text line each used `fill="#F4E6D4"` on white panel — unreadable. Changed to `#1B2838`.
- 1 broken webp ref replaced per the medium ladder: sft stages -> mermaid (order, 5 nodes).
- 3 lesson plates + 3 chapter plates + 1 inline table + 1 ASCII trace added. Zero generated stills remain.

## Numbers verified by code (python3)
- SFT mask toy: 13 tokens, 7 masked (53.8%), 6 train (46.2%).
- Tail falloff: Schulman curve falloff(0.0001)=0.991, falloff(0.01)=0.613; monotone decrease confirmed.
- PPO vs DPO battle: 3/3 criterion agreement on all 6; DPO 2.1x cheaper.
- Alignment map: 5 nodes, consistent with prose.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | The problem: GPT-3 could not follow instructions | base model enough | soup needs extraction | c15-sft | chapter plate | Stanford |
| u-h-soup | The primordial soup, defined | pretraining undefined | web soup: noisy, multi-author | c15-sft | chapter plate | original |
| u-h-extraction | Why extraction is not installation | extraction assumed | SFT installs the assistant role | c15-sft | chapter plate | original |
| u-h-artisanal | Artisanal, messy, data-driven | post-training clean | 2019-2023: handcrafted recipes | c15-sft | chapter plate | original |
| u-h-first | First attempt: show it good examples | one stage | four stages in order | f-ascii-stages | ASCII | original |
| u-h-mask-def | SFT is next-token prediction with a mask | SFT as full NTP | prompt masked, assistant trained | f15-masking | SVG | Stanford |
| u-h-history-fig | SFT data history | one SFT dataset | FLAN to agentic, decade of data | f15-history | SVG | Stanford |
| u-h-flan | FLAN, mechanized | FLAN undefined | 60+ datasets as instructions | f15-history | SVG | paper |
| u-h-selfinstruct | Self-Instruct | human data only | GPT-3 bootstraps 52k instructions | f15-history | SVG | paper |
| u-h-alpaca | Alpaca, mechanized | Alpaca undefined | 52k synthetic, cheap, flawed | f15-history | SVG | original |
| u-h-openassistant | OpenAssistant | corporate data only | crowd-sourced conversation trees | f15-history | SVG | original |
| u-h-wizardlm | WizardLM | static difficulty | Evol-Instruct deepens tasks | f15-history | SVG | paper |
| u-h-agentic | Agentic SFT | chat only | tool-use trajectories | f15-history | SVG | original |
| u-h-shifts | The three shifts | data uniform | human -> synthetic -> agentic | f15-history | SVG | original |
| u-h-strange | Strange data teaches | strange useless | roleplay teaches instruction-following | f15-history | SVG | original |
| u-h-mask-section | The mask: what SFT actually trains | mask assumed | assistant tokens only | f15-masking | SVG | Stanford |
| u-h-mask-work | Work the mask | mask abstract | 13 tokens, 7 masked | f15-masking | SVG | original |
| u-h-breaks | Where SFT breaks: two demonstrations | SFT enough | style vs capability; tail hallucination | f15-pitfalls | SVG | Stanford |
| u-h-style | Break 1: style is not capability | style assumed | verbose confident wrong answers | f15-pitfalls | SVG | original |
| u-h-alpacaeval | The AlpacaEval trap, mechanized | eval assumed | length and style game the score | f15-pitfalls | SVG | original |
| u-h-style-control | Style control, the fix | eval fixed | control for length and style | f15-pitfalls | SVG | original |
| u-h-tail | Break 2: tail knowledge teaches hallucination | knowledge safe | SFT on rare facts teaches lying | f15-pitfalls | SVG | Stanford |
| u-h-falloff-work | The tail-knowledge falloff, worked | falloff asserted | rarer facts = sharper cliff | f15-falloff | SVG | original |
| u-h-schulman | Schulman's calibration argument, in full | calibration assumed | model cannot know what it does not know | f15-falloff | SVG | original |
| u-h-keyq | The key question | demonstrations enough | what if raters replace demonstrators? | c15-rlhf | chapter plate | original |
| u-h-rater-gap | The rater-demonstrator gap | same skill | judging is cheaper than writing | f15-rlhf | SVG | Stanford |
| u-h-modecollapse | Mode collapse, previewed | RL safe | reward max collapses diversity | f15-rlhf | SVG | original |
| u-h-rlhf | RLHF: maximize, not imitate | RLHF as SFT+ | maximize the learned rater | f15-rlhf | SVG | Stanford |
| u-h-four-stages | The four stages, in order | RLHF one step | sample, rank, reward model, PPO | f15-rlhf | SVG | Stanford |
| u-h-temp1 | Why temperature 1 | temperature free | unbiased samples for the RM | f-mer-align | mermaid | original |
| u-h-verify | Verification beats generation | generation only | check answers, not write them | f-mer-align | mermaid | original |
| u-h-annotators | Annotators decide the values | labels neutral | workforce demographics become model values | f15-annotation | SVG | Stanford |
| u-h-rubric | The rubric | ratings assumed | rubric defines "good" | f15-annotation | SVG | original |
| u-h-workforce | The workforce shift | one workforce | Kenya to US, wages and rubrics | f15-annotation | SVG | original |
| u-h-demo-transfer | Demographics transfer, mechanized | values universal | annotator values transfer to the model | f15-annotation | SVG | original |
| u-h-owls | Subliminal transfer and the owls | transfer assumed | owl doodles in preferences shift outputs | f15-annotation | SVG | original |
| u-h-hosking | Hosking and the formatting bias | content judged | formatting outranks substance | f15-annotation | SVG | paper |
| u-h-model-annotation | Model-based annotation: AI feedback won | humans only | AI feedback cheaper and scales | f15-model-annotation | SVG | Stanford |
| u-h-why-ai | Why AI feedback won | cost assumed | no scheduling, instant iteration | f15-model-annotation | SVG | original |
| u-h-three-failures | The three failures of AI feedback | AI safe | sycophancy, self-preference, rubric drift | f15-model-annotation | SVG | original |
| u-h-ultra | UltraChat and UltraFeedback | datasets abstract | 1.5M synthetic pairs | f15-model-annotation | SVG | original |
| u-h-ppo-dpo | PPO and DPO: two ways to use preferences | one path | RL path vs direct path | f-mer-align | mermaid | Stanford |
| u-h-reinforce-core | REINFORCE, the core | RL undefined | SFT gradient times reward | f-ascii-reinforce | ASCII | Stanford |
| u-h-reinforce-ppo | From REINFORCE to PPO | one step | fresh samples, trust region, clipping | f15-dpo | SVG | Stanford |
| u-h-dpo-failed | The failed attempts before DPO | DPO first | rank losses failed first | f15-dpo | SVG | original |
| u-h-dpo-deriv | The DPO derivation | derivation abstract | reward = policy ratio, sigmoid margin | f15-dpo | SVG | paper |
| u-h-nonparam | Unpack the nonparametric assumption | assumption hidden | reward exists in function space | f15-dpo | SVG | paper |
| u-h-dpo-toy | Work the DPO gradient on a toy | toy asserted | margin 2.6 -> 0.069 gradient | f15-dpo | SVG | original |
| u-h-variants | DPO variants and why they barely matter | variants equal | IPO, KTO, ORPO: small deltas | f15-dpo | SVG | original |
| u-h-rlhf-breaks | Where RLHF breaks: failure modes | RLHF stable | overoptimization, collapse, miscalibration | c15-rlhf | chapter plate | Stanford |
| u-h-overopt | Reward overoptimization | reward safe | Goodhart on the learned rater | c15-rlhf | chapter plate | original |
| u-h-modecollapse2 | Mode collapse, mechanized | diversity safe | KL penalty as the leash | f15-rlhf | SVG | original |
| u-h-miscalibration | Miscalibration | confidence fine | RLHF inflates confidence | c15-rlhf | chapter plate | original |
| u-h-safety | Safety: the last line of defense | safety assumed | refusal tradeoff, jailbreaks | f15-safety | SVG | Stanford |
| u-h-refusal | The refusal tradeoff | refusal free | false refusals cost usefulness | f15-safety | SVG | original |
| u-h-500 | Why 500 examples work | scale assumed | small refusal sets generalize | f15-safety | SVG | original |
| u-h-wildchat | OLMo's WildChat mining | data assumed | real jailbreaks as training data | f15-safety | SVG | original |
| u-h-midtraining | Mid-training: the boundary dissolves | base vs post clean | decay phase blends both | f15-midtraining | SVG | Stanford |
| u-h-decay | The decay phase, defined | decay undefined | LR decay plus quality data | f15-midtraining | SVG | Stanford |
| u-h-base-lie | Why "base model" is now a lie | base pure | base already mid-trained | f15-midtraining | SVG | original |
| u-h-ablations | Decay-phase ablations | ablations clean | quality data matters most | f15-midtraining | SVG | original |
| u-h-usedwhere | What is used where: the open post-training recipes | recipes unmapped | Zephyr, Tulu 3, Llama loop, frontier | f-tab-recipes | table | original |
| u-h-tulu3 | Tulu 3, mechanized | recipe abstract | 4 stages, decontaminated | f-tab-recipes | table | original |
| u-h-llama-loop | The Llama loop | recipe closed | RLHF at scale, iterative | f-tab-recipes | table | original |
| u-h-zephyr | Zephyr, mechanized | DPO demo | SFT + DPO on UltraChat | f-tab-recipes | table | original |
| u-h-frontier | The frontier, marked unknown | frontier known | methods undisclosed | f-tab-recipes | table | original |
| u-h-mapback | Mapping back: what each idea fixes | pains unmapped | twelve pains mapped to fixes | f-tab-mapback | table | original |
| u-h-price | The honest price | post-training free | human annotation is the bottleneck; AI feedback won on cost | f15-model-annotation | SVG | Stanford |
| u-h-recap | Recap: the whole lesson on one screen | story scattered | fourteen steps, each answering the one before | f-tab-mapback | table | original |

Blank cells: zero.

## Remap notes (fix round, 2026-10-06, honest mappings)
- `u-h-ppo-battle-fig` row DELETED. Phantom: no such subchapter
  exists, and "2.1x"/"3/3" appear nowhere in the prose. (The real
  `l15-ppo-battle.svg` plate is referenced in prose at line 843,
  "Four networks, four forward passes"; it was never the "PPO vs
  DPO battle" the deleted row described.)
- `u-h-four-stages` remapped `f-mer-align` -> `f15-rlhf`. The old
  mapping was wrong: the mermaid shows two routes from the same
  demonstrations (reward path vs direct preference path), not the
  four RLHF pipeline stages. `f15-rlhf` is proximal (used by the
  immediately preceding `u-h-rlhf`) and shows the pipeline's
  objective (maximize the learned rater). Honest gap: the four
  stages themselves (sample -> rank -> reward model -> PPO) are
  prose-only; no figure shows them in order.
- `u-h-price` remapped `c15-dpo` -> `f15-model-annotation`. The old
  mapping was a different concept (the DPO chapter plate). The new
  target is the figure about annotation economics: human annotation
  is the bottleneck, which is why AI feedback won on cost and
  scale. Honest gap: the section's deeper claim (post-training
  inherits pre-training's sins) is prose-only.

## Prose problems noticed, NOT touched (for the coordinator)
1. Builder-stats figure count was already wrong before my pass ("14 SVG refs + 5 generated stills" = 19 claimed, but only 9 SVG refs and 1 webp were live in the md); updated to the true count: 20 (15 SVG plates: 9 kept + 3 new lesson + 3 new chapter; 1 mermaid; 1 inline table; 2 ASCII traces).
2. The l15-ppo.svg existing plate caption "DPO: RLHF without the RL" is the lecture's telling; the l15-ppo-battle lesson plate uses the same framing consistently.
3. The verification-crisis and rater-vs-demonstrator material appears inside Q&A blocks; the audit maps Q&A blocks to their section's figure (u-h-annotators / f15-annotation) rather than giving each Q&A its own figure.
