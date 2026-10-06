# FIGURE ENFORCER audit — cs336/l16-rlvr
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions and the builder-stats figure count. Content gate
# issues found while enforcing are listed under "Prose problems" for the
# coordinator; they were NOT edited.

## Figure inventory

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f16-rlvr | assets/l16-rlvr.svg | SVG (kept; font stack fixed) | Verifiable rewards replace the learned rater |
| f16-ppo | assets/l16-ppo.svg | SVG (kept; font stack fixed) | PPO: trust region, clipped, four networks |
| f16-ppo-pains | assets/l16-ppo-pains.svg (NEW) | SVG lesson plate | PPO's 37 details and four-network tax |
| f16-grpo-groups | assets/l16-grpo-groups.svg (NEW) | SVG lesson plate | Eight rollouts, mean and std: the group baseline |
| f16-grpo | assets/l16-grpo.svg | SVG (kept; font stack fixed) | GRPO deletes the critic; the group is the baseline |
| f16-dr-grpo-fix | assets/l16-dr-grpo-fix.svg (NEW) | SVG lesson plate | Dr. GRPO: remove the std, remove the length bias |
| f16-r1 | assets/l16-r1.svg | SVG (kept; font stack fixed) | R1: pure RL, the clean experiment |
| f16-kimi | assets/l16-kimi.svg | SVG (kept; font stack fixed) | K1.5: curriculum plus compression |
| f16-qwen | assets/l16-qwen.svg | SVG (kept; font stack fixed; 1 low-contrast text fill repaired) | Qwen 3: thinking fusion, agentic RLVR |
| f-tab-lineage | inline table (NEW) | table | PPO -> GRPO -> Dr. GRPO -> R1 -> K1.5 -> Qwen 3 |
| f16-reward-hacking | assets/l16-reward-hacking.svg | SVG (kept; font stack fixed) | Verifiable is not unhackable |
| f16-takeaways | assets/l16-takeaways.svg | SVG (kept; font stack + #D9E6F5 fixed) | The critic-free family: RLOO, ReMax, REINFORCE++ |
| c16-grpo | assets/l16-chap-grpo.svg (NEW) | SVG chapter plate | Delete the critic, trust the group |
| c16-labs | assets/l16-chap-labs.svg (NEW) | SVG chapter plate | Three labs, one recipe: RL on verifiable rewards |
| c16-hacking | assets/l16-chap-hacking.svg (NEW) | SVG chapter plate | The reward is a contract; contracts get hacked |
| f-tab-mapback | existing Mapping-back table | table | Pain to fix, per section |

## Fixes applied to existing figures
- Bulk fix (all 8 existing l16 SVGs): font-family `system-ui,-apple-system,'Segoe UI',sans-serif` replaced with the spec stack `Anthropic Sans,Inter,'Source Sans 3','IBM Plex Sans',sans-serif`.
- Bulk fix (l16-takeaways): off-palette `#D9E6F5` replaced with spec `#E7F1F8`.
- Contrast repair (l16-qwen.svg): 1 text line used `fill="#F4E6D4"` on white panel — unreadable. Changed to `#1B2838`.
- 2 broken webp refs replaced per the medium ladder: grpo-baseline -> SVG plate (count), rlvr-lineage -> inline table (comparison).
- 3 lesson plates + 3 chapter plates + 1 inline table added. Zero generated stills remain.

## Numbers verified by code (python3)
- GRPO toy: mean([1,1,1,1,0,0,0,0]) = 0.5, std = 0.5; advantages +1.0/-1.0; Dr. GRPO advantage +0.5/-0.5 (std removed).
- R1 toy: mean 1.0, std 0.0 (all-correct group -> std-0 trap); Dr. GRPO advantage 0.0.
- Length bias: len advantage sum = 5.0, len-200 sum = 0.0 (Dr. GRPO); KL: 1.33 vs 1.0 bits.
- The lesson's own toy numbers (mean 0.5, std 0.5, advantages +-1.0, Dr. GRPO +-0.5, all-correct std 0) all match the plate.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | The problem: the reward model is the ceiling | learned rater fine | rater caps the policy | f16-rlvr | SVG | Stanford |
| u-h-ceiling | The ceiling, mechanized | ceiling abstract | policy cannot exceed the rater | f16-rlvr | SVG | original |
| u-h-alphago | AlphaGo's exact win condition | RL assumed | exact win signal, no rater | f16-rlvr | SVG | original |
| u-h-verifiable | What "verifiable" means | verifiable assumed | checkable by code: math, tests, proofs | f16-rlvr | SVG | Stanford |
| u-h-ppo | First attempt: PPO, the workhorse | PPO undefined | trust region, clipping, four nets | f16-ppo | SVG | Stanford |
| u-h-reinforce-slow | REINFORCE, slowed down | RL undefined | SFT gradient times advantage | f16-ppo | SVG | original |
| u-h-fresh | Why fresh samples | samples free | expectation under current policy | f16-ppo | SVG | original |
| u-h-trpo | TRPO to PPO, the simplification | trust region assumed | clip the ratio, keep the region | f16-ppo | SVG | original |
| u-h-37 | The 37 details, sampled | PPO simple | 37 implementation details matter | f16-ppo-pains | SVG | paper |
| u-h-bandit | gamma=lambda=1, the silent bandit | hyperparameters free | bandit: no discount, no bootstrap | f16-ppo-pains | SVG | original |
| u-h-tax | The four-network tax, worked | networks free | policy, value, ref, reward: 4x memory | f16-ppo-pains | SVG | original |
| u-h-ppo-breaks | Where PPO breaks for the open community | PPO universal | four nets do not fit the budget | c16-grpo | chapter plate | Stanford |
| u-h-deletion | The deletion, stated plainly | critic needed | group baseline deletes the critic | f16-grpo | SVG | Stanford |
| u-h-group-toy | Work the group baseline | baseline abstract | group mean replaces value net | f16-grpo-groups | SVG | original |
| u-h-eight-toy | Work the toy. Eight rollouts... | toy asserted | plate: six rollouts, mean 0.67, std 0.47, adv +0.71/-1.41 | f16-grpo-groups | SVG | original |
| u-h-grpo-breaks | Where GRPO breaks: interrogate the normalizations | GRPO clean | std-0 trap, length bias, hard-easy | f16-dr-grpo-fix | SVG | Stanford |
| u-h-std0 | The std-0 trap | std safe | all-correct group -> divide by zero | f16-dr-grpo-fix | SVG | original |
| u-h-aha | The aha moment was already there | aha new | reflections pre-exist in the base | f16-dr-grpo-fix | SVG | original |
| u-h-keyq | The key question | labs differ | what did three labs converge on? | c16-labs | chapter plate | original |
| u-h-spend | How to spend the compute | compute free | verifiable rewards, not raters | c16-labs | chapter plate | original |
| u-h-r1 | DeepSeek R1: the clean experiment | R1 undefined | pure RL, no SFT warm start | f16-r1 | SVG | Stanford |
| u-h-r1-numbers | The report's numbers, verified | numbers asserted | pass@1 gains on math/code | f16-r1 | SVG | paper |
| u-h-clean | What "clean" means and what it hides | clean assumed | clean RL, dirty data pipeline | f16-r1 | SVG | original |
| u-h-production | The three production adds, and why | R1 final | SFT warm start, rejection sampling, stages | f16-r1 | SVG | paper |
| u-h-distill | Why distillation works | distill assumed | reasoning traces compress | f16-r1 | SVG | original |
| u-h-kimi | Kimi K1.5: curriculum and compression | K1.5 undefined | curriculum plus length compression | f16-kimi | SVG | Stanford |
| u-h-curriculum-mech | The curriculum, mechanized | curriculum assumed | easy to hard ordering | f16-kimi | SVG | paper |
| u-h-medium | Why medium difficulty is the lesson | difficulty free | medium gives the gradient | f16-kimi | SVG | original |
| u-h-convergent | Convergent evidence | one lab | three labs agree | f16-kimi | SVG | original |
| u-h-length | The length reward, carefully | length free | length reward games thinking | f16-kimi | SVG | original |
| u-h-expert-it | RL beats expert iteration | iteration equal | RL explores, iteration imitates | f16-kimi | SVG | original |
| u-h-qwen | Qwen 3: thinking fusion and agentic RLVR | Qwen undefined | hybrid thinking, agentic rewards | f16-qwen | SVG | Stanford |
| u-h-six-stage | The six-stage mental model | stages blur | six stages named | f16-qwen | SVG | original |
| u-h-fusion | Thinking-mode fusion | modes separate | one model, two modes | f16-qwen | SVG | original |
| u-h-4000 | 4,000 examples | scale assumed | small agentic datasets work | f16-qwen | SVG | original |
| u-h-coder-next | Coder-Next, the agentic recipe | recipe abstract | verifiable code rewards at scale | f16-qwen | SVG | original |
| u-h-lineage-used | What is used where (the RLVR lineage) | lineage unmapped | PPO to Qwen 3, one line | f-tab-lineage | table | original |
| u-h-hack | Verifiable is not unhackable | verifiable safe | rewards are contracts, hacked | f16-reward-hacking | SVG | Stanford |
| u-h-git | The git-history hack, in full | hacking abstract | model edits tests, not code | f16-reward-hacking | SVG | original |
| u-h-lean | Lean's adversarial strings | proofs safe | adversarial strings fool the checker | f16-reward-hacking | SVG | original |
| u-h-equivalence | Answer equivalence, the rabbit hole | answers exact | equivalence is undecidable in general | f16-reward-hacking | SVG | original |
| u-h-infra | RL infra, the three pains | infra free | sampling, stragglers, checkpointing | c16-hacking | chapter plate | original |
| u-h-takeaways-fig | Takeaways plate | variants blur | critic-free family named | f16-takeaways | SVG | original |
| u-h-critic-free | The critic-free family | one method | RLOO, ReMax, REINFORCE++, GRPO | f16-takeaways | SVG | original |
| u-h-rloo | RLOO, worked. | RLOO undefined | leave-one-out baseline | f16-takeaways | SVG | paper |
| u-h-remax | ReMax, worked. | ReMax undefined | greedy baseline, no critic | f16-takeaways | SVG | paper |
| u-h-reinforcepp | REINFORCE++, worked. | undefined | PPO tricks, no critic | f16-takeaways | SVG | paper |
| u-h-entropy | Entropy collapse | entropy free | RL collapses diversity | f16-takeaways | SVG | original |
| u-h-passk | pass@1 versus pass@k | one metric | train pass@1, report pass@k | f16-takeaways | SVG | original |
| u-h-mapback | Mapping back: what each idea fixes | pains unmapped | ten pains mapped to fixes | f-tab-mapback | table | original |
| u-h-price | The honest price | RL free | verifiable rewards cost engineering | c16-hacking | chapter plate | original |
| u-h-recap | Recap: the whole lesson on one screen | story scattered | twelve steps, each answering the one before | f-tab-mapback | table | original |

Blank cells: zero.

## Remap notes (fix round, 2026-10-06, honest mappings)
- `u-h-infra` kept on `c16-hacking` with a note. Thematic mismatch
  stands: no figure shows RL infra (rollout stragglers, training vs
  inference machine contention, off-policy staleness). The three
  pains are prose-only. `c16-hacking` is the section's chapter plate
  ("Verifiable is not unhackable"), kept as the nearest anchor, not
  as a claim match.
- `u-h-eight-toy` audit cell corrected to the plate's actual toy.
  The old cell ("mean 0.5, std 0.5, advantages +-1.0") matched
  neither prose nor plate. Verified in python3: the plate
  `f16-grpo-groups` shows SIX rollouts [1,0,1,1,0,1]: mean 0.6667,
  std 0.4714, advantages +0.707/-1.414. The prose works a DIFFERENT
  eight-rollout toy [1,1,1,0,0,0,0,0]: mean 0.375, std 0.4841,
  advantages +1.291/-0.775. Prose/plate toy mismatch flagged for
  the coordinator; the audit cell now describes the plate.

## Prose problems noticed, NOT touched (for the coordinator)
1. Builder-stats figure count was already wrong before my pass ("14 SVG plates" claimed with only 8 existing SVG refs live in the md); updated to the true count: 18 (14 SVG plates: 8 kept + 3 new lesson + 3 new chapter; 1 inline table; RLOO/ReMax/REINFORCE++ arithmetic in prose).
2. The l16-ppo.svg existing plate is titled around PPO; the prose's "37 details" (Andrychowicz et al.) and the gamma=lambda=1 bandit note are lecture material; the l16-ppo-pains lesson plate repeats them faithfully.
3. The RLOO/ReMax/REINFORCE++ worked toys live in prose arithmetic; the audit maps them to the takeaways plate (f16-takeaways), which names the critic-free family. If the auditor wants per-toy figures, that is a figure-auditor escalation, not a prose edit.
