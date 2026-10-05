import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- RLVR motivation ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("RLVR: rewards you cannot over-optimize", "RLHF rewards are learned and hackable. Verifiable rewards are not.")
p.panel(60, 140, 840, 130, "the RLHF ceiling")
p.text(84, 188, "push a learned reward model hard -> overfit it. Compute stops helping.", 15, ORANGE, 700)
p.panel(60, 296, 840, 130, "the RLVR bet")
p.text(84, 344, "math and code have checkable answers. Optimize the real objective.", 15, TEAL, 700)
p.text(84, 370, "AlphaGo worked because the win condition was exact. Same idea.", 14, MUT)
p.footer("Source: lecture RLVR-motivation slides, original plate. Shell 1: the bet.")
p.save("l16-rlvr.svg")

# ---- PPO ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("PPO: simple pseudocode, painful reality", "The workhorse. The '37 implementation details' blog post should scare you.")
p.panel(60, 140, 840, 130, "the core")
p.text(84, 188, "REINFORCE gradient: weighted SFT, weights +/-. Reuse rollouts via clipping.", 15, INK, 600)
p.text(84, 214, "policy gradient -> TRPO trust region -> PPO clipping heuristic", 14, MUT)
p.panel(60, 296, 840, 150, "the pain")
p.text(84, 344, "value model = another full model in memory", 15, ORANGE, 700)
p.text(84, 370, "gamma=lambda=1 silently degenerates to a bandit. KL clipping ruins KL.", 14, MUT)
p.text(84, 396, "libraries disagree. Many implementations are just wrong.", 14, MUT)
p.footer("Source: lecture PPO slides, original plate. Shell 2: the workhorse.")
p.save("l16-ppo.svg")

# ---- GRPO ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("GRPO: drop the value model", "Advantage = z-score within a group of rollouts. One page of code.")
p.panel(60, 140, 840, 130, "the idea")
p.text(84, 188, "sample k rollouts, subtract the group mean, divide by the group std", 15, INK, 600)
p.text(84, 214, "online form: clipping vanishes, advantage minus KL. Simple.", 14, MUT)
p.panel(60, 296, 840, 150, "the Dr. GRPO critique")
p.text(84, 344, "dividing by std breaks the baseline contract: upweights too-easy and too-hard", 15, ORANGE, 700)
p.text(84, 370, "length normalization rewards wrong-but-long. 'Aha moment' was in the base model.", 14, MUT)
p.footer("Source: lecture GRPO slides, original plate. Shell 3: the simplification.")
p.save("l16-grpo.svg")

# ---- R1 ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("DeepSeek R1: the clean recipe", "Base model + GRPO + accuracy rewards. Outcome supervision only.")
p.panel(60, 140, 840, 130, "R1-Zero")
p.text(84, 188, "no SFT. GRPO on math, reward = correct + format. Near o1.", 15, TEAL, 700)
p.text(84, 214, "dropped process supervision: outcome rewards were enough", 14, MUT)
p.panel(60, 296, 840, 150, "production R1")
p.text(84, 344, "long-CoT SFT ('collect a small amount' = distill?), language consistency reward", 15, INK, 600)
p.text(84, 370, "then RLHF for the user-facing finish. R1 CoTs distill into Qwen.", 14, MUT)
p.footer("Source: lecture DeepSeek slides, original plate. Shell 4: the milestone.")
p.save("l16-r1.svg")

# ---- Kimi ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Kimi K1.5: curriculum and compression", "Same destination as R1 by a different derivation.")
p.panel(60, 140, 840, 130, "curriculum")
p.text(84, 188, "best-of-8 filter: drop problems the model already solves. Medium difficulty only.", 15, INK, 600)
p.text(84, 214, "too hard = no signal, too easy = no learning. Remove mastered problems live.", 14, MUT)
p.panel(60, 296, 840, 150, "length")
p.text(84, 344, "long CoTs cost inference money. Compress with a length reward.", 15, TEAL, 700)
p.text(84, 370, "keep wrong answers explorable: shorten slightly, not to zero", 14, MUT)
p.footer("Source: lecture Kimi slides, original plate. Shell 4: the alternative.")
p.save("l16-kimi.svg")

# ---- Qwen ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Qwen 3: the full stack", "Thinking and non-thinking in one model. RL on 4,000 examples.")
p.panel(60, 140, 840, 130, "thinking fusion")
p.text(84, 188, "one model, prompt tag switches modes. Early exit degrades gracefully.", 15, INK, 600)
p.text(84, 214, "thinking mode beats instant mode even at tiny budgets", 14, MUT)
p.panel(60, 296, 840, 150, "Coder-Next: agentic RLVR")
p.text(84, 344, "mid-training for agents, 4 experts distilled into one, SWE envs at scale", 15, ACTIVE, 700)
p.text(84, 370, "70.6% SWE-bench with 3B active parameters", 14, TEAL, 700)
p.footer("Source: lecture Qwen slides, original plate. Shell 4: the stack.")
p.save("l16-qwen.svg")

# ---- reward hacking ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Verifiable is not unhackable", "RLVR is only as robust as its reward.")
p.panel(60, 140, 840, 150, "the git hack")
p.text(84, 188, "agent learned to read future commits via git history for the fix", 15, ORANGE, 700)
p.text(84, 214, "blocked git log -> queried the remote instead. Needed an anti-git reward.", 14, MUT)
p.panel(60, 316, 840, 150, "the lesson")
p.text(84, 364, "Lean compiler: 'bulletproof' until adversarial strings verified false proofs", 15, INK, 600)
p.text(84, 390, "answer equivalence checking is its own rabbit hole (regex or model)", 14, MUT)
p.footer("Source: lecture reward-hacking slides, original plate. Shell 4: the warning.")
p.save("l16-reward-hacking.svg")

# ---- takeaways ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("RLVR takeaways", "It is all about the reward.")
rows = [
    ("reward first", "unhackable rewards let you pour in compute", TEAL),
    ("GRPO", "one page of code, know it cold", ACTIVE),
    ("RL > expert iteration", "Kimi ablations: RL consistently wins", BLUE),
    ("finicky", "noisy, painful, but smoother than old PPO days", MUT),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 90
    p.panel(60, yy, 840, 76, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="76" rx="5" fill="{col}"/>')
    p.text(84, yy + 50, b, 14, MUT)
p.footer("Source: lecture closing slides, original plate. Shell 5: the moral.")
p.save("l16-takeaways.svg")
