import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- what is good ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("What is good?", "Four lenses. None is the answer. Each shapes what gets built.")
rows = [
    ("benchmarks", "Artificial Analysis intelligence index", "progress, gameable", TEAL),
    ("cost", "intelligence vs inference price", "correlated, not aligned", BLUE),
    ("preference", "Arena: which answer do people like", "style vs correctness", ACTIVE),
    ("usage", "OpenRouter: what people pay for", "economic, unrepresentative", ORANGE),
]
y0 = 140
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 36, b, 14, INK)
    p.text(84, yy + 62, c, 14, MUT)
p.text(60, 560-40, "Evaluation sets North Stars. Choose the star, choose the future.", 15, INK, 700)
p.footer("Source: lecture opening slides, original plate. Shell 1: define good first.")
p.save("l12-what-is-good.svg")

# ---- perplexity ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Perplexity: the distribution view", "A language model is P(x). Perplexity asks: how much mass on the test set?")
p.panel(60, 140, 840, 140, "the idea")
p.text(84, 190, "best possible = entropy of the true distribution, at p = t", 16, INK, 600)
p.text(84, 216, "drive perplexity down -> approach the truth -> solve everything?", 15, ORANGE, 700)
p.panel(60, 310, 840, 180, "the catches")
p.text(84, 360, "charges bits for boring tokens too (conditional perplexity focuses)", 15, INK)
p.text(84, 386, "cloze tasks are perplexity in disguise: LAMBADA, HellaSwag", 15, INK)
p.text(84, 412, "leaderboards need trust: unnormalized logprobs cheat", 15, ORANGE, 700)
p.footer("Source: lecture perplexity slides, original plate. Shell 2: the cleanest metric, with teeth.")
p.save("l12-perplexity.svg")

# ---- exam treadmill ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("The exam treadmill", "MMLU saturated. Pro saturated. GPQA saturated. HLE still has room. Multiple choice survives.")
rows = [
    ("MMLU (2020)", "57 subjects, few-shot", "barely above chance -> 90s", MUT),
    ("MMLU-Pro", "10 choices, chain of thought", "33 -> 88", BLUE),
    ("GPQA", "PhD contractors, diamond set", "experts 65%, models now 94", ACTIVE),
    ("Humanity's Last Exam", "crowdsourced, private held-out", "Mythos 64.7: still hard", TEAL),
]
y0 = 140
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 36, b, 14, INK)
    p.text(84, yy + 62, c, 14, MUT)
p.text(60, 556, "Exams miss real use: nobody asks HLE questions except on HLE.", 14, MUT)
p.footer("Source: lecture benchmark slides, original plate. Shell 4: benchmarks have shelf lives.")
p.save("l12-exam-treadmill.svg")

# ---- chat eval ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Judging open-ended chat", "Pairwise beats absolute. Judges have biases. Rubrics make it well-defined.")
rows = [
    ("Chatbot Arena", "humans pick A/B, ELO ranking", "real prompts; who are the raters? style vs correctness", TEAL),
    ("AlpacaEval", "LLM judge vs baseline, win rate", "length bias gamed it; debiased version", ACTIVE),
    ("WildBench", "checklists per prompt", "rubrics scope the judgment", BLUE),
]
y0 = 140
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 94, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="94" rx="5" fill="{col}"/>')
    p.text(84, yy + 40, b, 14, INK, 600)
    p.text(84, yy + 66, c, 13, MUT)
p.text(60, 496, "Evaluate the metric too: AlpacaEval correlated 0.98 with Arena. Sycophancy upweights pleasing lies.", 14, MUT)
p.footer("Source: lecture chat-eval slides, original plate. Shell 4: judge the judge.")
p.save("l12-chat-eval.svg")

# ---- agent eval ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Agents: evaluate what it does", "Environments with checkable outcomes. The scaffold is half the score.")
rows = [
    ("SWE-bench", "fix the issue, pass the tests", "16% -> 93% (Verified cleaned it)", TEAL),
    ("Terminal-Bench", "do it in a terminal", "crowdsourced, hours to weeks", BLUE),
    ("CyBench / MLE-bench", "CTF flags, Kaggle scores", "CyBench solved; scaffolds vary widely", ACTIVE),
]
y0 = 140
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 94, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="94" rx="5" fill="{col}"/>')
    p.text(84, yy + 40, b, 14, INK, 600)
    p.text(84, yy + 66, c, 13, MUT)
p.text(60, 496, "Same model, different scaffold, different score. Audit traces (Docent): empty responses scored 38% on TorchBench.", 14, ORANGE, 700)
p.footer("Source: lecture agent slides, original plate. Shell 4: the scaffold counts.")
p.save("l12-agent-eval.svg")

# ---- arc ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("ARC: reasoning without knowledge", "Grids a human solves in seconds, models could not. Reasoning models changed that.")
p.panel(60, 140, 840, 140, "the design")
p.text(84, 190, "100% human-solvable, knowledge-free, every task a snowflake", 15, INK, 600)
p.text(84, 216, "GPT-3: zero. o1/o3: ARC-1 solved. ARC-2 nearly. ARC-3: interactive, scores low.", 15, INK, 600)
p.panel(60, 310, 840, 140, "the caveat")
p.text(84, 360, "pure reasoning may not exist; still human-bounded, not superhuman", 15, MUT)
p.footer("Source: lecture ARC slides, original plate. Shell 4: isolate the faculty.")
p.save("l12-arc.svg")

# ---- contamination ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Contamination: assume the worst", "Models train on the internet. The test set is on the internet. Four defenses.")
rows = [
    ("detect", "question-order preference betrays memorization", TEAL),
    ("report", "norms: justify no train-test overlap with every claim", BLUE),
    ("fresh evals", "LiveCodeBench: scrape past the cutoff", ACTIVE),
    ("private evals", "internal code, rejected papers, HLE held-out", NEWTOK),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 56, b, 14, MUT)
p.text(60, 556, "Contamination is subtle: sources, not just the test set, leak in.", 14, ORANGE, 700)
p.footer("Source: lecture validity slides, original plate. Shell 3: trust, then verify.")
p.save("l12-contamination.svg")

# ---- purpose ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("No one eval rules all", "Purpose picks the benchmark. Declare the purpose first.")
rows = [
    ("buy a model", "ecological validity: GDPVal, clinician tasks", TEAL),
    ("measure intelligence", "exams, ARC: hard, clean, unreal", BLUE),
    ("improve the model", "perplexity: smooth, fast, internal", ACTIVE),
    ("ship safely", "HarmBench, AIR-Bench: refusal, taxonomy", ORANGE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 96
    p.panel(60, yy, 840, 80, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="80" rx="5" fill="{col}"/>')
    p.text(84, yy + 52, b, 14, MUT)
p.footer("Source: lecture closing slides, original plate. Shell 5: purpose first.")
p.save("l12-eval-purpose.svg")
