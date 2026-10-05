import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- SFT data history ----
p = Plate(960, 620); p.defs_arrow()
y = p.title("SFT data: a short history", "From repurposed NLP tasks to agentic tool calls.")
rows = [
    ("FLAN", "all NLP benchmarks as tasks. Unnatural, inherited deficiencies.", MUT),
    ("self-instruct", "models generate their own data. Better than some annotators.", MUT),
    ("Alpaca / Vicuna", "distill ChatGPT traces. Chat-like behavior, cheap.", BLUE),
    ("Open Assistant", "crowdsourced experts, Wikipedia-style. Stalled ~10k.", TEAL),
    ("WizardLM / Tulu3", "increasingly clever synthetic generation", ACTIVE),
    ("agentic (Nemotron)", "tool calls and to-do lists as supervised targets", NEWTOK),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 76
    p.panel(60, yy, 840, 64, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="64" rx="5" fill="{col}"/>')
    p.text(84, yy + 42, b, 13, MUT)
p.footer("Source: lecture SFT-history slides, original plate. Shell 4: the timeline.")
p.save("l15-sft-data-history.svg")

# ---- pitfalls ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("SFT pitfalls", "Style is not capability. Tail knowledge teaches hallucination.")
p.panel(60, 140, 840, 130, "style vs capability")
p.text(84, 188, "bullet lists and length win AlpacaEval, change no benchmark", 15, ORANGE, 700)
p.text(84, 214, "control style separately from capabilities", 14, MUT)
p.panel(60, 296, 840, 150, "tail knowledge -> hallucination")
p.text(84, 344, "teaching 'cite references' on unknown facts teaches fake citations", 15, INK, 600)
p.text(84, 370, "model generalizes the template, not the knowledge. RL can recalibrate.", 14, TEAL, 700)
p.footer("Source: lecture SFT-pitfalls slides, original plate. Shell 3: the traps.")
p.save("l15-sft-pitfalls.svg")

# ---- safety ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Safety: the last line of defense", "Balance violation rate against false refusals.")
p.panel(60, 140, 840, 120, "the tradeoff")
p.text(84, 186, "'how do I kill a Python process' must not refuse", 15, INK, 600)
p.text(84, 212, "few thousand examples (Llama 2). OLMo mined WildChat: 50k.", 14, MUT)
p.text(60, 290, "500 examples is surgical: malicious-instruction rate drops dramatically.", 16, TEAL, 700)
p.text(60, 324, "But fine-grained safety needs large-scale collection anyway.", 14, MUT)
p.footer("Source: lecture safety slides, original plate. Shell 2: the tradeoff.")
p.save("l15-safety.svg")

# ---- midtraining ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Mid-training: the blur", "Instruction data now lands in the decay phase of pre-training.")
p.panel(60, 140, 840, 150, "the decay phase")
p.text(84, 188, "highest quality data, lowest learning rate, closest to deployment", 15, INK, 600)
p.text(84, 214, "MiniCPM: internet mix -> chatty mix (UltraChat, StackExchange QA)", 14, MUT)
p.text(60, 320, "'Base model' is a lie: it already trained on chat data.", 16, ORANGE, 700)
p.text(60, 354, "Mixtures are trial and error. Decay ablations are cheap enough to run.", 14, MUT)
p.footer("Source: lecture mid-training slides, original plate. Shell 2: the blur.")
p.save("l15-midtraining.svg")

# ---- RLHF concept ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("SFT fits. RLHF maximizes.", "Generative modeling vs reward maximization: different games.")
p.panel(60, 140, 840, 130, "SFT / pre-training")
p.text(84, 188, "fit a distribution: predict the next token on the data", 15, INK, 600)
p.panel(60, 296, 840, 130, "RLHF")
p.text(84, 344, "maximize a reward: the policy may collapse to one answer per prompt", 15, ACTIVE, 700)
p.text(60, 450, "Why RL: raters prefer AI outputs over their own writing. Verification beats generation.", 15, INK, 700)
p.footer("Source: lecture RLHF-concept slides, original plate. Shell 1: the shift.")
p.save("l15-rlhf-concept.svg")

# ---- annotation ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Annotators shape the model", "Who rates you decides what you become.")
rows = [
    ("shift to experts", "bachelor/master majority, $50-100+/hr. Doctors, lawyers for white-collar tasks.", TEAL),
    ("ideology transfer", "InstructGPT annotators: SE Asian + West Coast -> models align Buddhist/Hindu/atheist", ORANGE),
    ("subliminal transfer", "'I like owls' data -> owl preference. Formatting vs factuality splits.", ORANGE),
    ("verification crisis", "annotators use ChatGPT; Bard raters had <1 min per long response", MUT),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 104
    p.panel(60, yy, 840, 88, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="88" rx="5" fill="{col}"/>')
    p.text(84, yy + 58, b, 13, MUT)
p.footer("Source: lecture annotation slides, original plate. Shell 4: the human layer.")
p.save("l15-annotation.svg")

# ---- model annotation ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Model-based annotation wins (for catch-up)", "GPT-4 annotates like humans at 1/10 the cost. Humans still needed at the frontier.")
p.panel(60, 140, 840, 130, "the verdict")
p.text(84, 188, "Zephyr tried human-only, ended with AI feedback. UltraChat/UltraFeedback standard.", 15, INK, 600)
p.text(84, 214, "Tulu3: model-based for the whole pipeline", 14, MUT)
p.panel(60, 296, 840, 120, "the catch")
p.text(84, 342, "length hacking: longer answers win model judges. RLHF on length alone works.", 15, ORANGE, 700)
p.footer("Source: lecture model-annotation slides, original plate. Shell 3: the verdict.")
p.save("l15-model-annotation.svg")

# ---- DPO ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("DPO: RLHF without the RL", "No reward model, no sampling. Just gradients: up on good, down on bad.")
p.panel(60, 140, 840, 140, "the derivation (one assumption)")
p.text(84, 188, "assume the policy can be anything -> closed form: tilt reference by reward", 15, INK, 600)
p.text(84, 214, "solve for the implied reward, plug into Bradley-Terry -> DPO loss", 14, MUT)
p.panel(60, 306, 840, 140, "the gradient intuition")
p.text(84, 354, "increase likelihood of the winner, decrease the loser", 15, ACTIVE, 700)
p.text(84, 380, "step size scales with surprise: wrong predictions move more", 14, MUT)
p.text(60, 480, "DPO vs PPO: fragile, setup-dependent. Variants (SimPO, length-norm) barely matter.", 14, MUT)
p.footer("Source: lecture DPO slides, original plate. Shell 3: the simplification.")
p.save("l15-dpo.svg")
