import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- loglog power law ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Log-log line = power law", "Error ~ 1/n^alpha. Mean estimation: -1. Neural nets: -0.1, nonparametric-like.")
p.panel(60, 140, 840, 200, "slope on log-log = exponent")
p.text(84, 190, "mean estimation: error = sigma^2 / n  ->  slope -1", 16, INK, 600)
p.text(84, 220, "neural nets (Kaplan/Hestness): slope ~ -0.1 to -0.3", 16, ORANGE, 700)
p.text(84, 250, "nonparametric in D dims: n^(-1/D). Nets learn like smooth functions in ~10 dims.", 15, INK)
p.text(84, 290, "Far from the noise floor: the line tapers at the asymptote.", 14, MUT)
p.footer("Source: lecture stats slides, original plate. Shell 2: the slope is the story.")
p.save("l09-loglog.svg")

# ---- history ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Scaling laws are old", "1993 to 2022: the same power law, rediscovered each decade.")
hist = [
    ("1993", "Cortes + Vapnik", "fit error decay, predict from small data", MUT),
    ("2001", "Banko + Brill", "more data beats better algorithms (NLP)", MUT),
    ("2012", "Kolachina et al.", "BLEU scaling, same power laws", MUT),
    ("2017", "Hestness et al.", "neural scaling across domains; emergence noted", TEAL),
    ("2020", "Kaplan (OpenAI)", "neural scaling laws; N^0.27 D^0.73", BLUE),
    ("2022", "Chinchilla (DeepMind)", "20 tokens/param; joint optimum fixed", ACTIVE),
]
y0 = 140
for i, (a, b, c, col) in enumerate(hist):
    yy = y0 + i * 64
    p.panel(60, yy, 840, 54, "")
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="54" rx="5" fill="{col}"/>')
    p.text(84, yy + 22, a + "  " + b, 14, INK, 700)
    p.text(84, yy + 42, c, 13, MUT)
p.footer("Source: lecture history slides, original plate. Shell 1: credit the lineage.")
p.save("l09-history.svg")

# ---- data engineering ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Data scaling for engineering", "Mixtures move intercepts. Repetition holds 4 epochs. Filters loosen with scale.")
rows = [
    ("mixtures", "slopes stay, intercepts shift. Best mix at small scale = best at large.", TEAL),
    ("repetition", "up to 4 epochs: no harm. Past it: worse than fresh data. Infinite compute: ensemble.", ORANGE),
    ("filtering", "small compute: filter hard. Big compute: loosen, or you rerun the same data.", BLUE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 120
    p.panel(60, yy, 840, 104, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="104" rx="5" fill="{col}"/>')
    p.text(84, yy + 44, b, 14, INK)
    p.text(84, yy + 70, "Lesson: interventions move intercepts, rarely slopes.", 13, MUT)
p.footer("Source: lecture data slides, original plate. Shell 4: slopes are stubborn.")
p.save("l09-data-uses.svg")

# ---- critical batch ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Critical batch size", "Noise-limited: perfect returns. Bias-limited: diminishing. B_crit grows as loss drops.")
p.panel(60, 140, 400, 200, "noise-limited")
p.text(84, 190, "extra examples cut", 15, INK)
p.text(84, 216, "gradient variance", 15, INK)
p.text(84, 250, "perfect scaling", 15, TEAL, 700)
p.panel(500, 140, 400, 200, "bias-limited")
p.text(524, 190, "local view vs global", 15, INK)
p.text(524, 216, "optimum disagree", 15, INK)
p.text(524, 250, "diminishing returns", 15, ORANGE, 700)
p.panel(60, 370, 840, 100, "B_crit = min examples / min steps; scales as a power law in loss")
p.text(84, 424, "Closer to the minimum, noise matters more. Big runs earn big batches.", 14, MUT)
p.footer("Source: lecture critical-batch slides, original plate. Shell 4: size the batch to the noise.")
p.save("l09-critical-batch.svg")

# ---- kaplan vs chinchilla ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Kaplan vs Chinchilla", "Same data-ish, different answers. Details decide: N^0.27 vs N^0.5.")
p.panel(60, 140, 400, 200, "Kaplan 2020")
p.text(84, 190, "N ~ C^0.27, D ~ C^0.73", 17, INK, 700)
p.text(84, 220, "train bigger models", 15, ORANGE, 700)
p.text(84, 250, "GPT-3 era giants", 14, MUT)
p.panel(500, 140, 400, 200, "Chinchilla 2022")
p.text(524, 190, "N ~ C^0.5, D ~ C^0.5", 17, INK, 700)
p.text(524, 220, "20 tokens per param", 15, TEAL, 700)
p.text(524, 250, "smaller, longer-trained", 14, MUT)
p.panel(60, 370, 840, 120, "why Kaplan lost")
p.text(84, 414, "excluded unembedding params; warmup too short to converge; fixed batch size suboptimal for small models.", 14, INK)
p.text(84, 440, "Scaling laws are lower bounds of a recipe. Bad recipe, bad law.", 14, ORANGE, 700)
p.footer("Source: lecture Chinchilla slides, original plate. Shell 4: details decide.")
p.save("l09-kaplan-vs-chinchilla.svg")

# ---- three methods ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Three ways to fit the joint law", "Lower envelope, IsoFLOP, parametric fit. IsoFLOP is the robust default.")
rows = [
    ("1. lower envelope", "best loss per FLOP across runs; scatter model sizes", MUT),
    ("2. IsoFLOP", "fix FLOPs, sweep N vs D, take minima, draw the line", TEAL),
    ("3. parametric fit", "fit the joint functional form; curve fitting is fiddly", ORANGE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 58, b, 14, MUT)
p.text(60, 470, "Epoch AI refit: method 3 was underfit in the paper. Redone, it agrees with 1 and 2.", 14, MUT)
p.footer("Source: lecture Chinchilla slides, original plate. Shell 4: robustify the fit.")
p.save("l09-three-methods.svg")

# ---- overtrain ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("Overtrain for serving", "Training-optimal is not serving-optimal. Small and capable beats big and bloated.")
p.panel(60, 140, 840, 120, "tokens per parameter")
p.text(84, 190, "GPT-3: 3  (undertrained)   ->   Chinchilla: 20   ->   modern: 100+ (overtrained)", 16, INK, 600)
p.panel(60, 290, 840, 140, "why")
p.text(84, 340, "most compute goes to R&D and serving, not training", 15, INK, 600)
p.text(84, 366, "serving wants small models that are capable", 15, TEAL, 700)
p.footer("Source: lecture discussion, original plate. Shell 5: optimize the deployment, not the paper.")
p.save("l09-overtrain.svg")

# ---- upstream downstream ----
p = Plate(960, 520); p.defs_arrow()
y = p.title("Perplexity is not downstream", "Scaling laws are clean for loss. Transfer to tasks is uncertain.")
p.panel(60, 140, 400, 200, "upstream: perplexity")
p.text(84, 190, "linear in log params", 15, TEAL, 700)
p.text(84, 216, "beautiful, predictable", 15, INK)
p.panel(500, 140, 400, 200, "downstream: tasks")
p.text(524, 190, "noisy, jagged", 15, ORANGE, 700)
p.text(524, 216, "NL12 best ppl, NL32XL best tasks", 14, INK)
p.text(60, 390, "Fit on perplexity. Verify transfer separately. Post-training inherits pre-training sins.", 15, INK, 700)
p.footer("Source: lecture Tay et al. slide, original plate. Shell 4: measure both, trust one.")
p.save("l09-upstream-downstream.svg")
