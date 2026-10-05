import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- transform ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Raw data is not text", "HTML, PDFs, repos in. Linearized text out. Every step loses something.")
p.panel(60, 140, 840, 130, "HTML -> text")
p.text(84, 188, "strip boilerplate, keep content. Tables are the hard case.", 15, INK, 600)
p.text(84, 214, "rule-based: fast, imperfect. Trafilatura / Resiliparse win.", 14, MUT)
p.panel(60, 296, 840, 130, "PDFs: rare, valuable")
p.text(84, 344, "often truncated in crawls. OCR with VLMs is expensive.", 15, INK, 600)
p.text(84, 370, "a PDF means someone had something to say: higher average quality", 15, TEAL, 700)
p.footer("Source: lecture transformation slides, original plate. Shell 1: raw to text.")
p.save("l14-transform.svg")

# ---- filtering ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Filtering: find more of what you want", "Target T (small, good), raw R (huge, messy). Keep the subset of R that looks like T.")
rows = [
    ("generative", "KenLM 5-gram on T, keep low perplexity (OpenMathText)", BLUE),
    ("discriminative", "fastText linear: T positive, R sample negative, threshold", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 94, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="94" rx="5" fill="{col}"/>')
    p.text(84, yy + 64, b, 14, MUT)
p.text(60, 380, "Must generalize AND run on 100T tokens. Language ID: easy. Math: 15B tokens that beat 20x.", 15, INK, 700)
p.text(60, 414, "Quality has no universal definition: define it, then filter for it.", 14, MUT)
p.footer("Source: lecture filtering slides, original plate. Shell 2: the recipe.")
p.save("l14-filtering.svg")

# ---- threshold ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("The threshold depends on the budget", "Train longer -> tolerate lower quality. Epoching flips the winner.")
p.panel(60, 140, 840, 140, "Michael Ryan experiment (157M params)")
p.text(84, 188, "early: DCLM (high quality) wins by a mile", 15, TEAL, 700)
p.text(84, 214, "late, after epoching: unfiltered Resiliparse catches up", 15, ORANGE, 700)
p.text(60, 310, "No optimal threshold exists. High-quality data is finite: epoching it overfits.", 15, INK, 700)
p.footer("Source: lecture filtering slides, original plate. Shell 2: budget matters.")
p.save("l14-quality-threshold.svg")

# ---- dedupe ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Deduplication", "Exact and near. Waste flops, memorize copyrighted text, or both.")
rows = [
    ("exact", "hash, remove all but one. C4: 3-sentence spans (breaks coherence!)", BLUE),
    ("near", "Jaccard > 0.99 via MinHash LSH. Licenses, headers, templates", ACTIVE),
    ("the exhibit", "one gas mask description appeared 61,000 times in C4", ORANGE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 94, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="94" rx="5" fill="{col}"/>')
    p.text(84, yy + 64, b, 14, MUT)
p.text(60, 496, "Dedupe across the whole dataset, not per source. Decontaminate the test set too.", 14, INK, 700)
p.footer("Source: lecture dedup slides, original plate. Shell 3: stop repeating yourself.")
p.save("l14-dedupe.svg")

# ---- minhash lsh ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("MinHash + LSH: near-dup in linear time", "Jaccard via hash collisions, sharpened into a phase transition.")
p.panel(60, 140, 840, 120, "MinHash")
p.text(84, 186, "P(collision) = Jaccard. Hash every element, keep the minimum.", 15, INK, 600)
p.text(84, 212, "permutation intuition: first row wins, ties broken by min", 14, MUT)
p.panel(60, 286, 840, 160, "LSH: b bands of r hashes")
p.text(84, 334, "band matches if ALL r agree; pair collides if ANY band matches", 15, INK, 600)
p.text(84, 360, "and-or structure sharpens: r right-shifts, b left-shifts the S-curve", 15, ACTIVE, 700)
p.text(84, 386, "threshold ~ (1/b)^(1/r). At the center, collision probability = 0.64.", 14, MUT)
p.text(60, 470, "Real setting: b=20, r=450. Linear time, no n-squared.", 15, INK, 700)
p.footer("Source: lecture MinHash/LSH slides, original plate. Shell 3: the algorithm.")
p.save("l14-minhash-lsh.svg")

# ---- mixing ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Data mixing: a distribution over sources", "Vibes, uniform, proportional. All can silently epoch you.")
p.panel(60, 140, 840, 150, "the 50-epoch trap")
p.text(84, 188, "10T low-quality + 10B high-quality, uniform, train 1T tokens", 15, INK, 600)
p.text(84, 214, "low: 5% touched once. high: every token 50 times. Overfit.", 15, ORANGE, 700)
p.panel(60, 316, 840, 130, "fixes")
p.text(84, 364, "UniMax: cap epochs per source (e.g. 20), reallocate", 15, INK, 600)
p.text(84, 390, "keep diversity: sources are incomparable, not rankable", 14, MUT)
p.footer("Source: lecture mixing slides, original plate. Shell 3: count your epochs.")
p.save("l14-mixing.svg")

# ---- regmix ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("RegMix: learn the mixture", "Swarm of small models -> regression -> optimize -> scale up.")
rows = [
    ("1. sample mixtures", "Dirichlet over domains, train small proxies (300M)", BLUE),
    ("2. fit regression", "mixture weights -> loss (log-linear works)", ACTIVE),
    ("3. optimize", "find the best mixture, train the big model", TEAL),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 56, b, 14, MUT)
p.text(60, 460, "Two leaps of faith: the optimum is out-of-distribution for the regressor, and small-scale optima may not transfer.", 14, ORANGE, 700)
p.text(60, 494, "Simulated epoching: downsample small runs so they feel large-scale scarcity.", 14, INK, 700)
p.footer("Source: lecture RegMix/Olmix slides, original plate. Shell 4: optimize the mix.")
p.save("l14-regmix.svg")

# ---- post-training ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Post-training data: environments, tasks, teachers", "Task-dependent data. Almost all synthetic in the open community.")
p.panel(60, 140, 840, 120, "the recipe")
p.text(84, 186, "environments (GitHub repos) -> tasks (prompts) -> responses (strong teacher)", 15, INK, 600)
p.text(84, 212, "human teachers are slow and expensive; frontier uses hybrid human-AI", 14, MUT)
rows = [
    ("OpenThoughts", "1.2M reasoning examples. Better model != better teacher (QwQ-32B > R1).", TEAL),
    ("SWE-smith", "50k synthetic tasks: bugs injected into real repos, verified", BLUE),
    ("SWE-Zero", "300k trajectories with NO execution (grep-only). 12M scaled up.", ACTIVE),
]
y0 = 286
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 96
    p.panel(60, yy, 840, 80, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="80" rx="5" fill="{col}"/>')
    p.text(84, yy + 52, b, 13, MUT)
p.footer("Source: lecture post-training slides, original plate. Shell 4: the new data economy.")
p.save("l14-posttraining.svg")
