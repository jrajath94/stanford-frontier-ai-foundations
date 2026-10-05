import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- pipeline ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Training comes in stages", "Large low-quality data first. Small high-quality data last.")
rows = [
    ("pre-training", "raw web documents, trillions of tokens", TEAL),
    ("mid-training", "higher quality data, long context, instruction data", BLUE),
    ("post-training", "chat transcripts, RL environments, safety", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 94, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="94" rx="5" fill="{col}"/>')
    p.text(84, yy + 64, b, 14, MUT)
    if i < 2: p.arrow(430, yy + 94, 430, yy + 110)
p.text(60, 496, "Base model = pre + mid. Instruct/chat = post. The lines are blurring.", 14, MUT)
p.footer("Source: lecture pipeline slides, original plate. Shell 1: stages.")
p.save("l13-pipeline.svg")

# ---- crawl barriers ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("You cannot crawl the whole internet", "'Trained on the internet' does not type-check.")
rows = [
    ("dynamic web", "apps, forms, Discord: crawling is not doing", MUT),
    ("deep web", "no hyperlinks to follow", MUT),
    ("auth walls", "Facebook, X, LinkedIn, NYT: walled gardens", MUT),
    ("robots.txt", "Consent in Crisis: ~50% fully restricted by mid-2023", ORANGE),
    ("anti-bot", "Cloudflare, CAPTCHA, IP blocks, rate limits", MUT),
    ("terms of service", "'no AI training' clauses rising since 2016", MUT),
    ("licenses", "everything is copyrighted by default", MUT),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 62
    p.panel(60, yy, 840, 50, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="50" rx="5" fill="{col}"/>')
    p.text(84, yy + 34, b, 13, INK)
p.footer("Source: lecture crawling + Consent in Crisis slides, original plate. Shell 1: barriers.")
p.save("l13-crawl.svg")

# ---- copyright ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("Copyright: license or fair use", "Everything on the web is copyrighted. That is not the end of the story.")
p.panel(60, 140, 840, 130, "license or fair use")
p.text(84, 188, "get a license: Creative Commons, or pay for one", 15, INK, 600)
p.text(84, 214, "or claim fair use: purpose, nature, amount, market effect", 15, INK, 600)
p.panel(60, 296, 840, 130, "fair use is semantics, not n-grams")
p.text(84, 344, "Harry Potter the character is copyrightable, not any book", 15, INK, 600)
p.text(84, 370, "Authors Guild v Google: snippets ruled fair use after 11 years", 15, INK, 600)
p.panel(60, 452, 840, 110, "2025 rulings")
p.text(84, 496, "training = fair use; pirating books = illegal (Anthropic $1.5B)", 15, ORANGE, 700)
p.text(84, 522, "Meta the same. Narrow rulings, active area.", 14, MUT)
p.footer("Source: lecture copyright + lawsuits slides, original plate. Shell 2: the law.")
p.save("l13-copyright.svg")

# ---- common crawl ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("Common Crawl: the open crawl", "Monthly web dumps since 2007. Raw, messy, indispensable.")
rows = [
    ("scale", "~3-5B pages per dump, ~300B total, ~372TB per dump text", TEAL),
    ("formats", "WARC: raw HTTP response. WET: lossy extracted text", BLUE),
    ("extraction matters", "trafilatura / resiliparse beat WET (DataComp)", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 94, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="94" rx="5" fill="{col}"/>')
    p.text(84, yy + 64, b, 14, MUT)
p.footer("Source: lecture Common Crawl slides, original plate. Shell 2: the raw material.")
p.save("l13-commoncrawl.svg")

# ---- quality pockets ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Three quality pockets", "Not uniform web: high-density sources worth special handling.")
rows = [
    ("Wikipedia", "67M articles, periodic dumps (do not crawl). Poisonable: Carlini dump-timing attack", TEAL),
    ("GitHub", "420M repos, 28M public. Permissive licenses only. Archive: issues, PRs, comments", BLUE),
    ("arXiv", "3M submissions since 1991. LaTeX source. Mostly Creative Commons", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 94, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="94" rx="5" fill="{col}"/>')
    p.text(84, yy + 62, b, 13, MUT)
p.text(60, 496, "Code matters for reasoning, not just coding.", 14, INK, 700)
p.footer("Source: lecture pockets slides, original plate. Shell 2: where the good stuff is.")
p.save("l13-quality-pockets.svg")

# ---- dataset history ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("A decade of pre-training datasets", "Bigger, messier, more filtered. The watershed: Books3 got Meta in trouble.")
rows = [
    ("BERT (2018)", "Wikipedia + Books (BookCorpus): documents, not sentences", MUT),
    ("GPT-2 (2019)", "Reddit >3 karma outgoing links: 40GB, 8M pages", MUT),
    ("CCNet (2019)", "Wikipedia-like LM classifier for quality", MUT),
    ("C4 (2019)", "rules only: 156B tokens, 800GB", MUT),
    ("GPT-3 (2020)", "CC + WebText + Books1/2 + Wiki: 500GB, quality classifier, fuzzy dedup", BLUE),
    ("The Pile (2020)", "grassroots: Books3 from a shadow library", ORANGE),
    ("Llama 1 (2022)", "1.2T tokens. Books3 announced -> the watershed", ORANGE),
    ("Refined/FineWeb", "web-only: 5T / 15T tokens, rules-first", TEAL),
    ("DCLM (2024)", "fastText classifier: keep 1.4% of 240T tokens", ACTIVE),
    ("Nemotron (2024)", "educational scoring + synthetic data: 6T tokens", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 44
    p.panel(60, yy, 840, 36, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="36" rx="5" fill="{col}"/>')
    p.text(84, yy + 25, b, 12, INK)
p.footer("Source: lecture dataset history, original plate. Shell 4: the timeline.")
p.save("l13-dataset-history.svg")

# ---- rules vs classifiers ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Rules vs classifiers", "Two schools of quality filtering. The funnel is the point.")
p.panel(60, 140, 840, 130, "rules")
p.text(84, 188, "C4: lines end in punctuation, 5+ words, no curly braces, no bad words", 15, INK, 600)
p.text(84, 214, "control, no ML bias. RefinedWeb, FineWeb, Gopher.", 14, MUT)
p.panel(60, 296, 840, 130, "classifiers")
p.text(84, 344, "CCNet: Wikipedia-like. GPT-3: high-quality. DCLM: OpenHermes+ELI5.", 15, INK, 600)
p.text(84, 370, "magical: 240T tokens in, 1.4% out, and it works better", 14, ORANGE, 700)
p.text(60, 470, "Filtering is the most important lever: 200T -> 3T.", 15, INK, 700)
p.footer("Source: lecture filtering slides, original plate. Shell 3: the lever.")
p.save("l13-rules-vs-classifiers.svg")

# ---- stack + common pile ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Code data and license-only data", "Two special projects: The Stack for code, Common Pile for the risk-averse.")
p.panel(60, 140, 840, 150, "The Stack")
p.text(84, 188, "137 repos -> 3TB. v2: issues, PRs, comments linearized as XML", 15, INK, 600)
p.text(84, 214, "low-resource languages paired with LLVM IR. PII redacted, bots filtered", 14, MUT)
p.panel(60, 316, 840, 160, "Common Pile")
p.text(84, 364, "only permissively licensed data: 8TB. Beats 2023 models, trails Qwen", 15, INK, 600)
p.text(84, 390, "watch for license laundering: collection licenses do not cover works", 15, ORANGE, 700)
p.text(84, 416, "no synthetic data: models were trained on unlicensed data anyway", 14, MUT)
p.footer("Source: lecture Stack + Common Pile slides, original plate. Shell 2: two projects.")
p.save("l13-stack-commonpile.svg")
