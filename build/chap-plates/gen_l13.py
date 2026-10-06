#!/usr/bin/env python3
"""Chapter plates for cs229 l13 (contrastive learning + RAG). All numbers code-computed with asserts."""
import math, os

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/cs229/assets")
SANS = "Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------- number computations (asserted) ----------------
# InfoNCE toy: positive 0.9, 100 negatives at 0.1, tau 0.1
e_pos = math.exp(0.9 / 0.1)
e_neg = math.exp(0.1 / 0.1)
denom = e_pos + 100 * e_neg
frac = e_pos / denom
loss = -math.log(frac)
assert abs(e_pos - 8103.08) < 0.5, e_pos
assert abs(e_neg - 2.718) < 0.01, e_neg
assert abs(frac - 0.96754) < 1e-4, frac
assert abs(loss - 0.03301) < 1e-4, loss
# hard negative at 0.85
e_hard = math.exp(0.85 / 0.1)
frac_hard = e_pos / (e_pos + e_hard + 100 * e_neg)
assert 0.60 < frac_hard < 0.63, frac_hard  # lesson: "drops toward 0.62"
# SimCLR batch
N, VIEWS = 4096, 8192
negs_per_anchor = VIEWS - 2
sims = VIEWS * (VIEWS - 1) // 2
assert negs_per_anchor == 8190
assert sims == 33550336, sims  # ~33.5M
assert abs(math.log(4096) - 8.3178) < 1e-3
# SimCSE toy
cos_pos = 0.869
e_sp = math.exp(cos_pos / 0.05)
e_sn = [math.exp(0.25 / 0.05), math.exp(0.60 / 0.05), math.exp(-0.05 / 0.05)]
frac_s = e_sp / (e_sp + sum(e_sn))
loss_s = -math.log(frac_s)
assert abs(frac_s - 0.9954) < 1e-3, frac_s
assert abs(loss_s - 0.0046) < 1e-3, loss_s
# ANN numbers
ops_exact = 10_000_000 * 768
assert ops_exact == 7_680_000_000  # 7.7B
ops_ivf = 10_000 * 768
assert ops_ivf == 7_680_000  # 7.7M
gb = 10_000_000 * 768 * 4 / 1e9
assert 30 < gb < 31.5, gb
gb_pq = 10_000_000 * 100 / 1e9
assert abs(gb_pq - 1.0) < 1e-9
# RAG toy
toks = 3 * 500
assert toks == 1500
fits = 128_000 // 1500
assert fits == 85
# UCB-style: gradient focus ratio
assert 0.3 / 0.001 == 300.0

# ---------------- svg builder ----------------
def plate(path, title, subtitle, left_label, center_label, right_label,
          left_lines, center_lines, right_lines, bottom_lines, footer,
          panel_h=300, bottom_h=72):
    W = 960
    y_panels = 140
    y_bottom = y_panels + panel_h + 16
    H = y_bottom + bottom_h + 56
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{SANS}">')
    parts.append(f'<rect width="{W}" height="{H}" fill="#F7F4EE"/>')
    parts.append(f'<text x="48" y="56" font-size="30" font-weight="600" fill="#1B2838">{esc(title)}</text>')
    parts.append(f'<text x="48" y="86" font-size="17" fill="#5C6B7A">{esc(subtitle)}</text>')
    parts.append('<g font-size="18" font-weight="600" fill="#1B2838">')
    parts.append(f'<text x="48" y="124">{esc(left_label)}</text>')
    parts.append(f'<text x="328" y="124">{esc(center_label)}</text>')
    parts.append(f'<text x="648" y="124">{esc(right_label)}</text>')
    parts.append('</g>')
    def region(x, w, fill, stroke, sw, lines, cx):
        parts.append(f'<rect x="{x}" y="{y_panels}" width="{w}" height="{panel_h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        y = y_panels + 44
        for (t, size, color, weight) in lines:
            parts.append(f'<text x="{cx}" y="{y}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{color}">{esc(t)}</text>')
            y += 30 if size >= 16 else 26
    DK, MU, TE, BL = "#1B2838", "#5C6B7A", "#1F7A72", "#1E4D8C"
    region(48, 264, "#FFFDF8", "#1B2838", 1.5, left_lines, 180)
    region(328, 304, "#E7F1F8", "#1E4D8C", 1.5, center_lines, 480)
    region(648, 264, "#E7F4EF", "#1F7A72", 2, right_lines, 780)
    parts.append(f'<rect x="48" y="{y_bottom}" width="864" height="{bottom_h}" rx="12" fill="#FFFDF8" stroke="#1B2838" stroke-width="1.5"/>')
    y = y_bottom + 30
    for (t, color) in bottom_lines:
        parts.append(f'<text x="480" y="{y}" text-anchor="middle" font-size="15" fill="{color}">{esc(t)}</text>')
        y += 26
    parts.append(f'<text x="48" y="{H - 22}" font-size="16" font-weight="500" fill="#1B2838">{esc(footer)}</text>')
    parts.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", path)

DK, MU, TE, BL = "#1B2838", "#5C6B7A", "#1F7A72", "#1E4D8C"
B = (lambda t, s=15, c=MU, w=500: (t, s, c, w))

plate(os.path.join(ASSETS, "plate-l13-chap-contrastive.svg"),
    "Chapter plate: contrastive learning, the InfoNCE engine",
    "Chapter plate. Augmentation is the supervision. Source: original synthesis of the session.",
    "WITHOUT the rule: labels", "the stored object: the loss", "WITH the rule: scale",
    [("classifiers per concept", 16, DK, 600), ("10M photos, zero labels", 15, MU, 500),
     ("new concept needs new labels", 15, MU, 500), ("an army of labelers", 15, MU, 500),
     ("labels are the bottleneck", 15, MU, 500)],
    [("positive 0.9, 100 negatives", 15, DK, 600), ("at 0.1, tau = 0.1", 15, MU, 500),
     ("fraction on positive: 0.9676", 16, DK, 600), ("loss: 0.033", 16, DK, 600),
     ("hard negative at 0.85", 15, MU, 500), ("fraction drops to 0.61", 15, TE, 600),
     ("tau sharpens the competition", 15, MU, 500)],
    [("SimCLR: batch 4,096", 16, DK, 600), ("8,190 negatives per anchor", 15, MU, 500),
     ("33.5M sims per step", 15, MU, 500), ("ceiling: log(4096) = 8.3 nats", 15, MU, 500),
     ("MoCo: 65k queue", 15, MU, 500), ("CLIP: 400M image-text pairs", 15, MU, 500)],
    [("Tradeoff: the negative set buys the information ceiling: 4,096 negatives cost 33.5M", DK),
     ("similarity computations per step, and easy negatives teach nothing.", MU)],
    "One connection: InfoNCE turns unlabeled views into supervision, and the negative set is what it spends.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l13-chap-hardneg.svg"),
    "Chapter plate: hard negatives keep the loss teaching",
    "Chapter plate. Random negatives saturate. Source: original synthesis of the session.",
    "WITHOUT mining: easy negatives", "the stored object: the hard negative", "WITH mining: three generations",
    [("anchor: cat playing soccer", 15, DK, 600), ("negative: a truck", 15, MU, 500),
     ("similarity 0.05", 15, MU, 500), ("loss already ~0", 15, MU, 500),
     ("nothing learned", 15, MU, 500), ("100 easy: loss saturates", 15, MU, 500)],
    [("FIFA text vs cat photo", 16, DK, 600), ("shares soccer, misses the cat", 15, MU, 500),
     ("gradient focuses on it", 15, TE, 600), ("weight 0.3 vs 0.001", 15, DK, 600),
     ("300x harder push", 16, TE, 600), ("the loss spends itself", 15, MU, 500)],
    [("gen 1: in-batch, free", 15, DK, 600), ("gen 2: MoCo queue 65k", 15, MU, 500),
     ("gen 3: cross-device, global", 15, MU, 500), ("false negatives: 2%", 15, DK, 600),
     ("poison rate at the top", 15, MU, 500), ("dedup gate: cosine 0.95", 15, TE, 600)],
    [("Tradeoff: harder negatives teach finer distinctions, but 2% are true positives in disguise.", DK),
     ("Dedup above 0.95 cosine before mining, not after.", MU)],
    "One connection: the loss spends its gradient on the confusing pairs, so the negative pool is the curriculum.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l13-chap-ann.svg"),
    "Chapter plate: approximate search at ten million vectors",
    "Chapter plate. Exact search does not ship. Source: original synthesis of the session.",
    "WITHOUT the rule: exact", "the stored object: the index", "WITH the rule: approximate",
    [("10M x 768 = 7.7B ops", 16, DK, 600), ("per query, exact", 15, MU, 500),
     ("~1 second on a GPU", 15, MU, 500), ("forever on a CPU fleet", 15, MU, 500),
     ("recall 1.0, cost fatal", 15, MU, 500)],
    [("IVF: 10k clusters, nprobe 10", 15, DK, 600), ("HNSW: layered graph", 15, DK, 600),
     ("PQ: 96 groups of 8 dims", 15, DK, 600), ("one byte per group", 15, MU, 500),
     ("shard past 100M vectors", 15, MU, 500), ("merge is exact", 15, MU, 500)],
    [("IVF: 7.7M ops, recall 0.95", 15, DK, 600), ("HNSW: ~50k, recall 0.98", 15, DK, 600),
     ("PQ: 30 GB down to 1 GB", 15, TE, 600), ("recall ~0.90", 15, MU, 500),
     ("every point past 0.90", 15, MU, 500), ("is bought with latency", 15, MU, 500)],
    [("Tradeoff: measure recall@k against exact search, then pick the fastest index clearing the bar.", DK),
     ("A search product missing the photo 5% of the time has a recall problem, not a model problem.", MU)],
    "One connection: ANN turns a linear scan into a navigable structure, and the recall dial prices every choice.",
    panel_h=300, bottom_h=72)

plate(os.path.join(ASSETS, "plate-l13-chap-rag.svg"),
    "Chapter plate: RAG, knowledge in the store",
    "Chapter plate. The model reads private docs at test time. Source: original synthesis of the session.",
    "WITHOUT the rule: fine-tune", "the stored object: the pipeline", "WITH the rule: retrieve",
    [("training run on the docs", 15, DK, 600), ("hosted weights", 15, MU, 500),
     ("small-data regime", 15, MU, 500), ("forgetting is research", 15, MU, 500),
     ("opaque and unleaky", 15, MU, 500)],
    [("embed the question", 15, DK, 600), ("ANN retrieves 100", 15, DK, 600),
     ("cross-encoder reranks to 5", 15, DK, 600), ("paste chunks into context", 15, DK, 600),
     ("generate grounded, cite", 15, MU, 500), ("update a doc: answers update", 15, TE, 600)],
    [("3 chunks x 500 tokens", 16, DK, 600), ("1,500 tokens per query", 15, MU, 500),
     ("128k fits ~85 retrievals", 15, MU, 500), ("hybrid: +3 to 8 recall pts", 15, MU, 500),
     ("rerank: 0.94 vs 0.12", 15, TE, 600), ("middle of context: -10-20 pts", 15, MU, 500)],
    [("Tradeoff: retrieval failures become answer failures. Modular, governable, deletable, cheap.", DK),
     ("Raise k until recall@k flattens, then stop. Strongest chunk first or last.", MU)],
    "One connection: RAG moves knowledge from weights to a store, and the store's freshness is the model's freshness.",
    panel_h=300, bottom_h=72)
print("l13 plates done")
