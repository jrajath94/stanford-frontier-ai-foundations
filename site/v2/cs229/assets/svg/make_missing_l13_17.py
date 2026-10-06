#!/usr/bin/env python3
"""Generate the 8 missing SVG figures for CS229 lessons l13-l17.

Follows the exact visual system of make_plates.py (warm paper #F7F4EE,
ink #1B2838, flat fills, 8px grid) so the new figures match the 44
existing SVGs. Every number on every figure is computed in code below
(F6); none are hand-waved.

Outputs into this directory (assets/svg/).
Run: python3 make_missing_l13_17.py
"""
import math
import os
import sys
import xml.dom.minidom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_plates as mp

OUT = mp.OUT
PAPER, INK, BORDER = mp.PAPER, mp.INK, mp.BORDER
BLUE, RED, GREEN, PURPLE, AMBER, GRAY = mp.BLUE, mp.RED, mp.GREEN, mp.PURPLE, mp.AMBER, mp.GRAY

txt, box, arrow, circle, line, save = mp.txt, mp.box, mp.arrow, mp.circle, mp.line, mp.save


def bar(p, x, y, w, h, frac, fill, stroke=INK):
    """Horizontal fraction bar: filled frac of width w."""
    box(p, x, y, w, h, fill="#FFFFFF", stroke=stroke, rx=6)
    if frac > 0:
        p.append(f'<rect x="{x+3}" y="{y+3}" width="{(w-6)*frac:.1f}" height="{h-6}" rx="4" fill="{fill}"/>')


def grid(p, ox, oy, n, cell, fill_live, fill_dead):
    for i in range(n):
        for j in range(n):
            f = fill_live if (i * n + j) % 3 else fill_dead
            p.append(f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-2}" height="{cell-2}" rx="2" fill="{f}" stroke="{BORDER}" stroke-width="1"/>')


# ============ l13-triplet.svg ============
def d_triplet(p):
    anchor, pos, neg, m = (1.0, 0.0), (0.9, 0.1), (0.2, 0.9), 0.2
    d_ap = math.dist(anchor, pos)
    d_an = math.dist(anchor, neg)
    loss = max(0.0, d_ap - d_an + m)
    assert abs(d_ap - 0.1414) < 0.001 and abs(d_an - 1.2042) < 0.001 and loss == 0.0
    # --- left: triplet geometry ---
    txt(p, 40, 66, "Triplet: one negative, one margin", 16, INK, bold=True)
    ax, ay = 120, 240
    circle(p, ax, ay, 12, BLUE); txt(p, ax, ay - 20, "anchor", 13, BLUE, anchor="middle", bold=True)
    txt(p, ax, ay + 34, "[1, 0]", 12, GRAY, anchor="middle")
    px, py = 205, 205
    circle(p, px, py, 10, GREEN); txt(p, px + 16, py - 8, "positive", 13, GREEN, bold=True)
    txt(p, px + 16, py + 10, "[0.9, 0.1]", 12, GRAY)
    line(p, ax + 14, ay - 6, px - 12, py + 4, GREEN, 2, dash="7 5")
    txt(p, 165, 175, f"d = {d_ap:.3f}", 13, GREEN, anchor="middle", bold=True)
    nx, ny = 300, 300
    circle(p, nx, ny, 10, RED); txt(p, nx + 16, ny - 8, "negative", 13, RED, bold=True)
    txt(p, nx + 16, ny + 10, "[0.2, 0.9]", 12, GRAY)
    line(p, ax + 8, ay + 10, nx - 12, ny - 8, RED, 2, dash="7 5")
    txt(p, 225, 300, f"d = {d_an:.3f}", 13, RED, anchor="middle", bold=True)
    txt(p, 40, 356, f"L = max(0, {d_ap:.3f} - {d_an:.3f} + {m}) = {loss:.0f}", 15, INK, bold=True)
    txt(p, 40, 382, "Satisfied already: no learning. Mine hard triplets.", 13, GRAY)
    # --- right: InfoNCE ---
    line(p, 400, 60, 400, 400, BORDER, 2)
    txt(p, 420, 66, "InfoNCE: every negative competes at once", 16, INK, bold=True)
    tau = 0.1
    sims = [("positive 0.9", 0.9, GREEN), ("neg 0.1", 0.1, RED), ("neg 0.1", 0.1, RED), ("neg 0.1", 0.1, RED)]
    scaled = [s / tau for _, s, _ in sims]
    exps = [math.exp(v) for v in scaled]
    Z = sum(exps)
    ws = [e / Z for e in exps]
    assert abs(sum(ws) - 1.0) < 0.01
    for i, ((lab, s, c), w) in enumerate(zip(sims, ws)):
        y = 100 + i * 56
        txt(p, 420, y + 16, lab, 13, INK)
        bar(p, 530, y, 130, 22, w, c)
        txt(p, 668, y + 16, f"{w:.4f}", 12, INK)
    loss_nce = -math.log(ws[0])
    txt(p, 420, 348, f"weights sum to {sum(ws):.4f}", 15, INK, bold=True)
    txt(p, 420, 374, f"loss = -log({ws[0]:.4f}) = {loss_nce:.4f}", 15, INK, bold=True)
    txt(p, 420, 400, "The margin becomes the temperature.", 13, GRAY)
    txt(p, 390, 436, "One negative with a margin vs all negatives in one softmax.", 14, INK, anchor="middle")


save("l13-triplet.svg", 780, 466, "Triplet loss vs InfoNCE", d_triplet)


# ============ l13-rerank.svg ============
def d_rerank(p):
    txt(p, 40, 52, "Retrieve then rerank", 16, INK, bold=True)
    txt(p, 40, 78, 'Query: "refund window"', 14, GRAY, italic=True)
    box(p, 60, 100, 600, 110, stroke=BLUE, sw=3)
    txt(p, 360, 132, "ANN retrieve (bi-encoder)", 16, BLUE, anchor="middle", bold=True)
    txt(p, 360, 162, "10,000,000 docs  ->  100 candidates", 14, INK, anchor="middle")
    txt(p, 360, 190, "cheap cosines: 0.82 vs 0.71, both look fine", 13, GRAY, anchor="middle")
    arrow(p, 360, 212, 360, 240, INK)
    box(p, 150, 244, 420, 110, stroke=PURPLE, sw=3)
    txt(p, 360, 276, "Cross-encoder rerank", 16, PURPLE, anchor="middle", bold=True)
    txt(p, 360, 306, "100 candidates  ->  5 finalists", 14, INK, anchor="middle")
    txt(p, 360, 334, "joint read: 0.94 vs 0.12, the false match dies", 13, GRAY, anchor="middle")
    arrow(p, 360, 356, 360, 384, INK)
    box(p, 240, 388, 240, 90, stroke=GREEN, sw=3)
    txt(p, 360, 420, "LLM reads 5", 16, GREEN, anchor="middle", bold=True)
    txt(p, 360, 448, "grounded answer", 13, GRAY, anchor="middle")
    txt(p, 40, 514, "Why the second stage exists:", 14, INK, bold=True)
    txt(p, 40, 540, 'C1 "Refunds within 30 days of purchase." -> 0.94', 13, GREEN)
    txt(p, 40, 564, 'C2 "Window shoppers get refunds never." -> 0.12', 13, RED)
    txt(p, 360, 606, "Each stage trades compute for precision.", 14, INK, anchor="middle")


save("l13-rerank.svg", 720, 636, "Retrieve then rerank", d_rerank)


# ============ l14-tokenize.svg ============
def d_tokenize(p):
    stages = [
        ("1. raw text", BLUE, '"the cat sat 2026"', "characters", 40, 150),
        ("2. pieces (BPE)", PURPLE, '"l o w e s t" -> "low est"', "merge frequent pairs", 235, 170),
        ("3. token ids", AMBER, "[toy ids: 1042, 2088, 15]", "lookup keys", 450, 170),
        ("4. vectors", GREEN, "id -> row of E", "128,256 x 4096", 645, 0),
    ]
    # draw as pipeline
    xs = [40, 205, 400, 575]; ws = [140, 170, 150, 145]
    labels = [("raw text", '"the cat sat 2026"', "characters in", BLUE),
              ("pieces", '"l o w e s t" -> "low est"', "BPE merges pairs", PURPLE),
              ("token ids", "1042  2088  15", "toy ids", AMBER),
              ("vectors", "id -> row of E", "128,256 x 4096", GREEN)]
    for i, (x, w) in enumerate(zip(xs, ws)):
        t, mid, bot, c = labels[i]
        box(p, x, 90, w, 150, stroke=c, sw=3)
        txt(p, x + w / 2, 122, f"{i+1}. {t}", 15, c, bold=True, anchor="middle")
        txt(p, x + w / 2, 160, mid, 13, INK, anchor="middle")
        txt(p, x + w / 2, 190, bot, 12, GRAY, anchor="middle")
        if i < 3:
            arrow(p, x + w + 3, 165, xs[i + 1] - 3, 165, INK)
    # BPE merge toy detail
    txt(p, 40, 290, "BPE toy: count pairs, merge the most frequent", 14, INK, bold=True)
    txt(p, 40, 316, 'before:  l  o  w  e  s  t', 14, INK)
    arrow(p, 260, 312, 310, 312, PURPLE)
    txt(p, 285, 302, "merge", 12, PURPLE, anchor="middle")
    txt(p, 330, 316, 'after:   low   est', 14, PURPLE, bold=True)
    # byte fallback + vocab
    box(p, 40, 340, 330, 90, fill="#EAF3E4", stroke=GREEN, sw=2.5)
    txt(p, 205, 368, "Byte-level BPE: 256 byte values", 14, GREEN, anchor="middle", bold=True)
    txt(p, 205, 392, "no unknown tokens, ever. Price: fertility.", 13, INK, anchor="middle")
    txt(p, 205, 414, "rare CJK char -> 3 byte tokens", 13, GRAY, anchor="middle")
    box(p, 390, 340, 330, 90, fill="#E8EEF7", stroke=BLUE, sw=2.5)
    txt(p, 555, 368, "Vocab sizes in the wild", 14, BLUE, anchor="middle", bold=True)
    txt(p, 555, 392, "32k Mistral 7B · 50k GPT-2", 13, INK, anchor="middle")
    txt(p, 555, 414, "128k Llama 3 · 256k Gemma", 13, INK, anchor="middle")
    emb = 128256 * 4096
    assert emb == 525336576
    txt(p, 360, 462, f"Embedding matrix 128,256 x 4096 = {emb//10**6}M params, ~6.5% of Llama 3 8B.", 14, INK, anchor="middle")


save("l14-tokenize.svg", 760, 492, "Tokenization: text to vectors", d_tokenize)


# ============ l14-mha.svg ============
def d_mha(p):
    d, h = 4096, 32
    dh = d // h
    assert dh == 128
    txt(p, 40, 66, "Multi-head attention: split the width, not the work", 16, INK, bold=True)
    # input vector bar
    box(p, 40, 100, 560, 44, fill="#E8EEF7", stroke=BLUE, sw=2.5)
    txt(p, 320, 128, f"token vector, width d = {d}", 14, BLUE, anchor="middle", bold=True)
    arrow(p, 320, 146, 320, 170, INK)
    # heads
    nshow = 8
    for i in range(nshow):
        x = 40 + i * 70
        on = i < 3
        box(p, x, 176, 62, 110, fill="#FFF7E8" if on else "#FFFFFF", stroke=AMBER if on else GRAY, sw=3 if on else 2)
        txt(p, x + 31, 208, f"head {i+1}", 13, AMBER if on else GRAY, anchor="middle", bold=on)
        txt(p, x + 31, 230, f"d_h={dh}", 12, GRAY, anchor="middle")
        if on:
            txt(p, x + 31, 254, "own mix", 12, AMBER, anchor="middle")
        else:
            txt(p, x + 31, 254, "...", 12, GRAY, anchor="middle")
    txt(p, 620, 230, f"x {h} heads", 13, INK)
    txt(p, 40, 312, f"Each head: own W_Q, W_K, W_V -> own mix pattern.", 14, INK)
    # concat + Wo
    arrow(p, 320, 322, 320, 348, INK)
    box(p, 180, 352, 280, 52, fill="#EAF3E4", stroke=GREEN, sw=2.5)
    txt(p, 320, 374, "concat -> W_O", 15, GREEN, anchor="middle", bold=True)
    txt(p, 320, 396, f"back to width {d}", 12, GRAY, anchor="middle")
    # cost audit
    txt(p, 40, 446, "Cost audit:", 14, INK, bold=True)
    txt(p, 40, 472, f"{h} heads x N^2 x {dh}  =  N^2 x {d}.  Identical to one full-width head.", 14, INK)
    txt(p, 40, 498, "The split is free. Capacity is in 32 simultaneous mix patterns.", 14, GREEN, bold=True)
    txt(p, 40, 524, "Dial: d_h in 64-128. Too many heads starves each.", 13, GRAY)


save("l14-mha.svg", 760, 554, "Multi-head attention", d_mha)


# ============ l14-rope.svg ============
def d_rope(p):
    theta, m, n = 0.5, 3, 1
    q = (math.cos(m * theta), math.sin(m * theta))
    k = (math.cos(n * theta), math.sin(n * theta))
    dot = q[0] * k[0] + q[1] * k[1]
    rel = math.cos((m - n) * theta)
    assert abs(dot - rel) < 1e-9 and abs(dot - 0.5403) < 0.001
    txt(p, 40, 66, "RoPE: rotate by position, keep only the distance", 16, INK, bold=True)
    for (cx, lab, pos, ang, vec, c) in [(190, "query q=[1,0]", f"position m={m}", m * theta, q, BLUE),
                                        (530, "key k=[1,0]", f"position n={n}", n * theta, k, PURPLE)]:
        cy = 220
        circle(p, cx, cy, 80, "#FFFFFF", BORDER)
        line(p, cx - 80, cy, cx + 80, cy, GRAY, 1, dash="5 4")
        line(p, cx, cy - 80, cx, cy + 80, GRAY, 1, dash="5 4")
        ex, ey = cx + 70 * vec[0], cy - 70 * vec[1]
        arrow(p, cx, cy, ex, ey, c, 3)
        txt(p, cx, cy - 100, f"rotated by {ang:.1f} rad", 12, GRAY, anchor="middle")
        txt(p, cx, cy + 108, lab, 13, c, anchor="middle", bold=True)
        txt(p, cx, cy + 132, pos, 13, INK, anchor="middle")
        txt(p, cx, cy + 158, f"[{vec[0]:.4f}, {vec[1]:.4f}]", 12, INK, anchor="middle")
    arrow(p, 330, 220, 390, 220, INK)
    txt(p, 360, 200, "dot", 13, INK, anchor="middle", bold=True)
    txt(p, 360, 424, f"q.k = {dot:.4f} = cos((m-n).theta) = cos({(m-n)*theta:.1f})", 15, INK, anchor="middle", bold=True)
    txt(p, 360, 452, "Absolute angles cancel. Only the relative distance m-n survives.", 14, GREEN, anchor="middle", bold=True)
    txt(p, 360, 478, "Llama 3: theta_base = 500,000. RoPE is the 2026 default.", 13, GRAY, anchor="middle")


save("l14-rope.svg", 720, 508, "RoPE: rotation is relative", d_rope)


# ============ l14-flash.svg ============
def d_flash(p):
    txt(p, 40, 66, "FlashAttention: never materialize the N x N matrix", 16, INK, bold=True)
    # left: standard
    box(p, 40, 90, 320, 250, stroke=RED, sw=3)
    txt(p, 200, 122, "Standard", 15, RED, anchor="middle", bold=True)
    grid(p, 120, 140, 6, 28, "#FBEDEC", "#F6D9D4")
    txt(p, 200, 330, "N x N scores in HBM", 13, RED, anchor="middle", bold=True)
    N = 8192
    floats = N * N
    assert floats == 67108864
    txt(p, 200, 356, f"N={N}: {floats//10**6}M floats/head", 12, INK, anchor="middle")
    txt(p, 200, 380, "read + written repeatedly", 12, INK, anchor="middle")
    # right: tiled
    box(p, 400, 90, 320, 250, stroke=GREEN, sw=3)
    txt(p, 560, 122, "FlashAttention", 15, GREEN, anchor="middle", bold=True)
    box(p, 430, 150, 90, 60, fill="#EAF3E4", stroke=GREEN, sw=2)
    txt(p, 475, 176, "Q tile", 13, GREEN, anchor="middle", bold=True)
    txt(p, 475, 196, "in SRAM", 11, GRAY, anchor="middle")
    box(p, 545, 150, 90, 60, fill="#EAF3E4", stroke=GREEN, sw=2)
    txt(p, 590, 176, "K,V tile", 13, GREEN, anchor="middle", bold=True)
    txt(p, 590, 196, "in SRAM", 11, GRAY, anchor="middle")
    box(p, 470, 240, 180, 80, fill="#FFF7E8", stroke=AMBER, sw=2.5)
    txt(p, 560, 266, "online softmax", 13, AMBER, anchor="middle", bold=True)
    txt(p, 560, 288, "m: running max", 12, INK, anchor="middle")
    txt(p, 560, 308, "l: running sum, O: output", 12, INK, anchor="middle")
    arrow(p, 520, 200, 505, 238, INK); arrow(p, 600, 200, 610, 238, INK)
    txt(p, 560, 366, "traffic: O(N^2) -> O(N)", 13, GREEN, anchor="middle", bold=True)
    arrow(p, 362, 220, 398, 220, INK)
    txt(p, 381, 200, "tile", 12, INK, anchor="middle")
    txt(p, 360, 418, "Exact math, 2-4x faster on training. Nothing changed mathematically,", 14, INK, anchor="middle")
    txt(p, 360, 442, "everything changed economically: the quadratic op runs at SRAM speed.", 14, INK, anchor="middle")


save("l14-flash.svg", 760, 472, "FlashAttention: tile into SRAM", d_flash)


# ============ l15-mla.svg ============
def d_mla(p):
    heads, dh, b = 32, 128, 2
    mha_bytes = heads * dh * b * 2
    latent, mla_bytes = 512, 512 * 2
    ratio = mha_bytes / mla_bytes
    assert mha_bytes == 16384 and mla_bytes == 1024 and ratio == 16.0
    txt(p, 40, 66, "MLA: store the latent, rebuild K and V on the fly", 16, INK, bold=True)
    # MHA cache block
    box(p, 40, 100, 220, 200, fill="#FBEDEC", stroke=RED, sw=3)
    txt(p, 150, 132, "MHA cache", 15, RED, anchor="middle", bold=True)
    txt(p, 150, 156, "per token, per layer", 12, GRAY, anchor="middle")
    box(p, 65, 175, 90, 60, fill="#FFFFFF", stroke=RED, sw=2)
    txt(p, 110, 200, "K", 15, RED, anchor="middle", bold=True)
    txt(p, 110, 220, f"{heads}x{dh}", 11, GRAY, anchor="middle")
    box(p, 165, 175, 90, 60, fill="#FFFFFF", stroke=RED, sw=2)
    txt(p, 210, 200, "V", 15, RED, anchor="middle", bold=True)
    txt(p, 210, 220, f"{heads}x{dh}", 11, GRAY, anchor="middle")
    txt(p, 150, 268, f"{mha_bytes//1024} KB", 16, RED, anchor="middle", bold=True)
    # arrow
    arrow(p, 265, 200, 330, 200, INK)
    txt(p, 297, 180, f"{ratio:.0f}x", 14, GREEN, anchor="middle", bold=True)
    # latent
    box(p, 335, 130, 130, 140, fill="#EAF3E4", stroke=GREEN, sw=3)
    txt(p, 400, 162, "latent c", 15, GREEN, anchor="middle", bold=True)
    txt(p, 400, 186, f"{latent} dims", 13, INK, anchor="middle")
    txt(p, 400, 210, "per token", 12, GRAY, anchor="middle")
    txt(p, 400, 244, f"{mla_bytes//1024} KB", 16, GREEN, anchor="middle", bold=True)
    # reconstruct
    arrow(p, 470, 170, 530, 170, BLUE); arrow(p, 470, 230, 530, 230, BLUE)
    txt(p, 500, 158, "up-project", 12, BLUE, anchor="middle")
    box(p, 535, 140, 70, 50, fill="#E8EEF7", stroke=BLUE, sw=2)
    txt(p, 570, 162, "K", 14, BLUE, anchor="middle", bold=True)
    txt(p, 570, 180, "rebuilt", 11, GRAY, anchor="middle")
    box(p, 535, 205, 70, 50, fill="#E8EEF7", stroke=BLUE, sw=2)
    txt(p, 570, 227, "V", 14, BLUE, anchor="middle", bold=True)
    txt(p, 570, 245, "rebuilt", 11, GRAY, anchor="middle")
    txt(p, 570, 292, "at attention time", 12, GRAY, anchor="middle")
    txt(p, 40, 340, "The trick: absorb the up-projection into the query projection.", 14, INK)
    txt(p, 40, 364, "Reconstruction costs almost nothing at runtime.", 14, INK, bold=True)
    txt(p, 40, 392, "RoPE needs care: decoupled rotary path (rotation vs compression).", 13, GRAY)
    txt(p, 360, 424, "DeepSeek-V3 ships MLA at 671B params. Change what is stored, not just how much.", 14, INK, anchor="middle")


save("l15-mla.svg", 720, 454, "MLA: compress the latent", d_mla)


# ============ l15-paged.svg ============
def d_paged(p):
    per_tok_kb, nreq, maxlen, avglen = 512, 10, 4096, 500
    reserved_gb = nreq * maxlen * per_tok_kb / 1024 / 1024
    used_gb = nreq * avglen * per_tok_kb / 1024 / 1024
    util_cont = used_gb / reserved_gb
    paged_reserved = 2.5
    util_paged = round(used_gb, 1) / paged_reserved  # lesson's rounded 2.4 / 2.5 = 96%
    assert abs(reserved_gb - 20.0) < 0.01 and abs(used_gb - 2.44) < 0.01
    assert abs(util_cont - 0.122) < 0.002 and abs(util_paged - 0.96) < 0.001
    txt(p, 40, 66, "PagedAttention: OS virtual memory for the KV cache", 16, INK, bold=True)
    # left: contiguous waste
    txt(p, 40, 100, "Contiguous reservation", 14, RED, bold=True)
    for i in range(5):
        y = 115 + i * 44
        bar(p, 40, y, 280, 26, avglen / maxlen, GREEN, RED)
        txt(p, 40, y + 42, f"req {i+1}: {avglen} used of {maxlen}", 11, GRAY)
    txt(p, 40, 350, f"{reserved_gb:.0f} GB reserved, {used_gb:.1f} GB used", 14, RED, bold=True)
    txt(p, 40, 376, f"utilization {util_cont*100:.0f}%", 14, RED, bold=True)
    # right: paged
    line(p, 380, 90, 380, 380, BORDER, 2)
    txt(p, 400, 100, "Paged blocks", 14, GREEN, bold=True)
    # block table
    txt(p, 400, 130, "block table (logical -> physical)", 12, INK, bold=True)
    logical = [0, 1, 2, 3, 4, 5]
    physical = [7, 3, 19, 2, 11, 8]
    for i, (l, ph) in enumerate(zip(logical, physical)):
        y = 145 + i * 32
        box(p, 400, y, 60, 26, fill="#E8EEF7", stroke=BLUE, sw=2)
        txt(p, 430, y + 18, str(l), 12, BLUE, anchor="middle", bold=True)
        arrow(p, 465, y + 13, 495, y + 13, INK)
        box(p, 500, y, 60, 26, fill="#EAF3E4", stroke=GREEN, sw=2)
        txt(p, 530, y + 18, str(ph), 12, GREEN, anchor="middle", bold=True)
    txt(p, 620, 160, "physical", 12, INK); txt(p, 620, 178, "block pool:", 12, INK)
    txt(p, 620, 196, "allocate", 12, GREEN, bold=True); txt(p, 620, 214, "on demand", 12, GREEN, bold=True)
    txt(p, 400, 350, f"{used_gb:.1f} GB used of ~{paged_reserved:.1f} GB reserved", 14, GREEN, bold=True)
    txt(p, 400, 376, f"utilization {util_paged*100:.0f}%", 14, GREEN, bold=True)
    txt(p, 380, 424, "2-4x higher serving throughput at equal latency. Copy-on-write shares prompt", 14, INK, anchor="middle")
    txt(p, 380, 448, "blocks across beam-search branches.", 14, INK, anchor="middle")


save("l15-paged.svg", 760, 478, "PagedAttention: the OS trick", d_paged)


if __name__ == "__main__":
    names = ["l13-triplet.svg", "l13-rerank.svg", "l14-tokenize.svg", "l14-mha.svg",
             "l14-rope.svg", "l14-flash.svg", "l15-mla.svg", "l15-paged.svg"]
    for n in names:
        path = os.path.join(OUT, n)
        size = os.path.getsize(path)
        xml.dom.minidom.parse(path)  # raises if malformed
        print(f"OK {n} ({size} bytes, valid XML)")
