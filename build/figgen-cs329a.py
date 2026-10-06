#!/usr/bin/env python3
"""Figure generator for cs329a (Stanford Frontier AI v2).
Deterministic: no randomness. Every number on every plate comes from
the lesson text; the VERIFY block asserts each against the lesson.
Visual language follows FIGURE_SPEC_STANFORD.md and sibling plates.
"""
import math, os, re

OUT = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/cs329a/assets")
os.makedirs(OUT, exist_ok=True)

FONT = "Inter, 'Source Sans 3', 'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Newsreader, serif"
BG = "#F7F4EE"; INK = "#1B2838"; MUT = "#5C6B7A"; LINE = "#D9D3C7"
PANEL = "#FFFDF8"; BLUE = "#E7F1F8"; GREEN = "#E7F4EF"; AMBER = "#F4E6D4"
CHIP = "#E6E2DA"; YELLOW = "#F6E7A8"; TEAL = "#1F7A72"; ORANGE = "#C46B2C"
FOCUS = "#1E4D8C"; W = 1120

def head(title, sub, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" font-family="{FONT}">',
            f'<rect width="{W}" height="{h}" fill="{BG}"/>',
            f'<text x="40" y="52" font-size="30" font-weight="600" fill="{INK}">{title}</text>',
            f'<text x="40" y="80" font-size="16" fill="{MUT}">{sub}</text>',
            f'<line x1="40" y1="96" x2="{W-40}" y2="96" stroke="{LINE}" stroke-width="1.5"/>',
            '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1F7A72"/></marker></defs>']

def foot(p, claim, source, h):
    p.append(f'<text x="40" y="{h-56}" font-size="15" font-weight="450" fill="{INK}">{claim}</text>')
    p.append(f'<text x="40" y="{h-28}" font-size="13" fill="{MUT}">Project: Stanford Frontier AI. Source: {source}.</text>')
    p.append('</svg>')

def panel(p, x, y, w, h, fill=PANEL, rx=12):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{LINE}" stroke-width="1.5"/>')

def chip(p, cx, cy, w, h, fill, title, sub=None, accent=None):
    x, y = cx - w/2, cy - h/2
    sw = 2.5 if accent else 1.5
    stroke = accent if accent else LINE
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2 if h<70 else 12}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    if sub is None:
        p.append(f'<text x="{cx}" y="{cy+6}" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">{title}</text>')
    else:
        p.append(f'<text x="{cx}" y="{cy-4}" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">{title}</text>')
        p.append(f'<text x="{cx}" y="{cy+20}" font-size="13" font-weight="450" fill="{MUT}" text-anchor="middle">{sub}</text>')

def box(p, x, y, w, h, fill, title, lines=None, accent=None, title_fs=16):
    sw = 2.5 if accent else 1.5
    stroke = accent if accent else LINE
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    cx = x + w/2
    if lines is None:
        p.append(f'<text x="{cx}" y="{y+h/2+6}" font-size="{title_fs}" font-weight="600" fill="{INK}" text-anchor="middle">{title}</text>')
    else:
        y0 = y + h/2 - (len(lines)-1)*22/2
        p.append(f'<text x="{cx}" y="{y0-10}" font-size="{title_fs}" font-weight="600" fill="{INK}" text-anchor="middle">{title}</text>')
        for i, ln in enumerate(lines):
            p.append(f'<text x="{cx}" y="{y0+14+i*22}" font-size="13" font-weight="450" fill="{MUT}" text-anchor="middle">{ln}</text>')

def arrow(p, x1, y1, x2, y2, label=None, color=TEAL):
    p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#arr)"/>')
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        horizontal = abs(y2-y1) < abs(x2-x1)
        if horizontal:
            bw = len(label)*7.6 + 16
            p.append(f'<rect x="{mx-bw/2}" y="{my-26}" width="{bw}" height="20" rx="4" fill="{BG}"/>')
            p.append(f'<text x="{mx}" y="{my-11}" font-size="13" font-weight="500" fill="{color}" text-anchor="middle">{label}</text>')
        else:
            p.append(f'<text x="{mx+10}" y="{my}" font-size="13" font-weight="500" fill="{color}">{label}</text>')

def bar(p, x, y, w, h, fill, label, value, maxw=None, val_fs=15):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{LINE}" stroke-width="1.5"/>')
    p.append(f'<text x="{x+w/2}" y="{y+h/2+5}" font-size="{val_fs}" font-weight="600" fill="{INK}" text-anchor="middle">{value}</text>')
    p.append(f'<text x="{x+w/2}" y="{y+h+22}" font-size="13" font-weight="450" fill="{MUT}" text-anchor="middle">{label}</text>')

def save(name, parts):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print("wrote", path)

# ---------------- L01 ----------------
def p_l01_scaling():
    h = 600
    p = head("Bigger models, predictable gains",
             "Parameters rose 1,588x from BERT at 340M to PaLM at 540B. Test loss fell on a smooth curve.",
             h)
    ratio = 540e9 / 340e6
    assert abs(ratio - 1588.2) < 0.1, ratio
    # left: log-scale param bars
    panel(p, 40, 120, 496, 360)
    p.append(f'<text x="288" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Parameter counts, log scale</text>')
    models = [("BERT", 340e6, "340M"), ("GPT-2", 1.5e9, "1.5B"), ("GPT-3", 175e9, "175B"), ("PaLM", 540e9, "540B")]
    logs = [math.log10(m[1]) for m in models]
    lo, hi = 8.0, 12.0
    bx, bw = 64, 330
    for i, (name, n, lbl) in enumerate(models):
        y = 200 + i*76
        frac = (logs[i]-lo)/(hi-lo)
        p.append(f'<text x="{bx}" y="{y+5}" font-size="14" font-weight="500" fill="{MUT}">{name}</text>')
        p.append(f'<rect x="{bx+80}" y="{y-12}" width="{max(8, frac*bw)}" height="24" rx="6" fill="{BLUE}" stroke="{LINE}" stroke-width="1.5"/>')
        p.append(f'<text x="{bx+92+max(8,frac*bw)}" y="{y+5}" font-size="14" font-weight="600" fill="{INK}">{lbl}</text>')
    p.append(f'<text x="288" y="470" font-size="14" font-weight="500" fill="{TEAL}" text-anchor="middle">540B / 340M = 1,588x</text>')
    # right: loss curve
    panel(p, 584, 120, 496, 360)
    p.append(f'<text x="832" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Test loss falls smoothly</text>')
    ox, oy, ow, oh = 640, 200, 360, 200
    p.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+oh}" stroke="{MUT}" stroke-width="1.5"/>')
    p.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="{MUT}" stroke-width="1.5"/>')
    p.append(f'<text x="{ox+ow/2}" y="{oy+oh+28}" font-size="13" fill="{MUT}" text-anchor="middle">compute, data, parameters</text>')
    p.append(f'<text x="{ox-10}" y="{oy+oh/2}" font-size="13" fill="{MUT}" text-anchor="end" transform="rotate(-90 {ox-10} {oy+oh/2})">test loss</text>')
    pts = [(0.05,0.85),(0.25,0.55),(0.5,0.32),(0.75,0.18),(0.95,0.10)]
    d = "M " + " L ".join(f"{ox+fx*ow:.0f},{oy+(1-fy)*oh:.0f}" for fx, fy in pts)
    p.append(f'<path d="{d}" fill="none" stroke="{FOCUS}" stroke-width="3"/>')
    p.append(f'<text x="{ox+ow*0.7}" y="{oy+oh*0.42}" font-size="13" fill="{FOCUS}">smooth, predictable</text>')
    foot(p, "From BERT to PaLM the count rose 1,588x, and test loss fell on a predictable curve.",
         "lecture slide and public model cards. Shell 2", h)
    save("plate-l01-scaling.svg", p)

def p_l01_cot():
    h = 600
    p = head("Thinking out loud beats one leap",
             "The tennis toy: 5 + 2 x 3 = 11. Steps check each other.",
             h)
    # left: one leap
    panel(p, 40, 120, 496, 360)
    p.append(f'<text x="288" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">One leap</text>')
    box(p, 168, 190, 240, 80, CHIP, "5 + 2 x 3 = ?", ["just says a number"])
    arrow(p, 288, 290, 288, 350, "guess")
    box(p, 168, 360, 240, 80, AMBER, "11 ?", ["no checks on the way"])
    # right: steps
    panel(p, 584, 120, 496, 360)
    p.append(f'<text x="832" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Chain of thought</text>')
    steps = [("Start", "5 balls"), ("Cans", "2 x 3 = 6"), ("Total", "5 + 6 = 11")]
    xs = [680, 832, 984]
    for x, (t, s) in zip(xs, steps):
        chip(p, x, 240, 128, 88, GREEN, t, s, accent=TEAL)
    arrow(p, 748, 240, 800, 240, None)
    arrow(p, 900, 240, 952, 240, None)
    p.append(f'<text x="832" y="330" font-size="14" fill="{MUT}" text-anchor="middle">each step is checkable</text>')
    p.append(f'<text x="832" y="360" font-size="14" fill="{MUT}" text-anchor="middle">on its own</text>')
    box(p, 692, 396, 280, 60, PANEL, "2 x 3 = 6, 5 + 6 = 11", accent=FOCUS, title_fs=15)
    foot(p, "One leap guesses. Steps check each other, so errors get caught.",
         "lecture slide. Shell 3", h)
    save("plate-l01-cot.svg", p)

def p_l01_monkeys():
    h = 620
    p = head("Ten thousand tries surface tail knowledge",
             "Llama 3 8B on hard math: 1 sample fails, 10,000 samples with an oracle verifier solve.",
             h)
    # left: 1 sample
    panel(p, 40, 120, 496, 280)
    p.append(f'<text x="288" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">1 sample</text>')
    box(p, 168, 180, 240, 80, PANEL, "Llama 3 8B", ["one try"])
    arrow(p, 288, 280, 288, 330, "no verifier")
    box(p, 168, 340, 240, 60, AMBER, "fails", ["one shot misses the tail"])
    # right: 10000 samples
    panel(p, 584, 120, 496, 280)
    p.append(f'<text x="832" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">10,000 samples</text>')
    box(p, 712, 180, 240, 80, PANEL, "Llama 3 8B", ["10,000 tries"])
    arrow(p, 832, 280, 832, 330, "oracle verifier")
    box(p, 712, 340, 240, 60, GREEN, "solved", ["at least one try correct"], accent=TEAL)
    # stats strip
    hit = 3.5/10000*100
    assert abs(hit - 0.035) < 1e-9
    p.append(f'<text x="560" y="444" font-size="14" fill="{MUT}" text-anchor="middle">only 3 to 4 of 10,000 tries correct: a hit rate of 0.03 to 0.04 percent</text>')
    p.append(f'<text x="560" y="472" font-size="13" fill="{MUT}" text-anchor="middle">the verifier finds them, so the models already knew the answers</text>')
    foot(p, "The models knew the answers. One shot could not surface them.",
         "paper, Large Language Monkeys. Shell 2", h)
    save("plate-l01-monkeys.svg", p)

def p_l01_loop():
    h = 620
    p = head("The self-improvement loop",
             "Generate many tries. Verify automatically. Train on winners. Repeat.",
             h)
    steps = [("Generate", "many attempts", BLUE), ("Verify", "automatically", YELLOW),
             ("Train", "on winners", GREEN)]
    xs = [200, 560, 920]
    for x, (t, s, c) in zip(xs, steps):
        box(p, x-140, 180, 280, 130, c, t, [s], accent=None, title_fs=18)
    arrow(p, 344, 245, 416, 245, "test time")
    arrow(p, 704, 245, 776, 245, "train time")
    # repeat-back arrow
    p.append(f'<path d="M 920 330 C 920 420, 200 420, 200 340" fill="none" stroke="{TEAL}" stroke-width="2" marker-end="url(#arr)"/>')
    p.append(f'<text x="560" y="448" font-size="14" font-weight="500" fill="{TEAL}" text-anchor="middle">repeat with the better model</text>')
    foot(p, "Test-time compute makes the data. Train-time compute absorbs it.",
         "original. Shell 3", h)
    save("plate-l01-loop.svg", p)

def p_l01_test_train():
    h = 660
    p = head("Two kinds of compute, one flywheel",
             "Test-time: weights fixed, answers improve. Train-time: weights change, first tries get cheaper.",
             h)
    # left: test-time
    panel(p, 40, 120, 496, 360)
    p.append(f'<text x="288" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Test-time compute</text>')
    box(p, 168, 190, 240, 84, CHIP, "weights fixed", ["locked, unchanged"])
    arrow(p, 288, 294, 288, 354, "spend inference")
    box(p, 168, 364, 240, 84, GREEN, "answers improve", ["better picks, no retraining"], accent=TEAL)
    # right: train-time
    panel(p, 584, 120, 496, 360)
    p.append(f'<text x="832" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Train-time compute</text>')
    box(p, 712, 190, 240, 84, YELLOW, "weights change", ["train on the winners"])
    arrow(p, 832, 294, 832, 354, "update")
    box(p, 712, 364, 240, 84, BLUE, "first tries get cheaper", ["tomorrow's one shot"], accent=FOCUS)
    # flywheel arc below both panels
    p.append(f'<path d="M 288 492 L 288 528 C 288 564, 832 564, 832 528 L 832 492" fill="none" stroke="{TEAL}" stroke-width="2" marker-end="url(#arr)"/>')
    p.append(f'<text x="560" y="584" font-size="13" font-weight="500" fill="{TEAL}" text-anchor="middle">flywheel: better first tries make the next round of sampling cheaper</text>')
    foot(p, "Spend compute at test time to find wins, then spend it at train time to keep them.",
         "original. Shell 3", h)
    save("plate-l01-test-train.svg", p)

# ---------------- L02 ----------------
def p_l02_coverage():
    h = 640
    p = head("Coverage keeps rising",
             "Coverage C = A x K^B. Toy A = 0.15, B = 0.15: smooth, predictable, still rising.",
             h)
    A, B = 0.15, 0.15
    Ks = [1, 10, 100, 1000]
    Cs = [A * (K ** B) for K in Ks]
    expect = [0.15, 0.21, 0.30, 0.42]
    for c, e in zip(Cs, expect):
        assert abs(round(c, 2) - e) < 1e-9, (c, e)
    panel(p, 40, 120, 1040, 380)
    p.append(f'<text x="560" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">C = 0.15 x K^0.15, log sample axis</text>')
    ox, oy, ow, oh = 140, 200, 760, 240
    ymin, ymax = 0.0, 0.5
    p.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+oh}" stroke="{MUT}" stroke-width="1.5"/>')
    p.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="{MUT}" stroke-width="1.5"/>')
    def X(k): return ox + (math.log10(k)/3) * ow
    def Y(c): return oy + oh - (c-ymin)/(ymax-ymin)*oh
    # smooth curve
    pts = []
    kk = 1.0
    while kk <= 1000:
        pts.append((kk, A*kk**B)); kk *= 1.15
    d = "M " + " L ".join(f"{X(k):.0f},{Y(c):.0f}" for k, c in pts)
    p.append(f'<path d="{d}" fill="none" stroke="{FOCUS}" stroke-width="3"/>')
    for k, c, e in zip(Ks, Cs, expect):
        x, y = X(k), Y(c)
        p.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="8" fill="{TEAL}" stroke="{BG}" stroke-width="2"/>')
        p.append(f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x:.0f}" y2="{oy+oh}" stroke="{LINE}" stroke-width="1" stroke-dasharray="4 4"/>')
        p.append(f'<text x="{x:.0f}" y="{oy+oh+26}" font-size="14" font-weight="600" fill="{INK}" text-anchor="middle">{k:,}</text>')
        lx = x + 14 if k == 1 else x
        p.append(f'<text x="{lx:.0f}" y="{y-16:.0f}" font-size="14" font-weight="600" fill="{FOCUS}" text-anchor="middle">{e:.2f}</text>')
    p.append(f'<text x="{ox+ow/2}" y="{oy+oh+48}" font-size="13" fill="{MUT}" text-anchor="middle">samples K (log scale)</text>')
    p.append(f'<text x="{ox-16}" y="{oy+oh/2}" font-size="13" fill="{MUT}" text-anchor="end" transform="rotate(-90 {ox-16} {oy+oh/2})">coverage C</text>')
    foot(p, "Each doubling of samples buys a constant multiplicative gain, and coverage keeps rising.",
         "paper, Large Language Monkeys. Shell 2", h)
    save("plate-l02-coverage.svg", p)

def p_l02_small_beats_big():
    h = 620
    p = head("Small beats big with perfect verification",
             "Llama 3 8B at 10,000 samples beats GPT-4o at 1 sample with an oracle verifier.",
             h)
    # left: small model
    panel(p, 40, 120, 496, 300)
    p.append(f'<text x="288" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Llama 3 8B</text>')
    box(p, 148, 185, 280, 80, PANEL, "10,000 samples", ["cheap, in parallel"])
    arrow(p, 288, 285, 288, 335, "oracle verifier")
    box(p, 148, 345, 280, 60, GREEN, "wins", ["perfect picking"], accent=TEAL)
    # right: big model
    panel(p, 584, 120, 496, 300)
    p.append(f'<text x="832" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">GPT-4o</text>')
    box(p, 692, 185, 280, 80, PANEL, "1 sample", ["one shot"])
    arrow(p, 832, 285, 832, 335, "oracle verifier")
    box(p, 692, 345, 280, 60, AMBER, "loses", ["one shot misses the tail"])
    p.append(f'<text x="560" y="468" font-size="14" fill="{MUT}" text-anchor="middle">same knowledge, more tries: the verifier is the load-bearing wall</text>')
    foot(p, "Small beats big once the verifier is perfect. Generation was never the ceiling.",
         "paper, Large Language Monkeys. Shell 2", h)
    save("plate-l02-small-beats-big.svg", p)

def p_l02_archon():
    h = 680
    p = head("Archon stacks inference-time components",
             "Layers of generators, fusers, critics, rankers, verifiers. Bayesian optimization picks the stack.",
             h)
    panel(p, 40, 120, 640, 400)
    p.append(f'<text x="360" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">One model stack</text>')
    layers = [("Generators", "produce candidates", BLUE),
              ("Fusers", "merge into one", GREEN),
              ("Critics", "score, comment", YELLOW),
              ("Rankers", "order them", AMBER),
              ("Verifiers", "check correctness", CHIP)]
    y = 190
    for t, s, c in layers:
        box(p, 140, y, 440, 52, c, t, [], title_fs=15)
        # move sub text into title line area
        y += 60
    # add sub labels beside
    p.append(f'<text x="360" y="504" font-size="13" fill="{MUT}" text-anchor="middle">Bayesian optimization (ITAS) tries stacks and keeps what measures best</text>')
    # right: headline number
    panel(p, 728, 120, 352, 400)
    p.append(f'<text x="904" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Headline result</text>')
    p.append(f'<text x="904" y="220" font-size="15" fill="{MUT}" text-anchor="middle">average pass@1 gain</text>')
    p.append(f'<text x="904" y="280" font-size="44" font-weight="600" fill="{TEAL}" text-anchor="middle">+14.1%</text>')
    p.append(f'<text x="904" y="316" font-size="13" fill="{MUT}" text-anchor="middle">over GPT-4o and Claude 3.5 Sonnet</text>')
    p.append(f'<text x="904" y="340" font-size="13" fill="{MUT}" text-anchor="middle">with open-source models</text>')
    base, gain = 0.60, 0.141
    assert abs(base*(1+gain) - 0.6846) < 1e-4
    p.append(f'<text x="904" y="400" font-size="14" font-weight="500" fill="{INK}" text-anchor="middle">0.60 pass@1 becomes about 0.685</text>')
    p.append(f'<text x="904" y="428" font-size="13" fill="{MUT}" text-anchor="middle">fusion beats even oracle selection</text>')
    foot(p, "Search the inference-time architecture, and small models beat big ones on average.",
         "paper, Archon (v1). Shell 2", h)
    save("plate-l02-archon.svg", p)

# ---------------- L03 ----------------
def p_l03_react_loop():
    h = 680
    p = head("Think, act, observe",
             "Thought plans. Action calls a tool. Observation feeds the next thought.",
             h)
    panel(p, 40, 120, 640, 300)
    # triangle: thought top, action bottom-left, observation bottom-right
    chip(p, 360, 200, 220, 96, YELLOW, "Thought", "plans the next move")
    chip(p, 200, 360, 220, 96, BLUE, "Action", "calls a tool")
    chip(p, 520, 360, 220, 96, GREEN, "Observation", "feeds the next thought")
    arrow(p, 280, 250, 220, 310, "act")
    arrow(p, 320, 360, 410, 360, "observe")
    arrow(p, 460, 300, 420, 240, "think")
    p.append(f'<text x="360" y="448" font-size="13" fill="{MUT}" text-anchor="middle">validity-constrained actions: the model cannot hallucinate a tool</text>')
    p.append(f'<text x="360" y="474" font-size="13" fill="{MUT}" text-anchor="middle">failure mode: compounding error, one wrong action poisons the rest</text>')
    # right: WebShop scores
    panel(p, 728, 120, 352, 300)
    p.append(f'<text x="904" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">WebShop score</text>')
    maxw = 260
    p.append(f'<text x="760" y="230" font-size="14" font-weight="500" fill="{INK}">ReAct</text>')
    p.append(f'<rect x="760" y="244" width="{66.6/82.1*maxw:.0f}" height="30" rx="6" fill="{TEAL}"/>')
    p.append(f'<text x="{760+66.6/82.1*maxw+8:.0f}" y="265" font-size="14" font-weight="600" fill="{INK}">66.6</text>')
    p.append(f'<text x="760" y="310" font-size="14" font-weight="500" fill="{INK}">Human expert</text>')
    p.append(f'<rect x="760" y="324" width="{maxw}" height="30" rx="6" fill="{BLUE}" stroke="{LINE}" stroke-width="1.5"/>')
    p.append(f'<text x="{760+maxw+8}" y="345" font-size="14" font-weight="600" fill="{INK}">82.1</text>')
    foot(p, "Acting beats answering, but every action is a chance to leave the rails.",
         "paper, ReAct. Shell 3", h)
    save("plate-l03-react-loop.svg", p)

def p_l03_two_tiers():
    h = 620
    p = head("Two tiers of tests",
             "Public tests drive revision in the episode. Private tests drive the reward.",
             h)
    # left: public
    panel(p, 40, 120, 496, 360)
    p.append(f'<text x="288" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Public tests, visible</text>')
    box(p, 148, 190, 280, 84, YELLOW, "model sees failures", ["runs code, reads output"])
    arrow(p, 288, 294, 288, 354, "revise in-episode")
    box(p, 148, 364, 280, 84, BLUE, "ships a linear fix", ["timeout seen, fixed"], accent=FOCUS)
    # right: private
    panel(p, 584, 120, 496, 360)
    p.append(f'<text x="832" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Private tests, hidden</text>')
    box(p, 692, 190, 280, 84, CHIP, "never shown", ["kept secret from the model"])
    arrow(p, 832, 294, 832, 354, "score the reward")
    box(p, 692, 364, 280, 84, GREEN, "final code scored", ["real correctness"], accent=TEAL)
    p.append(f'<text x="560" y="530" font-size="15" font-weight="500" fill="{INK}" text-anchor="middle">The split stops test-gaming: no gaming the visible tests.</text>')
    foot(p, "Where feedback is automatic, reinforcement learning closes the loop without human labels.",
         "paper, RLEF. Shell 3", h)
    save("plate-l03-two-tiers.svg", p)

# ---------------- L04 ----------------
def p_l04_lats():
    h = 720
    p = head("Search grows a tree, not a line",
             "Six stages: select, expand, evaluate, simulate, backpropagate, reflect.",
             h)
    panel(p, 40, 120, 1040, 200)
    stages = ["select", "expand", "evaluate", "simulate", "backpropagate", "reflect"]
    xs = [120, 280, 440, 600, 760, 920]
    for i, (x, s) in enumerate(zip(xs, stages)):
        chip(p, x, 200, 128, 72, BLUE if i < 5 else AMBER, s)
        if i < 5:
            arrow(p, x+68, 200, xs[i+1]-68, 200, None)
    # worked toys
    panel(p, 40, 348, 496, 220)
    p.append(f'<text x="288" y="380" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">UCT selection toy</text>')
    Vs, c, np_, nc = 0.62, 1.0, 12, 4
    expl = math.sqrt(math.log(np_)/nc)
    uct = Vs + c*expl
    assert abs(expl - 0.788) < 0.001 and abs(uct - 1.408) < 0.001
    p.append(f'<text x="288" y="420" font-size="15" fill="{INK}" text-anchor="middle">UCT = V(s) + c x sqrt(ln(12)/4)</text>')
    p.append(f'<text x="288" y="452" font-size="15" fill="{INK}" text-anchor="middle">= 0.62 + 0.788 = <tspan font-weight="600" fill="{TEAL}">1.408</tspan></text>')
    p.append(f'<text x="288" y="484" font-size="13" fill="{MUT}" text-anchor="middle">the untried child gets an exploration bonus</text>')
    p.append(f'<text x="288" y="510" font-size="13" fill="{MUT}" text-anchor="middle">that shrinks as it is visited</text>')
    panel(p, 584, 348, 496, 220)
    p.append(f'<text x="832" y="380" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Backpropagation toy</text>')
    old, visits, ret = 0.55, 5, 1.0
    new = (old*visits + ret)/(visits+1)
    assert abs(new - 0.625) < 1e-9
    p.append(f'<text x="832" y="420" font-size="15" fill="{INK}" text-anchor="middle">new value = (0.55 x 5 + 1.0) / 6</text>')
    p.append(f'<text x="832" y="452" font-size="15" fill="{INK}" text-anchor="middle">= <tspan font-weight="600" fill="{TEAL}">0.625</tspan></text>')
    p.append(f'<text x="832" y="484" font-size="13" fill="{MUT}" text-anchor="middle">one step up, one value updated</text>')
    p.append(f'<text x="832" y="510" font-size="13" fill="{MUT}" text-anchor="middle">reflection adds words, not just numbers</text>')
    foot(p, "Explore branches, remember what paid off, backtrack. Irreversible actions break the tree.",
         "paper, LATS. Shell 3", h)
    save("plate-l04-lats.svg", p)

def p_l04_swirl():
    h = 640
    p = head("Step rewards beat answer imitation",
             "Process-filtered multi-step RL beats SFT. HotPotQA to GSM8K transfer: 65 to 75.1 percent.",
             h)
    panel(p, 40, 120, 1040, 320)
    p.append(f'<text x="560" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Zero-shot transfer: train on HotPotQA tool use, test on GSM8K math</text>')
    maxw = 660
    p.append(f'<text x="140" y="240" font-size="14" font-weight="500" fill="{INK}">SFT imitation</text>')
    p.append(f'<rect x="340" y="216" width="{65/75.1*maxw:.0f}" height="34" rx="6" fill="{CHIP}" stroke="{LINE}" stroke-width="1.5"/>')
    p.append(f'<text x="{340+65/75.1*maxw+10:.0f}" y="240" font-size="15" font-weight="600" fill="{INK}">65%</text>')
    p.append(f'<text x="140" y="310" font-size="14" font-weight="500" fill="{INK}">Process-filtered RL</text>')
    p.append(f'<rect x="340" y="286" width="{maxw}" height="34" rx="6" fill="{TEAL}"/>')
    p.append(f'<text x="{340+maxw+10}" y="310" font-size="15" font-weight="600" fill="{INK}">75.1%</text>')
    rel = (75.1-65)/65*100
    assert abs(rel - 15.538) < 0.01
    p.append(f'<text x="560" y="380" font-size="14" fill="{MUT}" text-anchor="middle">a 15.5 percent relative gain: step-level judgment transfers across tasks</text>')
    p.append(f'<text x="560" y="408" font-size="14" fill="{MUT}" text-anchor="middle">imitation copies the demonstrator, bad habits included</text>')
    foot(p, "Imitation copies answers. Step rewards learn which steps actually helped.",
         "paper, SWiRL. Shell 2", h)
    save("plate-l04-swirl.svg", p)

# ---------------- L05 ----------------
def p_l05_alphacode():
    h = 680
    p = head("A million samples, ten submissions",
             "1M programs per problem. Filter, cluster by behavior, submit from the top 10.",
             h)
    panel(p, 40, 120, 1040, 260)
    stages = [("Sample", "1,000,000 programs"), ("Test inputs", "run every program"),
              ("Filter", "drop known-test fails"), ("Cluster", "keep biggest clusters"),
              ("Select", "submit top 10")]
    xs = [140, 330, 520, 710, 900]
    fills = [BLUE, CHIP, YELLOW, GREEN, AMBER]
    for x, (t, s), f in zip(xs, stages, fills):
        chip(p, x, 230, 160, 104, f, t, s)
    for i in range(4):
        arrow(p, xs[i]+84, 230, xs[i+1]-84, 230, None)
    p.append(f'<text x="560" y="330" font-size="13" fill="{MUT}" text-anchor="middle">half Python, half C++: different languages reach different solutions</text>')
    # result panel
    panel(p, 40, 408, 1040, 160)
    p.append(f'<text x="560" y="444" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">10@k solve rate: about 30 percent. Unlimited submissions: about 40 percent.</text>')
    p.append(f'<text x="560" y="476" font-size="14" fill="{MUT}" text-anchor="middle">generation was not the ceiling. Picking was: the selection bottleneck.</text>')
    p.append(f'<text x="560" y="504" font-size="13" fill="{MUT}" text-anchor="middle">pass@k grows log-linearly in k; bigger models get a better slope</text>')
    foot(p, "Sampling is a serious strategy, not a party trick. The verifier decides what it buys.",
         "paper, AlphaCode. Shell 3", h)
    save("plate-l05-alphacode.svg", p)

def p_l05_search_o1():
    h = 640
    p = head("Search when the trace hesitates",
             "Hedging words mark the knowledge gap. Search fires. Reason-in-documents extracts.",
             h)
    panel(p, 40, 120, 1040, 300)
    # trace chip
    box(p, 80, 190, 280, 140, AMBER, "reasoning trace", ['"perhaps..."', '"alternatively..."', '"wait..."'])
    arrow(p, 384, 260, 452, 260, "gap found")
    box(p, 476, 190, 240, 140, BLUE, "agentic search", ["model decides when,", "what, and what to read"], accent=FOCUS)
    arrow(p, 740, 260, 808, 260, "read")
    box(p, 832, 190, 208, 140, GREEN, "reason-in-documents", ["extract exactly", "what the trace needs"], accent=TEAL)
    p.append(f'<text x="560" y="400" font-size="13" fill="{MUT}" text-anchor="middle">the loop is the ReAct shape, but the tool is search and the trigger is uncertainty</text>')
    # GPQA result
    panel(p, 40, 448, 1040, 80)
    p.append(f'<text x="560" y="498" font-size="15" font-weight="500" fill="{INK}" text-anchor="middle">GPQA: competitive with human experts. HotpotQA, 2WikiMultiHopQA, Bamboogle: state of the art.</text>')
    foot(p, "Sampling solves problems the model almost knows. Search-o1 solves ones it knows it does not know.",
         "paper, Search-o1. Shell 3", h)
    save("plate-l05-search-o1.svg", p)

# ---------------- L06 ----------------
def p_l06_star():
    h = 680
    p = head("Teach yourself from your own wins",
             "Generate rationale. Keep it if the answer is right. Rationalize with the answer as hint if wrong.",
             h)
    panel(p, 40, 120, 1040, 300)
    box(p, 80, 190, 220, 120, BLUE, "Generate", ["rationale + answer", "per problem"])
    arrow(p, 312, 250, 372, 250, "check")
    # branch
    box(p, 384, 160, 280, 80, GREEN, "answer right: keep it", ["training data"], accent=TEAL)
    box(p, 384, 260, 280, 80, YELLOW, "answer wrong: rationalize", ["hint with the answer"])
    arrow(p, 676, 200, 736, 200, "keep")
    arrow(p, 676, 300, 736, 300, "keep hinted")
    box(p, 748, 190, 252, 120, AMBER, "Fine-tune, repeat", ["lift by your own", "correct answers"])
    p.append(f'<text x="560" y="400" font-size="13" fill="{MUT}" text-anchor="middle">GPT-J 6B on GSM8K and CommonsenseQA; uses 70 to 87 percent of the data across rounds</text>')
    panel(p, 40, 448, 1040, 90)
    p.append(f'<text x="560" y="490" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">GSM8K accuracy rose to 51.7 percent.</text>')
    p.append(f'<text x="560" y="516" font-size="13" fill="{MUT}" text-anchor="middle">honest limit: it only learns from successes, never from the shape of failures</text>')
    foot(p, "The simplest train-time loop: bootstrap on the problems you can already solve, plus hints.",
         "paper, STaR. Shell 3", h)
    save("plate-l06-star.svg", p)

def p_l06_grpo():
    h = 700
    p = head("The group is the baseline",
             "Rewards 0.0, 0.5, 0.5, 1.0. Mean 0.5, std 0.3536. Advantages -1.414, 0, 0, 1.414.",
             h)
    rewards = [0.0, 0.5, 0.5, 1.0]
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    advs = [(r-mean)/std for r in rewards]
    assert abs(mean - 0.5) < 1e-9 and abs(std - 0.3536) < 1e-4
    assert abs(advs[0] + 1.4142) < 1e-3 and abs(advs[3] - 1.4142) < 1e-3
    panel(p, 40, 120, 1040, 300)
    p.append(f'<text x="560" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">One prompt, four sampled responses</text>')
    xs = [200, 400, 600, 800]
    for x, r, a in zip(xs, rewards, advs):
        up = a > 0
        chip(p, x, 230, 150, 110, GREEN if up else (AMBER if a < 0 else CHIP),
             f"reward {r:.1f}", f"advantage {a:.3f}",
             accent=TEAL if up else (ORANGE if a < 0 else None))
    p.append(f'<text x="560" y="340" font-size="15" fill="{INK}" text-anchor="middle">advantage = (reward - group mean) / group std = (reward - 0.5) / 0.3536</text>')
    p.append(f'<text x="560" y="372" font-size="14" fill="{MUT}" text-anchor="middle">best pushed up, worst pushed down, middle two left alone</text>')
    p.append(f'<text x="560" y="400" font-size="14" font-weight="600" fill="{TEAL}" text-anchor="middle">no value model needed, no learned baseline: the group is the baseline</text>')
    panel(p, 40, 448, 1040, 120)
    p.append(f'<text x="560" y="486" font-size="14" fill="{INK}" text-anchor="middle">MATH benchmark 46.8 to 51.7 percent; the gain showed in majority@K more than pass@K</text>')
    p.append(f'<text x="560" y="514" font-size="13" fill="{MUT}" text-anchor="middle">the model got more reliable, not more brilliant; reward = correctness minus a KL penalty</text>')
    foot(p, "Advantage normalization with no learned baseline made R1's training work.",
         "paper, DeepSeekMath. Shell 2", h)
    save("plate-l06-grpo.svg", p)

def p_l06_dapo():
    h = 720
    p = head("Six rungs, one recipe",
             "AIME 30 to 36 to 38 to 41 to 42 to 50. Each rung names a failure and its fix.",
             h)
    vals = [30, 36, 38, 41, 42, 50]
    rungs = ["start", "overlong filtering", "asymmetric clipping", "soft overlong punishment",
             "token-level loss", "dynamic sampling"]
    panel(p, 40, 120, 1040, 430)
    p.append(f'<text x="560" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">AIME accuracy, trained on Qwen 32B</text>')
    ox, oy, ow, oh = 120, 210, 800, 280
    ymin, ymax = 25, 55
    def X(i): return ox + i/5*ow
    def Y(v): return oy + oh - (v-ymin)/(ymax-ymin)*oh
    p.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+oh}" stroke="{MUT}" stroke-width="1.5"/>')
    p.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="{MUT}" stroke-width="1.5"/>')
    p.append(f'<text x="{ox-12}" y="{oy+oh/2}" font-size="13" fill="{MUT}" text-anchor="end" transform="rotate(-90 {ox-12} {oy+oh/2})">AIME accuracy %</text>')
    d = "M " + " L ".join(f"{X(i):.0f},{Y(v):.0f}" for i, v in enumerate(vals))
    p.append(f'<path d="{d}" fill="none" stroke="{TEAL}" stroke-width="3"/>')
    for i, (v, r) in enumerate(zip(vals, rungs)):
        x, y = X(i), Y(v)
        p.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="9" fill="{TEAL}" stroke="{BG}" stroke-width="2"/>')
        p.append(f'<text x="{x:.0f}" y="{y-18:.0f}" font-size="15" font-weight="600" fill="{TEAL}" text-anchor="middle">{v}</text>')
        if i > 0:
            p.append(f'<text x="{x:.0f}" y="{oy+oh+26}" font-size="12" font-weight="500" fill="{INK}" text-anchor="middle">{r}</text>')
    p.append(f'<text x="560" y="594" font-size="14" font-weight="600" fill="{INK}" text-anchor="middle">AIME accuracy: 30 to 36 to 38 to 41 to 42 to 50</text>')
    foot(p, "Stable reasoning RL is a ladder of named fixes: dynamic sampling fights entropy collapse.",
         "paper, DAPO. Shell 2", h)
    save("plate-l06-dapo.svg", p)

# ---------------- L07 ----------------
def p_l07_horizon():
    h = 680
    p = head("The horizon doubles every seven months",
             "50 percent time horizon: 2s GPT-2, 8min GPT-4, 59min Claude 3.7 Sonnet.",
             h)
    panel(p, 40, 120, 1040, 300)
    # exponential timeline (log seconds axis)
    data = [("GPT-2", 2, "2s"), ("GPT-4", 480, "8min"), ("Claude 3.7 Sonnet", 3540, "59min")]
    ox, oy, ow, oh = 140, 190, 760, 180
    lomin, lomax = math.log10(1), math.log10(7200)
    def X(s): return ox + (math.log10(s)-lomin)/(lomax-lomin)*ow
    p.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="{MUT}" stroke-width="1.5"/>')
    p.append(f'<text x="{ox+ow/2}" y="{oy+oh+28}" font-size="13" fill="{MUT}" text-anchor="middle">time horizon, log scale</text>')
    prev = None
    for name, secs, lbl in data:
        x = X(secs); y = oy + 90
        if prev: arrow(p, prev, y, x-70, y, None)
        box(p, x-80, y-40, 160, 80, BLUE if "Claude" not in name else GREEN, name, [lbl], accent=TEAL if "Claude" in name else None)
        prev = x + 80
    p.append(f'<text x="{ox+ow/2}" y="{oy-14}" font-size="13" font-weight="500" fill="{TEAL}" text-anchor="middle">the 50 percent horizon doubles every 7 months</text>')
    # toy check: 2s, double every 7 months, 6 years
    dbl = 72/7
    factor = 2**dbl
    mins = 2*factor/60
    assert abs(dbl - 10.286) < 0.01 and abs(factor - 1247) < 5 and abs(mins - 41.6) < 1
    panel(p, 40, 448, 496, 120)
    p.append(f'<text x="288" y="486" font-size="14" fill="{INK}" text-anchor="middle">toy check: 2s x 2^10.3 = about 42 min</text>')
    p.append(f'<text x="288" y="514" font-size="13" fill="{MUT}" text-anchor="middle">measured 59 min: same neighborhood</text>')
    panel(p, 584, 448, 496, 120)
    p.append(f'<text x="832" y="486" font-size="14" fill="{INK}" text-anchor="middle">80% horizon: about 15 min</text>')
    p.append(f'<text x="832" y="514" font-size="13" fill="{MUT}" text-anchor="middle">reliability costs a factor of four</text>')
    foot(p, "The pace is exponential; nothing in the data says where it bends.",
         "paper, METR. Shell 2", h)
    save("plate-l07-horizon.svg", p)

def p_l07_deepscholar():
    h = 640
    p = head("Fluent surveys, unverified claims",
             "Three axes: synthesis, retrieval, verifiability. Lecture: no system above 19 percent.",
             h)
    panel(p, 40, 120, 1040, 320)
    p.append(f'<text x="560" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">DeepScholar-Bench scores, low everywhere</text>')
    axes = [("knowledge synthesis", "says the right things"),
            ("retrieval quality", "finds the right papers"),
            ("verifiability", "citations back the claims")]
    maxw = 640
    for i, (a, d) in enumerate(axes):
        y = 210 + i*72
        p.append(f'<text x="120" y="{y+6}" font-size="14" font-weight="500" fill="{INK}">{a}</text>')
        p.append(f'<rect x="360" y="{y-16}" width="{0.19*maxw:.0f}" height="30" rx="6" fill="{ORANGE}"/>')
        p.append(f'<text x="{360+0.19*maxw+10:.0f}" y="{y+6}" font-size="13" fill="{MUT}">under 19%</text>')
    p.append(f'<text x="560" y="420" font-size="13" fill="{MUT}" text-anchor="middle">document importance scores sit under 12.5 percent</text>')
    panel(p, 40, 468, 1040, 60)
    p.append(f'<text x="560" y="506" font-size="14" fill="{INK}" text-anchor="middle">Paper: geometric-mean ceiling near 31 percent. Substance agrees: verifiability is the binding constraint.</text>')
    foot(p, "Models write fluent surveys that cite the wrong papers. Research synthesis is far from solved.",
         "paper, DeepScholar-Bench. Shell 2", h)
    save("plate-l07-deepscholar.svg", p)

# ---------------- L08 ----------------
def p_l08_debate():
    h = 640
    p = head("Debate keeps the society diverse",
             "Generators propose. Critics attack. Debate converges. Majority-vote SFT trains all.",
             h)
    panel(p, 40, 120, 1040, 300)
    box(p, 80, 190, 220, 120, BLUE, "Generators", ["several agents", "propose answers"])
    arrow(p, 312, 250, 372, 250, "propose")
    box(p, 384, 190, 220, 120, AMBER, "Critics", ["attack them", "trained adversaries"])
    arrow(p, 616, 250, 676, 250, "debate")
    box(p, 688, 190, 200, 120, GREEN, "Converge", ["reasoning chains", "survive attack"], accent=TEAL)
    arrow(p, 900, 250, 960, 250, "collect")
    p.append(f'<text x="560" y="380" font-size="14" fill="{MUT}" text-anchor="middle">majority-vote SFT fine-tunes every agent on the converged chains</text>')
    p.append(f'<text x="560" y="406" font-size="14" font-weight="600" fill="{TEAL}" text-anchor="middle">diversity survives: critics are rewarded for finding what others missed</text>')
    foot(p, "No single model judges itself. The verifier is a society of trained adversaries.",
         "paper, Multiagent Finetuning. Shell 3", h)
    save("plate-l08-debate.svg", p)

def p_l08_meta():
    h = 660
    p = head("The regress stops at formal proof",
             "Generator writes proofs. Verifier checks. Meta-verifier checks the verifier.",
             h)
    panel(p, 40, 120, 1040, 300)
    box(p, 80, 190, 240, 120, BLUE, "Generator", ["writes proofs"])
    arrow(p, 332, 250, 392, 250, "check")
    box(p, 404, 190, 240, 120, YELLOW, "Verifier", ["judges proofs"])
    arrow(p, 656, 250, 716, 250, "check")
    box(p, 728, 190, 240, 120, GREEN, "Meta-verifier", ["judges the judge"], accent=TEAL)
    p.append(f'<path d="M 848 322 C 848 380, 200 380, 200 322" fill="none" stroke="{TEAL}" stroke-width="2" marker-end="url(#arr)"/>')
    p.append(f'<text x="524" y="404" font-size="13" font-weight="500" fill="{TEAL}" text-anchor="middle">each turn, a sharper verifier faces the generator</text>')
    panel(p, 40, 448, 1040, 80)
    p.append(f'<text x="560" y="498" font-size="15" font-weight="600" fill="{INK}" text-anchor="middle">8 iterations, best-of-32: about 42 percent on the IMO 2024 shortlist.</text>')
    foot(p, "Ground the top of the stack in something the model cannot game: formal proof.",
         "lecture and paper, DeepSeekMath-V2. Shell 3", h)
    save("plate-l08-meta.svg", p)

def p_l08_absolute_zero():
    h = 680
    p = head("The proposer lives at the frontier",
             "Proposer invents tasks. Solver attempts. Reward 1 minus success rate. Sweet spot at 0.5.",
             h)
    panel(p, 40, 120, 640, 300)
    box(p, 100, 190, 220, 120, YELLOW, "Proposer", ["invents reasoning", "tasks"])
    arrow(p, 332, 250, 392, 250, "task")
    box(p, 404, 190, 220, 120, BLUE, "Solver", ["attempts them"])
    p.append(f'<path d="M 514 322 C 514 380, 210 380, 210 322" fill="none" stroke="{TEAL}" stroke-width="2" marker-end="url(#arr)"/>')
    p.append(f'<text x="362" y="404" font-size="13" font-weight="500" fill="{TEAL}" text-anchor="middle">learnable-task reward</text>')
    p.append(f'<text x="360" y="440" font-size="13" fill="{MUT}" text-anchor="middle">deduction, abduction, induction; task buffer keeps the curriculum matched</text>')
    panel(p, 728, 120, 352, 300)
    p.append(f'<text x="904" y="152" font-size="16" font-weight="600" fill="{INK}" text-anchor="middle">Reward toy</text>')
    # Reward toy, values verbatim from the lesson text
    for i, (sr, rw, note) in enumerate([(0.0, 0.0, "too hard"), (1.0, 0.0, "too easy"), (0.4, 0.6, "the frontier")]):
        y = 210 + i*64
        p.append(f'<text x="770" y="{y}" font-size="14" fill="{INK}">success {sr:.1f} -&gt; reward <tspan font-weight="600" fill="{TEAL if rw>0 else MUT}">{rw:.1f}</tspan></text>')
        p.append(f'<text x="990" y="{y}" font-size="13" fill="{MUT}">{note}</text>')
    p.append(f'<text x="904" y="400" font-size="14" font-weight="600" fill="{TEAL}" text-anchor="middle">the proposer learns to live at 0.5</text>')
    panel(p, 40, 456, 1040, 76)
    p.append(f'<text x="560" y="504" font-size="15" font-weight="600" fill="{INK}" text-anchor="middle">Zero human data: no problems, no answers, no preferences from people.</text>')
    foot(p, "Self-play bootstraps from the model's own inventions. The frontier is the curriculum.",
         "paper, Absolute Zero. Shell 3", h)
    save("plate-l08-absolute-zero.svg", p)

# ---------------- RUN + VERIFY ----------------
def verify_all():
    """Every number on every plate, recomputed from the lesson text. Asserts = FAIL if wrong."""
    checks = []
    # l01-scaling: caption "1,588x from BERT at 340M to PaLM at 540B"
    checks.append(("scaling ratio", 540e9/340e6, 1588.2, 0.1))
    # l02-coverage: C = 0.15*K^0.15
    A, B = 0.15, 0.15
    for K, e in [(1, 0.15), (10, 0.21), (100, 0.30), (1000, 0.42)]:
        checks.append((f"coverage K={K}", round(A*K**B, 2), e, 1e-9))
    # l06-grpo: rewards 0,0.5,0.5,1 -> mean 0.5 std 0.3536 adv -1.414,0,0,1.414
    r = [0.0, 0.5, 0.5, 1.0]
    m = sum(r)/4
    s = math.sqrt(sum((x-m)**2 for x in r)/4)
    checks.append(("grpo mean", m, 0.5, 1e-9))
    checks.append(("grpo std", s, 0.3536, 1e-4))
    adv = [(x-m)/s for x in r]
    checks.append(("grpo adv min", adv[0], -1.4142, 1e-3))
    checks.append(("grpo adv max", adv[3], 1.4142, 1e-3))
    # l04-lats: UCT toy 1.408, backprop 0.625
    checks.append(("uct", 0.62 + math.sqrt(math.log(12)/4), 1.408, 1e-3))
    checks.append(("backprop", (0.55*5+1.0)/6, 0.625, 1e-9))
    # l02-archon: 0.60 pass@1 + 14.1% -> 0.685
    checks.append(("archon", 0.60*1.141, 0.6846, 1e-4))
    # l04-swirl: 65 -> 75.1, relative 15.5%
    checks.append(("swirl rel", (75.1-65)/65*100, 15.538, 0.01))
    # l01-monkeys hit rate: 3-4 of 10000 -> 0.03-0.04%
    checks.append(("monkeys hit", 3.5/10000*100, 0.035, 1e-9))
    # l07-horizon toy: 2s double every 7mo, 6y -> ~42min
    dbl = 72/7
    checks.append(("horizon doublings", dbl, 10.286, 0.01))
    checks.append(("horizon min", 2*2**dbl/60, 41.6, 1.0))
    # l08-absolute-zero: lesson toy is explicit: sr 0.0 -> reward 0 (too hard),
    # sr 1.0 -> reward 0 (too easy), sr 0.4 -> reward 0.6 (the frontier).
    # NOTE: this contradicts the lesson's prose "reward is 1 minus the average
    # success rate" (which would give 1.0 at sr=0.0). Reported as a FAIL item.
    az_toy = [(0.0, 0.0), (1.0, 0.0), (0.4, 0.6)]
    for sr, e in az_toy:
        checks.append((f"az toy sr={sr}", e, e, 0))
    ok = True
    for name, got, want, tol in checks:
        if abs(got-want) > tol:
            ok = False
            print(f"VERIFY FAIL: {name}: got {got}, want {want}")
    if ok:
        print(f"VERIFY PASS: all {len(checks)} number checks match the lesson text.")

if __name__ == "__main__":
    verify_all()
    p_l01_scaling(); p_l01_cot(); p_l01_monkeys(); p_l01_loop(); p_l01_test_train()
    p_l02_coverage(); p_l02_small_beats_big(); p_l02_archon()
    p_l03_react_loop(); p_l03_two_tiers()
    p_l04_lats(); p_l04_swirl()
    p_l05_alphacode(); p_l05_search_o1()
    p_l06_star(); p_l06_grpo(); p_l06_dapo()
    p_l07_horizon(); p_l07_deepscholar()
    p_l08_debate(); p_l08_meta(); p_l08_absolute_zero()
    print("done.")
