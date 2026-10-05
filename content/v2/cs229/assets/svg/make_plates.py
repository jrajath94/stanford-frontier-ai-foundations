#!/usr/bin/env python3
"""Hand-drawn warm-paper SVG plates for CS229 (per VISUAL_SYSTEM.md medium ladder).
One atomic unit = one plate. Warm paper #F7F4EE, ink #1B2838, flat fills.
Captions (source + shell) live in the lesson markdown, NOT in the plates.
Run: python3 make_plates.py  (outputs into this directory)
"""
import os

OUT = os.path.dirname(os.path.abspath(__file__))
PAPER = "#F7F4EE"
INK = "#1B2838"
BORDER = "#D8D2C4"
BLUE = "#2B5EA7"
RED = "#B3402E"
GREEN = "#2E7D4F"
PURPLE = "#6C4BA6"
AMBER = "#C77F1A"
GRAY = "#8A8FA0"

def plate_start(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Georgia, serif">',
            f'<rect x="2" y="2" width="{w-4}" height="{h-4}" rx="14" fill="{PAPER}" stroke="{BORDER}" stroke-width="2"/>',
            f'<text x="24" y="38" font-size="21" font-weight="bold" fill="{INK}">{title}</text>']

def plate_end(p):
    return "\n".join(p) + "\n</svg>\n"

def txt(p, x, y, s, size=15, fill=INK, bold=False, anchor="start", italic=False):
    st = f'font-weight="bold"' if bold else ""
    st += ' font-style="italic"' if italic else ""
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" {st}>{s}</text>')

def box(p, x, y, w, h, fill="#FFFFFF", stroke=INK, rx=10, sw=2):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def arrow(p, x1, y1, x2, y2, color=INK, sw=2.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')
    import math
    ang = math.atan2(y2-y1, x2-x1); L = 11
    for da in (0.42, -0.42):
        a = ang + da
        p.append(f'<line x1="{x2}" y1="{y2}" x2="{x2-L*math.cos(a)}" y2="{y2-L*math.sin(a)}" stroke="{color}" stroke-width="{sw}"/>')

def circle(p, x, y, r, fill, stroke="none"):
    p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>')

def line(p, x1, y1, x2, y2, color=INK, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')

def save(name, w, h, title, draw):
    p = plate_start(w, h, title)
    draw(p)
    open(os.path.join(OUT, name), "w").write(plate_end(p))

# ---------------- L01 ----------------
def d_paradigms(p):
    cols = [("Supervised", GREEN, "(x, y) pairs", "predict y from x", "regression,\nclassification"),
            ("Unsupervised", PURPLE, "x only", "find structure", "clustering,\ndensity"),
            ("Reinforcement", BLUE, "act, get reward", "sequential decisions", "robots, games,\nLLM reasoning")]
    for i,(t,c,inn,verb,ex) in enumerate(cols):
        x = 30 + i*230
        box(p, x, 60, 210, 300, stroke=c, sw=3)
        txt(p, x+105, 92, t, 18, c, bold=True, anchor="middle")
        box(p, x+25, 108, 160, 44, fill="#FFFFFF", stroke=GRAY)
        txt(p, x+105, 135, inn, 14, INK, anchor="middle")
        arrow(p, x+105, 152, x+105, 178, c)
        box(p, x+25, 182, 160, 44, fill="#FFFFFF", stroke=GRAY)
        txt(p, x+105, 209, verb, 14, INK, anchor="middle")
        for j,ln in enumerate(ex.split("\n")):
            txt(p, x+105, 252+j*22, ln, 13, GRAY, anchor="middle")
save("l01-paradigms.svg", 720, 390, "Three learning paradigms", d_paradigms)

def d_roadmap(p):
    segs = [("L2-8", "Supervised:\nregression, GLMs,\nneural nets", GREEN, 30, 300),
            ("L9-10", "Unsupervised:\nk-means, GMM,\nPCA", PURPLE, 330, 180),
            ("L11", "Diffusion", AMBER, 510, 90),
            ("L12-15", "LLMs:\ntransformers,\nSFT", BLUE, 600, 140)]
    # simplified: three blocks
    blocks = [("L02-08", "Supervised learning", GREEN, 30, 330),
              ("L09-10", "Unsupervised", PURPLE, 360, 200),
              ("L11-17", "Generative + RL:\ndiffusion, LLMs, RL", BLUE, 560, 0)]
    # draw as arrow timeline
    y = 200
    for i,(lab,t,c,x,w) in enumerate(blocks):
        box(p, x, y-55, w, 110, stroke=c, sw=3)
        txt(p, x+w/2, y-28, lab, 16, c, bold=True, anchor="middle")
        for j,ln in enumerate(t.split("\n")):
            txt(p, x+w/2, y-2+j*20, ln, 13, INK, anchor="middle")
        if i < 2:
            arrow(p, x+w+4, y, x+blocks[i+1][3]-4, y, INK)
save("l01-roadmap.svg", 720, 300, "Course arc: three blocks", d_roadmap)

# ---------------- L02 ----------------
def d_losschip(p):
    # THE LOSS CHIP — cross-course symbol, first defined in CS229 L02. Keep stable.
    box(p, 60, 80, 600, 220, fill="#FFFFFF", stroke=INK, sw=3, rx=18)
    txt(p, 360, 122, "LOSS", 24, INK, bold=True, anchor="middle")
    line(p, 90, 140, 630, 140, GRAY, 1.5)
    txt(p, 360, 190, "J(θ) = ½ Σᵢ ( h_θ(x⁽ⁱ⁾) − y⁽ⁱ⁾ )²", 26, INK, anchor="middle")
    txt(p, 360, 232, "prediction − truth, squared, averaged (½ = convention)", 15, GRAY, anchor="middle")
    txt(p, 360, 262, "Smaller J(θ) = better fit. Training = minimize J.", 15, INK, anchor="middle")
save("l02-loss-chip.svg", 720, 330, "The loss chip: J(θ)", d_losschip)

def d_gd(p):
    # contour bowl with gradient steps
    cx, cy = 360, 225
    for r,c in [(150, "#E4E9F2"),(110, "#D3DBEA"),(70, "#C2CFE3")]:
        p.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{r}" ry="{r*0.62}" fill="{c}" stroke="{BORDER}"/>')
    circle(p, cx, cy, 7, RED)
    txt(p, cx+12, cy+5, "θ*", 16, RED, bold=True)
    pts = [(160,120),(210,150),(260,178),(305,198),(335,212)]
    for x,y in pts: circle(p, x, y, 6, BLUE)
    for i in range(len(pts)-1):
        arrow(p, pts[i][0]+8, pts[i][1]+5, pts[i+1][0]-8, pts[i+1][1]-5, BLUE)
    arrow(p, pts[-1][0]+8, pts[-1][1]+4, cx-12, cy-6, BLUE)
    txt(p, 90, 360, "θ ← θ − α · ∇J(θ): step opposite the gradient.", 15, INK)
    txt(p, 90, 384, "α = learning rate (step size). Too big = overshoot.", 15, GRAY)
save("l02-gd.svg", 720, 410, "Gradient descent on a bowl", d_gd)

def d_sgd(p):
    box(p, 30, 60, 320, 280, stroke=BLUE, sw=3)
    txt(p, 190, 92, "Batch GD", 17, BLUE, bold=True, anchor="middle")
    txt(p, 190, 130, "gradient over ALL data", 14, INK, anchor="middle")
    txt(p, 190, 160, "one exact step", 14, INK, anchor="middle")
    txt(p, 190, 200, "slow on huge data", 14, RED, anchor="middle")
    txt(p, 190, 240, "cost per step ∝ m", 14, GRAY, anchor="middle")
    box(p, 370, 60, 320, 280, stroke=GREEN, sw=3)
    txt(p, 530, 92, "SGD / minibatch", 17, GREEN, bold=True, anchor="middle")
    txt(p, 530, 130, "gradient over a batch", 14, INK, anchor="middle")
    txt(p, 530, 160, "many noisy steps", 14, INK, anchor="middle")
    txt(p, 530, 200, "fast; noise helps escape", 14, GREEN, anchor="middle")
    txt(p, 530, 240, "shuffle each epoch", 14, GRAY, anchor="middle")
save("l02-sgd.svg", 720, 370, "Batch vs stochastic gradient descent", d_sgd)

def d_normaleq(p):
    box(p, 60, 70, 600, 120, fill="#FFFFFF", stroke=INK, sw=2.5)
    txt(p, 360, 125, "θ = (XᵀX)⁻¹ Xᵀ y", 26, INK, anchor="middle")
    txt(p, 360, 162, "Closed form. One shot, no iterations.", 14, GRAY, anchor="middle")
    box(p, 60, 210, 600, 120, fill="#FFF7E8", stroke=AMBER, sw=2.5)
    txt(p, 90, 248, "Catches:", 16, AMBER, bold=True)
    txt(p, 90, 278, "• XᵀX must be invertible: need more data points than parameters", 14, INK)
    txt(p, 90, 302, "• Singular → null space → infinitely many θ, all equally valid", 14, INK)
save("l02-normaleq.svg", 720, 360, "Normal equations", d_normaleq)

# ---------------- L03 ----------------
def d_mle(p):
    box(p, 40, 70, 190, 120, fill="#FFFFFF", stroke=BLUE, sw=2.5)
    txt(p, 135, 105, "Model", 15, BLUE, bold=True, anchor="middle")
    txt(p, 135, 132, "y = θᵀx + ε", 16, INK, anchor="middle")
    txt(p, 135, 158, "ε ~ N(0,σ²)", 15, INK, anchor="middle")
    arrow(p, 230, 130, 280, 130, INK)
    box(p, 290, 70, 190, 120, fill="#FFFFFF", stroke=PURPLE, sw=2.5)
    txt(p, 385, 105, "Likelihood", 15, PURPLE, bold=True, anchor="middle")
    txt(p, 385, 132, "L(θ) = Πᵢ p(yⁱ|xⁱ;θ)", 15, INK, anchor="middle")
    txt(p, 385, 158, "iid → product", 13, GRAY, anchor="middle")
    arrow(p, 480, 130, 530, 130, INK)
    box(p, 540, 70, 150, 120, fill="#EAF3E4", stroke=GREEN, sw=2.5)
    txt(p, 615, 105, "MLE", 15, GREEN, bold=True, anchor="middle")
    txt(p, 615, 132, "max log L(θ)", 15, INK, anchor="middle")
    txt(p, 615, 158, "= least squares", 14, GREEN, bold=True, anchor="middle")
    txt(p, 360, 240, "Gaussian noise + MLE = the squared loss of L02. Not a coincidence.", 15, INK, anchor="middle")
    txt(p, 360, 268, "Maximum likelihood is the bedrock. — L03", 14, GRAY, anchor="middle", italic=True)
save("l03-mle.svg", 720, 300, "MLE: Gaussian noise gives least squares", d_mle)

def d_sigmoid(p):
    ox, oy = 90, 280
    line(p, ox, 60, ox, oy+20, INK); line(p, ox-10, oy, 560, oy, INK)
    # sigmoid curve: map z in [-6,6] to x, g(z) to y
    pts = []
    for i in range(61):
        z = -6 + 12*i/60
        g = 1/(1+2.718281828**(-z))
        pts.append((ox + (z+6)/12*450, oy - g*200))
    pl = " ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    p.append(f'<polyline points="{pl}" fill="none" stroke="{BLUE}" stroke-width="3.5"/>')
    line(p, ox, oy-100, 540, oy-100, RED, 1.5, dash="6 5")
    txt(p, 548, oy-96, "0.5", 13, RED)
    line(p, ox+225, 60, ox+225, oy, GRAY, 1.5, dash="6 5")
    txt(p, ox+218, oy+18, "θᵀx = 0", 13, GRAY, anchor="middle")
    txt(p, 330, 340, "h(x) = 1/(1+e^(−θᵀx)) = P(y=1|x). Predict 1 iff h ≥ 0.5.", 15, INK, anchor="middle")
save("l03-sigmoid.svg", 720, 370, "Logistic regression: the sigmoid", d_sigmoid)

def d_newton(p):
    ox, oy = 80, 270
    line(p, ox, 60, ox, oy+15, INK); line(p, ox-8, oy, 620, oy, INK)
    # convex curve J
    pts = []
    for i in range(61):
        x = i/60*520
        y = 0.004*(x-260)**2
        pts.append((ox+x, oy-y))
    pl = " ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    p.append(f'<polyline points="{pl}" fill="none" stroke="{INK}" stroke-width="3"/>')
    # GD zigzag
    gx = [ox+40, ox+90, ox+135, ox+175]
    for i,gxx in enumerate(gx):
        y = oy - 0.004*(gxx-ox-260)**2
        circle(p, gxx, y, 5, BLUE)
    txt(p, 120, 330, "GD: many small steps", 14, BLUE)
    # Newton: two big steps
    nx1, ny1 = ox+470, oy-0.004*(470-260)**2
    circle(p, nx1, ny1, 6, RED)
    circle(p, ox+260, oy, 7, RED)
    arrow(p, nx1-10, ny1+2, ox+272, oy-4, RED)
    txt(p, 470, 330, "Newton: jump to the bottom", 14, RED, anchor="middle")
    txt(p, 360, 360, "θ := θ − f(θ)/f′(θ). No step size. Uses curvature (Hessian).", 15, INK, anchor="middle")
save("l03-newton.svg", 720, 390, "Newton vs gradient descent", d_newton)

# ---------------- L04 ----------------
def d_expfam(p):
    box(p, 40, 60, 640, 130, fill="#FFFFFF", stroke=PURPLE, sw=3)
    txt(p, 360, 110, "p(y; η) = b(y) · exp( ηᵀT(y) − a(η) )", 24, INK, anchor="middle")
    txt(p, 130, 152, "carrier", 13, GRAY, anchor="middle"); txt(p, 285, 152, "natural param × sufficient stat", 13, GRAY, anchor="middle"); txt(p, 520, 152, "log-partition", 13, GRAY, anchor="middle")
    items = [("Gaussian", "T(y)=y, a(η)=η²/2", 90), ("Bernoulli", "T(y)=y, a(η)=log(1+e^η)", 300), ("Multinomial", "T(y)=one-hot, a(η)=log Σe^ηᵏ", 510)]
    for t, d, x in items:
        box(p, x-85, 210, 170, 90, fill="#FFFFFF", stroke=GRAY)
        txt(p, x, 240, t, 15, PURPLE, bold=True, anchor="middle")
        txt(p, x, 268, d, 12, INK, anchor="middle")
    txt(p, 360, 340, "One form, many distributions. GLMs pick one and link η = θᵀx.", 15, INK, anchor="middle")
save("l04-expfam.svg", 720, 370, "The exponential family", d_expfam)

def d_glm(p):
    steps = [("1. Pick\ndistribution", "Gaussian? Bernoulli?\nMultinomial?", BLUE),
             ("2. Link η = θᵀx", "linear predictor\ninto natural param", PURPLE),
             ("3. Fit by MLE", "maximize log L;\ngradient or Newton", GREEN)]
    for i,(t,d,c) in enumerate(steps):
        x = 40 + i*225
        box(p, x, 70, 200, 150, stroke=c, sw=3)
        for j,ln in enumerate(t.split("\n")):
            txt(p, x+100, 105+j*22, ln, 16, c, bold=True, anchor="middle")
        for j,ln in enumerate(d.split("\n")):
            txt(p, x+100, 165+j*20, ln, 13, INK, anchor="middle")
        if i < 2: arrow(p, x+204, 145, x+221, 145, INK)
    txt(p, 360, 270, "Least squares, logistic regression, softmax: all GLMs.", 15, INK, anchor="middle")
save("l04-glm.svg", 720, 300, "The GLM recipe", d_glm)

def d_softmax(p):
    logits = [2.0, 1.0, 0.1]; exps = [7.39, 2.72, 1.11]; s = sum(exps)
    labels = ["cat", "dog", "car"]
    for i,(lg, e, lab) in enumerate(zip(logits, exps, labels)):
        y = 80 + i*80
        txt(p, 80, y+8, lab, 15, INK, anchor="end")
        txt(p, 150, y+8, f"z={lg}", 14, GRAY)
        w = e/s*380
        p.append(f'<rect x="230" y="{y-14}" width="{w:.0f}" height="28" rx="6" fill="{BLUE if i==0 else "#9DB8DD"}"/>')
        txt(p, 245, y+8, f"{e/s:.2f}", 14, "#FFFFFF" if i==0 else INK, bold=(i==0))
    txt(p, 360, 340, "softmax(z)ₖ = e^zₖ / Σⱼe^zⱼ. Positive, sums to 1. Winner is a distribution.", 15, INK, anchor="middle")
save("l04-softmax.svg", 720, 375, "Softmax: logits to probabilities", d_softmax)

# ---------------- L05 ----------------
def d_gda(p):
    # two gaussian blobs, linear boundary
    import math
    def blob(cx, cy, n, spread, color, seed):
        import random
        random.seed(seed)
        for _ in range(n):
            x = cx + random.gauss(0, spread); y = cy + random.gauss(0, spread*0.8)
            circle(p, x, y, 5, color)
    blob(220, 200, 60, 42, "#9DB8DD", 1)
    blob(500, 200, 60, 42, "#E8A0A0", 2)
    txt(p, 220, 130, "class 0", 14, BLUE, anchor="middle", bold=True)
    txt(p, 500, 130, "class 1", 14, RED, anchor="middle", bold=True)
    line(p, 360, 60, 360, 320, INK, 3, dash="10 7")
    txt(p, 360, 345, "shared Σ → linear boundary", 14, INK, anchor="middle")
    txt(p, 360, 372, "separate Σₖ → quadratic boundary", 14, GRAY, anchor="middle")
save("l05-gda.svg", 720, 400, "GDA: model each class as a Gaussian", d_gda)

def d_nb(p):
    box(p, 40, 70, 640, 110, fill="#FFFFFF", stroke=INK, sw=2.5)
    txt(p, 360, 115, "P(spam | words) ∝ P(spam) · Πᵢ P(wordᵢ | spam)", 20, INK, anchor="middle")
    txt(p, 360, 150, "Naive = features independent given the class. Rarely true, often works.", 14, GRAY, anchor="middle")
    box(p, 40, 200, 300, 100, fill="#FFF7E8", stroke=AMBER, sw=2.5)
    txt(p, 190, 232, "Zero-count trap", 15, AMBER, bold=True, anchor="middle")
    txt(p, 190, 260, "unseen word → P=0 kills product", 13, INK, anchor="middle")
    box(p, 380, 200, 300, 100, fill="#EAF3E4", stroke=GREEN, sw=2.5)
    txt(p, 530, 232, "Laplace smoothing", 15, GREEN, bold=True, anchor="middle")
    txt(p, 530, 260, "add 1 to every count", 13, INK, anchor="middle")
save("l05-naivebayes.svg", 720, 330, "Naive Bayes spam filter", d_nb)

# ---------------- L06 ----------------
def d_biasvar(p):
    labs = [("low bias\nlow variance", GREEN, 1), ("low bias\nhigh variance", AMBER, 2),
            ("high bias\nlow variance", AMBER, 3), ("high bias\nhigh variance", RED, 4)]
    import random
    for i,(t,c,seed) in enumerate(labs):
        x = 60 + i*165; y = 200
        circle(p, x, y, 55, "#FFFFFF", BORDER); circle(p, x, y, 36, "#FFFFFF", BORDER); circle(p, x, y, 18, "#FFFFFF", BORDER)
        circle(p, x, y, 4, RED)
        random.seed(seed)
        spread = 14 if "low variance" in t else 44
        off = 0 if "low bias" in t else 30
        for _ in range(14):
            circle(p, x+off+random.gauss(0,spread), y+random.gauss(0,spread), 4, BLUE)
        for j,ln in enumerate(t.split("\n")):
            txt(p, x, 290+j*20, ln, 13, c, anchor="middle", bold=True)
save("l06-biasvar.svg", 720, 350, "Bias vs variance: four archers", d_biasvar)

def d_dd(p):
    ox, oy = 80, 300
    line(p, ox, 50, ox, oy+10, INK); line(p, ox-8, oy, 640, oy, INK)
    txt(p, 40, 80, "test", 13, INK); txt(p, 40, 100, "error", 13, INK)
    txt(p, 350, 345, "model size →", 13, INK, anchor="middle")
    pts = []
    for i in range(81):
        x = i/80*520
        y = 200 - 130*2.718**(-((x-170)/130)**2) - 60*2.718**(-((x-420)/90)**2) + 40
        pts.append((ox+x, oy-y+90))
    pl = " ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    p.append(f'<polyline points="{pl}" fill="none" stroke="{BLUE}" stroke-width="3.5"/>')
    line(p, ox+170, 50, ox+170, oy, RED, 1.5, dash="6 5")
    txt(p, ox+170, 330, "interpolation", 12, RED, anchor="middle")
    txt(p, 480, 140, "classical U", 13, GRAY); txt(p, 560, 200, "second descent", 13, GREEN, bold=True)
save("l06-dd.svg", 720, 370, "Double descent", d_dd)

def d_hyperband(p):
    # successive halving brackets
    txt(p, 60, 80, "configs", 13, GRAY); txt(p, 60, 300, "budget →", 13, GRAY)
    for b,(n,keep) in enumerate([(8,4),(4,2)]):
        x0 = 120 + b*260
        txt(p, x0+60, 80, f"bracket {b+1}", 14, INK, anchor="middle", bold=True)
        for r in range(3):
            y = 120 + r*70
            ncfg = max(1, n//(2**r))
            w = 60 + ncfg*14
            p.append(f'<rect x="{x0}" y="{y-14}" width="{w}" height="28" rx="6" fill="{BLUE if r==0 else ("#9DB8DD" if r==1 else "#D3DBEA")}"/>')
            txt(p, x0+8, y+5, f"{ncfg} cfgs", 12, "#FFFFFF" if r==0 else INK)
            if r < 2: arrow(p, x0+w/2, y+16, x0+w/2, y+52, INK)
        txt(p, x0+60, 330, "keep top 1/2, double budget", 12, GRAY, anchor="middle")
save("l06-hyperband.svg", 720, 360, "Hyperband: successive halving", d_hyperband)

# ---------------- L07 ----------------
def d_neuron(p):
    for i,y in enumerate([140, 200, 260]):
        circle(p, 90, y, 20, "#FFFFFF", BLUE)
        txt(p, 90, y+6, f"x{i+1}", 14, BLUE, anchor="middle")
        line(p, 112, y, 250, 200, GRAY, 1.5)
        txt(p, 170, y-8 if i!=1 else y+18, f"w{i+1}", 13, GRAY, anchor="middle")
    circle(p, 290, 200, 34, "#EAF3E4", GREEN)
    txt(p, 290, 196, "Σ", 22, GREEN, bold=True, anchor="middle"); txt(p, 290, 218, "+b", 13, GREEN, anchor="middle")
    arrow(p, 326, 200, 380, 200, INK)
    box(p, 380, 165, 90, 70, fill="#FFFFFF", stroke=PURPLE, sw=2.5)
    txt(p, 425, 195, "σ", 26, PURPLE, anchor="middle", bold=True)
    txt(p, 425, 218, "ReLU etc.", 12, GRAY, anchor="middle")
    arrow(p, 470, 200, 540, 200, INK)
    circle(p, 575, 200, 22, "#FFFFFF", RED)
    txt(p, 575, 206, "a", 16, RED, anchor="middle")
    txt(p, 360, 300, "a = σ(w·x + b). Linear part, then bend.", 15, INK, anchor="middle")
save("l07-neuron.svg", 720, 330, "One neuron", d_neuron)

def d_mlp(p):
    layers = [(90,"input x"),(250,"hidden"),(410,"hidden"),(570,"output")]
    ys = {0:[140,200,260], 1:[110,170,230,290], 2:[110,170,230,290], 3:[170,230]}
    for li,(x,lab) in enumerate(layers):
        for y in ys[li]:
            circle(p, x, y, 16, "#FFFFFF", BLUE if li<3 else RED)
        txt(p, x, 340, lab, 13, GRAY, anchor="middle")
        if li < 3:
            for y1 in ys[li]:
                for y2 in ys[li+1]:
                    line(p, x+16, y1, layers[li+1][0]-16, y2, "#C9D4E8", 1)
    txt(p, 360, 60, "Every neuron sees every neuron before it.", 15, INK, anchor="middle")
save("l07-mlp.svg", 720, 370, "A multilayer perceptron", d_mlp)

def d_residual(p):
    box(p, 120, 120, 200, 120, fill="#FFFFFF", stroke=BLUE, sw=3)
    txt(p, 220, 165, "F(x)", 20, BLUE, anchor="middle", bold=True)
    txt(p, 220, 195, "two layers", 13, GRAY, anchor="middle")
    txt(p, 60, 190, "x", 18, INK, anchor="middle")
    arrow(p, 80, 180, 118, 180, INK)
    arrow(p, 322, 180, 420, 180, INK)
    txt(p, 445, 190, "x + F(x)", 18, INK, anchor="middle")
    # skip
    p.append(f'<path d="M 90 150 Q 260 60 430 150" fill="none" stroke="{RED}" stroke-width="3"/>')
    arrow(p, 425, 148, 438, 168, RED)
    txt(p, 260, 60, "skip connection", 14, RED, anchor="middle", bold=True)
    txt(p, 360, 300, "Learn the change, not the whole map. Trains very deep nets.", 15, INK, anchor="middle")
save("l07-residual.svg", 720, 330, "Residual block", d_residual)

# ---------------- L08 ----------------
def d_backprop(p):
    mods = ["M₁: matmul", "M₂: σ", "M₃: matmul", "M₄: loss"]
    for i,m in enumerate(mods):
        x = 60 + i*160
        box(p, x, 130, 130, 70, fill="#E8EEF7", stroke=BLUE, sw=2.5)
        txt(p, x+65, 160, m.split(":")[0], 14, BLUE, anchor="middle", bold=True)
        txt(p, x+65, 182, m.split(":")[1].strip(), 12, GRAY, anchor="middle")
        if i < 3: arrow(p, x+132, 165, x+158, 165, BLUE)
    txt(p, 360, 100, "forward: compute u, then J", 14, BLUE, anchor="middle")
    for i in range(4):
        x = 60 + i*160
        if i < 3: arrow(p, x+130, 225, x+104, 225, RED)
    txt(p, 360, 260, "backward: ∂J/∂u flows back through each module's B[M, u]", 14, RED, anchor="middle")
    txt(p, 360, 300, "Cost of gradient = cost of forward: both O(#params).", 15, INK, anchor="middle", bold=True)
save("l08-backprop.svg", 720, 330, "Backprop as modules", d_backprop)

def d_rank1(p):
    box(p, 60, 80, 110, 170, fill="#E8EEF7", stroke=BLUE, sw=2.5)
    txt(p, 115, 160, "error", 14, BLUE, anchor="middle"); txt(p, 115, 182, "δ", 18, BLUE, anchor="middle")
    txt(p, 195, 165, "⊗", 24, INK, anchor="middle")
    box(p, 225, 140, 220, 50, fill="#E8EEF7", stroke=BLUE, sw=2.5)
    txt(p, 335, 165, "input", 14, BLUE, anchor="middle"); txt(p, 335, 185, "xᵀ", 16, BLUE, anchor="middle")
    txt(p, 465, 165, "=", 24, INK, anchor="middle")
    box(p, 495, 80, 170, 170, fill="#F3E8F0", stroke=PURPLE, sw=2.5)
    txt(p, 580, 150, "∂J/∂W", 18, PURPLE, anchor="middle", bold=True)
    txt(p, 580, 180, "rank 1", 16, PURPLE, anchor="middle")
    txt(p, 360, 300, "One example, one weight matrix: gradient is always an outer product.", 15, INK, anchor="middle")
save("l08-rank1.svg", 720, 330, "The gradient is rank 1", d_rank1)

# ---------------- L09 ----------------
def d_elbow(p):
    ox, oy = 90, 280
    line(p, ox, 50, ox, oy+10, INK); line(p, ox-8, oy, 580, oy, INK)
    txt(p, 45, 70, "distortion", 12, INK); txt(p, 330, 322, "k →", 12, INK, anchor="middle")
    ks = [1,2,3,4,5,6,7,8]; ds = [95,55,32,22,17,14,12,11]
    pts = [(ox+k/8*460, oy-d/100*200) for k,d in zip(ks,ds)]
    pl = " ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    p.append(f'<polyline points="{pl}" fill="none" stroke="{BLUE}" stroke-width="3.5"/>')
    for x,y in pts: circle(p, x, y, 5, BLUE)
    circle(p, pts[2][0], pts[2][1], 9, RED)
    txt(p, pts[2][0]+18, pts[2][1]-8, "elbow", 14, RED, bold=True)
    txt(p, 360, 352, "More clusters always fit better. Stop where the gain bends.", 14, INK, anchor="middle")
save("l09-elbow.svg", 720, 380, "Choosing k: the elbow method", d_elbow)

def d_gmm(p):
    txt(p, 190, 70, "k-means: hard", 16, BLUE, bold=True, anchor="middle")
    txt(p, 530, 70, "GMM: soft", 16, PURPLE, bold=True, anchor="middle")
    import random
    random.seed(7)
    for cx in (190, 530):
        for _ in range(20):
            circle(p, cx+random.gauss(-40,18), 190+random.gauss(0,40), 6, "#9DB8DD")
            circle(p, cx+random.gauss(40,18), 190+random.gauss(0,40), 6, "#E8A0A0")
    circle(p, 190, 290, 8, BLUE); txt(p, 210, 296, "= cluster 1, 100%", 14, INK)
    circle(p, 530, 290, 8, PURPLE); txt(p, 550, 296, "= 60% / 40%", 14, INK)
    txt(p, 360, 340, "Responsibilities: each point partially belongs to every cluster.", 15, INK, anchor="middle")
save("l09-gmm.svg", 720, 370, "Hard vs soft clustering", d_gmm)

# ---------------- L10 ----------------
def d_elbo(p):
    ox, oy = 80, 280
    line(p, ox, 60, ox, oy+10, INK); line(p, ox-8, oy, 620, oy, INK)
    txt(p, 360, 322, "θ →", 13, INK, anchor="middle")
    pts = []
    for i in range(61):
        x = i/60*520
        y = 170 - 0.0022*(x-300)**2 + 18*2.718**(-((x-180)/40)**2)
        pts.append((ox+x, oy-y+60))
    pl = " ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    p.append(f'<polyline points="{pl}" fill="none" stroke="{INK}" stroke-width="3"/>')
    txt(p, 560, 120, "log-likelihood", 13, INK)
    # ELBO touching at theta_t
    tx = ox+180
    epts = [(tx-140, oy-30),(tx-70, oy-55),(tx, oy-75),(tx+70, oy-62),(tx+140, oy-40)]
    epl = " ".join(f"{x:.0f},{y:.0f}" for x,y in epts)
    p.append(f'<polyline points="{epl}" fill="none" stroke="{GREEN}" stroke-width="3" stroke-dasharray="9 6"/>')
    circle(p, tx, oy-75, 7, GREEN)
    txt(p, tx, oy-95, "θₜ", 15, GREEN, bold=True, anchor="middle")
    line(p, tx, 60, tx, oy, GRAY, 1.5, dash="5 5")
    txt(p, 200, 200, "ELBO (touches here)", 13, GREEN)
    txt(p, 360, 352, "E-step: tighten the bound. M-step: climb it.", 15, INK, anchor="middle")
save("l10-elbo.svg", 720, 380, "EM and the ELBO", d_elbo)

def d_pca(p):
    import random, math
    random.seed(3)
    cx, cy = 300, 190
    for _ in range(90):
        a = random.gauss(0, 1); b = random.gauss(0, 0.35)
        x = cx + a*130*math.cos(0.5) - b*130*math.sin(0.5)
        y = cy + a*130*math.sin(0.5) + b*130*math.cos(0.5)
        circle(p, x, y, 4, "#9DB8DD")
    arrow(p, cx-150, cy-75, cx+150, cy+75, RED, 3)
    txt(p, cx+160, cy+85, "u₁ (max variance)", 14, RED, bold=True)
    arrow(p, cx+40, cy-105, cx-40, cy+105, AMBER, 2.5)
    txt(p, cx-120, cy-95, "u₂", 14, AMBER, bold=True)
    txt(p, 560, 150, "center: subtract", 13, INK); txt(p, 560, 172, "the mean", 13, INK)
    txt(p, 560, 200, "rescale: feet vs miles", 13, INK); txt(p, 560, 222, "must match", 13, RED, bold=True)
    txt(p, 360, 340, "Keep top k eigenvectors of the covariance. 1000 dims → 10.", 15, INK, anchor="middle")
save("l10-pca.svg", 720, 370, "PCA: the axes of variation", d_pca)

# ---------------- L11 ----------------
def d_diffusion(p):
    xs = [80, 210, 340, 470, 600]
    for i,x in enumerate(xs):
        alpha = 1 - i*0.22
        shade = int(247 - (247-180)*(1-alpha))
        box(p, x-45, 90, 90, 90, fill=f"rgb({shade},{shade},{shade+5})", stroke=INK, sw=2)
        txt(p, x, 205, f"x_{i if i<4 else 'T'}", 15, INK, anchor="middle", italic=True)
        if i < 4:
            arrow(p, x+48, 135, xs[i+1]-48, 135, RED)
    txt(p, 340, 60, "forward q: add a little noise each step (fixed)", 14, RED, anchor="middle")
    for i in range(4):
        arrow(p, xs[i+1]-48, 165, xs[i]+48, 165, BLUE)
    txt(p, 340, 250, "reverse p_θ: learned denoising, step by step", 14, BLUE, anchor="middle")
    txt(p, 340, 290, "Train with the ELBO. Sample: noise → clean image.", 15, INK, anchor="middle")
save("l11-diffusion.svg", 720, 320, "Diffusion: noise up, learn to denoise", d_diffusion)

# ---------------- L12 ----------------
def d_fm(p):
    box(p, 30, 70, 300, 150, stroke=BLUE, sw=3)
    txt(p, 180, 105, "Pre-training", 17, BLUE, bold=True, anchor="middle")
    txt(p, 180, 135, "massive unlabeled data", 13, INK, anchor="middle")
    txt(p, 180, 158, "one big model", 13, INK, anchor="middle")
    txt(p, 180, 181, "messy internet text", 13, GRAY, anchor="middle")
    arrow(p, 334, 145, 386, 145, INK)
    box(p, 390, 70, 300, 150, stroke=GREEN, sw=3)
    txt(p, 540, 105, "Adaptation", 17, GREEN, bold=True, anchor="middle")
    txt(p, 540, 135, "many tasks", 13, INK, anchor="middle")
    txt(p, 540, 158, "probing / finetuning", 13, INK, anchor="middle")
    txt(p, 540, 181, "prompts / SFT / RL", 13, GRAY, anchor="middle")
    txt(p, 360, 270, "Good foundation first, then adapt. Never train per task from scratch.", 15, INK, anchor="middle")
save("l12-fm.svg", 720, 300, "The foundation-model paradigm", d_fm)

def d_probe(p):
    box(p, 40, 80, 200, 120, fill="#E8EEF7", stroke=BLUE, sw=2.5)
    txt(p, 140, 120, "φ(x)", 20, BLUE, anchor="middle", bold=True)
    txt(p, 140, 148, "frozen", 13, GRAY, anchor="middle")
    txt(p, 140, 170, "representation", 13, INK, anchor="middle")
    arrow(p, 244, 140, 296, 140, INK)
    box(p, 300, 80, 200, 120, fill="#FFFFFF", stroke=GREEN, sw=2.5)
    txt(p, 400, 120, "w", 20, GREEN, anchor="middle", bold=True)
    txt(p, 400, 148, "trained", 13, GRAY, anchor="middle")
    txt(p, 400, 170, "linear probe", 13, INK, anchor="middle")
    arrow(p, 504, 140, 556, 140, INK)
    txt(p, 620, 146, "w·φ(x)", 18, INK, anchor="middle")
    txt(p, 360, 250, "Linear probing: cheap test of what the representation knows.", 15, INK, anchor="middle")
    txt(p, 360, 278, "Also used to interpret LLMs (mechanistic interpretability).", 14, GRAY, anchor="middle")
save("l12-probe.svg", 720, 310, "Linear probing", d_probe)

# ---------------- L13 ----------------
def d_contrastive(p):
    box(p, 60, 90, 150, 110, fill="#E8EEF7", stroke=BLUE, sw=2.5)
    txt(p, 135, 135, "anchor", 15, BLUE, anchor="middle", bold=True)
    txt(p, 135, 160, "cat photo", 13, INK, anchor="middle")
    box(p, 300, 90, 150, 110, fill="#EAF3E4", stroke=GREEN, sw=2.5)
    txt(p, 375, 135, "positive", 15, GREEN, anchor="middle", bold=True)
    txt(p, 375, 160, "same cat, cropped", 13, INK, anchor="middle")
    box(p, 530, 90, 150, 110, fill="#FBEDEC", stroke=RED, sw=2.5)
    txt(p, 605, 135, "negative", 15, RED, anchor="middle", bold=True)
    txt(p, 605, 160, "airplane", 13, INK, anchor="middle")
    arrow(p, 215, 145, 295, 145, GREEN); txt(p, 255, 130, "pull", 12, GREEN, anchor="middle")
    arrow(p, 455, 145, 525, 145, RED); txt(p, 490, 130, "push", 12, RED, anchor="middle")
    txt(p, 360, 260, "Diagonal big, off-diagonal small. No labels needed.", 15, INK, anchor="middle")
save("l13-contrastive.svg", 720, 290, "Contrastive learning", d_contrastive)

def d_rag(p):
    box(p, 30, 80, 130, 90, fill="#FFFFFF", stroke=BLUE, sw=2.5)
    txt(p, 95, 115, "query", 15, BLUE, anchor="middle", bold=True)
    txt(p, 95, 140, "your question", 12, GRAY, anchor="middle")
    arrow(p, 164, 125, 206, 125, INK)
    box(p, 210, 80, 150, 90, fill="#E8EEF7", stroke=PURPLE, sw=2.5)
    txt(p, 285, 115, "retriever", 15, PURPLE, anchor="middle", bold=True)
    txt(p, 285, 140, "5–10 docs", 12, GRAY, anchor="middle")
    arrow(p, 364, 125, 406, 125, INK)
    box(p, 410, 60, 130, 130, fill="#FFF7E8", stroke=AMBER, sw=2.5)
    txt(p, 475, 100, "corpus", 14, AMBER, anchor="middle", bold=True)
    txt(p, 475, 125, "private docs", 12, INK, anchor="middle")
    txt(p, 475, 145, "never trained", 12, GREEN, anchor="middle", bold=True)
    txt(p, 475, 165, "easy to delete", 12, GREEN, anchor="middle")
    arrow(p, 544, 125, 586, 125, INK)
    box(p, 590, 80, 110, 90, fill="#FFFFFF", stroke=GREEN, sw=2.5)
    txt(p, 645, 115, "LLM", 15, GREEN, anchor="middle", bold=True)
    txt(p, 645, 140, "answers", 12, GRAY, anchor="middle")
    txt(p, 360, 230, "RAG: private data stays out of training; rides along as context.", 15, INK, anchor="middle")
save("l13-rag.svg", 720, 260, "Retrieval-augmented generation", d_rag)

# ---------------- L14 ----------------
def d_attn(p):
    for i,(lab,y,c) in enumerate([("query q",110,BLUE),("key k",180,PURPLE),("value v",250,GREEN)]):
        circle(p, 130, y, 22, "#FFFFFF", c)
        txt(p, 130, y+6, lab.split()[0], 15, c, anchor="middle", bold=True)
    txt(p, 320, 80, "score = q·k  →  softmax  →  weights", 14, INK, anchor="middle")
    for i,y in enumerate([110,180,250]):
        arrow(p, 155, y, 260, 140, GRAY, 1.5, dash="5 4")
    box(p, 260, 110, 120, 60, fill="#FFF7E8", stroke=AMBER, sw=2.5)
    txt(p, 320, 136, "weights", 14, AMBER, anchor="middle", bold=True)
    txt(p, 320, 156, "Σ = 1", 12, GRAY, anchor="middle")
    arrow(p, 382, 140, 460, 140, INK)
    circle(p, 510, 140, 26, "#EAF3E4", GREEN)
    txt(p, 510, 146, "out", 15, GREEN, anchor="middle", bold=True)
    txt(p, 360, 300, "out = Σ weightsᵢ · vᵢ. Attend to what is relevant.", 15, INK, anchor="middle")
save("l14-attn.svg", 720, 330, "Attention: query, key, value", d_attn)

def d_mask(p):
    n = 6; s = 56; ox, oy = 170, 70
    for i in range(n):
        for j in range(n):
            fill = "#E8EEF7" if j <= i else "#FFFFFF"
            stroke = BLUE if j <= i else "#D8D2C4"
            p.append(f'<rect x="{ox+j*s}" y="{oy+i*s}" width="{s-4}" height="{s-4}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    for i in range(n):
        txt(p, ox-14, oy+i*s+34, f"t{i+1}", 12, GRAY, anchor="end")
        txt(p, ox+i*s+26, oy-12, f"t{i+1}", 12, GRAY, anchor="middle")
    txt(p, 360, 450, "Causal mask: position t sees only positions ≤ t.", 15, INK, anchor="middle")
    txt(p, 360, 478, "The future is masked out. No cheating.", 14, GRAY, anchor="middle")
save("l14-mask.svg", 720, 510, "Causal masking", d_mask)

def d_t2(p):
    ox, oy = 90, 300
    line(p, ox, 50, ox, oy+10, INK); line(p, ox-8, oy, 620, oy, INK)
    txt(p, 360, 345, "sequence length T →", 13, INK, anchor="middle")
    txt(p, 45, 70, "compute", 12, INK)
    pts = [(ox + i/40*500, oy - (i/40*500)**2/500*230 - 10) for i in range(41)]
    pl = " ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    p.append(f'<polyline points="{pl}" fill="none" stroke="{RED}" stroke-width="3.5"/>')
    txt(p, 480, 130, "O(T²·d)", 18, RED, bold=True)
    txt(p, 360, 380, "The T×T attention matrix is the bottleneck. 4× length = 16× compute.", 15, INK, anchor="middle")
save("l14-t2.svg", 720, 410, "Attention is quadratic in T", d_t2)

# ---------------- L15 ----------------
def d_kvcache(p):
    txt(p, 150, 70, "without cache", 15, RED, anchor="middle", bold=True)
    txt(p, 520, 70, "with KV cache", 15, GREEN, anchor="middle", bold=True)
    for i in range(3):
        y = 120 + i*60
        box(p, 60, y, 180, 44, fill="#FBEDEC", stroke=RED)
        txt(p, 150, y+27, "recompute all K,V", 13, INK, anchor="middle")
        box(p, 430, y, 180, 44, fill="#EAF3E4", stroke=GREEN)
        txt(p, 520, y+27, "reuse stored K,V", 13, INK, anchor="middle")
    txt(p, 360, 340, "Decode step t only needs the NEW token's K,V. Store the rest.", 15, INK, anchor="middle")
save("l15-kvcache.svg", 720, 370, "KV cache", d_kvcache)

def d_moe(p):
    circle(p, 110, 180, 26, "#FFFFFF", BLUE); txt(p, 110, 186, "token", 14, BLUE, anchor="middle")
    box(p, 200, 150, 110, 60, fill="#FFF7E8", stroke=AMBER, sw=2.5)
    txt(p, 255, 176, "router", 15, AMBER, anchor="middle", bold=True)
    txt(p, 255, 196, "top-2", 12, GRAY, anchor="middle")
    arrow(p, 138, 180, 198, 180, INK)
    for i in range(4):
        y = 80 + i*70
        on = i in (0, 2)
        box(p, 360, y, 150, 52, fill="#EAF3E4" if on else "#FFFFFF", stroke=GREEN if on else GRAY, sw=3 if on else 2)
        txt(p, 435, y+32, f"expert {i+1}", 14, GREEN if on else GRAY, anchor="middle", bold=on)
        arrow(p, 312, 180, 358, y+26, GREEN if on else GRAY, 2.5 if on else 1.5)
    txt(p, 600, 180, "Σ weighted", 14, INK, anchor="middle")
    arrow(p, 512, 180, 560, 180, INK)
    txt(p, 360, 380, "Sparse: each token uses 2 of N experts. Capacity without the FLOPs.", 15, INK, anchor="middle")
save("l15-moe.svg", 720, 410, "Mixture of experts", d_moe)

def d_icl(p):
    box(p, 30, 60, 320, 200, stroke=BLUE, sw=3)
    txt(p, 190, 95, "Zero-shot", 16, BLUE, bold=True, anchor="middle")
    txt(p, 60, 130, "Task: translate to French.", 13, INK)
    txt(p, 60, 155, "Input: hello", 13, INK)
    txt(p, 60, 185, "→ bonjour", 13, GREEN, bold=True)
    txt(p, 60, 215, "no examples given", 12, GRAY)
    box(p, 370, 60, 320, 200, stroke=PURPLE, sw=3)
    txt(p, 530, 95, "Few-shot", 16, PURPLE, bold=True, anchor="middle")
    txt(p, 400, 130, "cat → chat", 13, INK)
    txt(p, 400, 155, "dog → chien", 13, INK)
    txt(p, 400, 185, "hello → ?", 13, INK)
    txt(p, 400, 215, "examples in the prompt", 12, GRAY)
    txt(p, 360, 310, "No parameter updates. The prompt IS the training set.", 15, INK, anchor="middle")
save("l15-icl.svg", 720, 340, "Zero-shot vs few-shot", d_icl)

def d_sft(p):
    pairs = [("Translate: hello", "bonjour"), ("Summarize: ...", "TL;DR ...")]
    for i,(x,y) in enumerate(pairs):
        yy = 80 + i*90
        box(p, 60, yy, 260, 60, fill="#E8EEF7", stroke=BLUE, sw=2)
        txt(p, 190, yy+26, "x: "+x, 13, BLUE, anchor="middle"); txt(p, 190, yy+46, "seen, not predicted", 11, GRAY, anchor="middle")
        box(p, 380, yy, 260, 60, fill="#EAF3E4", stroke=GREEN, sw=2)
        txt(p, 510, yy+26, "y: "+y, 13, GREEN, anchor="middle"); txt(p, 510, yy+46, "loss only here", 11, GRAY, anchor="middle")
        arrow(p, 322, yy+30, 378, yy+30, INK)
    txt(p, 360, 300, "SFT = instruction tuning. Continue from checkpoint; predict y given x.", 15, INK, anchor="middle")
save("l15-sft.svg", 720, 330, "Supervised fine-tuning", d_sft)

# ---------------- L16 ----------------
def d_mdp(p):
    circle(p, 170, 170, 55, "#E8EEF7", BLUE)
    txt(p, 170, 164, "state", 15, BLUE, anchor="middle", bold=True); txt(p, 170, 188, "s", 18, BLUE, anchor="middle", italic=True)
    box(p, 420, 130, 150, 80, fill="#FFF7E8", stroke=AMBER, sw=2.5)
    txt(p, 495, 162, "action a", 15, AMBER, anchor="middle", bold=True); txt(p, 495, 184, "π(a|s)", 14, INK, anchor="middle", italic=True)
    # loop arrows
    p.append(f'<path d="M 225 150 Q 330 90 420 130" fill="none" stroke="{INK}" stroke-width="2.5"/>')
    arrow(p, 415, 128, 428, 142, INK)
    txt(p, 330, 85, "act", 13, INK, anchor="middle")
    p.append(f'<path d="M 495 210 Q 400 300 225 200" fill="none" stroke="{GREEN}" stroke-width="2.5"/>')
    arrow(p, 230, 203, 218, 190, GREEN)
    txt(p, 380, 290, "world: s′ ~ P(·|s,a), reward r", 13, GREEN, anchor="middle")
    txt(p, 360, 345, "Goal: pick π to maximize expected total reward.", 15, INK, anchor="middle")
save("l16-mdp.svg", 720, 375, "The MDP loop", d_mdp)

def d_pg(p):
    box(p, 40, 70, 300, 110, fill="#EAF3E4", stroke=GREEN, sw=2.5)
    txt(p, 190, 108, "good trajectory", 15, GREEN, anchor="middle", bold=True)
    txt(p, 190, 136, "return = +10", 14, INK, anchor="middle")
    txt(p, 190, 160, "↑ P(actions)", 14, GREEN, anchor="middle", bold=True)
    box(p, 380, 70, 300, 110, fill="#FBEDEC", stroke=RED, sw=2.5)
    txt(p, 530, 108, "bad trajectory", 15, RED, anchor="middle", bold=True)
    txt(p, 530, 136, "return = −3", 14, INK, anchor="middle")
    txt(p, 530, 160, "↓ P(actions)", 14, RED, anchor="middle", bold=True)
    box(p, 120, 210, 480, 80, fill="#FFFFFF", stroke=INK, sw=2.5)
    txt(p, 360, 240, "∇J = E[ Σₜ ∇log π(aₜ|sₜ) · R ]", 20, INK, anchor="middle")
    txt(p, 360, 268, "sample, score, shift probability toward winners", 14, GRAY, anchor="middle")
save("l16-pg.svg", 720, 320, "Policy gradient intuition", d_pg)

# ---------------- L17 ----------------
def d_ppo(p):
    ox, oy = 80, 270
    line(p, ox, 60, ox, oy+10, INK); line(p, ox-8, oy, 600, oy, INK)
    txt(p, 360, 315, "ratio r = π_new / π_old →", 13, INK, anchor="middle")
    txt(p, 45, 80, "gain", 12, INK)
    line(p, ox+220, 60, ox+220, oy, GRAY, 1.5, dash="6 5")
    txt(p, ox+220, 335, "r = 1", 12, GRAY, anchor="middle")
    # unclipped line
    line(p, ox+60, oy-40, ox+380, oy-260, RED, 2, dash="8 6")
    # clipped
    pts = [(ox+60, oy-40),(ox+220, oy-150),(ox+300, oy-180),(ox+380, oy-180)]
    pl = " ".join(f"{x},{y}" for x,y in pts)
    p.append(f'<polyline points="{pl}" fill="none" stroke="{GREEN}" stroke-width="3.5"/>')
    txt(p, 480, 120, "clipped: stop here", 13, GREEN, bold=True)
    txt(p, 180, 200, "unclipped", 12, RED)
    txt(p, 360, 365, "PPO: reuse old samples, but clip the ratio. Stay proximal.", 15, INK, anchor="middle")
save("l17-ppo.svg", 720, 395, "PPO clipping", d_ppo)

def d_verify(p):
    box(p, 40, 80, 200, 100, fill="#FFFFFF", stroke=BLUE, sw=2.5)
    txt(p, 140, 118, "model output", 14, BLUE, anchor="middle", bold=True)
    txt(p, 140, 146, "<answer>42</answer>", 14, INK, anchor="middle")
    arrow(p, 244, 130, 296, 130, INK)
    box(p, 300, 80, 140, 100, fill="#FFF7E8", stroke=AMBER, sw=2.5)
    txt(p, 370, 118, "parser", 14, AMBER, anchor="middle", bold=True)
    txt(p, 370, 146, "extract 42", 13, INK, anchor="middle")
    arrow(p, 444, 130, 496, 130, INK)
    box(p, 500, 80, 180, 100, fill="#EAF3E4", stroke=GREEN, sw=2.5)
    txt(p, 590, 112, "42 == 42?", 15, GREEN, anchor="middle", bold=True)
    txt(p, 590, 142, "reward = 1 : 0", 16, INK, anchor="middle", bold=True)
    txt(p, 360, 240, "Verifiable rewards: the task itself grades the answer. No human labeler.", 15, INK, anchor="middle")
save("l17-verify.svg", 720, 270, "Verifiable reward for math", d_verify)

if __name__ == "__main__":
    print("plates written to", OUT)
