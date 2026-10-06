#!/usr/bin/env python3
"""Chunk C: L05-L08 plates."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathml_a import *
from plates_mathml_b import usedwhere

# ---------------- L05 ----------------
def l05():
    bab("plate-l05-conditional.svg", "Conditional probability restricts the world",
        "P(A|B) = P(A and B) / P(B). First keep only B, then measure A.",
        "The confusion", [("test 95% accurate", None),
                          ("P(positive | sick) = 0.95", None),
                          ("gut: P(sick | positive) = 0.95?", ORANGE)],
        "flip", "The right question",
        [("P(sick | positive) = ?", TEAL),
         ("restrict to the positives", None),
         ("then count the sick", None)], footer="One claim: condition first, measure second. Shell 2. Project: Stanford Frontier AI.")

    tp = 100 * 0.95; fp = 9900 * 0.10
    post = tp / (tp + fp); assert round(post, 3) == 0.088
    vbar("plate-l05-bayes-counts.svg", "990 false alarms drown 95 true hits",
         "Posterior = 95 / 1085 = 8.8%. The base rate dominates the test.",
         "Counts out of 10,000 people. Shell 3. Source: original arithmetic. Project: Stanford Frontier AI.",
         [("true positives", tp, TEAL), ("false positives", fp, PINK)])

    bab("plate-l05-bayes-flip.svg", "Bayes flips likelihood into posterior",
        "Posterior = likelihood x prior / evidence. Name the pieces every time.",
        "What the test gives", [("P(positive | sick) = 0.95", None),
                                ("likelihood: data given hypothesis", MUTED)],
        "Bayes", "What you want",
        [("P(sick | positive) = 0.088", TEAL),
         ("= 0.95 * 0.01 / 0.1085", None),
         ("prior 1% -> posterior 8.8%", None)], footer="One claim: multiply prior by likelihood, renormalize. Shell 3. Project: Stanford Frontier AI.")

    probs = [0.1]*5 + [0.5]; faces = [1, 2, 3, 4, 5, 6]
    E = sum(f * p for f, p in zip(faces, probs))
    V = sum(p * (f - E) ** 2 for f, p in zip(faces, probs))
    assert round(E, 2) == 4.50 and round(V, 2) == 3.25
    bab("plate-l05-expectation.svg", "Two numbers summarize a random variable",
        "Loaded die: mean 4.5, variance 3.25. Both computed term by term.",
        "Expectation (average)", [("E = 1*.1+2*.1+3*.1+4*.1+5*.1+6*.5", None),
                                  ("E = 4.5  (fair die: 3.5)", TEAL)],
        "spread", "Variance (spread)",
        [("V = .1*(1-4.5)^2 + ... + .5*(6-4.5)^2", None),
         ("V = 3.25, std = 1.80", TEAL),
         ("E[X+Y] = E[X]+E[Y], always", MUTED)], footer="One claim: mean locates, variance spreads. Shell 2. Project: Stanford Frontier AI.")

    usedwhere("plate-l05-used-where.svg", "Probability: what is used where",
              "Every classifier is a bet priced by these rules.",
              "Shell 5. Source: standard ML practice. Project: Stanford Frontier AI.",
              [("Bayes rule", "naive Bayes (CS229 L05)", "priors x likelihoods, classify by posterior"),
               ("base rate", "rare-event classifiers", "fraud, disease: posteriors stay small"),
               ("expectation", "SGD proofs", "linearity splits loss over batches"),
               ("posterior", "LLM next token", "the model outputs P(token | context)")])

# ---------------- L06 ----------------
def l06():
    var = lambda p: p * (1 - p)
    assert round(var(0.5), 2) == 0.25 and round(var(0.03), 4) == 0.0291
    curve("plate-l06-bernoulli.svg", "Bernoulli variance peaks at p = 0.5",
          "p(1-p): maximum uncertainty at 0.5, zero at certainty. Ad click p=0.03: 0.0291.",
          "Variance of one yes/no trial. Shell 2. Source: original arithmetic. Project: Stanford Frontier AI.",
          var, 0.0, 1.0, [(0.03, "0.0291", FOCUS), (0.5, "max 0.25", TEAL)],
          xlabel="p", ylabel="p(1-p)")

    from math import comb
    p = 0.2; k = 3; n = 10
    prob = comb(n, k) * p**k * (1 - p)**(n - k)
    assert round(prob, 3) == 0.201
    bab("plate-l06-binomial.svg", "Binomial: exactly k yeses in n trials",
        "P(3 spam of 10) = 120 * 0.2^3 * 0.8^7 = 0.201. Mean 2, variance 1.6.",
        "Count the ways", [("C(10,3) = 120 orders", None),
                           ("0.2^3 = 0.008 (the yeses)", None),
                           ("0.8^7 = 0.2097 (the nos)", None)],
        "multiply", "Probability and moments",
        [("P = 120*0.008*0.2097 = 0.201", TEAL),
         ("E = np = 2", None), ("Var = np(1-p) = 1.6", None)], footer="One claim: ways times per-way probability. Shell 2. Project: Stanford Frontier AI.")

    def gauss(x, mu=170.0, s=10.0):
        return math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    curve("plate-l06-gaussian.svg", "The Gaussian: 68-95-99.7",
          "Heights N(170, 100): 95% fall between 150 and 190 cm.",
          "Bell curve of human heights. Shell 2. Source: original arithmetic. Project: Stanford Frontier AI.",
          gauss, 130, 210, [(160, "68%", TEAL), (150, "95%", FOCUS), (140, "99.7%", ORANGE)],
          xlabel="height (cm)", ylabel="density")

    def L(p): return p**2 * (1 - p)**3
    assert abs(L(0.4) - 0.03456) < 1e-9
    assert L(0.4) > L(0.5) and L(0.4) > L(0.3)
    curve("plate-l06-mle.svg", "Maximum likelihood: the peak is the estimate",
          "Data [1,0,0,1,0]: L(p) = p^2 (1-p)^3 peaks at p = 0.4, the sample mean.",
          "Likelihood of p given 2 clicks in 5 views. Shell 3. Source: original arithmetic. Project: Stanford Frontier AI.",
          L, 0.01, 0.99, [(0.4, "peak 0.0346", TEAL), (0.5, "0.0313", MUTED), (0.3, "0.0309", MUTED)],
          xlabel="p", ylabel="likelihood")

    pts = [(1, 2), (2, 4), (3, 6)]
    mx = sum(x for x, y in pts) / 3; my = sum(y for x, y in pts) / 3
    cov = sum((x - mx) * (y - my) for x, y in pts) / 3
    assert round(cov, 2) == 1.33
    vx = sum((x - mx) ** 2 for x, y in pts) / 3; vy = sum((y - my) ** 2 for x, y in pts) / 3
    vsum = vx + vy + 2 * cov; assert round(vsum, 2) == 6.00
    P = Plate(760, 440, "Covariance: how two variables move together",
              "Cov = 1.33 on three collinear points. Var(X+Y) = 6: the +2Cov term added 2.67.",
              "Shell 3. Source: original arithmetic. Project: Stanford Frontier AI.")
    P.rect(24, 96, 360, 260, PANEL)
    ax, ay, aw, ah = 56, 130, 280, 200
    def pxx(x): return ax + (x - 0.5) / 3 * aw
    def pyy(y): return ay + ah - (y - 1.5) / 5 * ah
    P.line(ax, ay, ax, ay + ah, INK, 1.5); P.line(ax, ay + ah, ax + aw, ay + ah, INK, 1.5)
    P.line(pxx(0.5), pyy(0), pxx(3.5), pyy(2 * 3.5), TEAL, 2.5)
    for x, y in pts:
        P.p.append(f'<circle cx="{pxx(x):.1f}" cy="{pyy(y):.1f}" r="8" fill="{FOCUS}" stroke="{INK}" stroke-width="1.5"/>')
        P.text(pxx(x) + 10, pyy(y) - 8, f"({x},{y})", 13, 500, INK)
    P.rect(408, 96, 328, 260, NEWOBJ)
    P.text(424, 126, "The variance inflation", 16, 600, INK)
    for i, (s, c) in enumerate([("Cov = [(1-2)(2-4) + 0 + (1)(2)]/3", None),
                                ("    = 1.33 (positive: rising)", TEAL),
                                ("Var(X)=0.67, Var(Y)=2.67", None),
                                ("Var(X+Y) = 0.67+2.67+2*1.33", None),
                                ("         = 6.00", TEAL)]):
        P.mono(424, 158 + i * 30, s, 14.5, c or INK)
    P.save("plate-l06-covariance.svg")

    usedwhere("plate-l06-used-where.svg", "Distributions: what is used where",
              "Noise assumptions choose your loss; counting trains your model.",
              "Shell 5. Source: standard ML practice. Project: Stanford Frontier AI.",
              [("Gaussian init", "weight initialization", "sums stay controlled (CLT)"),
               ("MLE", "model training", "naive Bayes counts; deep nets optimize it"),
               ("68-95-99.7", "anomaly detection", "2-3 sigma thresholds"),
               ("covariance", "PCA L03", "covariance eigenvectors are components")])

# ---------------- L07 ----------------
def l07():
    dfdx = 2 * 2 + 3 * 1; dfdy = 3 * 2 + 2 * 1
    assert (dfdx, dfdy) == (7, 8)
    bab("plate-l07-partial.svg", "Partial derivative: freeze all but one",
        "f = x^2+3xy+y^2 at (2,1): df/dx = 7, df/dy = 8. Nudge x by 0.01, f rises ~0.07.",
        "Freeze y", [("f(x,y) = x^2 + 3xy + y^2", None),
                     ("df/dx = 2x + 3y", None),
                     ("at (2,1): 4 + 3 = 7", TEAL)],
        "freeze", "Freeze x",
        [("df/dy = 3x + 2y", None),
         ("at (2,1): 6 + 2 = 8", TEAL),
         ("each slope holds the other still", MUTED)], footer="One claim: a partial derivative is a one-axis slope. Shell 2. Project: Stanford Frontier AI.")

    bab("plate-l07-gradient.svg", "The gradient points steepest uphill",
        "grad f = [7, 8] at (2,1). Walk opposite: that is gradient descent.",
        "Stack the slopes", [("grad f = [df/dx, df/dy]", None),
                             ("at (2,1): [7, 8]", TEAL)],
        "use", "Direction + steepness",
        [("direction [7,8]: steepest increase", None),
         ("length: how steep", None),
         ("descent walks -[7,8] (L08)", MUTED)], footer="One claim: the gradient is the compass; descent walks against it. Shell 2. Project: Stanford Frontier AI.")

    J = [[4, 1], [3, 6]]
    bab("plate-l07-jacobian.svg", "The Jacobian: one gradient per output row",
        "g = [x^2+y, 3xy] at (2,1): J = [4 1; 3 6]. Small steps map to J times the step.",
        "Vector output g", [("g1 = x^2 + y,  g2 = 3xy", None),
                            ("each output has a gradient", None)],
        "stack", "The m x n matrix",
        [("J = [dg1/dx dg1/dy; dg2/dx dg2/dy]", None),
         ("at (2,1): [4 1; 3 6]", TEAL),
         ("m=1: J is the gradient, transposed", MUTED)], footer="One claim: the Jacobian is the linear map near a point. Shell 2. Project: Stanford Frontier AI.")

    P = Plate(760, 440, "Backprop by hand: forward stores, backward multiplies",
              "x=2, w1=3, w2=4: y=24. dy/dw2=6, dy/dw1=4*2=8. Bump test agrees.",
              "Shell 3. Source: original toy. Project: Stanford Frontier AI.")
    P.rect(24, 96, 712, 120, PANEL)
    P.chip(60, 146, 90, 44, "x = 2")
    P.chip(250, 146, 90, 44, "h = 6", NEWOBJ)
    P.chip(440, 146, 110, 44, "y = 24", COUNT)
    P.arrow(150, 168, 250, 168, "w1 = 3", TEAL)
    P.arrow(340, 168, 440, 168, "w2 = 4", TEAL)
    P.text(60, 130, "forward: store h", 13, 500, MUTED)
    P.rect(24, 236, 712, 120, PANEL)
    P.arrow(440, 296, 340, 296, "dy/dh = 4", ORANGE)
    P.arrow(250, 296, 150, 296, "dy/dw1 = 4*2 = 8", ORANGE)
    P.text(560, 296, "dy/dw2 = 6", 15, 600, ORANGE)
    P.text(60, 270, "backward: each layer multiplies local deriv x incoming signal", 13, 500, MUTED)
    P.mono(60, 340, "bump test: w1 -> 3.01, y -> 24.08: slope 8. agrees.", 13.5, TEAL)
    P.save("plate-l07-backprop.svg")

    usedwhere("plate-l07-used-where.svg", "Calculus: what is used where",
              "Derivatives are the engine; the chain rule is the transmission.",
              "Shell 5. Source: standard ML practice. Project: Stanford Frontier AI.",
              [("gradient", "every optimizer", "descent walks against it (L08)"),
               ("chain rule", "backprop", "one backward sweep per network"),
               ("Jacobian-vector", "autograd", "never form the full matrix"),
               ("partial derivatives", "sensitivity", "which input moves the output")])


# ---------------- L08 ----------------
def l08():
    bab("plate-l08-convex.svg", "Convex: one bowl, no traps",
        "f'' >= 0 everywhere. Every local minimum is the global minimum.",
        "Bowl: f(x) = (x-3)^2", [("f'' = 2 > 0: convex", TEAL),
                                 ("one valley at x = 3", None),
                                 ("GD cannot get trapped", None)],
        "test", "Not convex: f(x) = x^3",
        [("f'' = 6x: negative for x < 0", ORANGE),
         ("dips that are not the bottom", None),
         ("no guarantees for walkers", None)], footer="One claim: convex means the only dip is the bottom. Shell 2. Project: Stanford Frontier AI.")

    xs, fs = [], []
    x = 0.0
    for _ in range(6):
        xs.append(x); fs.append((x - 3) ** 2); x = x - 0.1 * 2 * (x - 3)
    assert [round(v, 3) for v in xs[:5]] == [0.0, 0.6, 1.08, 1.464, 1.771]
    trace("plate-l08-gd-trace.svg", "Gradient descent converges geometrically",
          "eta=0.1: x closes 20% of the gap per step. Loss x0.64 per step.",
          "x walking to the minimum at 3. Shell 3. Source: original trace. Project: Stanford Frontier AI.",
          [("x", xs, TEAL, False), ("target 3", [3.0] * len(xs), MUTED, True)],
          xlabel="step", ylabel="x")

    xd, fd = [0.0], [9.0]
    x = 0.0
    for _ in range(3):
        x = x - 1.1 * 2 * (x - 3); xd.append(x); fd.append((x - 3) ** 2)
    assert [round(v, 2) for v in xd] == [0.0, 6.6, -1.32, 8.18]
    assert [round(v, 2) for v in fd] == [9.0, 12.96, 18.66, 26.87]
    trace("plate-l08-diverge.svg", "eta = 1.1 diverges: the sharp boundary",
          "Loss 9.0 -> 12.96 -> 18.66 -> 26.87. Each step overshoots further.",
          "Overshoot grows every step. Shell 3. Source: original trace. Project: Stanford Frontier AI.",
          [("loss f(x)", fd, ORANGE, False)], xlabel="step", ylabel="loss")

    bab("plate-l08-lr-rule.svg", "The step-size rule: eta < 2/L",
        "Curvature L = 2 here, so eta < 1. eta=0.1 converges; eta=1.1 explodes.",
        "Stable: eta = 0.1", [("eta < 2/L = 1: inside", TEAL),
                              ("gap shrinks x0.8 per step", None)],
        "rule", "Unstable: eta = 1.1",
        [("eta > 1: overshoot grows", ORANGE),
         ("11% over the limit is enough", None),
         ("loss explodes: debug the step", MUTED)], footer="One claim: the learning rate boundary is sharp. Shell 3. Project: Stanford Frontier AI.")

    bab("plate-l08-logreg.svg", "Logistic regression: convex by construction",
        "Hessian = X^T D X with D > 0: eigenvalues >= 0. One bowl, any start works.",
        "Sigmoid", [("p = 1 / (1 + e^-z), z = w.x", None),
                    ("outputs a probability", None)],
        "convex", "The loss",
        [("loss = -[y log p + (1-y) log(1-p)]", None),
         ("Hessian X^T D X, D diagonal positive", TEAL),
         ("GD from any start: global min", None)], footer="One claim: the logistic loss has exactly one valley. Shell 3. Project: Stanford Frontier AI.")

    usedwhere("plate-l08-used-where.svg", "Convexity: what is used where",
              "Convex losses train reliably; the rest need the optimizer zoo.",
              "Shell 5. Source: standard ML practice. Project: Stanford Frontier AI.",
              [("logistic regression", "CS229 L05", "convex: GD always works"),
               ("SVM", "hinge loss", "convex: global optimum guaranteed"),
               ("deep networks", "non-convex", "Adam, momentum, restarts"),
               ("learning rate", "all training", "too big diverges: the 1.1 lesson")])

if __name__ == "__main__":
    l05(); l06(); l07(); l08()
    print("L05-L08 plates done")
