#!/usr/bin/env python3
"""Chunk D: L09, L10, cheatsheet, crash-course plates."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathml_a import *
from plates_mathml_b import usedwhere

# ---------------- L09 ----------------
def l09():
    pts = [(1, 1.1), (2, 1.9), (3, 3.2)]
    XtX = sum(x * x for x, y in pts); Xty = sum(x * y for x, y in pts)
    w = Xty / XtX
    assert XtX == 14 and round(Xty, 1) == 14.5 and round(w, 4) == 1.0357
    scatter("plate-l09-spring.svg", "Least squares: the best line through noisy points",
            f"y = {w:.4f} x: normal equation X^T X w = X^T y gives 14 w = 14.5.",
            "Spring force vs stretch. Shell 3. Source: original toy. Project: Stanford Frontier AI.",
            [(x, y, "", FOCUS) for x, y in pts], line=(w, 0.0),
            xlabel="force x (N)", ylabel="stretch y (cm)", xrange=(0, 3.5), yrange=(0, 3.7))

    preds = [w * x for x, y in pts]
    ss = sum((p - y) ** 2 for p, (x, y) in zip(preds, pts))
    assert round(ss, 4) == 0.0421
    bab("plate-l09-normal.svg", "The normal equation: differentiate, set to zero",
        "X^T X w = X^T y. Toy: 14 w = 14.5, w = 1.0357, error 0.0421.",
        "loss(w) = ||Xw - y||^2", [("gradient: 2 X^T (Xw - y)", None),
                                   ("convex: zero gradient = minimum", None)],
        "solve", "One linear system",
        [("X^T X = 14,  X^T y = 14.5", None),
         ("w = 14.5/14 = 1.0357", TEAL),
         ("error 0.0421 beats every guess", None)], footer="One claim: one linear system replaces all guessing. Shell 3. Project: Stanford Frontier AI.")

    r = [y - w * x for x, y in pts]
    col = [x for x, y in pts]
    d = sum(ri * ci for ri, ci in zip(r, col))
    assert abs(d) < 0.01
    bab("plate-l09-projection.svg", "Least squares IS projection",
        "Residual dots the column space to ~0. Prediction is the shadow; error is perpendicular.",
        "y not in the column space", [("y = [1.1, 1.9, 3.2]", None),
                                      ("column: [1, 2, 3]", None),
                                      ("y is off the line (noise)", None)],
        "project", "Orthogonality check",
        [(f"residual = [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]", None),
         (f"residual.[1,2,3] = {d:.3f} ~ 0", TEAL),
         ("X^T(Xw - y) = 0: the normal equation", MUTED)], footer="One claim: the error is perpendicular to the model's reach. Shell 3. Project: Stanford Frontier AI.")

    bab("plate-l09-singular.svg", "Duplicate features break the formula",
        "X^T X = [14 28; 28 56], det = 0. Infinite solutions fit identically.",
        "Second feature = 2x first", [("X^T X = [14 28; 28 56]", None),
                                      ("row 2 = 2 x row 1", ORANGE),
                                      ("det = 14*56 - 28*28 = 0", None)],
        "fix", "No unique w",
        [("w=[1.036, 0] and w=[0, 0.518] tie", None),
         ("fixes: drop, regularize, min-norm", TEAL),
         ("100 feats, 10 ex: rank <= 10", MUTED)], footer="One claim: dependence makes the normal equation singular. Shell 3. Project: Stanford Frontier AI.")

    usedwhere("plate-l09-used-where.svg", "Least squares: what is used where",
              "The oldest learning algorithm, still the first fit.",
              "Shell 5. Source: standard ML practice. Project: Stanford Frontier AI.",
              [("normal equation", "small-d regression", "exact, O(d^3)"),
               ("gradient descent", "large-d regression", "iterate instead (L08)"),
               ("ridge", "CS229 L03", "add lambda I: fixes singularity"),
               ("Gaussian MLE", "probabilistic view", "squares = Gaussian noise (L06)")])

# ---------------- L10 ----------------
def l10():
    curve("plate-l10-surprise.svg", "Surprise: rarer events carry more bits",
          "-log2(p): a 50/50 event is 1 bit; a 1% event is 6.64 bits.",
          "Surprise curve. Shell 2. Source: original arithmetic. Project: Stanford Frontier AI.",
          lambda p: -math.log2(p), 0.005, 1.0,
          [(0.5, "1 bit", TEAL), (0.25, "2 bits", FOCUS), (0.01, "6.64 bits", ORANGE)],
          xlabel="probability p", ylabel="-log2(p) bits")

    H = lambda ps: -sum(p * math.log2(p) for p in ps)
    hf, hb = H([0.5, 0.5]), H([0.25, 0.75])
    assert round(hf, 3) == 1.0 and round(hb, 3) == 0.811
    vbar("plate-l10-entropy.svg", "Entropy: expected surprise",
         "Fair coin: 1 bit. Biased 0.25 coin: 0.811 bits. Bias teaches less per flip.",
         "Bits per outcome. Shell 2. Source: original arithmetic. Project: Stanford Frontier AI.",
         [("fair coin", hf, TEAL), ("biased 0.25", hb, FOCUS)])

    p = [0.25, 0.75]; q = [0.5, 0.5]
    klpq = sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q))
    klqp = sum(qi * math.log2(qi / pi) for pi, qi in zip(p, q))
    assert round(klpq, 3) == 0.189 and round(klqp, 3) == 0.208
    bab("plate-l10-kl.svg", "KL divergence is asymmetric",
        f"KL(p||q) = {klpq:.3f} bits, KL(q||p) = {klqp:.3f} bits. Not a distance.",
        "Believe q, truth p", [(f"KL(p||q) = {klpq:.3f} bits", TEAL),
                               ("cost of your wrong belief", None)],
        "direction", "Flip it",
        [(f"KL(q||p) = {klqp:.3f} bits", ORANGE),
         ("different question, different price", None),
         ("KL >= 0; zero only if p = q", MUTED)], footer="One claim: KL has a direction. Shell 3. Project: Stanford Frontier AI.")

    ce = lambda qt: -math.log2(qt)
    vbar("plate-l10-crossentropy.svg", "Cross-entropy punishes confident errors",
         "Truth [1,0]. Predict 0.99: 0.014 bits. Predict 0.7: 0.515. Predict 0.01: 6.64.",
         "Loss in bits vs predicted probability of truth. Shell 3. Source: original arithmetic. Project: Stanford Frontier AI.",
         [("confident right", ce(0.99), TEAL), ("decent", ce(0.7), FOCUS), ("confident wrong", ce(0.01), PINK)])

    usedwhere("plate-l10-used-where.svg", "Information theory: what is used where",
              "Bits price every prediction a model makes.",
              "Shell 5. Source: standard ML practice. Project: Stanford Frontier AI.",
              [("cross-entropy", "LM training", "next-token loss (CS336)"),
               ("perplexity", "LM eval", "perplexity = 2^(cross-entropy)"),
               ("KL penalty", "RLHF", "keep the policy near the reference"),
               ("bits", "compression", "entropy = the coding limit")])

# ---------------- cheatsheet ----------------
def cheat():
    P = Plate(760, 560, "MATH-ML: the ten numbers that matter",
              "One number per lesson. If you remember these, you remember the course.",
              "Shell 5. Source: the ten lessons. Project: Stanford Frontier AI.")
    rows = [("L01 cosine", "0.77", "direction, not size"),
            ("L02 rotation", "[-2, 1]", "[0 -1; 1 0] turns [1,2]"),
            ("L03 eigen", "l = 3, 2", "[1,1] triples, [1,0] doubles"),
            ("L04 SVD", "s = 5, 0", "rank 1; drop s2, error = s2"),
            ("L05 Bayes", "8.8%", "95% test, 1% base rate"),
            ("L06 binomial", "0.201", "exactly 3 of 10 at p=0.2"),
            ("L07 backprop", "dy/dw1 = 8", "chain: 4 x 2"),
            ("L08 step", "eta < 1", "eta=1.1 diverges"),
            ("L09 fit", "w = 1.0357", "14 w = 14.5"),
            ("L10 loss", "0.515 bits", "truth [1,0], pred [0.7,0.3]")]
    y = 118
    for lab, num, note in rows:
        P.rect(24, y - 20, 130, 40, CHIP, LINE, 999)
        P.text(36, y + 5, lab, 14, 600, INK)
        P.mono(180, y + 5, num, 16, TEAL)
        P.text(360, y + 5, note, 14, 450, MUTED)
        y += 46
    P.save("plate-cheat-numbers.svg")

    P = Plate(760, 640, "MATH-ML: if this, then that",
              "Decision rules. Read the left, do the right.",
              "Shell 5. Source: the ten lessons. Project: Stanford Frontier AI.")
    rows = [("features look duplicated", "check det(X^T X); drop or regularize"),
            ("loss explodes mid-training", "learning rate too big: halve eta"),
            ("rare class, 99% accuracy", "report precision and recall, not accuracy"),
            ("need direction, not size", "cosine, not dot product"),
            ("matrix not square", "SVD, not eigen-decomposition"),
            ("slow-decaying spectrum", "do not truncate: compression will hurt"),
            ("noise has outliers", "squared loss is wrong: use absolute"),
            ("q assigns 0 to a live event", "smooth: infinite loss otherwise"),
            ("KL looks symmetric", "it is not: KL(p||q) != KL(q||p)"),
            ("covariance is zero", "not independence: try y = x^2")]
    y = 118
    for cond, act in rows:
        P.rect(24, y - 20, 300, 40, PINK, LINE, 8)
        P.text(36, y + 5, "IF " + cond, 13.5, 600, INK)
        P.arrow(330, y, 352, y, None, TEAL)
        P.rect(356, y - 20, 380, 40, NEWOBJ, LINE, 8)
        P.text(368, y + 5, "THEN " + act, 13.5, 500, INK)
        y += 46
    P.save("plate-cheat-decisions.svg")

# ---------------- crash course ----------------
def crash():
    P = Plate(760, 520, "The arc: five kinds of arithmetic, one trenchcoat",
              "Data becomes vectors; matrices move them; eigen/SVD compress; probability bets; calculus trains.",
              "Shell 5. Source: the ten lessons. Project: Stanford Frontier AI.")
    steps = [("L01-02", "vectors +", "matrices", "hold and move data"),
             ("L03-04", "eigen + SVD", "compress", "keep big directions"),
             ("L05-06", "probability", "bet", "price uncertainty"),
             ("L07-08", "calculus", "train", "walk downhill"),
             ("L09-10", "fit + bits", "score", "project, then price")]
    x = 24
    for a, b, c, d in steps:
        P.rect(x, 130, 128, 190, PANEL)
        P.text(x + 12, 160, a, 13, 600, TEAL)
        P.mono(x + 12, 190, b, 14, INK)
        P.mono(x + 12, 214, c, 14, INK)
        P.text(x + 12, 244, d, 13, 450, MUTED)
        if x < 600:
            P.arrow(x + 128, 225, x + 148, 225, None, TEAL)
        x += 148
    P.text(24, 380, "Exam rule of thumb:", 16, 600, INK)
    for i, s in enumerate(["every interview question is one of these five moves;",
                           "name the move, write its formula, work its toy number."]):
        P.text(24, 408 + i * 26, s, 15, 450, INK)
    P.save("plate-crash-arc.svg")

    P = Plate(760, 640, "Never-confuse pairs",
              "The six traps interviewers set. Each pair differs by exactly one idea.",
              "Shell 5. Source: the ten lessons. Project: Stanford Frontier AI.")
    rows = [("dot product", "cosine similarity", "size included vs divided out"),
            ("KL(p||q)", "KL(q||p)", "direction of the wrongness"),
            ("variance", "std deviation", "squared units vs same units"),
            ("gradient", "Jacobian", "vector output=1 row vs m rows"),
            ("likelihood P(E|H)", "posterior P(H|E)", "test's view vs your question"),
            ("zero covariance", "independence", "no linear link vs no link at all"),
            ("accuracy", "precision/recall", "lies on rare classes vs tells truth"),
            ("convex", "non-convex", "one valley vs traps"),
            ("rank", "dimension", "used directions vs room size"),
            ("MLE", "MAP", "no prior vs prior included")]
    y = 118
    for a, b, diff in rows:
        P.rect(24, y - 20, 190, 40, CHIP, LINE, 999); P.text(36, y + 5, a, 13.5, 600, INK)
        P.text(224, y + 5, "vs", 13, 500, MUTED)
        P.rect(258, y - 20, 190, 40, CHIP, LINE, 999); P.text(270, y + 5, b, 13.5, 600, INK)
        P.text(462, y + 5, diff, 13.5, 450, ORANGE)
        y += 46
    P.save("plate-crash-traps.svg")

if __name__ == "__main__":
    l09(); l10(); cheat(); crash()
    print("L09-L10-cheat-crash plates done")
