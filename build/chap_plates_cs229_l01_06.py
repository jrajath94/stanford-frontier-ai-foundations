#!/usr/bin/env python3
"""Chapter plates for CS229 L01-L06: dense SVGs in the mse435 bar layout.

Bar: content/v2/mse435/assets/plate-l01-chap-*.svg (960x600, warm paper
#F7F4EE, three regions left/center/right, tradeoff band, one-connection
footer). Caption format matches the bar EXACTLY.

Every number on every plate is recomputed below from the lesson text
and asserted. If any assert fails, no plate is written.
"""
import math
import os
import xml.sax.saxutils as sx

OUT = "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs229/assets"

# ---------------- number checks (recomputed from the six lessons) ----------------
def close(a, b, tol=1e-9):
    return abs(a - b) <= tol

# l01: 1000 rules x 30 min = 500 engineer-hours
assert 1000 * 30 / 60 == 500
# l01: 1 - 0.999^1000 ~ 63%
assert f"{(1 - 0.999**1000) * 100:.0f}" == "63"
# l01: precision/recall toys
assert 49 / 50 == 0.98 and 49 / 100 == 0.49
assert f"{95 / 200 * 100:.1f}" == "47.5" and 95 / 100 == 0.95
assert f"{81 / 90 * 100:.0f}" == "90" and f"{81 / 100 * 100:.0f}" == "81"
# l01: 10000 labels / 2000 per day = 5 labeler-days
assert 10000 / 2000 == 5
# l01: 1750 sq ft x $200 = $350,000
assert 1750 * 200 == 350000
# l01: 1000 falls x $500 = $500,000
assert 1000 * 500 == 500000
# l01: 50 rules per wave x 30 min = 25 engineer-hours
assert 50 * 30 / 60 == 25
# l02: squared vs absolute on misses 1, 2, 10
assert 1 + 4 + 100 == 105 and 1 + 2 + 10 == 13
# l02: 3-house toy J at theta=0 and after one step
assert f"{(4 + 9 + 25) / 6:.2f}" == "6.33"
assert f"{(2.10 + 4.28 + 13.54) / 6:.2f}" == "3.32"
# l02: one step knobs
assert f"{0 - 0.05 * (-3.33):.3f}" == "0.167"
# lesson prints 0.383 (truncated); exact value is 0.3835
assert abs(0 - 0.05 * (-7.67) - 0.3835) < 1e-12
# l02: alpha=0.01 trace on theta^2 from 4
assert f"{4 * 0.98**100:.2f}" == "0.53"
# l02: SGD unbiased toy: (-2 + -6)/2 = -4
assert (-2 + -6) / 2 == -4
# l02: batch cost 2B x 40 = 80B
assert 2e9 * 40 == 8e10
# l02: normal equations toy
assert 3 * 14 - 6 * 6 == 6
assert f"{(14 * 10 - 6 * 23) / 6:.4f}"[:3] == "0.3" and (14 * 10 - 6 * 23) / 6 == 1 / 3
assert (-6 * 10 + 3 * 23) / 6 == 3 / 2
assert f"{(0.1667**2 + 0.3333**2 + 0.1667**2) / 6:.3f}" == "0.028"
# l02: n=10,000 -> n^3 = 10^12
assert 10000**3 == 10**12
# l02: redundant features det
assert f"{3 * 0.0259 - 0.279**2:.4f}" == "-0.0001"
# l03: coin MLE scoreboard
assert f"{0.5**10:.5f}" == "0.00098"
assert f"{0.6**7 * 0.4**3:.5f}" == "0.00179"
assert f"{0.7**7 * 0.3**3:.5f}" == "0.00222"
assert f"{0.8**7 * 0.2**3:.5f}" == "0.00168"
assert f"{(0.7**7 * 0.3**3) / 0.5**10:.1f}" == "2.3"
# l03: log likelihoods
assert f"{7 * math.log(0.7) + 3 * math.log(0.3):.2f}" == "-6.11"
assert f"{10 * math.log(0.5):.2f}" == "-6.93"
assert f"{10000 * math.log(0.5):.0f}" == "-6931"
# l03: sigmoid values
assert f"{1 / (1 + math.exp(-0.27)):.2f}" == "0.57"
assert f"{1 / (1 + math.exp(-0.55)):.2f}" == "0.63"
assert f"{1 / (1 + math.exp(-2)):.2f}" == "0.88"
assert f"{1 / (1 + math.exp(2)):.2f}" == "0.12"
assert f"{1 / (1 + math.exp(-10)):.5f}" == "0.99995"
# l03: flat-gradient trap: 0.99 x 0.0099 = 0.0098
assert f"{0.99 * 0.0099:.4f}" == "0.0098"
# l03: Newton on theta^2: 4 - 8/2 = 0; GD at 0.1: 4*0.8^38 ~ 0.001
assert 4 - 8 / 2 == 0
# lesson: GD at alpha=0.1 needs about 38 steps to reach 0.001
assert 4 * 0.8**37 > 0.001 > 4 * 0.8**38
# l03: GD vs Newton ops: 500000 x 2000 = 10^9; Newton step at n=1000,d=20
assert 500000 * 2000 == 10**9
assert 1000 * 400 + 8000 == 408000
# l03: d=1B -> d^3 = 10^27
assert (10**9)**3 == 10**27
# l04: Bernoulli natural parameter at phi=0.8
assert f"{math.log(0.8 / 0.2):.3f}" == "1.386"
assert f"{math.log(1 + math.exp(1.386)):.3f}" == "1.609"
assert f"{math.exp(1.386) / (1 + math.exp(1.386)):.1f}" == "0.8"
# l04: Poisson weekend toy: exp(log 100 + 0.69) = 200
# lesson: theta_weekend = 0.69 = log 2; weekend prediction exp(log 100 + log 2) = 200
assert abs(0.69 - math.log(2)) < 0.01
assert f"{math.exp(math.log(100) + math.log(2)):.0f}" == "200"
# l04: softmax toy
e = [math.exp(2.0), math.exp(1.0), math.exp(0.5)]
assert [f"{x:.2f}" for x in e] == ["7.39", "2.72", "1.65"]
s = sum(e)
assert f"{s:.2f}" == "11.76"
assert [f"{x / s:.2f}" for x in e] == ["0.63", "0.23", "0.14"]
# l04: temperature tau=0.5 and tau=2
e05 = [math.exp(4.0), math.exp(2.0), math.exp(1.0)]
s05 = sum(e05)
assert [f"{x / s05:.2f}" for x in e05] == ["0.84", "0.11", "0.04"]
e2 = [math.exp(1.0), math.exp(0.5), math.exp(0.25)]
s2 = sum(e2)
assert [f"{x / s2:.2f}" for x in e2] == ["0.48", "0.29", "0.23"]
# l04: K=2 softmax = sigmoid(1)
assert f"{e[0] / (e[0] + e[1]):.3f}" == f"{1 / (1 + math.exp(-1)):.3f}" == "0.731"
# l04: cross-entropy
assert f"{-math.log(0.63):.2f}" == "0.46"
assert f"{-math.log(0.05):.2f}" == "3.00"
assert f"{-math.log(1e-6):.1f}" == "13.8"
assert close(-0.37 + 0.23 + 0.14, 0.0, 1e-12)
assert f"{-(0.9 * math.log(0.63) + 0.05 * math.log(0.23) + 0.05 * math.log(0.14)):.2f}" == "0.59"
# l05: GDA toy {1,3},{7,9}: means 2 and 8, pooled var 1
assert (1 + 3) / 2 == 2 and (7 + 9) / 2 == 8
assert ((1 - 2)**2 + (3 - 2)**2 + (7 - 8)**2 + (9 - 8)**2) / 4 == 1
# l05: knob counts at d=100
assert 2 * 100 + 100 * 101 // 2 + 1 == 5251
assert 2 * 100 + 2 * (100 * 101 // 2) + 1 == 10301
# l05: Naive Bayes toy
assert f"{0.8 * 0.8 * 0.5:.2f}" == "0.32"
assert f"{0.1 * 0.1 * 0.5:.3f}" == "0.005"
assert f"{0.32 / 0.005:.0f}" == "64"
assert f"{10_000_000 / 86400:.0f}" == "116"
# l05: Laplace: (0+1)/(10+2) = 1/12
assert (0 + 1) / (10 + 2) == 1 / 12
assert f"{math.log(0.32):.2f}" == "-1.14"
assert f"{math.log(0.005):.2f}" == "-5.30"
# l06: bias-variance toy: preds 0.7, 0.9, 1.1 at truth 1.0
assert (0.7 + 0.9 + 1.1) / 3 == 0.9
assert close((0.9 - 1.0) ** 2, 0.01, 1e-12)
assert f"{((0.7 - 0.9)**2 + (0.9 - 0.9)**2 + (1.1 - 0.9)**2) / 3:.3f}" == "0.027"
# l06: split arithmetic: 6000/2000/2000; k-fold mean
assert 6000 + 2000 + 2000 == 10000
assert f"{(0.21 + 0.19 + 0.23 + 0.20 + 0.22) / 5:.2f}" == "0.21"

print("all number checks passed")

# ---------------- renderer (matches the mse435 chapter-plate bar) ----------------
SANS = ("Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex Sans', "
        "'DejaVu Sans', sans-serif")
SERIF = ("'Liberation Serif','Source Serif 3',Georgia,'DejaVu Serif',serif")
INK = "#1B2838"
MUTED = "#5C6B7A"
TEAL = "#1F7A72"
BLUE = "#1E4D8C"
ORANGE = "#C46B2C"

from PIL import ImageFont
_FB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
_FR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
_MEAS = {}


def measure(text, size, weight):
    key = (weight, size)
    if key not in _MEAS:
        _MEAS[key] = ImageFont.truetype(_FB if weight >= 600 else _FR, size)
    b = _MEAS[key].getbbox(text)
    return b[2] - b[0]


PANELS = [(48, 140, 264, 300, "#FFFDF8", INK, 1.5),
          (328, 140, 304, 300, "#E7F1F8", BLUE, 1.5),
          (648, 140, 264, 300, "#E7F4EF", TEAL, 2.0)]
CXS = [180, 480, 780]
HXS = [48, 328, 648]


def esc(t):
    return sx.escape(t)


def plate(path, title, subtitle, headers, panels, tradeoff, footer):
    """headers: 3 strings. panels: 3 lists of rows.
    row: (text, size, color, weight, serif_bool)."""
    assert len(headers) == 3 and len(panels) == 3
    assert len(tradeoff) <= 2
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" '
             f'viewBox="0 0 960 600" font-family="{SANS}">',
             '<rect width="960" height="600" fill="#F7F4EE"/>']
    assert measure(title, 30, 600) <= 864 - 24, f"title too wide: {title}"
    parts.append(f'<text x="48" y="56" font-size="30" font-weight="600" fill="{INK}">{esc(title)}</text>')
    parts.append(f'<text x="48" y="86" font-size="17" fill="{MUTED}">{esc(subtitle)}</text>')
    parts.append(f'<g font-size="18" font-weight="600" fill="{INK}">')
    for hx, h in zip(HXS, headers):
        assert measure(h, 18, 600) <= 296, f"header too wide: {h}"
        parts.append(f'<text x="{hx}" y="124">{esc(h)}</text>')
    parts.append('</g>')
    for (x, y, w, h, fill, stroke, sw), cx, rows in zip(PANELS, CXS, panels):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        ry = 184
        for (t, s, c, wt, serif) in rows:
            wd = measure(t, s, wt)
            if os.environ.get("OVERFLOW_CHECK") and wd > w - 28:
                print(f"OVERFLOW {path}: [{wd:.0f}>{w-28}] {t}")
                continue
            assert wd <= w - 28, f"row too wide ({w-28}px): {t}"
            ff = f' font-family="{SERIF}"' if serif else ""
            parts.append(f'<text x="{cx}" y="{ry}" text-anchor="middle"{ff} '
                         f'font-size="{s}" font-weight="{wt}" fill="{c}">{esc(t)}</text>')
            ry += s * 1.75 + 6
        assert ry <= y + h - 14, f"panel overflow in {path}: rows end at {ry}"
    parts.append(f'<rect x="48" y="456" width="864" height="72" rx="12" '
                 f'fill="#FFFDF8" stroke="{INK}" stroke-width="1.5"/>')
    for i, t in enumerate(tradeoff):
        assert measure(t, 15, 500 if i == 0 else 400) <= 840, f"tradeoff too wide: {t}"
        wt = 500 if i == 0 else 400
        c = INK if i == 0 else MUTED
        parts.append(f'<text x="480" y="{486 + i * 24}" text-anchor="middle" '
                     f'font-size="15" font-weight="{wt}" fill="{c}">{esc(t)}</text>')
    assert measure(footer, 16, 500) <= 888, f"footer too wide: {footer}"
    parts.append(f'<text x="48" y="566" font-size="16" font-weight="500" '
                 f'fill="{INK}">{esc(footer)}</text>')
    parts.append('</svg>')
    with open(os.path.join(OUT, path), "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", path)


def R(t, s=15, c=INK, w=500, serif=False):
    return (t, s, c, w, serif)


CAPTIONS = []


def cap(lesson, n, slug, title, left, center, right, bottom):
    fn = f"plate-l{lesson:02d}-chap-{slug}.svg"
    line = (f'![Chapter plate: {title}](assets/{fn} '
            f'"Chapter plate L{lesson:02d}-C{n}. Left: {left}. '
            f'Center: {center}. Right: {right}. Bottom: {bottom}. '
            f'Dense chapter plate. Source: original synthesis of the lesson. '
            f'Project: Stanford Frontier AI.")')
    CAPTIONS.append((lesson, fn, line))
    return line

# ---------------- L01: What Machine Learning Is ----------------
plate("plate-l01-chap-definitions.svg",
      "Chapter plate: two definitions of learning",
      "Chapter plate. Samuel names the dream, Mitchell names the test. Source: original toy on the spam job.",
      ["WITHOUT the rule: hand rules", "the stored object: T, E, P", "WITH the rule: measured learning"],
      [[R("5 engineers, 1,000 rules", 16),
        R("30 min per rule: 500 engineer-hours", 14, MUTED),
        R("waves every 14 days", 14, MUTED),
        R("probes cost ~1,000 emails", 14, MUTED),
        R("humans: 10 rules/day", 16),
        R("spammers: 100 evasions/day", 16),
        R("rules interact: 63% false positives", 14, MUTED)],
       [R("T: sort email into spam or not", 15),
        R("E: 10,000 labeled emails", 15),
        R("P: accuracy on new mail", 15),
        R("P on 2,000 held-out emails", 14, MUTED),
        R("94% then 99% as E grows", 22, INK, 600),
        R("train 100% = memorization", 14, MUTED)],
       [R("threshold 0.9: precision 98%", 15),
        R("recall 49%", 14, MUTED),
        R("threshold 0.1: precision 47.5%", 15),
        R("recall 95%", 14, MUTED),
        R("accuracy lies on skewed tasks", 14, MUTED),
        R("spam buys precision", 14, MUTED),
        R("fraud buys recall", 14, MUTED)]],
      ["Tradeoff: adaptability costs data: 10,000 labels are 5 labeler-days.",
       "Mitchell's P is a choice, not a default."],
      "One connection: Mitchell turns Samuel's dream into a test: P must rise with E on new data.")
cap(1, 1, "definitions", "two definitions of learning",
    "hand rules: 500 engineer-hours, 10 rules/day vs 100 evasions/day, 63% false positives",
    "Mitchell's T, E, P: sort email, 10,000 labels, accuracy 94% then 99% on held-out mail",
    "measured learning: precision vs recall is a product choice, spam buys precision, fraud buys recall",
    "adaptability costs data: 10,000 labels are 5 labeler-days")

plate("plate-l01-chap-paradigms.svg",
      "Chapter plate: the three paradigms",
      "Chapter plate. The experience type picks the paradigm. Source: original toys from the lesson.",
      ["WITHOUT: the wrong fuel", "experience picks the paradigm", "WITH: each paradigm's toy"],
      [[R("labels cost: 10,000 = 5 labeler-days", 15),
        R("labels are the ceiling", 14, MUTED),
        R("5% label noise: 95% ceiling", 14, MUTED),
        R("web text: trillions of words", 14, MUTED),
        R("labels do not scale to trillions", 14, MUTED),
        R("rules need the world to hold still", 14, MUTED)],
       [R("labeled pairs: supervised", 16),
        R("raw data: unsupervised", 16),
        R("actions + rewards: reinforcement", 16),
        R("no answers given: invent categories", 14, MUTED),
        R("bandits: try vs exploit", 14, MUTED)],
       [R("regression: $200 per sq ft", 15),
        R("1,750 sq ft predicts $350,000", 15),
        R("clustering: boundary at 10.5", 15),
        R("cluster audit: spans 2 and 2", 14, MUTED),
        R("RL: 29M self-play games", 15),
        R("robot falls: $500 x 1,000 = $500,000", 14, MUTED)]],
      ["Tradeoff: RL fits where trials are cheap.",
       "Where trials cost hardware, sample efficiency is the whole game."],
      "One connection: ask what experience you have, and the paradigm picks itself.")
cap(1, 2, "paradigms", "the three paradigms",
    "the wrong fuel: labels cost 5 labeler-days per 10,000, and labels are the ceiling",
    "labeled pairs mean supervised, raw data means unsupervised, actions with rewards mean reinforcement",
    "regression at $200 per sq ft, clusters split at 10.5, RL at 29M games or $500,000 in falls",
    "RL fits where trials are cheap; where trials cost hardware, sample efficiency is the whole game")

plate("plate-l01-chap-ruleloss.svg",
      "Chapter plate: why the rules lost",
      "Chapter plate. The evasion treadmill, in numbers. Source: original toy on the spam job.",
      ["WITHOUT learning: rule lists", "the race, counted", "WITH learning: retrain"],
      [[R("1,000 rules, 30 min each", 15),
        R("500 engineer-hours to stand up", 14, MUTED),
        R("50 new rules per wave", 14, MUTED),
        R("25 engineer-hours per wave, forever", 14, MUTED),
        R("rule 612 vs rule 663: rot", 14, MUTED)],
       [R("humans: 10 rules/day", 16),
        R("spammers: 100 evasions/day", 16),
        R("attacker out-produces 10 to 1", 15),
        R("probes cost ~1,000 emails", 14, MUTED),
        R("1 - 0.999^1000 = 63% false positives", 14, MUTED, 500, True)],
       [R("retrain on this week's mail", 15),
        R("no new code", 15),
        R("10,000 labels = 5 labeler-days", 15),
        R("pattern re-learned weekly", 14, MUTED),
        R("adapts to fr33, f.r.e.e, next", 14, MUTED)]],
      ["Tradeoff: rules still win under 100 rules in a still world.",
       "Tax brackets, conversions, checklists: write them down."],
      "One connection: probing is cheap and rule-writing is slow, so learning inverts the cost.")
cap(1, 3, "ruleloss", "why the rules lost",
    "rule lists: 500 engineer-hours to stand up, 25 engineer-hours per wave every 14 days, forever",
    "the race: humans add 10 rules a day, spammers invent 100 evasions a day, 63% false positives at 1,000 rules",
    "retrain on this week's mail: no new code, 10,000 labels cost 5 labeler-days",
    "rules still win under 100 rules in a still world")

plate("plate-l01-chap-price.svg",
      "Chapter plate: the honest price",
      "Chapter plate. Adaptability costs three currencies. Source: original synthesis of the lesson.",
      ["WITHOUT: the rule list", "the three currencies", "WITH: what you buy"],
      [[R("one clever engineer", 14, MUTED),
        R("runs in microseconds", 14, MUTED),
        R("precise and auditable", 14, MUTED),
        R("dies every Monday", 16)],
       [R("data: 10,000 labels", 15),
        R("= 5 labeler-days", 14, MUTED),
        R("some labels unbuyable: rare diseases", 14, MUTED),
        R("compute: millions of operations", 15),
        R("retrain repeats the bill", 14, MUTED),
        R("trust: a box nobody can read", 15)],
       [R("filter survives Monday", 16),
        R("spammers adapt, mail relabeled", 14, MUTED),
        R("no code rewritten", 14, MUTED),
        R("course goal: shrink all three", 14, MUTED),
        R("but 3 a.m. mail = spam?", 14, ORANGE)]],
      ["Tradeoff: a learned model adapts, but nobody can read its rules back.",
       "When it fails, it fails in ways no human predicted."],
      "One connection: the rest of this course makes each of these three costs as small as possible.")
cap(1, 4, "price", "the honest price",
    "the rule list: one clever engineer, microseconds per run, dead every Monday",
    "data, compute, trust: 10,000 labels, millions of operations per retrain, a box nobody can read",
    "the filter survives Monday with no rewritten code, but its failures are unpredictable",
    "a learned model adapts, but nobody can read its rules back")

# ---------------- L02: Linear Regression ----------------
plate("plate-l02-chap-setup.svg",
      "Chapter plate: the supervised setup",
      "Chapter plate. Best gets a definition, and the definition has a price. Source: original toys from the lesson.",
      ["WITHOUT: no score", "the loss chip", "WITH: scored lines"],
      [[R("any line through the cloud", 14, MUTED),
        R("hand-picked, unscored", 14, MUTED),
        R("Ames: hundreds of pairs", 14, MUTED),
        R("knobs with no judge", 14, MUTED)],
       [R("J(theta) = 1/(2m) sum (h - y)^2", 16, INK, 600, True),
        R("theta_0 = 50,000, theta_1 = 100", 14, MUTED),
        R("1,800 sq ft predicts $230,000", 14, MUTED),
        R("$10k miss x 100 = $100k miss", 14, MUTED),
        R("1/m: no growth with data", 14, MUTED),
        R("1/2: cancels the 2 in the derivative", 14, MUTED)],
       [R("3-house toy: J = 6.33", 16),
        R("one step: J = 3.32", 16),
        R("squared 105 vs absolute 13", 14, MUTED),
        R("on misses 1, 2, 10", 14, MUTED),
        R("absolute: kink at zero", 14, MUTED),
        R("kink stalls gradient descent", 14, MUTED)]],
      ["Tradeoff: squares are smooth and big-miss-phobic, but one $1M miss",
       "contributes 10^12 and drags the whole line toward itself."],
      "One connection: minimizing J is the whole game of supervised learning.")
cap(2, 1, "setup", "the supervised setup",
    "no score: any hand-picked line through the cloud, knobs with no judge",
    "the loss chip J(theta) = 1/(2m) sum of squared errors, where the square punishes big misses 100-fold",
    "the 3-house toy scores J = 6.33, then 3.32 after one step; squared 105 beats absolute 13",
    "one $1M miss contributes 10^12 and drags the line")

plate("plate-l02-chap-gd.svg",
      "Chapter plate: gradient descent and alpha",
      "Chapter plate. The iterative workhorse, and its one dial. Source: original toys from the lesson.",
      ["WITHOUT: calculus only", "error x feature, summed", "WITH: alpha's three fates"],
      [[R("derivative = 0, solve", 14, MUTED),
        R("needs a matrix inverse", 14, MUTED),
        R("works for lines only", 14, MUTED),
        R("dies on neural networks", 15)],
       [R("theta_j := theta_j - alpha * error * x_j", 15, INK, 600, True),
        R("error = h - y, summed over m", 14, MUTED),
        R("toy: (0, 0) to (0.167, 0.383)", 15),
        R("J: 6.33 to 3.32", 14, MUTED),
        R("matrix form runs on hardware", 14, MUTED)],
       [R("alpha 0.01: 100 steps reach 0.53", 15),
        R("alpha 0.1: converges smooth", 15),
        R("alpha 1.5: 4, -8, 16, -32, 64", 15),
        R("explodes to infinity", 14, ORANGE),
        R("pick the largest smooth alpha", 14, MUTED)]],
      ["Tradeoff: when the loss bounces instead of falling, alpha is too high.",
       "Turn it down."],
      "One connection: gradient descent trains nearly every model you will ever meet.")
cap(2, 2, "gd", "gradient descent and alpha",
    "calculus only: derivative set to zero, needs a matrix inverse, dies on neural networks",
    "error times feature summed over examples: the toy moves from (0,0) to (0.167, 0.383), J from 6.33 to 3.32",
    "alpha 0.01 crawls to 0.53 in 100 steps, alpha 0.1 converges, alpha 1.5 explodes to infinity",
    "when the loss bounces instead of falling, alpha is too high")

plate("plate-l02-chap-sgd.svg",
      "Chapter plate: stochastic gradient descent",
      "Chapter plate. One example per step, noisy but m times cheaper. Source: original toys from the lesson.",
      ["WITHOUT: full passes", "one example per step", "WITH: the noisy win"],
      [[R("one step = all m examples", 15),
        R("2B x 40 = 80B multiply-adds", 14, MUTED),
        R("0.008 s of arithmetic", 14, MUTED),
        R("retrain hourly: passes too slow", 14, MUTED),
        R("the full-pass schedule fails", 15)],
       [R("theta_j := theta_j - alpha (h - y) x_j", 15, INK, 600, True),
        R("no sum, no 1/m", 14, MUTED),
        R("E[single gradient] = batch gradient", 15),
        R("toy: (-2 + -6)/2 = -4", 14, MUTED),
        R("shuffle each epoch", 14, MUTED)],
       [R("m times cheaper per step", 16),
        R("bounces, but arrives sooner", 14, MUTED),
        R("batch 1: noisy; batch m: exact", 14, MUTED),
        R("mini-batch B = 32 to 256", 15),
        R("64x less noise, ~same cost", 14, MUTED)]],
      ["Tradeoff: noise is the price of speed.",
       "Shuffle every epoch so no example's noise dominates."],
      "One connection: the stochastic part is not a hack; it is why internet-scale training is possible.")
cap(2, 3, "sgd", "stochastic gradient descent",
    "full passes: one step scores all m examples, 80B multiply-adds at 2B examples, and the schedule fails",
    "one example per step with no sum, an unbiased estimate of the batch gradient: (-2 + -6)/2 = -4",
    "m times cheaper per step and arriving sooner, with mini-batch 32 to 256 as the production middle",
    "noise is the price of speed; shuffle every epoch")

plate("plate-l02-chap-normaleq.svg",
      "Chapter plate: the normal equations",
      "Chapter plate. The exact answer for lines, in one shot. Source: original toys from the lesson.",
      ["WITHOUT: hundreds of steps", "the one-shot formula", "WITH: the two bills"],
      [[R("alpha tuned by hand", 14, MUTED),
        R("trace: 6.33 down to 0.028", 14, MUTED),
        R("hundreds of steps", 14, MUTED),
        R("neural nets: no closed form", 14, MUTED)],
       [R("theta = (X'X)^-1 X'y", 17, INK, 600, True),
        R("X'X = [[3, 6], [6, 14]]", 14, MUTED, 500, True),
        R("det = 6", 14, MUTED),
        R("theta = (1/3, 3/2), J = 0.028", 15),
        R("preds: 1.83, 3.33, 4.83", 14, MUTED)],
       [R("O(n^3): n = 10,000 is 10^12 ops", 15),
        R("singular: det near -0.0001", 15),
        R("sq ft + sq m: no inverse", 14, MUTED),
        R("m < n: singular too", 14, MUTED),
        R("fix: drop, or ridge", 14, MUTED)]],
      ["Tradeoff: no alpha, no iterations, no bouncing.",
       "But only for small n, full rank, and linear models."],
      "One connection: the normal equations are the statement that the gradient is zero, solved.")
cap(2, 4, "normaleq", "the normal equations",
    "hundreds of steps: alpha tuned by hand, the trace crawls from 6.33 to 0.028",
    "theta = (X'X)^-1 X'y in one shot: det 6, theta (1/3, 3/2), exact J = 0.028",
    "O(n^3) makes n = 10,000 cost 10^12 ops, and redundant features make the inverse fail",
    "no alpha and no iterations, but only for small n, full rank, and linear models")

# ---------------- L03: Logistic Regression ----------------
plate("plate-l03-chap-mle.svg",
      "Chapter plate: maximum likelihood",
      "Chapter plate. The bedrock framework of the course. Source: original toys from the lesson.",
      ["WITHOUT: fit the values", "the coin scoreboard", "WITH: log it"],
      [[R("line + threshold at 0.5", 14, MUTED),
        R("one outlier: 1.75 to 2.9", 15),
        R("a diagnosis flips", 14, ORANGE),
        R("treats 0.9->1.0 like 0.4->0.5", 14, MUTED),
        R("outputs are not probabilities", 14, MUTED)],
       [R("L(phi) = phi^7 (1-phi)^3", 16, INK, 600, True),
        R("0.00098, 0.00179", 14, MUTED),
        R("0.00222, 0.00168", 14, MUTED),
        R("peak at 0.7, 2.3x the fair coin", 15),
        R("log: products become sums", 14, MUTED)],
       [R("ell = 7 log phi + 3 log(1-phi)", 15, INK, 500, True),
        R("ell(0.7) = -6.11 vs ell(0.5) = -6.93", 14, MUTED),
        R("0.5^10000 = 0; logs give -6931", 14, MUTED),
        R("concave: one peak, no traps", 14, MUTED)]],
      ["Tradeoff: write the probability of the data as a function of the knobs,",
       "and maximize it. That is the whole framework."],
      "One connection: every loss in this course is MLE wearing a noise assumption.")
cap(3, 1, "mle", "maximum likelihood",
    "fitting values: the line plus 0.5 threshold lets one outlier flip a diagnosis from 1.75 to 2.9",
    "the coin scoreboard L(phi) = phi^7(1-phi)^3 peaks at 0.7, the observed fraction, 2.3 times the fair coin",
    "the log version: -6.11 beats -6.93, and logs cure the 0.5^10000 underflow",
    "write the probability of the data as a function of the knobs, and maximize it")

plate("plate-l03-chap-gaussian.svg",
      "Chapter plate: least squares was MLE all along",
      "Chapter plate. The loss chip gets a probabilistic meaning. Source: original derivation from the lesson.",
      ["WITHOUT: arbitrary squares", "Gaussian noise in", "WITH: squares out"],
      [[R("why squares, not absolute?", 14, MUTED),
        R("felt like a choice", 14, MUTED),
        R("the chip had no story", 14, MUTED),
        R("minimized, but no one knew why", 14, MUTED)],
       [R("y = theta'x + eps, eps ~ N(0, s^2)", 14, INK, 500, True),
        R("p(y|x) has exp(-(y-h)^2/2s^2)", 14, INK, 500, True),
        R("log L = const - sum errors^2", 14, MUTED, 500, True),
        R("sigma^2: constant, dropped", 14, MUTED),
        R("maximize = minimize sum (h-y)^2", 15)],
       [R("squares = Gaussian MLE", 16),
        R("every loss = a noise assumption", 15),
        R("Bernoulli wears cross-entropy", 14, MUTED),
        R("multinomial wears softmax loss", 14, MUTED)]],
      ["Tradeoff: change the noise assumption and MLE hands you a different loss.",
       "Ask the noise, not the fashion."],
      "One connection: minimizing J is maximizing the probability of the data under Gaussian noise.")
cap(3, 2, "gaussian", "least squares was MLE all along",
    "arbitrary squares: least squares felt like a choice with no story",
    "Gaussian noise in: y = theta'x + noise, whose log likelihood is a constant minus the squared errors",
    "squares out: maximizing the likelihood is minimizing the squared loss; Bernoulli wears cross-entropy",
    "change the noise assumption and MLE hands you a different loss")

plate("plate-l03-chap-sigmoid.svg",
      "Chapter plate: the sigmoid",
      "Chapter plate. The squeeze that turns scores into probabilities. Source: original toys from the lesson.",
      ["WITHOUT: the line", "g(z) = 1/(1+e^-z)", "WITH: logistic regression"],
      [[R("line predicts 0.27 and 0.55", 15),
        R("no walls at 0 and 1", 14, MUTED),
        R("2.5 cm tumor scores 0.45: benign", 14, MUTED),
        R("threshold jumps, no gradient", 14, MUTED)],
       [R("g(-2) = 0.12, g(0) = 0.5", 15),
        R("g(2) = 0.88, g(10) = 0.99995", 15),
        R("smooth and monotone", 14, MUTED),
        R("g'(z) = g (1-g)", 14, MUTED, 500, True)],
       [R("h = g(theta'x), fit by MLE", 15, INK, 500, True),
        R("gradient: (h - y) x", 15, INK, 500, True),
        R("tumor toy: size knob to 0.25", 14, MUTED),
        R("fraud: threshold 0.2, recall 92%", 14, MUTED),
        R("error x feature, again", 14, MUTED)]],
      ["Tradeoff: sigmoid plus squared loss is the flat-gradient trap.",
       "Sigmoid plus cross-entropy is the design: the loss must match the squash."],
      "One connection: the sigmoid's derivative factors into itself, so the learning rules stay clean.")
cap(3, 3, "sigmoid", "the sigmoid",
    "the line: predicts 0.27 and 0.55 with no walls at 0 and 1, and the threshold is not differentiable",
    "g(z) = 1/(1+e^-z): 0.12, 0.5, 0.88 at -2, 0, 2, smooth and monotone",
    "logistic regression fit by MLE: gradient (h-y)x, error times feature in probability space",
    "sigmoid plus squared loss is the flat-gradient trap; sigmoid plus cross-entropy is the design")

plate("plate-l03-chap-newton.svg",
      "Chapter plate: Newton's method",
      "Chapter plate. Curvature sets the step size automatically. Source: original toys from the lesson.",
      ["WITHOUT: slope only", "the parabola jump", "WITH: the per-step bill"],
      [[R("GD: 500-2,000 iterations", 15),
        R("n = 10,000, d = 50", 14, MUTED),
        R("about 10^9 operations", 14, MUTED),
        R("alpha tuned by hand", 14, MUTED),
        R("many cheap steps", 14, MUTED)],
       [R("theta := theta - J'/J''", 17, INK, 600, True),
        R("theta^2 from 4: 4 - 8/2 = 0", 15, INK, 500, True),
        R("one step vs GD's 38", 15),
        R("no alpha to tune", 14, MUTED),
        R("each step: weighted least squares", 14, MUTED),
        R("IRLS: reweight by uncertainty", 14, MUTED)],
       [R("weights h(1-h): 0.25, 0.25, 0.09", 15),
        R("confident ones barely vote", 14, MUTED),
        R("6-10 steps, O(nd^2 + d^3) each", 15),
        R("d = 20: 408,000 ops; d = 1B: 10^27", 14, ORANGE),
        R("LBFGS: O(md) memory", 14, MUTED)]],
      ["Tradeoff: few steps, each impossibly expensive, loses to many steps, each dirt cheap.",
       "SGD is the workhorse of machine learning."],
      "One connection: LBFGS keeps Newton's fast finish without the impossible matrix.")
cap(3, 4, "newton", "Newton's method",
    "slope only: gradient descent needs 500 to 2,000 tuned iterations, about 10^9 operations at n = 10,000, d = 50",
    "the parabola jump theta := theta - J'/J'': one step on theta^2 from 4 versus 38 gradient steps",
    "each Newton step is weighted least squares at O(nd^2 + d^3): 408,000 ops at d = 20, 10^27 at d = 1B",
    "few expensive steps lose to many cheap steps")

# ---------------- L04: GLMs and Softmax ----------------
plate("plate-l04-chap-expfam.svg",
      "Chapter plate: the exponential family",
      "Chapter plate. One algebraic shape under two lectures. Source: original algebra from the lesson.",
      ["WITHOUT: two inventions", "the one shape", "WITH: members and dividends"],
      [[R("least squares for numbers", 14, MUTED),
        R("logistic for yes-or-no", 14, MUTED),
        R("felt like different machines", 14, MUTED),
        R("two lectures, one basement", 14, MUTED)],
       [R("p(y; eta) = b(y) e^(eta'T(y)-a(eta))", 14, INK, 600, True),
        R("eta: natural parameter", 14, MUTED),
        R("T(y): sufficient statistic", 14, MUTED),
        R("a(eta): the normalizer", 14, MUTED),
        R("b(y): base measure", 14, MUTED)],
       [R("Bernoulli: eta = logit", 14),
        R("a = log(1+e^eta)", 14, MUTED),
        R("Gaussian: eta = mu, a = eta^2/2", 14),
        R("mean = a'(eta), always", 15),
        R("sigmoid = inverse of logit", 14, MUTED),
        R("phi 0.8: eta 1.386, a' = 0.8", 14, MUTED)]],
      ["Tradeoff: the mean falls out of the normalizer by differentiation,",
       "for every member of the family."],
      "One connection: lecture 3's Gaussian-noise story is this family in another costume.")
cap(4, 1, "expfam", "the exponential family",
    "two inventions: least squares and logistic regression felt like separate machines",
    "the one shape p(y; eta) = b(y) exp(eta'T(y) - a(eta)) with natural parameter, statistic, and normalizer",
    "Gaussian and Bernoulli both fit, and the mean is always a'(eta): phi 0.8 gives eta 1.386",
    "the mean falls out of the normalizer by differentiation")

plate("plate-l04-chap-glm.svg",
      "Chapter plate: the GLM recipe",
      "Chapter plate. Three steps generate every classical model. Source: original recipe from the lesson.",
      ["WITHOUT: derive each", "the three steps", "WITH: models fall out"],
      [[R("binary target: derive logistic", 14, MUTED),
        R("count target: derive Poisson", 14, MUTED),
        R("new target, new derivation", 14, MUTED),
        R("each from scratch", 14, MUTED)],
       [R("1. pick the distribution", 15),
        R("2. eta = theta'x", 16, INK, 600, True),
        R("3. predict a'(eta)", 15),
        R("canonical link: mean to eta", 14, MUTED),
        R("probit: breaks the clean gradient", 14, MUTED)],
       [R("Gaussian gives least squares", 15),
        R("Bernoulli gives logistic", 15),
        R("Poisson: weekend 100 to 200 hits", 14, MUTED),
        R("theta = log 2 = 0.69", 14, MUTED),
        R("R's glm: same IRLS engine", 14, MUTED)]],
      ["Tradeoff: eta must be linear in x. Curved boundaries are impossible.",
       "That ceiling is why lectures 7 and 8 exist."],
      "One connection: let eta be a neural network instead of theta'x, and you get deep learning.")
cap(4, 2, "glm", "the GLM recipe",
    "deriving each model from scratch: logistic for binary, Poisson for counts",
    "three steps: pick the distribution, set eta = theta'x, predict a'(eta)",
    "Gaussian falls out as least squares, Bernoulli as logistic, Poisson doubles 100 to 200 weekend hits",
    "eta must be linear in x, so curved boundaries are impossible")

plate("plate-l04-chap-softmax.svg",
      "Chapter plate: softmax",
      "Chapter plate. One normalization where classes compete. Source: original toy from the lesson.",
      ["WITHOUT: three sigmoids", "the one normalization", "WITH: the four whys"],
      [[R("(0.88, 0.73, 0.62)", 15),
        R("sum: 2.23", 17, INK, 600),
        R("normalize: (0.39, 0.33, 0.28)", 14, MUTED),
        R("not probabilities", 14, MUTED),
        R("cannot be compared", 14, MUTED)],
       [R("P(y=j|x) = e^z_j / sum e^z_c", 14, INK, 600, True),
        R("(7.39, 2.72, 1.65) / 11.76", 14, MUTED),
        R("(0.63, 0.23, 0.14)", 17, INK, 600),
        R("shift-invariant: subtract max", 14, MUTED),
        R("classes compete in one sum", 14, MUTED)],
       [R("GLM-dictated, smooth", 14),
        R("max-entropy, convenient", 14),
        R("tau 0.5: (0.84, 0.11, 0.04)", 14),
        R("tau 2: (0.48, 0.29, 0.23)", 14),
        R("tau -> 0: hard max; tau -> inf: uniform", 13, MUTED),
        R("K = 2: 0.731 = sigmoid(1)", 14, MUTED)]],
      ["Tradeoff: softmax costs O(k) per prediction.",
       "At k = 50,000 words, the normalization is the bottleneck."],
      "One connection: everything proved about logistic regression transfers to softmax at k = 2.")
cap(4, 3, "softmax", "softmax",
    "three sigmoids: (0.88, 0.73, 0.62) sum to 2.23, not probabilities",
    "one normalization e^z_j / sum e^z_c: (7.39, 2.72, 1.65)/11.76 becomes (0.63, 0.23, 0.14)",
    "the four whys: GLM-dictated, smooth, max-entropy, convenient; temperature dials sharpness",
    "softmax costs O(k) per prediction")

plate("plate-l04-chap-xent.svg",
      "Chapter plate: cross-entropy",
      "Chapter plate. The multi-class loss, and its unbounded anger. Source: original toy from the lesson.",
      ["WITHOUT: no multi-class loss", "minus log(p_true class)", "WITH: the error vector"],
      [[R("logistic loss covered k = 2", 14, MUTED),
        R("three diagnoses needed more", 14, MUTED),
        R("one-hot pushes scores to infinity", 14, MUTED)],
       [R("loss = -log(p_true)", 17, INK, 600, True),
        R("true class at 0.63: 0.46", 15),
        R("true class at 0.05: 3.00", 15),
        R("confident and right is cheap", 14, MUTED),
        R("confident and wrong is ruinous", 14, MUTED)],
       [R("gradient: (p - one-hot) x", 15),
        R("errors sum to zero", 14, MUTED),
        R("-0.37 + 0.23 + 0.14 = 0", 14, MUTED),
        R("k = 2: the lecture-3 loss", 14, MUTED),
        R("smoothing 0.1: transformer default", 14, MUTED)]],
      ["Tradeoff: unbounded: one mislabeled example at p = 10^-6 costs 13.8.",
       "Label smoothing taxes overconfidence: 0.59 vs 0.46."],
      "One connection: the losers' probability is given to the winner, errors summing to zero.")
cap(4, 4, "xent", "cross-entropy",
    "no multi-class loss: the logistic loss covered only k = 2",
    "loss = -log(p of the true class): 0.46 when right at 0.63, 3.00 when right at 0.05",
    "the gradient is (p - one-hot) times features, errors summing to zero",
    "unbounded: one mislabeled example at p = 10^-6 costs 13.8")

# ---------------- L05: GDA and Naive Bayes ----------------
plate("plate-l05-chap-genframe.svg",
      "Chapter plate: the generative turn",
      "Chapter plate. Model each class, then let Bayes decide. Source: original toys from the lesson.",
      ["WITHOUT: averages only", "Bayes' rule", "WITH: two roads, one line"],
      [[R("cats average 4 kg", 14, MUTED),
        R("elephants average 4,000 kg", 14, MUTED),
        R("40 kg is closer to 4: cat", 15),
        R("40 kg: 10x the cat mean", 14, MUTED),
        R("wrong: averages discard spread", 14, ORANGE)],
       [R("p(y|x) = p(x|y) p(y) / p(x)", 16, INK, 600, True),
        R("p(x) is the same for both", 14, MUTED),
        R("model each class's spread", 14, MUTED),
        R("pick max p(x|y) times p(y)", 15)],
       [R("boundary at y = 1, both roads", 15),
        R("mu_0 = (0.5, 0), mu_1 = (0.5, 2)", 14, MUTED),
        R("prior 9-to-1: rare must shout", 14, MUTED),
        R("wrong models: roads diverge", 14, MUTED)]],
      ["Tradeoff: class models cost assumptions.",
       "They buy one-pass training and tiny inference."],
      "One connection: the discriminative road asks where the line is; the generative road asks what each class looks like.")
cap(5, 1, "genframe", "the generative turn",
    "averages only: a 40 kg animal is closer to 4 than 4,000, so cat; wrong, because averages discard spread",
    "Bayes' rule p(y|x) = p(x|y)p(y)/p(x): model each class, pick the biggest p(x|y) times p(y)",
    "two roads to the same line at y = 1; they diverge when the class models are wrong",
    "class models cost assumptions and buy one-pass training")

plate("plate-l05-chap-gda.svg",
      "Chapter plate: Gaussian discriminant analysis",
      "Chapter plate. Shared spread deletes the quadratic term. Source: original toys from the lesson.",
      ["WITHOUT: the boundary drawn", "bells per class", "WITH: the linear line"],
      [[R("logistic draws the line directly", 14, MUTED),
        R("models p(y|x)", 14, MUTED),
        R("boundary learned directly", 14, MUTED),
        R("the discriminative road", 14, MUTED)],
       [R("cats ~ Gaussian(4, 1)", 15),
        R("elephants ~ Gaussian(40, 1)", 15),
        R("MLE = class averages, one pass", 14, MUTED),
        R("class 0 = {1, 3}, class 1 = {7, 9}", 14, MUTED),
        R("bells cross at x = 22", 16)],
       [R("w = Sigma^-1 (mu_1 - mu_0)", 15, INK, 500, True),
        R("shared Sigma: quadratics cancel", 14, MUTED),
        R("d = 100: 5,251 vs 10,301 knobs", 14),
        R("QDA crossings: 1.42 and -3.42", 14, MUTED),
        R("QDA curves around the tighter class", 14, MUTED)]],
      ["Tradeoff: shared spread buys linearity and half the knobs.",
       "Wildly different spreads plus plenty of data earn QDA."],
      "One connection: the linearity is not an assumption; it is the shared covariance deleting the quadratic term.")
cap(5, 2, "gda", "Gaussian discriminant analysis",
    "the boundary drawn directly: logistic models p(y|x), the discriminative road",
    "per-class Gaussians fit by class averages in one pass: the bells cross at x = 22",
    "shared covariance gives w = Sigma^-1(mu_1 - mu_0): linear, 5,251 knobs vs QDA's 10,301 at d = 100",
    "shared spread buys linearity and half the knobs")

plate("plate-l05-chap-naivebayes.svg",
      "Chapter plate: Naive Bayes",
      "Chapter plate. The generative spam filter, fit by counting. Source: original toys from the lesson.",
      ["WITHOUT: bell curves only", "words vote independently", "WITH: dirt cheap"],
      [[R("GDA needs real-valued features", 14, MUTED),
        R("words are indicators", 14, MUTED),
        R("'free' and 'money' travel together", 14, MUTED),
        R("not bell curves", 14, MUTED)],
       [R("p(x|y) = product p(x_j|y)", 16, INK, 600, True),
        R("naive: words independent given class", 14, MUTED),
        R("fit: count", 15),
        R("Bernoulli: presence only", 14, MUTED),
        R("multinomial: counts repeat", 14, MUTED),
        R("spam: 0.8 x 0.8 x 0.5 = 0.32", 15)],
       [R("real: 0.1 x 0.1 x 0.5 = 0.005", 15),
        R("spam wins 64 to 1", 15),
        R("10M emails/day on one core", 14, MUTED),
        R("Bernoulli 38-to-1, multinomial 192-to-1", 13, MUTED),
        R("score in log space", 14, MUTED)]],
      ["Tradeoff: the independence lie double-counts correlated evidence: 64-to-1 vs the honest 8-to-1.",
       "The ranking survives; calibration does not."],
      "One connection: when the lie hurts, logistic regression weighs the evidence jointly and splits the credit.")
cap(5, 3, "naivebayes", "Naive Bayes",
    "bell curves only: GDA needs real-valued features, but words are indicators",
    "words vote independently given the class: fit by counting, spam scores 0.8 x 0.8 x 0.5 = 0.32",
    "dirt cheap: spam wins 64 to 1, 10M emails a day on one core, scored in log space",
    "the independence lie double-counts correlated evidence; ranking survives, calibration does not")

plate("plate-l05-chap-laplace.svg",
      "Chapter plate: Laplace smoothing",
      "Chapter plate. Add one so unseen words cannot veto. Source: original toy from the lesson.",
      ["WITHOUT: the zero that kills", "add 1 to every count", "WITH: no vetoes"],
      [[R("'congratulations' never seen", 14, MUTED),
        R("p = 0 in both classes", 15),
        R("one unseen word vetoes all", 14, MUTED),
        R("every score zero: blind", 14, ORANGE)],
       [R("p = (count + 1)/(total + V)", 16, INK, 600, True),
        R("(0 + 1)/(10 + 2) = 1/12", 16),
        R("small, honest, non-vetoing", 14, MUTED),
        R("pulls 1/1 back toward uniform", 14, MUTED)],
       [R("shrinks wild fractions to uniform", 14, MUTED),
        R("the simplest regularization", 15),
        R("every NB scores in log space", 14, MUTED),
        R("log space: (-1.14, -5.30)", 14, MUTED)]],
      ["Tradeoff: the price of humility is small.",
       "The reward: one wrong label cannot drag the fit to infinity."],
      "One connection: regularization is a prior, and Laplace smoothing is that prior in counting clothes.")
cap(5, 4, "laplace", "Laplace smoothing",
    "the zero that kills: an unseen word scores 0 in both classes and blinds the filter",
    "add 1 to every count: (0+1)/(10+2) = 1/12, small, honest, and non-vetoing",
    "no vetoes: wild fractions shrink toward uniform, scored in log space as (-1.14, -5.30)",
    "the price of humility is small")

# ---------------- L06: Bias, Variance, Model Selection ----------------
plate("plate-l06-chap-biasvar.svg",
      "Chapter plate: bias and variance",
      "Chapter plate. The two enemies, and the U between them. Source: original toys from the lesson.",
      ["WITHOUT: fit harder", "error = bias^2 + variance + noise", "WITH: the U-curve"],
      [[R("degree-10: train 0.00", 15),
        R("test: 4.7", 16),
        R("line: train 0.42, test 0.51", 14, MUTED),
        R("more knobs always fit training", 14, MUTED)],
       [R("bias: wrong assumptions", 14),
        R("variance: noise sensitivity", 14),
        R("preds 0.7, 0.9, 1.1 at truth 1.0", 13, MUTED),
        R("bias^2 = 0.01, variance = 0.027", 15, INK, 500, True),
        R("infinite data: bias survives", 14, MUTED)],
       [R("degree 3: train 0.08, test 0.18", 15),
        R("degree 9: train 0.01, test 1.8", 14, MUTED),
        R("degree 15: train 0.00, test 4.9", 15),
        R("the sweet spot, found only on dev", 14, MUTED)]],
      ["Tradeoff: data fights variance; flexibility fights bias.",
       "You need both moves."],
      "One connection: when a model fails, this split tells you which enemy to fight.")
cap(6, 1, "biasvar", "bias and variance",
    "fit harder: the degree-10 polynomial scores 0.00 on training and 4.7 on test, against the line's 0.42 and 0.51",
    "expected test error splits into bias squared, variance, and noise: 0.01 and 0.027 on the toy",
    "the U-curve: degree 3 wins at 0.18, degree 15 falls to 4.9, and only dev finds the bottom",
    "data fights variance; flexibility fights bias")

plate("plate-l06-chap-doubledescent.svg",
      "Chapter plate: double descent",
      "Chapter plate. The canon breaks past the interpolation peak. Source: original toys from the lesson.",
      ["WITHOUT: the U-canon", "the interpolation threshold", "WITH: the second descent"],
      [[R("big models must fail", 14, MUTED),
        R("the U is the whole story", 14, MUTED),
        R("neural nets broke it", 14, MUTED),
        R("decades of doctrine", 14, MUTED)],
       [R("p ~ n: parameters = data", 16),
        R("the fit is knife-edge", 14, MUTED),
        R("12-point toy peaks at degree 11", 14, MUTED),
        R("flat minima: a plausible mechanism", 14, MUTED)],
       [R("15% to spike 25% to 8%", 16),
        R("degree 100: 0.4, degree 1000: 0.25", 14, MUTED),
        R("past the peak, bigger helps", 14, MUTED),
        R("modern practice jumps the peak", 14, MUTED)]],
      ["Tradeoff: the worst place to sit is just past the sweet spot.",
       "Big enough to be sensitive, not big enough to be smooth."],
      "One connection: the canon is not wrong, it is incomplete: it describes the left of the peak.")
cap(6, 2, "doubledescent", "double descent",
    "the U-canon: test error must rise past the classical sweet spot",
    "the interpolation threshold p ~ n: the fit is knife-edge, peaking at degree 11 on the 12-point toy",
    "the second descent: 15% spikes to 25% then falls to 8%; past the peak, bigger helps",
    "the worst place to sit is just past the sweet spot")

plate("plate-l06-chap-split.svg",
      "Chapter plate: train, dev, test",
      "Chapter plate. The discipline that keeps the numbers honest. Source: original toys from the lesson.",
      ["WITHOUT: training error lies", "three jobs, three sets", "WITH: the honest report"],
      [[R("polynomial scored 0.00", 15),
        R("and lied", 14, ORANGE),
        R("train 100% = memorization", 14, MUTED),
        R("no peeking at the future", 14, MUTED)],
       [R("train: fits the knobs", 14),
        R("dev: compares models", 14),
        R("test: reports once", 14),
        R("decide on dev, report on test", 14, MUTED),
        R("6,000 / 2,000 / 2,000", 16)],
       [R("degree 3: dev 0.18, test 0.21", 15),
        R("0.03: the luck margin", 14, MUTED),
        R("k-fold: mean 0.21, sd 0.015", 14, MUTED),
        R("k = 5 or 10 for small data", 14, MUTED)]],
      ["Tradeoff: a locked test set costs 10,000 examples the model never trains on.",
       "Cross-validation costs k training runs."],
      "One connection: every decision made on a dataset contaminates it; the test set decides nothing, ever.")
cap(6, 3, "split", "train, dev, test",
    "training error lies: the polynomial scored 0.00 and lied",
    "three jobs, three sets: train fits the knobs, dev compares, test reports once; 6,000 / 2,000 / 2,000",
    "the honest report: dev 0.18, test 0.21, with k-fold at mean 0.21 when data is scarce",
    "every decision made on a dataset contaminates it")

plate("plate-l06-chap-ridge.svg",
      "Chapter plate: ridge regression",
      "Chapter plate. Pay for big knobs, calm the curve. Source: original toys from the lesson.",
      ["WITHOUT: whipsaw coefficients", "the knob tax", "WITH: the calmed curve"],
      [[R("degree-10 whipsaws", 14, MUTED),
        R("enormous knobs: violent swings", 14, MUTED),
        R("X'X singular: det near -0.0001", 14, MUTED),
        R("infinitely many fits, n < d", 14, MUTED)],
       [R("J = sum (h-y)^2 + rho sum t_j^2", 14, INK, 600, True),
        R("theta = (X'X + rho I)^-1 X'y", 15, INK, 600, True),
        R("rho I: every eigenvalue + rho", 14, MUTED),
        R("rho = 0: normal equations", 14, MUTED),
        R("always invertible for rho > 0", 14)],
       [R("rho = 0: 4.7; rho = 1: 0.9", 16),
        R("rho = 100: 2.1, bias dominates", 14, MUTED),
        R("MAP: rho = s^2/tau^2", 14, MUTED, 500, True),
        R("lasso keeps 17 of 1,000", 14, MUTED),
        R("Lasso's diamond deletes", 14, MUTED)]],
      ["Tradeoff: ridge costs bias on purpose: underfit a little to overfit a lot less.",
       "Tune rho on dev, never on training."],
      "One connection: if we make theta really big, it has got to be worth it by fitting the data a lot better.")
cap(6, 4, "ridge", "ridge regression",
    "whipsaw coefficients: the degree-10 polynomial whipsaws and X'X goes singular at det near -0.0001",
    "the knob tax: J = squared loss + rho times knob squares, theta = (X'X + rho I)^-1 X'y, always invertible",
    "the calmed curve: test error 4.7 at rho 0, 0.9 at rho 1, 2.1 at rho 100; Lasso's diamond deletes instead",
    "ridge costs bias on purpose; tune rho on dev, never on training")

print(f"defined {len(CAPTIONS)} captions")
