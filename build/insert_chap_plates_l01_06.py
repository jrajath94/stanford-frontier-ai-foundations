#!/usr/bin/env python3
"""Insert the 24 chapter-plate caption lines into CS229 L01-L06 lessons.

Each plate lands at the end of its concept's section, before the next
## heading (or before the Q&A block for L01-C4).
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
import chap_plates_cs229_l01_06 as gen

LESSON_FILE = {
    1: "l01-introduction.md",
    2: "l02-linear-regression.md",
    3: "l03-logistic-regression.md",
    4: "l04-glms-softmax.md",
    5: "l05-gda-naive-bayes.md",
    6: "l06-bias-variance.md",
}
BASE = "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs229"

# (lesson, filename, anchor, mode)
JOBS = [
    (1, "plate-l01-chap-definitions.svg", "## The three paradigms", "before"),
    (1, "plate-l01-chap-paradigms.svg", "## Why the old way broke, in numbers", "before"),
    (1, "plate-l01-chap-ruleloss.svg", "## What is used where", "before"),
    (1, "plate-l01-chap-price.svg", "while keeping the adaptability.", "after"),
    (2, "plate-l02-chap-setup.svg", "## First attempt: solve it with calculus", "before"),
    (2, "plate-l02-chap-gd.svg", "## The key question", "before"),
    (2, "plate-l02-chap-sgd.svg", "## The normal equations: the exact answer for lines", "before"),
    (2, "plate-l02-chap-normaleq.svg", "## Mapping back: three ways to fit a line", "before"),
    (3, "plate-l03-chap-mle.svg", "## Least squares was MLE all along", "before"),
    (3, "plate-l03-chap-gaussian.svg", "## Logistic regression: the sigmoid", "before"),
    (3, "plate-l03-chap-sigmoid.svg", "## Newton's method: use the curvature", "before"),
    (3, "plate-l03-chap-newton.svg", "## Mapping back", "before"),
    (4, "plate-l04-chap-expfam.svg", "## The GLM recipe: three steps", "before"),
    (4, "plate-l04-chap-glm.svg", "## Softmax: the multi-class answer", "before"),
    (4, "plate-l04-chap-softmax.svg", "## Cross-entropy: the multi-class loss", "before"),
    (4, "plate-l04-chap-xent.svg", "## The honest price", "before"),
    (5, "plate-l05-chap-genframe.svg", "## Gaussian discriminant analysis", "before"),
    (5, "plate-l05-chap-gda.svg", "## The key question", "before"),
    (5, "plate-l05-chap-naivebayes.svg", "## Where it breaks: the zero that kills", "before"),
    (5, "plate-l05-chap-laplace.svg", "## The honest price", "before"),
    (6, "plate-l06-chap-biasvar.svg", "## Where the canon breaks: double descent", "before"),
    (6, "plate-l06-chap-doubledescent.svg", "## The key question", "before"),
    (6, "plate-l06-chap-split.svg", "## Ridge: pay for big knobs", "before"),
    (6, "plate-l06-chap-ridge.svg", "## Hyperband: stop wasting compute on losers", "before"),
]

caps = {(les, fn): line for les, fn, line in gen.CAPTIONS}
assert len(caps) == 24

for lesson, fn, anchor, mode in JOBS:
    path = f"{BASE}/{LESSON_FILE[lesson]}"
    with open(path) as f:
        text = f.read()
    line = caps[(lesson, fn)]
    assert line not in text, f"already inserted: {fn}"
    assert text.count(anchor) == 1, f"anchor not unique in {path}: {anchor}"
    if mode == "before":
        new = text.replace(anchor, line + "\n\n" + anchor, 1)
    else:  # after: insert right after the anchor line
        new = text.replace(anchor + "\n", anchor + "\n\n" + line + "\n", 1)
    assert new != text
    with open(path, "w") as f:
        f.write(new)
    print("inserted", fn, "->", LESSON_FILE[lesson])
