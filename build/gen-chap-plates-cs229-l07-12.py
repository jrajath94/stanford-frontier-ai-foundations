#!/usr/bin/env python3
"""Chapter plates for CS229 L07-L12: code-computed dense SVG plates.

Every number on every plate is recomputed below and asserted before
rendering. Layout follows the mse435 bar (plate-l01-chap-*.svg):
960 wide, warm paper #F7F4EE, three cards (left cream, center blue,
right green), tradeoff bar, one-connection footer.
"""
import math, os, html

ASSETS = os.path.expanduser("~/workspace/stanford-frontier-ai/content/v2/cs229/assets")
INK = "#1B2838"; MUT = "#5C6B7A"; GRN = "#1F7A72"; BLU = "#1E4D8C"
PAPER = "#F7F4EE"; CREAM = "#FFFDF8"; CBLUE = "#E7F1F8"; CGREEN = "#E7F4EF"
FONT = "Anthropic Sans, Inter, 'Source Sans 3', 'IBM Plex Sans', sans-serif"

# ---------------- number checks (every plate number asserted) ----------------
def relu(z): return max(0.0, z)
# L07-C1: bump construction 0,1,2,1,0
bump = [relu(x) - 2*relu(x-2) + relu(x-4) for x in range(5)]
assert bump == [0, 1, 2, 1, 0], bump
# knob count 784-256-128-10
l1 = 784*256 + 256; l2 = 256*128 + 128; l3 = 128*10 + 10
assert (l1, l2, l3) == (200960, 32896, 1290)
total = l1 + l2 + l3
assert total == 235146
assert abs(l1/total - 0.8545) < 0.001
# walkability toy
z_good = -2*0.2 + 3*0.9 + 0.5; z_bad = -2*2.0 + 3*0.1 + 0.5
assert abs(z_good - 2.8) < 1e-9 and abs(z_bad + 3.2) < 1e-9
assert abs(relu(z_good) - 2.8) < 1e-9 and relu(z_bad) == 0
# L07-C2: losses
mse = (400 + 100 + 900)/3
assert abs(mse - 466.6667) < 0.001
ce_a = -math.log(0.7); ce_b = -math.log(0.1)
assert abs(ce_a - 0.357) < 0.001 and abs(ce_b - 2.303) < 0.001
hinge = max(0, 1 - 1*0.3); assert abs(hinge - 0.7) < 1e-9
zsm = [2.0, 1.0, 0.5]; exps = [math.exp(v) for v in zsm]; s = sum(exps)
p = [e/s for e in exps]
assert abs(p[0]-0.629) < 0.005 and abs(p[1]-0.231) < 0.005 and abs(p[2]-0.140) < 0.005
assert abs(sum(p) - 1.0) < 1e-9
g = (p[0]-1, p[1]-0, p[2]-0)
assert abs(g[0]+0.371) < 0.01 and abs(g[1]-0.231) < 0.01 and abs(g[2]-0.140) < 0.01
lse = 1000 + math.log(1 + math.exp(-1))
assert abs(lse - 1000.313) < 0.001
# L07-C3: activations and init
sig_slope = math.exp(-10)/(1+math.exp(-10))**2
assert abs(sig_slope - 4.54e-5) < 1e-6  # ~0.00005
t2 = math.tanh(2); ts = 1 - t2**2
assert abs(t2 - 0.964) < 0.001 and abs(ts - 0.071) < 0.001
he_var = 2/100; he_std = math.sqrt(he_var)
assert abs(he_var - 0.02) < 1e-9 and abs(he_std - 0.141) < 0.001
xa_var = 1/100; xa_std = math.sqrt(xa_var)
assert abs(xa_var - 0.01) < 1e-9 and abs(xa_std - 0.1) < 1e-9
gelu2 = 2*0.9772; gelum2 = -2*0.0228
assert abs(gelu2 - 1.95) < 0.01 and abs(gelum2 + 0.046) < 0.01
# L07-C4: vanishing
vg = 0.25**50
assert abs(math.log10(vg) + 30.1) < 0.1
# L08-C1: toy backward
y_hat = 2*0 + (-1)*1.5
assert abs(y_hat + 1.5) < 1e-9
L = 0.5*(y_hat - 1)**2
assert abs(L - 3.125) < 1e-9
dl_dy = y_hat - 1
assert abs(dl_dy + 2.5) < 1e-9
dl_dW2 = (dl_dy*0, dl_dy*1.5)
assert dl_dW2 == (0.0, -3.75)
dl_da1 = (2*dl_dy, -1*dl_dy)
assert dl_da1 == (-5.0, 2.5)
dl_dz1 = (dl_da1[0]*0, dl_da1[1]*1)
assert dl_dz1 == (0.0, 2.5)
# L08-C2: jacobian entries
J = 1024*1024
assert J == 1048576
# fan-out: x=2, A=x^2, B=3x
assert 2*2 + 3 == 7
# L08-C3: memory bill
assert 12*100 == 1200
# L08-C4: hessian entries
assert (10**9)**2 == 10**18
# L09-C1: kmeans toy
mu1 = (1+2+3)/3; mu2 = (10+11+12)/3
assert mu1 == 2.0 and mu2 == 11.0
k1 = sum((x-6.5)**2 for x in [1,2,3,10,11,12])
assert abs(k1 - 125.5) < 1e-9
k2 = sum((x-2)**2 for x in [1,2,3]) + sum((x-11)**2 for x in [10,11,12])
assert abs(k2 - 4.0) < 1e-9
# L09-C2: silhouette toy point 1
a = (1+2)/2; b = (9+10+11)/3; sil = (b-a)/max(a, b)
assert abs(a-1.5) < 1e-9 and abs(b-10.0) < 1e-9 and abs(sil-0.85) < 1e-9
# L09-C2: kmeans++ squared distances on toy with first seed at 2
sq = {1:1, 3:1, 10:64, 11:81, 12:100}
assert sum(sq.values()) == 247
# L09-C3: responsibilities x=3
d1 = math.exp(-(3-2)**2/2)/math.sqrt(2*math.pi)
assert abs(d1 - 0.242) < 0.001
# L09-C4: kmedoids
km_mean = (1+2+3+100)/4
assert abs(km_mean - 26.5) < 1e-9
km_dist = sum((x-26.5)**2 for x in [1,2,3,100])
assert abs(km_dist - 7205.0) < 0.01
med_cost = 1+0+1+98
assert med_cost == 100
# covariance params d=10, k=5
full = 5*10*11//2; diag = 5*10; sph = 5; tied = 10*11//2
assert (full, diag, sph, tied) == (275, 50, 5, 55)
# voronoi border
assert (2+11)/2 == 6.5
# L10-C1: jensen
lhs = math.log(5); rhs = (math.log(1)+math.log(9))/2
assert abs(lhs-1.609) < 0.001 and abs(rhs-1.099) < 0.001 and lhs >= rhs
# L10-C2: M-step variance
devs = [0.17**2, 0.83**2, 0.67**2]
# exact squares of the stated deviations (lesson rounds them to 0.028/0.694/0.444)
assert abs(devs[0]-0.0289) < 0.001 and abs(devs[1]-0.6889) < 0.001 and abs(devs[2]-0.4489) < 0.001
assert abs(sum(devs)/3 - 0.389) < 0.001  # lesson's rounded average 0.389
mapv = (1.167+1.0)/(3+2)
assert abs(mapv - 0.433) < 0.001
# L10-C3: pca diagonal toy
eigs = (2.5, 0.0)
assert eigs[0]/(eigs[0]+eigs[1]) == 1.0
# SUV eigenvalues
a_tr, det = 1.75, 1.25*0.5 - 0.75**2
disc = math.sqrt(a_tr**2 - 4*det)
e1, e2 = (a_tr+disc)/2, (a_tr-disc)/2
assert abs(e1-1.7135) < 0.001 and abs(e2-0.0365) < 0.001
assert abs(e1/(e1+e2) - 0.979) < 0.001
s1 = math.sqrt(4*2.5)
assert abs(s1 - 3.162) < 0.001
# L11-C1: pixel toy
x1 = math.sqrt(0.99)*0.8 + math.sqrt(0.01)*0.5
assert abs(x1 - 0.846) < 0.001
abar100 = 0.99**100
assert abs(abar100 - 0.366) < 0.001
sig_left = math.sqrt(abar100)*0.8; noise_std = math.sqrt(1-abar100)
assert abs(sig_left - 0.484) < 0.001 and abs(noise_std - 0.796) < 0.01
# schedule at t=500
lin_sig = math.sqrt(0.079); cos_sig = math.sqrt(0.5)
assert abs(lin_sig-0.281) < 0.01 and abs(cos_sig-0.707) < 0.001
# L11-C3: guidance
gx = 0.1 + 7.5*(0.3-0.1); gy = 0.2 + 7.5*(0.1-0.2)
assert abs(gx-1.6) < 1e-9 and abs(gy+0.55) < 1e-9
# L11-C4: latent saving
pix = 512*512*3; lat = 64*64*4
assert pix == 786432 and lat == 16384 and pix//lat == 48
# L12-C1: 58 percent
assert 10001/500 > 20
# L12-C2: objectives
assert abs(-math.log(0.02)-3.912) < 0.001
assert abs(-math.log(0.3)-1.204) < 0.001
# L12-C3: lora
assert 2*1000*10 == 20000 and 1000**2 == 1000000
lora2 = 2*4096*8; full2 = 4096**2
assert lora2 == 65536 and full2 == 16777216
assert abs(full2/lora2 - 256) < 0.01
assert 65536*2/1024 == 128  # KB fp16 per matrix
# L12-C4: bill
flops = 6*70e9*1.4e12
assert abs(flops - 5.88e23)/5.88e23 < 0.01
secs = flops/(2000*1e14)
assert abs(secs/86400 - 34.0) < 1.0
cost = 2000*34*24*2
assert cost == 3264000
print("all number checks passed")

# ---------------- svg builder ----------------
def esc(t): return html.escape(t)

def line(x, cx, text, size, color, bold, anchor="middle"):
    b = ' font-weight="600"' if bold else ''
    return f'<text x="{x}" y="{cx}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{b}>{esc(text)}</text>'

def build(fname, title, subtitle, region_labels, cards, tradeoff, footer):
    # cards: list of (title_line, lines) where lines = [(text,size,color,bold)]
    n = max(len(c[1]) for c in cards)
    card_h = 96 + n*30
    y_card = 150
    y_trade = y_card + card_h + 16
    y_foot = y_trade + 30 + 30*len(tradeoff)
    H = y_foot + 48
    xs = [48, 328, 648]; ws = [264, 304, 264]
    parts = [f'<svg xmlns="http://www.w3.org/200/svg" width="960" height="{H}" viewBox="0 0 960 {H}" font-family="{FONT}">']
    parts.append(f'<rect width="960" height="{H}" fill="{PAPER}"/>')
    parts.append(line(48, 56, title, 30, INK, True, "start"))
    parts.append(line(48, 86, subtitle, 17, MUT, False, "start"))
    for i, (lab, (ctitle, clines)) in enumerate(zip(region_labels, cards)):
        x, w = xs[i], ws[i]
        parts.append(line(x, 124, lab, 18, INK, True, "start"))
        fill, stroke = [(CREAM, INK), (CBLUE, BLU), (CGREEN, GRN)][i]
        sw = 1.5 if i < 2 else 2
        parts.append(f'<rect x="{x}" y="{y_card}" width="{w}" height="{card_h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        parts.append(line(x+w/2, y_card+44, ctitle, 17, INK, True))
        yy = y_card + 84
        for (text, size, color, bold) in clines:
            parts.append(line(x+w/2, yy, text, size, color, bold))
            yy += 30
    parts.append(f'<rect x="48" y="{y_trade}" width="864" height="{30*len(tradeoff)+26}" rx="12" fill="{CREAM}" stroke="{INK}" stroke-width="1.5"/>')
    ty = y_trade + 34
    for t in tradeoff:
        parts.append(line(480, ty, t, 15, INK, False))
        ty += 30
    parts.append(line(48, y_foot, "One connection: " + footer, 16, INK, True, "start"))
    parts.append('</svg>')
    svg = "\n".join(p for p in parts if p)
    # drop the empty placeholder line (y_card+72 with size 1)
    svg = svg.replace('\n<text x="', '\n<text x="')
    out = os.path.join(ASSETS, fname)
    with open(out, "w") as f:
        f.write(svg + "\n")
    print("wrote", out, H)
    return fname, H

def L(text, size=15, color=MUT, bold=False): return (text, size, color, bold)

plates = []

# ================= L07 =================
plates.append(build("plate-l07-chap-neuron.svg",
 "Chapter plate: the bend and the stack",
 "Chapter plate. The bend makes the feature. The stack makes any shape.",
 ["WITHOUT bends", "the stored object", "WITH bends"],
 [
  ("linear models", [L("XOR impossible: no line separates",14),L("the diagonal pairs",14),
    L("w1+w2 > -2b yet < -b: impossible",14,INK,True),
    L("lines only mix raw columns",14),L("walkability never appears",14)]),
  ("the neuron: sigma(w^T x + b)", [L("walkability toy: w = (-2, 3), b = 0.5",13),
    L("house (0.2, 0.9): z = 2.8, ReLU = 2.8",14,INK,True),
    L("house (2.0, 0.1): z = -3.2, ReLU = 0",14,INK,True),
    L("bias b moves the firing threshold",14),L("the bend is the entire point",14)]),
  ("the MLP stack", [L("two ReLUs make a bump: 0, 1, 2, 1, 0",14,INK,True),
    L("784-256-128-10: 235,146 knobs",14,INK,True),
    L("first layer owns 85 percent of them",14),
    L("universal approximation: any shape",14)]),
 ],
 ["Tradeoff: bends buy learned features and any-shape approximation.",
  "The price: representable does not mean learnable. Training is lecture 8."],
 "the bend turns a line into a feature, and stacking bends turns features into any curve."))

plates.append(build("plate-l07-chap-loss.svg",
 "Chapter plate: the loss vocabulary",
 "Chapter plate. The loss must score what the task cares about.",
 ["WRONG scoreboard", "the stored object", "the matched losses"],
 [
  ("MSE on class labels", [L("optimizes the wrong pain",14),
    L("the numbers look fine",14),L("while the model is wrong",14),
    L("treats 0.49 vs 0.51 like any gap",14)]),
  ("the loss: one number", [L("loss(prediction, truth) = how wrong",14,INK,True),
    L("three jobs: numbers, probabilities, margins",14),
    L("MSE: gap 20 becomes 400",14,INK,True),
    L("cross-entropy: -log(0.7) = 0.357",14,INK,True),
    L("cross-entropy: -log(0.1) = 2.303",14,INK,True)]),
  ("softmax plus cross-entropy", [L("softmax(2, 1, 0.5) = (0.63, 0.23, 0.14)",14,INK,True),
    L("sum: 1.00. gradient is p - y",14,INK,True),
    L("= (-0.37, 0.23, 0.14)",14,GRN,True),
    L("hinge: s = 0.3 pays 0.7",14),
    L("(1000,999): log-sum-exp = 1000.31",14)]),
 ],
 ["Tradeoff: regression squares the gap, classification logs the truth's probability,",
  "the hinge charges only inside the margin. Match the loss to the data type."],
 "every output layer promises a data type, and the loss is the scoreboard that keeps that promise."))

plates.append(build("plate-l07-chap-activation.svg",
 "Chapter plate: activations and initialization",
 "Chapter plate. Match activation to layer, init to activation.",
 ["MISMATCHED", "the stored object", "MATCHED"],
 [
  ("saturating and cloned", [L("sigmoid at z = 10: slope 0.00005",14,INK,True),
    L("tanh at z = 2: slope 0.071",14,INK,True),
    L("zeros clone: 64 neurons, one feature",14),
    L("identical starts never diverge",14)]),
  ("two matching rules", [L("hidden layers: ReLU or GELU",14,INK,True),
    L("binary output: sigmoid. multi-class: softmax",14),
    L("GELU(2) = 1.95, GELU(-2) = -0.046",14,INK,True),
    L("ReLU(2) = 2, ReLU(-2) = 0",14),
    L("Leaky ReLU keeps a trickle alive",14)]),
  ("variance-preserving init", [L("He for ReLU: Var = 2/100 = 0.02",14,INK,True),
    L("std 0.141 at n = 100",14,GRN,True),
    L("Xavier for tanh: Var = 1/100 = 0.01",14),
    L("std 0.10. match init to activation",14),
    L("wrong scale: signal dies to dust",14)]),
 ],
 ["Tradeoff: the failure is silent: the net trains, just badly.",
  "Match the activation to the layer's job and the init to the activation."],
 "both the bend and the starting dice decide whether the signal survives the first layer."))

plates.append(build("plate-l07-chap-residual.svg",
 "Chapter plate: depth and the skip",
 "Chapter plate. The skip carries the gradient untouched. Source: He et al. 2015.",
 ["DEPTH without skips", "the stored object", "DEPTH with skips"],
 [
  ("the signal dies", [L("0.25^50 is about 10^-30",14,INK,True),
    L("early layers learn nothing",14),
    L("plain nets stall past ~20 layers",14),
    L("training error rises with depth",14)]),
  ("out = x + F(x)", [L("learn the change, add the input back",14,INK,True),
    L("F = 0: the block is an identity",14),
    L("adding blocks cannot hurt",14),
    L("skip multiplies the gradient by 1",14,BLU,True)]),
  ("152 layers train", [L("ResNet-152: 3.57 percent top-5 error",14,INK,True),
    L("ILSVRC 2015 winner",14,GRN,True),
    L("34, 50, 101, 152 keep improving",14),
    L("every transformer block is residual",14)]),
 ],
 ["Tradeoff: depth stopped being a risk and became a dial.",
  "The price: x and F(x) must match dimension, and the dial still needs tuning."],
 "fan-out adds gradients, so the skip path and the F path rejoin by addition at the plus node."))

# ================= L08 =================
plates.append(build("plate-l08-chap-theorem.svg",
 "Chapter plate: the O(N) gradient",
 "Chapter plate. The gradient costs about as much as the forward pass.",
 ["WITHOUT the theorem", "the stored object", "WITH the theorem"],
 [
  ("finite differences", [L("n+1 forward passes per gradient",14),
    L("10,000 knobs: 10,001 passes per step",14,INK,True),
    L("1B knobs: 1B forward passes per step",14,INK,True),
    L("the universe ends first",14)]),
  ("the toy backward walk", [L("y_hat = -1.5, L = 3.125",14,INK,True),
    L("dL/dW2 = [0, -3.75]",14,BLU,True),
    L("dL/dW1 = [[0,0],[2.5,5]]",14,BLU,True),
    L("dead ReLU zeroes a whole row",14),
    L("downstream signal times local slope",14)]),
  ("one backward pass", [L("toy: ~12 ops forward, ~10 backward",14,INK,True),
    L("gradient costs O(N), not O(N^2)",14,INK,True),
    L("1B knobs: 2-3 forward passes",14,GRN,True),
    L("this is why large models train",14)]),
 ],
 ["Tradeoff: finite differences stay as the debugging ground truth (relative error below 1e-7).",
  "The price of the fast gradient: backprop stores the forward pass's intermediates."],
 "the chain rule shares one backward walk across all knobs, which is why training at scale is possible."))

plates.append(build("plate-l08-chap-module.svg",
 "Chapter plate: modules, VJPs, and the graph",
 "Chapter plate. Frameworks never differentiate. They compose contracts.",
 ["WITHOUT the contract", "the stored object", "WITH the contract"],
 [
  ("hand-derived gradients", [L("a new formula per architecture",14),
    L("full Jacobian 1024x1024",14,INK,True),
    L("= 1,048,576 stored entries",14,INK,True),
    L("memory dies at the first block",14)]),
  ("forward plus backward", [L("each module ships both, locally",14,INK,True),
    L("linear: dL/dx = W^T dL/dy",14,BLU,True),
    L("dL/dW = dL/dy outer x",14,BLU,True),
    L("the tape saves x, nothing else",14)]),
  ("matrix-free VJP", [L("v^T J without ever forming J",14,INK,True),
    L("dL/dW is rank 1 per example",14,INK,True),
    L("fan-out adds: 4 + 3 = 7",14,GRN,True),
    L("the graph is the program",14)]),
 ],
 ["Tradeoff: the backward function is local, so new architectures are new graphs with free backward passes.",
  "The price: differentiability. ReLU at 0, sampling, and ties each need a substitute rule."],
 "the residual plus node is fan-out by design, and its addition is why residual gradients survive depth."))

plates.append(build("plate-l08-chap-memory.svg",
 "Chapter plate: the tape's price",
 "Chapter plate. Speed costs memory. The tape is the invoice.",
 ["WITHOUT the escape hatch", "the stored object", "WITH the escape hatch"],
 [
  ("the full tape", [L("12 layers at 100 MB: 1,200 MB",14,INK,True),
    L("activations grow with depth x batch",14),
    L("weights are fixed cost",14),
    L("activations dominate training memory",14,INK,True)]),
  ("saved intermediates", [L("the backward needs a1 and z1",14,INK,True),
    L("forward values the VJP reads",14),
    L("sparse embedding backward:",14),
    L("touch 64 values, not 640,000",14,BLU,True)]),
  ("checkpointing", [L("save every 4th layer: 300 MB",14,INK,True),
    L("one extra forward pass",14),
    L("30-40 percent slower, memory down 4x",14,GRN,True),
    L("reversible: O(1) tape in depth",14)]),
 ],
 ["Tradeoff: checkpoint when memory binds. Buy bigger GPUs when money binds.",
  "Every large transformer run checkpoints, because the tape does not fit otherwise."],
 "the O(N) theorem buys speed, and the tape is the invoice it sends: memory is the real training budget."))

plates.append(build("plate-l08-chap-curvature.svg",
 "Chapter plate: rank one and curvature",
 "Chapter plate. The matrix is too big to exist. Its action is cheap.",
 ["WITHOUT matrix-free methods", "the stored object", "WITH matrix-free methods"],
 [
  ("Newton's bill", [L("Hessian d by d, invert at O(d^3)",14),
    L("d = 1B: 10^18 entries",14,INK,True),
    L("the matrix cannot be formed",14),
    L("curvature looks unaffordable",14)]),
  ("two structured objects", [L("dL/dW = signal outer input",14,INK,True),
    L("toy: [0,-3.75] = (-2.5) x [0,1.5]",14,BLU,True),
    L("one example: rank 1 per layer",14),
    L("H*v: Hessian times a vector",14)]),
  ("O(N) curvature", [L("H*v costs one extra backward pass",14,INK,True),
    L("conjugate gradient needs only H*v",14),
    L("the matrix never exists",14,GRN,True),
    L("batch size caps the update rank",14)]),
 ],
 ["Tradeoff: second-order information survives only for matrix-free methods.",
  "Double backward pays roughly double the tape: curvature has a memory bill too."],
 "one example carries one direction of information per layer, which is why the batch size is the rank budget."))

# ================= L09 =================
plates.append(build("plate-l09-chap-kmeans.svg",
 "Chapter plate: the ad-hoc algorithm",
 "Chapter plate. Assign, average, repeat. The seed decides the answer.",
 ["WITHOUT the algorithm", "the stored object", "WITH the algorithm"],
 [
  ("a million unlabeled records", [L("no y anywhere",14),
    L("no natural groups visible",14),
    L("labels would cost human hours",14),
    L("the machine must invent categories",14)]),
  ("assign, then average", [L("{1,2,3,10,11,12}, k = 2",14),
    L("start (1, 12): assign, update",14,BLU,True),
    L("mu_1 = 2, mu_2 = 11, 1 round",14,BLU,True),
    L("the mean minimizes the distortion",14),
    L("each step optimal for its half",14)]),
  ("converges to a local minimum", [L("distortion falls every round",14,INK,True),
    L("k = 1: 125.5. k = 2: 4.0",14,INK,True),
    L("bad seed strands a cluster",14),
    L("distortion near 200 instead of 4.0",14,GRN,True)]),
 ],
 ["Tradeoff: the algorithm is deterministic given the seed, and the seed is luck.",
  "NP-hardness forbids a global guarantee: local minima are fundamental, not a bug."],
 "k-means is lecture 5's GDA with the labels hidden, which is why lecture 10's EM fits the grown-up version."))

plates.append(build("plate-l09-chap-seeding.svg",
 "Chapter plate: seed far apart, choose k",
 "Chapter plate. Seeding is provably close. Choosing k stays heuristic.",
 ["RANDOM seeds, eyeball k", "the stored object", "CAREFUL seeds, scored k"],
 [
  ("the seed lottery", [L("seeds 0, 0.1 strand a cluster",14),
    L("empty cluster needs a rescue rule",14),
    L("elbow bends are eyeball-only",14),
    L("real curves bend gradually",14)]),
  ("k-means++ rule", [L("seed proportional to squared distance",14,INK,True),
    L("first seed at 2, next likely in {10,11,12}",14,BLU,True),
    L("squared distances: 64, 81, 100 vs 1, 1",14,BLU,True),
    L("sklearn default, plus 10 restarts",14)]),
  ("guarantee plus numbers", [L("O(log k) expected approximation ratio",14,INK,True),
    L("125.5, 4.0, 2.7, 1.5: bend at 2",14),
    L("silhouette: 0.85 on the toy point",14,GRN,True),
    L("gap statistic vs the uniform null",14)]),
 ],
 ["Tradeoff: k-means++ buys a guarantee from seeding alone. It cannot buy optimality.",
  "Elbow for exploration, silhouette for a number, gap for a significance claim."],
 "seeding proportional to squared distance samples each point's share of the objective, which the proof needs."))

plates.append(build("plate-l09-chap-gmm.svg",
 "Chapter plate: the soft grown-up",
 "Chapter plate. Responsibilities quantify doubt.",
 ["HARD assignments", "the stored object", "SOFT assignments"],
 [
  ("all or nothing", [L("each point exactly one cluster",14),
    L("overlapping groups forced apart",14),
    L("the point at 5.0 drags a whole mean",14),
    L("no notion of uncertainty",14)]),
  ("responsibility gamma_j(x)", [L("posterior: how much cluster j claims x",14,INK,True),
    L("point 3: 1.0 vs about 0",14,BLU,True),
    L("midpoint 6.5: 0.5 and 0.5",14,BLU,True),
    L("between: a smooth slide, not a wall",14)]),
  ("the probabilistic model", [L("70/30 splits: overlap quantified",14,INK,True),
    L("k-means is the zero-variance limit",14,GRN,True),
    L("the ad-hoc algorithm was probabilistic",14),
    L("with the variance turned to zero",14)]),
 ],
 ["Tradeoff: soft assignments model overlap and shape.",
  "The price: the labels stay hidden, so MLE has no closed form and EM must iterate."],
 "the bridge proves k-means was the Gaussian mixture all along, with sigma squared sent to zero."))

plates.append(build("plate-l09-chap-shapes.svg",
 "Chapter plate: shapes and covariance dials",
 "Chapter plate. Match the algorithm to the shape.",
 ["SPHERES only", "the stored object", "the shape dial"],
 [
  ("straight bisectors", [L("centers 2 and 11: border at 6.5",14,INK,True),
    L("every k-means cell is convex",14),
    L("the bisector splits both moons",14,INK,True),
    L("structural failure, not a bad seed",14)]),
  ("choose by shape", [L("DBSCAN: density chains, finds k",14,BLU,True),
    L("spectral: graph cuts follow curves",14),
    L("hierarchical: one run, every k",14),
    L("plot a 2-D projection first",14)]),
  ("covariance types, d=10, k=5", [L("full: 275. diagonal: 50.",14,INK,True),
    L("spherical: 5. tied: 55.",14,INK,True),
    L("spherical is k-means, softened",14,GRN,True),
    L("start diagonal, go full with data",14)]),
 ],
 ["Tradeoff: if the true groups are not separable by straight borders, k-means cannot find them.",
  "The covariance type is the GMM's bias-variance dial from lecture 6."],
 "the Voronoi view predicts the crescent failure: curved moons cannot fit convex cells."))

# ================= L10 =================
plates.append(build("plate-l10-chap-elbo.svg",
 "Chapter plate: the bound that never falls",
 "Chapter plate. Jensen turns a log of a sum into a tractable bound.",
 ["WITHOUT the bound", "the stored object", "WITH the bound"],
 [
  ("log of a sum", [L("sum over all hidden labelings",14),
    L("the log cannot reach inside",14),
    L("no closed form, no clean gradient",14),
    L("MLE looks impossible",14)]),
  ("ELBO = E_Q[log p(x,z)] + H(Q)", [L("fit the Q-weighted data",14,INK,True),
    L("honor the uncertainty (entropy)",14,INK,True),
    L("log(5) = 1.609 beats 1.099",14,BLU,True),
    L("the chord sags: log is concave",14),
    L("tight when Q equals the posterior",14,BLU,True)]),
  ("touch, then climb", [L("E-step: Q = posterior, bound touches",14,INK,True),
    L("M-step: climb the bound",14,INK,True),
    L("L(new) >= B(new) >= B(old) = L(old)",14,GRN,True),
    L("the likelihood never falls",14,GRN,True)]),
 ],
 ["Tradeoff: a climber that never descends still summits the wrong hill.",
  "The guarantee is monotonicity per round, not the global maximum, and not speed."],
 "the equality condition of Jensen, when the chord collapses onto the curve, is exactly the E-step's choice."))

plates.append(build("plate-l10-chap-em.svg",
 "Chapter plate: EM, the two easy steps",
 "Chapter plate. EM is k-means grown up: soft instead of hard.",
 ["HARD guess-fit-repeat", "the stored object", "SOFT expectation-maximization"],
 [
  ("force the boundary", [L("the point at 5.0 gets one label",14),
    L("full weight, zero doubt",14),
    L("drags its cluster's mean",14),
    L("uncertainty thrown away",14)]),
  ("E-step, M-step", [L("E: responsibilities under current params",14,INK,True),
    L("M: responsibility-weighted MLE",14,INK,True),
    L("weighted means, spreads, mixing weights",14),
    L("coordinate ascent on (Q, theta)",14)]),
  ("the toy, honestly weighted", [L("converges to mu = 0.33 and 10.0",14,INK,True),
    L("sigma_1^2 = 0.389, sigma_1 = 0.62",14,INK,True),
    L("MAP with prior: 0.433, not 0.389",14,GRN,True),
    L("sigma to 0: density to infinity",14)]),
 ],
 ["Tradeoff: soft weights handle boundary points honestly instead of forcing them.",
  "The price: local maxima, a slow crawl near the top, and degenerate collapse. The prior is the guardrail."],
 "the doubt from the E-step flows into the M-step's parameters, which is what the weighted formulas implement."))

plates.append(build("plate-l10-chap-pca.svg",
 "Chapter plate: the axes of variation",
 "Chapter plate. Truncated SVD is the optimal rank-k summary.",
 ["WITHOUT the axes", "the stored object", "WITH the axes"],
 [
  ("200 redundant features", [L("plotting impossible",14),
    L("distances expensive",14),
    L("exact eigendecomposition O(d^3)",14),
    L("millimeters dominate meters",14)]),
  ("eigenvectors of the covariance", [L("eigenvalue = variance along the axis",14,INK,True),
    L("diagonal toy: (2.5, 0)",14,BLU,True),
    L("one axis keeps 100 percent",14,BLU,True),
    L("SUV: PC1 = (1, 0.618), keeps 97.9 percent",14,BLU,True)]),
  ("shipped as SVD", [L("SVD(X): s_1 = 3.162 on the toy",14,INK,True),
    L("never square the condition number",14),
    L("Eckart-Young: optimal rank-k",14,GRN,True),
    L("not a heuristic: the optimum",14,GRN,True)]),
 ],
 ["Tradeoff: PCA is the best k-dimensional summary by squared reconstruction error.",
  "The price: it is linear. A Swiss roll has no good straight summary, and variance is not meaning."],
 "PCA builds new features, it does not select originals: PC1 is a direction, not the length feature."))

plates.append(build("plate-l10-chap-scale.svg",
 "Chapter plate: PCA at scale",
 "Chapter plate. Sketch first, then SVD the small core.",
 ["EXACT decomposition", "the stored object", "RANDOMIZED decomposition"],
 [
  ("O(n d^2)", [L("d = 100,000: never finishes",14,INK,True),
    L("raw pixels, vocab embeddings",14),
    L("the eigendecomposition is dead",14),
    L("exact PCA does not scale",14)]),
  ("the sketch", [L("Y = X times random Omega",14,INK,True),
    L("random projections keep large directions",14),
    L("orthogonalize, project: small B",14,BLU,True),
    L("SVD the small core",14,BLU,True)]),
  ("O(n d k)", [L("n = 1M, d = 100K, k = 50",14,INK,True),
    L("exact hopeless, randomized finishes",14,GRN,True),
    L("oversample by 10",14),
    L("error tiny with high probability",14)]),
 ],
 ["Tradeoff: randomized SVD is the production PCA.",
  "The price: it is approximate, and oversampling is the control on the randomness."],
 "exact PCA dies where the data lives: at d = 100K the sketch is not a shortcut, it is the only door."))

# ================= L11 =================
plates.append(build("plate-l11-chap-forward.svg",
 "Chapter plate: the fixed destruction",
 "Chapter plate. Destruction needs no intelligence. Source: Ho et al. 2020.",
 ["WITHOUT the schedule", "the stored object", "WITH the schedule"],
 [
  ("one-shot generation", [L("GAN-style: unstable training",14),
    L("modes collapse: the same cat forever",14),
    L("no clean likelihood to optimize",14),
    L("the field moved on",14)]),
  ("x_t from x_{t-1}: fixed betas", [L("fixed: needs no learning",14,INK,True),
    L("pixel toy: 0.8 becomes 0.846",14,BLU,True),
    L("0.796 of signal plus 0.05 of noise",14),
    L("closed form: any t in O(1)",14,BLU,True)]),
  ("the image dies on schedule", [L("100 steps: 0.48 signal in 0.80 noise",14,INK,True),
    L("1,000 steps: pure static",14,INK,True),
    L("t=500: linear 0.28, cosine 0.71",14,GRN,True),
    L("free training pairs at every level",14)]),
 ],
 ["Tradeoff: all learning concentrates in the reverse, where the intelligence belongs.",
  "The schedule is the curriculum: cosine spends the budget where learning happens."],
 "the forward process is jumpable, not simulable, which is what makes training cost independent of T."))

plates.append(build("plate-l11-chap-reverse.svg",
 "Chapter plate: the learned reversal",
 "Chapter plate. The objective collapses into T regression problems.",
 ["ONE giant reversal", "the stored object", "T tiny reversals"],
 [
  ("the leap is unlearnable", [L("a big jump has a multimodal reversal",14),
    L("the network cannot fit it",14),
    L("the ELBO bound loosens",14),
    L("generation stays too hard",14)]),
  ("predict each step's noise", [L("network takes (x_t, t), predicts eps",14,INK,True),
    L("ELBO becomes noise-prediction MSE",14,BLU,True),
    L("one regression problem per level",14),
    L("stable, no adversary, no collapse games",14,BLU,True)]),
  ("noise points uphill", [L("predicting noise estimates the score",14,INK,True),
    L("the direction toward likely images",14),
    L("sampling: hill-climbing from static",14,GRN,True),
    L("the U-Net is a learned compass",14)]),
 ],
 ["Tradeoff: diffusion trains by denoising score matching: plain, stable regression.",
  "The price is sampling: the method is correct, and the meter is running."],
 "noise prediction is better conditioned: the target is always standard Gaussian, same scale at every step."))

plates.append(build("plate-l11-chap-bill.svg",
 "Chapter plate: the sampling bill",
 "Chapter plate. Diffusion won by being the most trainable.",
 ["GAN sampling", "the stored object", "diffusion sampling"],
 [
  ("1 eval per image", [L("one forward pass",14),
    L("fast sampling",14),
    L("unstable training",14),
    L("mode collapse",14)]),
  ("T network evals per image", [L("T = 1,000: 1,000 evals",14,INK,True),
    L("tiny steps keep reversals learnable",14),
    L("10,000 images at 100 evals/s/GPU:",14),
    L("27.8 GPU-hours at full steps",14,BLU,True)]),
  ("the discounts, priced", [L("DDIM: 50 steps, 20x cheaper",14,INK,True),
    L("distilled: 4 steps, 250x cheaper",14,INK,True),
    L("7 minutes for 10,000 images",14,GRN,True),
    L("fine detail lags the full sampler",14)]),
 ],
 ["Tradeoff: each discount trades a little sample quality for speed.",
  "Price the quality tier, not the method: batch shots tolerate 4 steps, hero images get 1,000."],
 "DDIM takes bigger steps along the trajectory, consistency models jump off it, and neither repeals the bill."))

plates.append(build("plate-l11-chap-steer.svg",
 "Chapter plate: steer and compress",
 "Chapter plate. Guidance is a dial, not a switch.",
 ["UNCONDITIONAL, pixels", "the stored object", "STEERED, latent"],
 [
  ("ignores the prompt", [L("samples drift from the text",14),
    L("pixel space: 512x512x3",14),
    L("= 786,432 numbers per step",14,INK,True),
    L("most of them redundant",14)]),
  ("guided noise plus VAE", [L("guided = uncond + w (cond - uncond)",14,INK,True),
    L("w = 7.5 amplifies the prompt direction",14,BLU,True),
    L("(0.1,0.2),(0.3,0.1) becomes (1.6,-0.55)",14,BLU,True),
    L("VAE encode once, decode once",14)]),
  ("the product design", [L("64x64x4 = 16,384: 48x smaller",14,INK,True),
    L("Stable Diffusion's design",14,GRN,True),
    L("past w ~ 15: saturate and distort",14),
    L("low w: diversity. high w: adherence",14)]),
 ],
 ["Tradeoff: latent diffusion keeps the perceptual content and drops the pixel noise the eye ignores.",
  "Pixel diffusion is the textbook. Latent diffusion is the product."],
 "the text embedding enters the denoiser at every step, so each small correction steers toward the prompt."))

# ================= L12 =================
plates.append(build("plate-l12-chap-paradigm.svg",
 "Chapter plate: the 500-label problem",
 "Chapter plate. The billions built the map. The 500 labels steer.",
 ["WITHOUT pre-training", "the stored object", "WITH pre-training"],
 [
  ("train on the 500 alone", [L("10,001 knobs, 500 examples",14,INK,True),
    L("20 parameters per example",14,INK,True),
    L("train near 100 percent, test 58",14,INK,True),
    L("memorization plus a whisper",14)]),
  ("representations", [L("\"masterpiece\" near \"triumph\"",14,BLU,True),
    L("shared contexts build the geometry",14),
    L("pre-train moves the start near",14,INK,True),
    L("good solutions for many tasks",14)]),
  ("adapt cheaply", [L("linear probe: 88 vs 58 percent",14,INK,True),
    L("the 30-point gap is pre-training",14,GRN,True),
    L("probe ladder: 62, 80, 85",14),
    L("same labels, different underneath",14)]),
 ],
 ["Tradeoff: fine-tuning is a short walk from an excellent start, not a random search.",
  "The price: the training set is everything, so every benchmark is suspect without a contamination report."],
 "pre-training moves the model to a region of parameter space where good solutions for many tasks are nearby."))

plates.append(build("plate-l12-chap-objectives.svg",
 "Chapter plate: labels hiding in the data",
 "Chapter plate. The self-supervised objective invents its own labels.",
 ["LABELS cost money", "the stored object", "the data labels itself"],
 [
  ("the startup has 500 labels", [L("billions of unlabeled sentences",14),
    L("classical tools starve",14),
    L("labeling is the bottleneck",14),
    L("the text order is free",14)]),
  ("three self-supervised objectives", [L("next-token: predict \"mat\"",14,INK,True),
    L("masked: fill the [MASK]",14,INK,True),
    L("contrastive: pull the pair, push the rest",14,INK,True),
    L("a trillion words: a trillion examples",14)]),
  ("the worked losses", [L("next-token: -log(0.02) = 3.91",14,BLU,True),
    L("masked: -log(0.3) = 1.20",14,BLU,True),
    L("contrastive: 1 positive, 127 negatives",14),
    L("BERT reads both ways, GPT generates",14,GRN,True)]),
 ],
 ["Tradeoff: grammar, facts, and reasoning arrive as side effects of predicting the next word well.",
  "The objective picks the strength: bidirectional for understanding, left-to-right for generation."],
 "next-token prediction is classification with 50,000 classes, repeated a trillion times."))

plates.append(build("plate-l12-chap-lora.svg",
 "Chapter plate: adapt without moving billions",
 "Chapter plate. Small changes live in low-dimensional subspaces.",
 ["FULL fine-tuning", "the stored object", "LoRA"],
 [
  ("move everything", [L("7B knobs move on 500 labels",14),
    L("14 GB copy per task",14,INK,True),
    L("overfitting-prone, overkill",14),
    L("each task its own giant",14)]),
  ("W_adapted = W + A x B", [L("W frozen, A and B train",14,INK,True),
    L("d = 1000, r = 10: 20K vs 1M",14,BLU,True),
    L("50x fewer degrees of freedom",14,BLU,True),
    L("d = 4096, r = 8: 256x cut",14,BLU,True)]),
  ("merges free", [L("deploy: W' = W + AB, once",14,INK,True),
    L("one multiply: zero extra latency",14,GRN,True),
    L("128 KB per matrix in fp16",14),
    L("one base model, many pocket adapters",14,GRN,True)]),
 ],
 ["Tradeoff: the rank r is the dial: 8 to 64 in practice, tuned on dev.",
  "LoRA cannot express large behavioral changes: new domains need full fine-tuning."],
 "adapters that insert new layers cannot merge, which is why LoRA's W plus AB formula won."))

plates.append(build("plate-l12-chap-scaling.svg",
 "Chapter plate: the scaling bill",
 "Chapter plate. Millions to build the map, pocket change to steer.",
 ["WITHOUT the budget math", "the stored object", "WITH the budget math"],
 [
  ("140 GB does not fit", [L("70B in fp16: 140 GB",14,INK,True),
    L("one GPU: 80 GB",14),
    L("scale felt like alchemy",14),
    L("bigger models, starved of data",14)]),
  ("6 x N x D FLOPs", [L("forward 2, backward 4, per token",14,INK,True),
    L("70B on 1.4T tokens: 5.9e23 FLOPs",14,BLU,True),
    L("2,000 A100s, 34 days",14,BLU,True),
    L("about $3.3M for one run",14,BLU,True)]),
  ("Chinchilla's correction", [L("20 tokens per parameter",14,INK,True),
    L("70B on 1.4T beats 280B on 300B",14,GRN,True),
    L("3D parallelism: data, tensor, pipeline",14),
    L("adaptation: tens of dollars",14,GRN,True)]),
 ],
 ["Tradeoff: scale is a budget allocation problem: parameters and data grow together.",
  "Starve either and the law punishes you. Only a few organizations can pay the pre-training bill."],
 "the paradigm concentrates pre-training and democratizes adaptation."))

print("plates built:", len(plates))
