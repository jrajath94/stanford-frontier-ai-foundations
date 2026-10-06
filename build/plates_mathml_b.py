#!/usr/bin/env python3
"""Chunk B: L01-L04 plates. Run after plates_mathml_a.py."""
import sys, math
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/build")
from plates_mathml_a import *

def usedwhere(name, title, claim, footer, rows, source="original"):
    P = Plate(760, 440, title, claim, footer, source)
    y = 116
    P.text(24, y, "Math idea", 14, 600, MUTED)
    P.text(250, y, "Where it appears", 14, 600, MUTED)
    P.text(520, y, "Why there", 14, 600, MUTED)
    P.line(24, y + 8, 736, y + 8, LINE, 1.5)
    y += 30
    for m, w, why in rows:
        P.rect(24, y - 18, 200, 40, CHIP, LINE, 999)
        P.text(34, y + 6, m, 14, 600, INK)
        P.text(250, y - 2, w, 14, 500, INK)
        P.text(250, y + 16, why, 13, 450, MUTED)
        y += 52
    return P.save(name)

# ---------------- L01 ----------------
def l01():
    A = [2, 1, 0]; B = [1, 1, 1]
    dot = sum(a * b for a, b in zip(A, B)); assert dot == 3
    bab("plate-l01-dot-product.svg", "The dot product measures agreement",
        "Multiply matching components, add. Attention is this operation.",
        "Word counts", [("A = [2, 1, 0]  (great:2, film:1)", None),
                        ("B = [1, 1, 1]", None)],
        "dot", "Agreement score",
        [("2*1 + 1*1 + 0*1 = 3", TEAL),
         ("x.y = x1*y1 + ... + xn*yn", None),
         ("attention: q.k is a dot product", MUTED)], footer="One claim: the dot product is a weighted count of matches. Shell 2. Project: Stanford Frontier AI.")

    nA = math.sqrt(sum(a * a for a in A)); assert round(nA, 2) == 2.24
    A3 = [6, 3, 0]; nA3 = math.sqrt(sum(a * a for a in A3)); assert round(nA3, 2) == 6.71
    bab("plate-l01-norm.svg", "The norm measures size alone",
        "Square, sum, square-root. A3 is 3x longer than A.",
        "Lengths", [("||A||  = sqrt(4+1+0)  = 2.24", None),
                    ("||A3|| = sqrt(36+9+0) = 6.71", None)],
        "norm", "Size, no direction",
        [("A3 is 3.00x longer than A", TEAL),
         ("||x|| = sqrt(sum xi^2)", None),
         ("weight decay penalizes ||w||", MUTED)], footer="One claim: the norm is the length of the arrow. Shell 2. Project: Stanford Frontier AI.")

    nB = math.sqrt(3)
    c1 = dot / (nA * nB); c2 = sum(a * b for a, b in zip(A3, B)) / (nA3 * nB)
    assert round(c1, 2) == 0.77 and round(c2, 2) == 0.77
    bab("plate-l01-cosine.svg", "Cosine similarity divides the size out",
        "A and its tripled copy score identically: 0.77.",
        "Dot product (fooled)", [("A.B  = 3", None), ("A3.B = 9  (3x, no new content)", ORANGE)],
        "divide", "Cosine (direction only)",
        [(f"cos(A,B)  = 3/(2.24*1.73) = {c1:.2f}", TEAL),
         (f"cos(A3,B) = 9/(6.71*1.73) = {c2:.2f}", TEAL),
         ("-1 opposite, 0 perpendicular, 1 parallel", MUTED)], footer="One claim: cosine compares direction, ignores length. Shell 3. Project: Stanford Frontier AI.")

    bab("plate-l01-lincomb.svg", "Linear combination: scale, then add",
        "3*[1,0] + 2*[0,1] = [3,2]. Every vector in R^2 is such a mix.",
        "Standard vectors", [("[1,0]  the x-axis", None), ("[0,1]  the y-axis", None)],
        "mix", "Any vector in R^2",
        [("3*[1,0] + 2*[0,1]", None), ("= [3,2]", TEAL),
         ("[c1,c2] = c1*[1,0] + c2*[0,1]", MUTED)], footer="One claim: scaling and adding vectors reaches every point. Shell 2. Project: Stanford Frontier AI.")

    bab("plate-l01-independence.svg", "Independence: no vector is a copy",
        "[1,1] and [2,2] are dependent; [1,0] and [0,1] are not.",
        "Dependent pair", [("v1 = [1,1], v2 = [2,2]", None),
                           ("2*v1 - 1*v2 = [0,0]", ORANGE),
                           ("v2 adds no new direction", None)],
        "test", "Independent pair",
        [("v1 = [1,0], v2 = [0,1]", None),
         ("no nonzero mix gives [0,0]", TEAL),
         ("each adds a new direction", None)], footer="One claim: dependence means a secret copy. Shell 2. Project: Stanford Frontier AI.")

    usedwhere("plate-l01-used-where.svg", "Vectors: what is used where",
              "The same three tools run embeddings, search, and training.",
              "Shell 5. Source: public model docs and standard practice. Project: Stanford Frontier AI.",
              [("dot product", "attention scores q.k", "every transformer layer (CS229S L02)"),
               ("cosine similarity", "retrieval ranking", "FAISS / vector search compares direction"),
               ("norm ||w||", "weight decay, clipping", "L2 penalty and gradient clipping"),
               ("independence", "feature design", "duplicate features break L09")])

# ---------------- L02 ----------------
def l02():
    M = [[0, -1], [1, 0]]; v = [1, 2]
    r = [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]
    assert r == [-2, 1]
    bab("plate-l02-rotation.svg", "A matrix rotates the vector",
        "Rows dot the vector: [0 -1; 1 0] turns [1,2] into [-2,1].",
        "Machine M, input v", [("M = [ 0 -1 ; 1 0 ]", None), ("v = [1, 2]", None)],
        "rows.v", "Output Mv",
        [("row1.v = 0*1 + -1*2 = -2", None), ("row2.v = 1*1 + 0*2 = 1", None),
         ("Mv = [-2, 1]  (90 deg rotation)", TEAL)], footer="One claim: each row dots the input vector. Shell 2. Project: Stanford Frontier AI.")

    M2 = [[2, 1], [0, 3]]; v2 = [3, 4]
    r2 = [2 * 3 + 1 * 4, 0 * 3 + 3 * 4]; assert r2 == [10, 12]
    bab("plate-l02-columns.svg", "Columns are where the axes land",
        "Mv = 3*[2,0] + 4*[1,3] = [10,12]. Same answer as rows dot v.",
        "Row view", [("row1.v = 2*3 + 1*4 = 10", None), ("row2.v = 0*3 + 3*4 = 12", None)],
        "same", "Column view",
        [("3 * column1 + 4 * column2", None), ("= 3*[2,0] + 4*[1,3]", None),
         ("= [10, 12]", TEAL)], footer="One claim: the answer mixes the columns by the input. Shell 2. Project: Stanford Frontier AI.")

    Arot = [[0, -1], [1, 0]]; Bs = [[2, 0], [0, 2]]
    BA = [[sum(Bs[i][k] * Arot[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert BA == [[0, -2], [2, 0]]
    out = [BA[0][0] * 1 + BA[0][1] * 2, BA[1][0] * 1 + BA[1][1] * 2]; assert out == [-4, 2]
    bab("plate-l02-compose.svg", "Composition is multiplication",
        "Rotate then scale-2: one matrix BA does both. (BA)v = B(Av).",
        "Two steps", [("v = [1,2]", None), ("A v = [-2,1]  (rotate)", None), ("B(Av) = [-4,2]  (scale)", None)],
        "BA", "One matrix",
        [("BA = [ 0 -2 ; 2 0 ]", None), ("(BA) v = [-4, 2]", TEAL),
         ("compose once, apply to millions", MUTED)], footer="One claim: chaining transformations is matrix multiplication. Shell 3. Project: Stanford Frontier AI.")

    b = [3, 4]; a = [1, 0]
    ab = sum(x * y for x, y in zip(a, b)); aa = sum(x * x for x in a)
    proj = [ab / aa * x for x in a]; assert proj == [3, 0]
    left = [b[i] - proj[i] for i in range(2)]; assert left == [0, 4]
    bab("plate-l02-projection.svg", "Projection: the shadow formula",
        "[3,4] onto the x-axis is [3,0]; the leftover [0,4] is perpendicular.",
        "Drop the perpendicular", [("b = [3,4], a = [1,0]", None),
                                   ("a.b = 3,  a.a = 1", None)],
        "project", "Shadow + leftover",
        [("proj = (3/1)*[1,0] = [3,0]", TEAL),
         ("leftover = [0,4], perpendicular", None),
         ("[3,0] + [0,4] = [3,4]: nothing lost", MUTED)], footer="One claim: every vector splits along the line and across it. Shell 3. Project: Stanford Frontier AI.")

    M3 = [[2, 1], [0, 3]]
    Mt = [[M3[j][i] for j in range(2)] for i in range(2)]; assert Mt == [[2, 0], [1, 3]]
    A = [[1, 2], [0, 1]]; B = [[3, 0], [1, 1]]
    AB = [[sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    ABt = [[AB[j][i] for j in range(2)] for i in range(2)]
    Bt = [[B[j][i] for j in range(2)] for i in range(2)]; At = [[A[j][i] for j in range(2)] for i in range(2)]
    BtAt = [[sum(Bt[i][k] * At[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert ABt == BtAt == [[5, 1], [2, 1]]
    bab("plate-l02-transpose.svg", "Transpose flips rows and columns",
        "(AB)^T = B^T A^T: both equal [5 1; 2 1]. Order flips.",
        "Flip M", [("M  = [ 2 1 ; 0 3 ]", None), ("M^T = [ 2 0 ; 1 3 ]", TEAL)],
        "order", "(AB)^T vs B^T A^T",
        [("(AB)^T  = [ 5 1 ; 2 1 ]", None), ("B^T A^T = [ 5 1 ; 2 1 ]", TEAL),
         ("M^T M never has negative eigenvalues", MUTED)], footer="One claim: the transpose flips, and reverses product order. Shell 2. Project: Stanford Frontier AI.")

    usedwhere("plate-l02-used-where.svg", "Matrices: what is used where",
              "Every neural layer is one of these multiplications.",
              "Shell 5. Source: public model docs. Project: Stanford Frontier AI.",
              [("Wq, Wk, Wv", "attention projections", "CS229S L02: queries, keys, values"),
               ("MX batch", "GPU inference", "one call transforms the batch"),
               ("embedding table", "token lookup", "rows are token vectors (CS336)"),
               ("projection", "least squares L09", "prediction is the shadow")])

# ---------------- L03 ----------------
def l03():
    A = [[2, 1], [0, 3]]
    v1 = [1, 1]; Av1 = [A[0][0] * v1[0] + A[0][1] * v1[1], A[1][0] * v1[0] + A[1][1] * v1[1]]
    assert Av1 == [3, 3]
    v2 = [1, 0]; Av2 = [A[0][0] * v2[0] + A[0][1] * v2[1], A[1][0] * v2[0] + A[1][1] * v2[1]]
    assert Av2 == [2, 0]
    bab("plate-l03-eigen.svg", "Eigenvectors: stretched, never rotated",
        "A[1,1] = [3,3] = 3*[1,1]. A[1,0] = [2,0] = 2*[1,0].",
        "Test a vector", [("A = [ 2 1 ; 0 3 ]", None), ("try v = [1, 1]", None)],
        "Av = lv", "Eigen pairs",
        [("A[1,1] = [3,3] = 3*[1,1]", TEAL), ("eigenvalue l = 3", None),
         ("A[1,0] = [2,0] = 2*[1,0]", TEAL), ("eigenvalue l = 2", None)], footer="One claim: an eigenvector keeps its direction. Shell 2. Project: Stanford Frontier AI.")

    bab("plate-l03-charpoly.svg", "The characteristic equation finds them",
        "det(A - lI) = (2-l)(3-l) = 0 gives l = 3 and l = 2.",
        "det(A - lI) = 0", [("A - lI = [2-l 1; 0 3-l]", None),
                            ("det = (2-l)(3-l) - 0 = 0", None)],
        "roots", "Eigenvalues then vectors",
        [("l1 = 3, l2 = 2", TEAL),
         ("l=3: -x + y = 0 -> v = [1,1]", None),
         ("l=2: y = 0 -> v = [1,0]", None)], footer="One claim: the eigenvalues are the roots of one equation. Shell 3. Project: Stanford Frontier AI.")

    pts = [(2, 0), (-2, 0), (0, 1), (0, -1)]
    vx = sum(x * x for x, y in pts) / 4; vy = sum(y * y for x, y in pts) / 4
    assert (vx, vy) == (2.0, 0.5)
    frac = vx / (vx + vy); assert round(frac, 2) == 0.80
    P = Plate(760, 440, "PCA by hand: four points, real variance numbers",
              "Top component (x-axis) holds 80% of the variance. Keep it, drop the rest.",
              "Shell 3. Source: original toy. Project: Stanford Frontier AI.")
    P.rect(24, 96, 440, 260, PANEL)
    P.text(40, 126, "Four centered points", 16, 600, INK)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax, ay, aw, ah = 60, 150, 360, 190
    def pxx(x): return ax + (x + 2.5) / 5 * aw
    def pyy(y): return ay + ah - (y + 1.5) / 3 * ah
    P.line(ax, pyy(0), ax + aw, pyy(0), LINE, 1.5)
    P.line(pxx(0), ay, pxx(0), ay + ah, LINE, 1.5)
    P.arrow(pxx(-2.2), pyy(0), pxx(2.2), pyy(0), "PC1: variance 2.0", TEAL)
    for x, y in pts:
        P.p.append(f'<circle cx="{pxx(x):.1f}" cy="{pyy(y):.1f}" r="8" fill="{FOCUS}" stroke="{INK}" stroke-width="1.5"/>')
    P.rect(492, 96, 244, 260, NEWOBJ)
    P.text(508, 126, "Read the answer", 16, 600, INK)
    for i, (s, c) in enumerate([("cov = [2 0; 0 0.5]", None), ("eigenvalues: 2, 0.5", None),
                                ("2 / 2.5 = 80% kept", TEAL), ("drop y: 2-D -> 1-D", None)]):
        P.mono(508, 158 + i * 30, s, 14.5, c or INK)
    P.save("plate-l03-pca.svg")

    bab("plate-l03-symmetric.svg", "Symmetric matrices always cooperate",
        "Covariance matrices are symmetric: perpendicular, complete eigenvectors. PCA is safe.",
        "General square matrix", [("may be defective", ORANGE),
                                  ("eigenvectors may not span", None),
                                  ("A = V L V^-1 can fail", None)],
        "symmetry", "Symmetric (covariance)",
        [("eigenvectors perpendicular", TEAL),
         ("always complete: V^-1 = V^T", TEAL),
         ("PCA eigen-decomposition exists", None)], footer="One claim: symmetry guarantees the decomposition. Shell 3. Project: Stanford Frontier AI.")

    usedwhere("plate-l03-used-where.svg", "Eigen-decomposition: what is used where",
              "Stretch directions run compression, ranking, and curvature.",
              "Shell 5. Source: public docs and papers. Project: Stanford Frontier AI.",
              [("PCA", "sklearn.decomposition", "top eigenvectors of covariance"),
               ("PageRank", "Google ranking", "top eigenvector of the link matrix"),
               ("spectral clustering", "graph ML", "eigenvectors of the Laplacian"),
               ("Hessian eigenvalues", "optimization L08", "curvature = second derivatives")])

# ---------------- L04 ----------------
def l04():
    A = [[4, 0], [3, 0]]
    AtA = [[sum(A[k][i] * A[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert AtA == [[25, 0], [0, 0]]
    s1, s2 = math.sqrt(25), math.sqrt(0); assert (s1, s2) == (5.0, 0.0)
    u1 = [4 / 5, 3 / 5]; assert u1 == [0.8, 0.6]
    bab("plate-l04-svd.svg", "A = U Sigma V^T: rotation, stretch, rotation",
        "Singular values 5 and 0. u1 = [0.8, 0.6]. Entry (1,1): 0.8*5*1 = 4.",
        "From A^T A", [("A^T A = [25 0; 0 0]", None),
                       ("eigenvalues 25, 0", None),
                       ("singular values: 5, 0", TEAL)],
        "build U", "The three pieces",
        [("u1 = A[1,0]/5 = [0.8, 0.6]", None),
         ("Sigma = [5 0; 0 0]", None),
         ("check: 0.8*5 = 4, 0.6*5 = 3", TEAL)], footer="One claim: every matrix is rotate, stretch, rotate. Shell 2. Project: Stanford Frontier AI.")

    bab("plate-l04-rank.svg", "Rank counts the nonzero singular values",
        "Sigma = [5, 0]: rank 1. The zero exposes the dead column.",
        "Singular values [5, 0]", [("sigma1 = 5: real direction", None),
                                   ("sigma2 = 0: nothing flows", ORANGE)],
        "count", "Rank and meaning",
        [("rank = 1, not 2", TEAL),
         ("column 2 of A is all zeros", None),
         ("rank = true dimensionality", MUTED)], footer="One claim: zeros in Sigma are dead directions. Shell 2. Project: Stanford Frontier AI.")

    b11, b21 = 0.8 * 5, 0.6 * 5; assert (b11, b21) == (4.0, 3.0)
    bab("plate-l04-eckart.svg", "Truncation error equals the dropped value",
        "Drop sigma2 = 1: error ||B - B1|| = 1 exactly. Best possible.",
        "Keep top piece", [("B1 = [0.8;0.6] [5] [1 0]", None),
                           ("= [4 0; 3 0]", TEAL)],
        "drop", "Error = sigma2",
        [("||B - B1|| = 1", TEAL),
         ("no rank-1 matrix is closer", None),
         ("Eckart-Young: optimal", MUTED)], footer="One claim: the top-k SVD is the best rank-k compression. Shell 3. Project: Stanford Frontier AI.")

    full = 4096 * 4096; low = 64 * (4096 + 4096 + 1)
    assert full == 16777216 and low == 524352
    vbar("plate-l04-compression.svg", "Low rank compresses 32x",
         "4096x4096 weights: 16.8M numbers. Rank-64: 0.52M. Same story as LoRA.",
         "Parameters, in millions. Shell 3. Source: original arithmetic. Project: Stanford Frontier AI.",
         [("full", round(full / 1e6, 1), PINK), ("rank-64", round(low / 1e6, 2), TEAL)])

    usedwhere("plate-l04-used-where.svg", "SVD: what is used where",
              "The compression trick behind adapters, recommenders, and PCA.",
              "Shell 5. Source: public papers and docs. Project: Stanford Frontier AI.",
              [("LoRA", "LLM fine-tuning", "train low-rank updates, freeze base (Hu et al.)"),
               ("matrix factorization", "recommenders", "user/item factors from the SVD"),
               ("PCA via SVD", "sklearn", "SVD of centered data = components"),
               ("image compression", "JPEG-style", "drop small singular values")])


if __name__ == "__main__":
    l01(); l02(); l03(); l04()
    print("L01-L04 plates done")
