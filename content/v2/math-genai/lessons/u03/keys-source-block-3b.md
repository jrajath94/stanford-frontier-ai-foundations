# keys-source-block-3b.md, U03 source block answer keys

Date: 2026-10-06. Closed-book answers for lesson 03b. Keep
separate from the lesson file. All numbers computed
2026-10-06, numpy 1.26.4, float64.

## E01

Two neural networks train each other: a generator G maps
noise to samples and minimizes the game value. A
discriminator D maps samples to (0,1) and maximizes it.
No likelihood is used.

## E02

min_G max_D V(D,G) with V = E_{p_data}[log2 D(x)] +
E_z[log2(1-D(G(z)))]. Inner: max over D (best referee for
the current G). Outer: min over G.

## E03

Fix x. Maximize a log y + b log(1-y), a = p_data(x), b =
p_g(x). d/dy = a/y - b/(1-y) = 0 gives y = a/(a+b).
Hence D*(x) = p_data(x)/(p_data(x)+p_g(x)).

## E04

(1) G(z): guards against constant output (C01 failure).
(2) D(x) with sigmoid: guards against unbounded outputs
that break the logs. (3) The two losses in bits:
guards against the nat/bit factor error. (4) The k-loop:
guards against a frozen or lazy D (C02 failure). (5) The
eps clip on D: guards against log-boundary explosions
(C07).

## E05

D*/(1-D*) = p_data/p_g. At x = 0: 0.7870/0.2130 = 3.69
(data country). At x = 1: 0.2327/0.7673 = 0.30 (fake
country). G's mass should move from x = 1 toward x = 0,
following the ratio uphill.

## E06

No. The citation is invalid: no transcript of W2_T5 was
inspected (source_gaps.md G2), so nothing about the
tutorial's content is source-confirmed. The
non-saturating loss is authored lesson content. The
valid citation is "U03-C05 of this build".

## L01

Game: min_G max_D V(D,G). Toy: V = -1.3537, D*(0) =
0.7870. Derivation: E03 plus the four-line JS identity
(E07 of the main lesson). Code: the C04 snippet. Assert
|V-(2JS-2)| < 1e-12. Compare: the titles W2_L6/W2_L7/W3L8
promise the idea, the formulation, and the sampler view.
the mapping is title-level only, so every mechanism
above is authored, not lecture-confirmed. Debug: with D
frozen the game is not adversarial: G minimizes against
a fixed function and the min-max structure is gone.
Critique: "adversarial" requires the inner max to track
G. A frozen D makes the title inaccurate. Design: an
inspected transcript (or slide deck) of W2_L7 showing
the minimax equation would promote the formulation row
to source-confirmed. A W2_T5 code listing would promote
the implementation row.
