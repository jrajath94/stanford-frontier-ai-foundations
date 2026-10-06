# Lab keys 03, calculus and optimization computations

Date: 2026-10-06. Computed values, numpy 1.26.4, float64.

## Task 1

(a) w* = 39/30 = 1.3.
(b) Gradient at w = 1: -2.25. Gradient at w*: 4.440892098500626e-16,
within 1e-12 of zero. Float dust, not a remainder.
(c) New w = 1 - 0.1 * (-2.25) = 1.225. The cost fell: the gradient
was negative, so w rose toward w* = 1.3.

## Task 2

(a) Final J: eta = 0.05 -> 0.7178979876918524. eta = 0.5 -> 0.0.
eta = 1.5 -> 150994944.0.
(b) Multipliers: 0.9 (shrinks), 0.0 (lands), 2.0 (doubles the
distance each step). Prediction before measuring: slow crawl,
one-step landing, explosion.
(c) Strong answer: "Your fixed step crossed the stability boundary
mid-run: the valley narrowed, the curvature grew, and 1/L fell below
your eta. Cut the rate and check whether the divergence step matches
the |1 - 2 eta| arithmetic."

## Task 3

(a) Iterates: 1.0, 1.333333, 1.236715, 1.224916, 1.224745,
1.224745. Final value 1.224745.
(b) sqrt(1.5) = 1.224744871391589. Match within 1e-9.
(c) From x0 = 0.1, Newton converges to 0.0, the saddle. The Hessian
at 0.1 is 12*0.01 - 6 = -5.88, negative: the local parabola opens
downward, so Newton aims at its peak, the hill, not a bowl. Wrong
target, correct arithmetic.
