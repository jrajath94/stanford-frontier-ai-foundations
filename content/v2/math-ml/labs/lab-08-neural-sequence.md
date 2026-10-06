# Lab 08, neural and sequence architectures

Unit: math-ml-U08. Date: 2026-10-06. Keys in labs/keys-lab-08.md.
All numbers computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## Task 1, backprop by hand on a new scalar net

Net: o = w2 tanh(w1 x + b1) + b2, loss
L = 0.5 (o - y)^2. Values: w1 = 0.4, b1 = 0.2,
w2 = -1.5, b2 = 0.5, x = 1.0, y = 1.0.

(a) Forward: compute z, a, o, L.
Reference: z = 0.6, a = 0.5370495670,
o = -0.3055743505, L = 0.8522621923.

(b) Backward: compute dw2, db2, dw1, db1.
Reference: dw2 = -0.7011581396, db2 =
-1.3055743505, dw1 = 1.3935265128, db1 =
1.3935265128.

(c) Check dw1 and dw2 against central finite
differences (h = 1e-7) to 6 digits.

## Task 2, attention by hand on a new toy

Q = [[1, 1]], K = [[1, 0], [0, 1], [1, 1]],
V = [[2, 0], [0, 2], [1, 1]]. d = 2.

(a) Compute the score row, the stable softmax
weights, and the output row. Reference:
scores [0.7071, 0.7071, 1.4142], weights
[0.2483, 0.2483, 0.5034], output [1.0, 1.0].

(b) Assert the weights sum to 1.0 and explain
in one sentence why the third key dominates.

## Task 3, Adam versus SGD on a new gradient

g = [0.8, -0.2], beta1 = 0.9, beta2 = 0.999,
eta = 0.02, eps = 1e-8, from zero state.

(a) Compute the Adam update and the SGD
update with eta = 0.02. Reference: Adam
[0.02, -0.02], SGD [-0.016, 0.004].

(b) Explain in two sentences what the two
updates reveal about scale handling.
