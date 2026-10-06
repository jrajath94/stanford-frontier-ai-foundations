# Keys, lab 08 neural and sequence architectures

Date: 2026-10-06.

## Task 1

(a) z = 0.4(1.0) + 0.2 = 0.6. a = tanh(0.6) =
0.5370495670. o = -1.5(0.5370495670) + 0.5 =
-0.3055743505. L = 0.5(-1.3055743505)^2 =
0.8522621923.

(b) do = -1.3055743505. dw2 = do a =
-0.7011581396. db2 = do = -1.3055743505.
da = do w2 = 1.9583615257. dz = da(1 - a^2)
= 1.9583615257(0.711577) = 1.3935265128.
dw1 = dz x = 1.3935265128. db1 = dz =
1.3935265128.

(c) Central differences agree to 9 digits
(1.3935265131 vs 1.3935265128).

## Task 2

(a) Scores: [1, 1, 2]/sqrt(2) = [0.7071,
0.7071, 1.4142]. Max-subtracted: [-0.7071,
-0.7071, 0]. Exp: [0.4931, 0.4931, 1.0],
sum 1.9862. Weights [0.2483, 0.2483, 0.5034].
Output: 0.2483[2,0] + 0.2483[0,2] +
0.5034[1,1] = [1.0, 1.0].

(b) Weights sum to 1.0. The third key scores
twice the dot product (it matches both query
items), so after the softmax it holds half
the mass.

## Task 3

(a) m = [0.08, -0.02], v = [6.4e-4, 4e-5],
m_hat = [0.8, -0.2], v_hat = [0.64, 0.04],
update = 0.02[0.8/0.8, -0.2/0.2] = [0.02,
-0.02]. SGD: -0.02[0.8, -0.2] = [-0.016,
0.004].

(b) Adam erases the 4:1 scale ratio and steps
eta per coordinate. SGD keeps the ratio and
steps 4x farther on the steep coordinate.
Adam's first step is always a signed step of
size eta from zero state.
