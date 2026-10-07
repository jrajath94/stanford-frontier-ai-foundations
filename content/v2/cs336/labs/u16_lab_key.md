# U16 lab key

Executed 2026-10-06, numpy CPU. Run `python3 u16_lab_run.py`, outputs
must match exactly.

```
T1 BT P=0.731
T2 pair = (prompt, chosen, rejected), ties discarded
T3 KL 0.0253 nats, penalty 0.00253 at beta=0.1
T4 clipped terms: [ 0.7  0.9 -1.   1.2  1.2]
T5 A = r - V = 1 - 0.7 = 0.3
T6 margin 0.139, loss 0.626
T7 advantages: [ 1. -1.  1. -1.]
T8 rewards [1.0, 1.0, 0.0, 1.0, 0.0, 1.0] mean 0.67
T9 batch cost 4096 slot-seconds (toy)
T10 drift max 0.006 nats: healthy, halt past budget
T11 gap 0.24 at step 100, opens at step 40
T12 gate: safety 0.98, reasoning no-regress, gap<0.05: SHIP
```
