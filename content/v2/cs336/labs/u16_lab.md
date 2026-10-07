# U16 lab , alignment arithmetic on synthetic toys

Instructions: implement each task as a small numpy function, then run
`python3 u16_lab_run.py`. Your outputs must match `u16_lab_key.md`
exactly. No RL runs, synthetic logits and rewards.

## Task 1 , Bradley-Terry (C01)

1a. r_w=1.0, r_l=0.0: P=0.731.

## Task 2 , pairs (C02)

2a. Pair = (prompt, chosen, rejected), ties discarded.

## Task 3 , KL penalty (C03)

3a. KL 0.0253 nats, penalty 0.00253 at beta=0.1.

## Task 4 , PPO clip (C04)

4a. Clipped terms: [0.7, 0.9, -1.0, 1.2, 1.2].

## Task 5 , advantage (C05)

5a. A = 1 - 0.7 = 0.3.

## Task 6 , DPO (C06)

6a. Margin 0.139, loss 0.626.

## Task 7 , GRPO (C07)

7a. Rewards [1,0,1,0] -> advantages [1,-1,1,-1].

## Task 8 , verifiable (C08)

8a. Rewards [1,1,0,1,0,1], mean 0.67.

## Task 9 , rollout cost (C09)

9a. Toy batch cost 4096 slot-seconds.

## Task 10 , staleness (C10)

10a. Drift max 0.006 nats: healthy.

## Task 11 , hacking (C11)

11a. Gap 0.24 at step 100, opens at step 40.

## Task 12 , ship gate (C12)

12a. Safety 0.98, no reasoning regression, gap<0.05: SHIP.
