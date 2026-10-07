# U07 lab , tiled attention checks

Instructions: implement each task as a small numpy function, then run
`python3 u07_lab_run.py`. Your outputs must match `u07_lab_key.md`
exactly. CPU numpy only, Triton is pseudocode here.

## Task 1 , tile sizing (C01)

1a. Implement `tile_size(sram, tiles, dh, bytes)`.
1b. Check Bc=200 for 100KB/4 tiles/dh=64/bf16, 21 blocks for T=4096.

## Task 2 , online softmax (C02)

2a. Implement `online_softmax(blocks)`.
2b. Check [1,2],[3,4] gives m=4.0, l=1.5530, matching naive.

## Task 3 , tiled forward (C03)

3a. Implement `flash_attn_fwd(Q,K,V,Bc)` with the nested loops.
3b. Check it matches naive to 3.33e-16 on the (1,32,8) toy.
3c. State the traffic: naive 537 MB, tiled 25.2 MB.

## Task 4 , backward check (C04)

4a. Write the analytic dQ/dK/dV for the toy.
4b. Check dQ against central differences: max rel err 3.41e-07.

## Task 5 , block skipping (C05)

5a. Count blocks for Bc=200, T=4096: 21.
5b. With a 3-block window: 18 skipped.

## Task 6 , fusion (C06)

6a. Compute the traffic for 3 separate versus 1 fused pass on
(B=2,T=4096,d=2048) bf16: 101 MB vs 34 MB.

## Task 7 , Triton analog (C07)

7a. Implement the masked block add in numpy.
7b. Check the tail is correct.

## Task 8 , protocol (C08, written)

8a. List the four protocol layers.

## Task 10 , SRAM table (C10)

10a. Tabulate Bc for 3/4/5 tiles at dh=64: 266/200/160.
10b. Add dh=128, 4 tiles: 100.

## Task 11 , accumulator dtype (C11)

11a. Compare fp16- versus fp32-accumulated scores on the toy:
max deviation 3.77e-03.

## Task 12 , invariants (C12)

12a. Implement the four invariant checks.
12b. Check all pass on the tiled toy.
