# FIGURE ENFORCER audit — cs336/l07-parallelism
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions (Shell + Source added, F5) and added inline
# figure tables (F1/F2).

## Figure inventory (14 plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f01 | assets/l07-collectives.svg | SVG lesson plate (kept) | Three collectives run training: all-gather assembles, reduce-scatter sums and splits, all-reduce does both |
| f02 | assets/l07-ring-allreduce.svg | SVG lesson plate (NEW, replaces webp) | The ring all-reduce moves 2x the data; bandwidth independent of world size |
| f03 | assets/l07-memory-hierarchy.svg | SVG lesson plate (kept) | Distance grows, bandwidth shrinks: registers to HBM to NVLink to IB to Ethernet |
| f04 | assets/l07-topology.svg | SVG lesson plate (kept, GPU labels fixed) | 8 GPUs per node on NVSwitch; InfiniBand between pods; NVL72 = 72 in one NVLink domain |
| f05 | assets/l07-rdma.svg | SVG lesson plate (kept) | Ethernet copies via CPU; RDMA lets GPUs write each other's memory directly |
| f06 | assets/l07-data-parallel.svg | SVG lesson plate (kept) | Split rows, all-reduce gradients, average; parameters stay identical everywhere |
| f07 | assets/l07-tensor-parallel.svg | SVG lesson plate (kept) | Column split forward, all-gather; reduce-scatter backward; comm per layer |
| f08 | assets/l07-pipeline-parallel.svg | SVG lesson plate (kept) | Each rank owns layer groups; micro-batches shrink the idle bubble |
| f09 | assets/l07-pipeline-bubble.svg | SVG lesson plate (NEW, replaces webp) | Idle = (p-1)/(m+p-1): 75% at m=1, 27% at m=8 |
| f10 | assets/l07-where-which.svg | SVG lesson plate (kept) | Hardware picks the cut: TP in-node, DP across, PP if needed |
| f11 | assets/l07-who-trains-how.svg | SVG lesson plate (NEW, replaces webp) | DeepSeek-V3: EP replaces TP. Llama 3 405B: dense, TP splits the layers |
| c1 | assets/l07-chap-data-parallel.svg | SVG chapter plate (NEW) | DDP: 12 bytes replicated per GPU; one all-reduce; batch must exceed world size |
| c2 | assets/l07-chap-tensor-parallel.svg | SVG chapter plate (NEW) | Column-up/row-down, one sync per pair; NVLink-only traffic decides viability |
| c3 | assets/l07-chap-pipeline-parallel.svg | SVG chapter plate (NEW) | 75% idle to 27%; schedules trade memory for utilization; bubbles never fully die |
| f-tab-fsdp | inline table (NEW) | table | FSDP 3P (2.6GB) vs DDP 2P (1.75GB): the extra all-gather is the sharding price |
| f-tab-regime | inline table (NEW) | table | 1KB: latency wins 10,000:1. 400MB: bandwidth wins 44:1 |
| f-tab-nccl | inline table (NEW) | table | Message size picks topology (tree/ring) and protocol (LL/Simple) |
| f-tab-12bytes | inline table (NEW) | table | 2+2+4+4 = 12 bytes per parameter, line by line |
| f-tab-bucket | inline table (NEW) | table | Naive 70ms dead time vs bucketed 0.125ms exposed; same bytes, different schedule |
| f-tab-tp-traffic | inline table (NEW) | table | 7B: ~3GB per rank/step. Llama-70B: 86GB; 48ms on NVLink, 1.7s on IB |
| f-tab-gpipe | inline table (NEW) | table | GPipe 27% twice vs 1F1B 37.5% once; GPipe holds m, 1F1B holds p |
| f-tab-history | inline table (NEW) | table | Megatron wired it, GPT-3 proved the blend, DeepSpeed removed replication |
| f-eq-latbw | inline prose equation (kept) | equation | time = latency + size/bandwidth |
| f-eq-bubble | inline prose equation (kept) | equation | idle fraction = (p-1)/(m+p-1) |
| f-tab-mapping | inline table (kept) | table | Cut, what splits, communication, hardware it needs |

## Fixes applied to existing figures
- Font stack reordered to spec (Anthropic Sans first) in all 48 l07-l12 SVGs.
- l07-topology.svg: GPU labels were GPU1/3/5/7/5/7/9/11 (duplicates); fixed to GPU0-GPU7.
- Captions on all 11 kept plates: added "Shell N." and "Source:" per F5.
- Three generated webp stills DELETED (ring-allreduce, pipeline-bubble, who-trains-how) and redrawn as flat SVG lesson plates per the medium ladder (F2): each claim is position/count/state-change, which SVG passes first. No generated stills remain in l07.

## Medium-ladder justification for new figures
- f02 ring all-reduce: table no (the claim is the ring's motion + the 2x accounting); equation no; ASCII borderline (4 ranks x 6 steps exceeds a clean trace); mermaid cannot carry the per-rank MB accounting; SVG passes first.
- f09 pipeline bubble: table no (idle tax is a fill/drain state); ASCII could show 4x1 but the 4x8 pipe plus the GPipe/1F1B contrast is positional; SVG passes first.
- f11 who-trains-how: table could compare the recipes, but the claim is two architectural choices with degrees; the lesson plate carries the [uncertain] marks visibly; SVG passes first.
- f-tab-* tables: every one is a comparison of values (FSDP vs DDP traffic, regimes, protocols, byte lines, schedules, systems): table is the first medium that passes.
- f-eq-latbw / f-eq-bubble: definitions: equation is the first medium that passes; kept as inline prose equations.
- c1-c3: chapter plates mandated by the spec; SVG keeps text crisp and matches the site plate system.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | Two reasons for many GPUs: model does not fit, or speed | one GPU assumed | 1T params x 12B = 12TB vs 192GB B200 | c1 | chapter plate | lecture |
| u-def-collective | Collectives: communication templates; rank, world size | GPUs talk how? | named patterns, NCCL picks topology | f01 | SVG | lecture |
| u-def-allgather | all-gather: each rank's piece ends on every rank | pattern unnamed | 4-rank toy: all hold [1,2,3,4] | f01 | SVG | lecture |
| u-def-reducescatter | reduce-scatter: reduce per shard, scatter | pattern unnamed | rank r holds summed shard r | f01 | SVG | lecture |
| u-def-allreduce | all-reduce = reduce-scatter + all-gather | pattern unnamed | every rank holds the sum, 10 | f01, f02 | SVG | lecture |
| u-def-alltoall | all-to-all routes tokens to experts in MoE | pattern unnamed | arbitrary bytes rank to rank | f01 | SVG | lecture |
| u-h-whycollectives | Collectives name the pattern; NCCL picks the topology | point-to-point hand scheduling | one call, library writes the plan | f01 | SVG | lecture |
| u-h-fsdp | FSDP: all-gather params forward, reduce-scatter grads | monolithic all-reduce | never holds the full model | f-tab-fsdp | table | original toy |
| u-h-fsdp-cost | FSDP 2.6GB vs DDP 1.75GB per rank per layer | cost unworked | 3P vs 2P; overlap hides it | f-tab-fsdp | table | original toy |
| u-eq-latbw | time = latency + size/bandwidth | two costs unnamed | the one equation | f-eq-latbw | equation | lecture |
| u-h-latbw-1k | 1KB over NVLink: latency dominates 10,000:1 | link speed assumed to matter | 0.0006us vs 5us startup | f-tab-regime | table | original toy |
| u-h-latbw-400m | 400MB over NVLink: bandwidth dominates 44:1 | startup assumed to matter | 222us vs 5us | f-tab-regime | table | original toy |
| u-h-topo-rule | Small messages want trees; large want rings | topology as preference | two-regime arithmetic decides | f02, f-tab-nccl | SVG/table | lecture |
| u-h-ring | Ring all-reduce: n-1 + n-1 steps; 2(n-1)/n per rank | algorithm unnamed | 4-rank toy: 6MB moved per rank | f02 | SVG | lecture |
| u-h-ring-tree | Ring bandwidth-optimal; tree latency-optimal | one topology assumed | NCCL picks per message size | f02 | SVG | lecture |
| u-h-nccl | NCCL: rings, trees, double binary tree, LL vs Simple | one ring assumed | hierarchy: fast rings in-node, trees across | f-tab-nccl | table | lecture |
| u-h-memhier | Memory hierarchy: registers to Ethernet | HBM only | 8 TB/s HBM, 1.8 TB/s NVLink, IB, Ethernet | f03 | SVG | lecture |
| u-h-topo | 8 GPUs per node on NVSwitch; IB between pods | flat cluster assumed | NVL72: 72 GPUs, one NVLink domain | f04 | SVG | lecture |
| u-def-rdma | RDMA: GPU writes GPU memory, no CPU | Ethernet assumed | RoCE: cheap answer to InfiniBand | f05 | SVG | lecture |
| u-h-nccl-api | torch.distributed: spawn, barrier, two asynchronies | API unnamed | sync kernels and processes to benchmark | f05 | SVG | lecture |
| u-h-bweff | 100M-element all-reduce at ~400 GB/s | bandwidth unmeasured | 2(n-1)/n x size / duration | f02 | SVG | lecture |
| u-h-dp | Data parallel: split rows, all-reduce, average | batch as one unit | 32 rows per rank; params identical | f06, c1 | SVG | lecture code |
| u-code-ddp | The one line: all_reduce then divide by world size | sync unnamed | sum then average, every step | f06 | SVG | lecture_07.py |
| u-h-bucket | Bucketing: 25MB buckets overlap backward | sync waits for full backward | exposed sync 70ms to 0.125ms | f-tab-bucket | table | original toy |
| u-h-bucket-fail | Dead branch stalls its bucket: find_unused_parameters | hang mysterious | unused params need the flag | f-tab-bucket | table | original toy |
| u-h-12bytes | 12 bytes/param: 2+2+4+4 | memory uncounted | four lines memorized | f-tab-12bytes, c1 | table | Lecture 2 |
| u-h-dp-break | DDP breaks: 70B x 12 = 840GB > 192GB | replication assumed fine | the first cut fails on memory | c1 | chapter plate | original toy |
| u-h-critbatch | Past critical batch size, data parallel wastes compute | bigger batch always better | ceiling data cannot lift | c1 | chapter plate | lecture |
| u-h-keyq | Split the model; match cuts to hardware | data split only | three cuts answer | f10 | SVG | lecture |
| u-h-tp | Tensor parallel: column split, all-gather fwd, reduce-scatter bwd | layer as one unit | comm per layer per step | f07, c2 | SVG | lecture |
| u-h-tp-traffic | 7B: ~3GB per rank/step; Llama-70B: 86GB | traffic unworked | NVLink 48ms vs IB 1.7s | f-tab-tp-traffic, c2 | table | original toy |
| u-h-megatron | Megatron wiring: column-up, row-down, one all-reduce per pair | wiring folk art | nonlinearity stays local | c2 | chapter plate | Megatron-LM |
| u-h-seqpar | Sequence parallel: shard the elementwise leftovers, free | 10sbh stubborn | rides TP's existing sync | c2 | chapter plate | Korthikanti et al. |
| u-h-seqpar-nums | 2.68GB to 336MB at s=4096,b=4,h=8192,t=8 | leftover unworked | divides, does not delete | c2 | chapter plate | original toy |
| u-h-pp | Pipeline: rank owns layer group, send/recv | layers as one unit | point-to-point activations | f08, c3 | SVG | lecture |
| u-h-bubble | 75% idle with 1 batch; micro-batches shrink it | pipe assumed full | 3 of 4 ranks idle | f08, f09, c3 | SVG | lecture |
| u-eq-bubble | idle fraction = (p-1)/(m+p-1) | tax unformula'd | 3/4=75%, 3/11=27% | f-eq-bubble, f09 | equation/SVG | lecture |
| u-h-interleave | Interleaved: bubble (p-1)/(v x m); DualPipe both ends | one schedule assumed | v=2: 37.5% to 18.75% | c3 | chapter plate | lecture |
| u-h-gpipe | GPipe 27% twice; 1F1B 37.5% once; memory m vs p | schedules unnamed | 1F1B wins both axes | f-tab-gpipe, c3 | table | papers |
| u-h-where | Hardware picks: TP in-node, DP across, PP if needed | cut as preference | match comm to links | f10 | SVG | lecture |
| u-tab-mapping | Mapping table: cut, split, comm, hardware | scattered claims | one-screen mapping | f-tab-mapping | table | original synthesis |
| u-h-recipes | DeepSeek-V3 EP64/PP16/ZeRO-1; Llama3 4D [uncertain] | one recipe assumed | architecture picks the cuts | f11 | SVG | papers, Oct 2026 |
| u-h-decentral | Decentralized: pipeline across the world | datacenter assumed | only cut surviving the internet | f10 | SVG | lecture |
| u-tab-history | Megatron, GPT-3, DeepSpeed contributions | toolbox timeless | 2019-2020, problem was young | f-tab-history | table | papers |
| u-h-price | Every cut taxes something: batch, interconnect, bubbles, engineer | cuts free | honest price stated | c1, c2, c3 | chapter plates | original synthesis |
| u-h-coverage | Coverage map: claim to section and line | claims scattered | traceability table | n/a - coverage unit | table | original |
| u-h-recap | 8-step recap | lesson as sequence | one-screen consolidation | c1, c2, c3 | chapter plates | original synthesis |
| u-h-qa | 8+ interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: nocookie embed + 4 links | n/a (media law unit) | video + links | n/a - media unit | video/links | papers |
| u-h-official | Official sources and further reading | n/a (sourcing unit) | sources + caveats | n/a - sourcing unit | links | lecture/docs |
| u-h-conn | Connections to other courses | n/a (sourcing unit) | cross-links | n/a - sourcing unit | prose | original |

## Prose problems noticed but NOT touched (for the coordinator)
1. The lesson states NVLink 5 at 1.8 TB/s and B200 HBM at 8 TB/s as demo measurements; the plate repeats them as lecture figures. If the builder updates the numbers, the plates follow.
2. The ring-allreduce plate's 4MB toy matches the lecture's 4 ranks x 4MB; the prose's bandwidth-independence claim is asymptotic ((n-1)/n -> 1), which the plate states.
3. l07-topology.svg's "9 trays of 8 = 72" matches the prose; the prose's "buyers with deep pockets" is editorial, not in the plate.
