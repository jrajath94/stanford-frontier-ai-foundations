# FIGURE ENFORCER audit — mse435/l01-electrons-to-tokens
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched except
# figure captions (F1, F4 number corrections) and two added inline figures.

## Figure inventory (13 plates + inline figures)

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f02 | assets/plate-l01-cobb-douglas.svg | SVG lesson plate | Labor was the only term that could not move on an investment horizon; now it can |
| f03 | assets/plate-l01-digital-labor.svg | SVG lesson plate | Traditional labor waits on the birth rate; digital labor scales with CapEx |
| f04 | assets/plate-l01-abilene.svg | SVG lesson plate | Stranded power is a location decision: build where the electrons already are |
| f05 | assets/plate-l01-bottleneck.svg | SVG lesson plate | The bottleneck moves (chips, memory, energized shells); labor binds underneath |
| f06 | assets/plate-l01-power-path.svg | SVG lesson plate (NEW) | Each stage trades voltage for current; backup is sized to what must survive |
| f07 | assets/plate-l01-cooling-path.svg | SVG lesson plate (NEW) | Chips turn power into heat; the loop carries it to air and brings cold water back |
| f08 | assets/plate-l01-h100.svg | SVG lesson plate (NEW) | The rental market priced obsolescence fear first, then agent demand repriced it |
| f09 | assets/plate-l01-services.svg | SVG lesson plate (NEW) | The factory owner who sells the service captures the rent and the margin |
| c1 | assets/plate-l01-chap-digitallabor.svg | SVG chapter plate (NEW) | Digital labor turns the capped labor term investable; factories are the vehicle |
| c2 | assets/plate-l01-chap-energy.svg | SVG chapter plate (NEW) | Energy-first won the sites; power availability, not chips, set their size |
| c3 | assets/plate-l01-chap-bottleneck.svg | SVG chapter plate (NEW) | The binding constraint moves; the hedge is owning the layers it can move across |
| c4 | assets/plate-l01-chap-megawatt.svg | SVG chapter plate (NEW) | The megawatt is the unit price of the whole business |
| f-eq-prodfn | inline ascii equation block | equation | AI = data + algorithms + compute + energy + data centers |
| f-eq-cobb | inline ascii equation block | equation | GDP growth = change in labor + change in capital + change in technology |
| f-eq-cobbformal | inline prose equation | equation | Formal multiplicative Cobb-Douglas agrees with the guest's point |
| f-code-toy | inline ascii block | ASCII | 3.0% vs 3.5% compounding: $1.34 vs $1.41 per $1 in ten years |
| f-code-payback | inline ascii block | ASCII | 60 / 15 = about 4 years payback on a revenue basis |
| f-mer-quan | inline mermaid block | mermaid | Across-the-meter: wind farm feeds campus first, grid firms the rest |
| f-ascii-volt | inline ascii block (NEW) | ASCII | Voltage ladder: 345 kV AC down to 900 V DC; every rung is old tech |
| f-tab-capex | inline table | table | 2026 hyperscaler guidance: ~$730B combined, +78% YoY |
| f-tab-ingredient | inline table | table | Ingredient plain meanings, economic roles, and per-MW prices (spending concentrates on the last three) |
| f-tab-abilene | inline table | table | Abilene since the session: capped at 1.2 GW, 450,000 GB200s |
| f-tab-neocloud | inline table | table | Neocloud scale, October 2026 |
| f-tab-water | inline table | table | Water: ~1M gallons in the loop, filled once, household-scale use |
| f-tab-20m | inline table | table | $20M/MW building stack, layer by layer |
| f-tab-40m | inline table | table | $40M/MW machine stack, layer by layer |
| f-tab-spark | inline table | table | Stick-built campus vs Crusoe Spark |
| f-tab-commodity | inline table | table | Compute commoditization three-way verdict |
| f-tab-breaks | inline table | table | Payback sensitivity: $30M to $7.5M revenue rows |
| f-tab-lifelength | inline table | table | Useful-life views: books 5-6 yrs, Research Affiliates ~3, guest "as long as it earns" |
| f-tab-space | inline table | table | What vanishes and what stays in space |
| f-tab-advice | inline table | table | Crusoe values: mountaineer thinking, infinite growth loop |
| f-tab-cooling | inline table (NEW) | table | Cooling method decision rule by rack power band |
| f-tab-coveragemap | inline table | table | Every session claim mapped to section and file line |

## Fixes applied to existing figures
- f01 SUPERSEDED in fix round 1 (see below): the plate file is deleted; the unit now maps to f-tab-ingredient (table). The old fixes (machinery sub-label, bar rescale, caption wording, font stack) are moot.
- f03 SUPERSEDED in fix round 1 (see below): the webp is deleted, redrawn as a flat SVG lesson plate per the spec.
- f02: "growth 3.0% > 3.5%" corrected to "growth 3.0% -> 3.5%"; third top box widened 272->280 so all three peer boxes are equal width. Font stack reordered.
- f04: off-palette #F6E7A8 replaced with spec #F4E6D4; footer cut from two sentences to the one claim (eight-buildings/gas-plant facts moved to the md caption); "2.1 GW" box corrected to "1.2 GW delivered / of 2.1 GW planned; grid capped it" per the October 2026 update; bottom-row boxes equalized to 280px; md caption extended with the cap and the moved facts. Font stack reordered.
- f05: timeline boxes equalized (was 200/200/200/108; now 4x192 with equal gaps). Font stack reordered.

## Fix round 1 (figure auditor FAIL, F2 medium ladder, 2026-10-06)
1. f01 plate-l01-production-function.svg DELETED. The claim ("Five inputs make AI; spending concentrates on the last three") is a comparison of values, not a move: the ladder mandates a table. The unit now points at the existing ingredient table f-tab-ingredient; a `$/MW` column was added (data/algorithms: no per-MW price; compute: about $40M; energy and data centers: about $20M combined), matching the lesson prose ("about $20M per MW for the factory" + "about $40M per MW for the machines inside"). The `![]()` reference and its caption were removed from the md; nothing else in the prose was touched.
2. f03 plate-l01-digital-labor.webp DELETED and redrawn as plate-l01-digital-labor.svg: flat SVG lesson plate in the sibling style (warm paper #F7F4EE, Anthropic Sans first, 8px grid, one claim). Left = before (traditional labor: workers=headcount, capped by birth rate, ~20-year lead time); center = the named rule (digital labor scales with CapEx; task -> agent -> tokens; tokens cost power, chips, buildings; the wage becomes a CapEx line); right = after (digital labor: labor from CapEx, lead time quarters, moves when money moves). All numbers from the lesson. Caption names Shell 3 and the source. crash-course.md reference updated (webp -> svg); prose untouched.

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-chart | Hyperscaler AI CapEx is one of the largest investments ever, second only to US defense | no scale anchor for the $650B | $650B framed vs space program, highways, Manhattan Project | f-tab-capex | table | original (company guidance) |
| u-h-chart-oct | October 2026 guidance steepens the chart: ~$730B combined, +78% YoY, 5.2x 2023 | spring $650B host estimate | ~$730B guided; 2027 near $1.2T (Goldman analyst estimate) | f-tab-capex | table | original (company guidance) |
| u-def-capex | CapEx = money to build long-lived assets | term undefined | defined; anchors the whole chart | f-tab-capex | table | original |
| u-def-hyperscaler | Hyperscaler = Amazon, Alphabet, Meta, Microsoft, Oracle | term undefined | the five named | f-tab-capex | table | original |
| u-def-megawatt | Megawatt = 1M watts draw; the unit price of the business | term undefined | "per seat" of the data center business | f-tab-ingredient | table | original |
| u-h-prodfn | AI needs five inputs; money lands on the last three | "what does it take to produce AI?" | data, algorithms, compute, energy, data centers | f-tab-ingredient | table | original |
| u-eq-prodfn | AI = data + algorithms + compute + energy + data centers | prose list | the recipe as one equation | f-eq-prodfn | equation | original |
| u-tab-ingredient | Each ingredient's plain meaning and economic role | five bare nouns | raw material / process tech / machinery / fuel / factory | f-tab-ingredient | table | original |
| u-def-nn | Neural network = layers of simple math units learning from examples | term undefined | defined at first use | f-tab-ingredient | table | original |
| u-def-training | Training = one giant run setting the model's numbers | term undefined | defined at first use | f-tab-ingredient | table | original |
| u-def-backprop | Backpropagation = error at output pushed back as corrections | term undefined | defined at first use | f-tab-ingredient | table | original |
| u-def-transformer | Transformer = architecture behind modern language models | term undefined | defined at first use | f-tab-ingredient | table | original |
| u-h-prodfn-data | Data and algorithms do not absorb factory-scale CapEx | five inputs look equal | money flows to inputs that gate output | f-tab-ingredient | table | original |
| u-h-prodfn-money | Last three inputs bought by the megawatt: ~$20M factory, ~$40M machines | no unit price | the megawatt as unit price | f-tab-ingredient | table | original |
| u-h-growth | Cobb-Douglas splits GDP growth into labor + capital + technology | why is the spend rational? | a supercycle moves one term hard | f02 | SVG plate | original toy |
| u-eq-cobb | GDP growth = change in labor + change in capital + change in technology | prose | the guest's additive form | f-eq-cobb | equation | original |
| u-eq-cobbformal | Formal multiplicative form agrees; labor share ~0.6 in the US | intuitive form only | textbook background recorded | f-eq-cobbformal | equation | original |
| u-def-supercycle | Supercycle = long investment wave from a general-purpose technology | term undefined | PC, internet, mobile, now AI | f02 | SVG plate | original |
| u-h-growth-toy | 3% toy: buying 0.5 pt of labor growth = 5% larger economy in 10 yrs | 3.0% split three ways, labor capped | 3.5%; $1.34 vs $1.41 per $1 | f-code-toy | ASCII | original toy |
| u-h-growth-slow | Labor was the slow term: birth rate, 20-year lead time | three terms look symmetric | labor fixed on any planning horizon | f02 | SVG plate | original |
| u-h-digilab | Digital labor: the labor term becomes an investment decision | labor moves only via birth rate | agents doing real work = labor from CapEx | f03 | SVG plate | original |
| u-def-token | Token = basic unit of text (~3/4 word); every token costs factories | term undefined | defined; cost chain attached | f03 | SVG plate | original |
| u-h-digilab-mech | Mechanism: task -> agent -> tokens -> power/chips/buildings -> CapEx line | thesis stated | three steps; the wage becomes a CapEx line | f03 | SVG plate | original |
| u-h-digilab-replace | Substitutes routine cognitive work; not presence, trust, judgment | "does it replace everything?" | bar is low: a slice moves the term | f03 | SVG plate | original |
| u-fig-gs-token | Goldman: token consumption ~24-fold by 2030, ~120 quadrillion/month | thesis without demand evidence | enterprise agents drive demand; unbalanced to H2 2027 | c1 | chapter plate | original synthesis |
| u-h-energy | Rank inputs by scarcity; move the factory to the scarcest one | thesis without strategy | decision rule stated | c2 | chapter plate | original synthesis |
| u-h-energy-hubs | Hubs filled first: Northern Virginia crowded and pricey | why not build in hubs? | crowded trade; energy mispriced by location | c2 | chapter plate | original synthesis |
| u-def-inference | Inference = running the trained model for users, billions of times | term undefined | distinguished from one-time training | c2 | chapter plate | original |
| u-def-pow | Proof-of-work = burning energy to validate crypto transactions | term undefined | crypto also turns electricity into output | c2 | chapter plate | original |
| u-h-energy-inversion | Inversion: move computers to energy, not energy to computers | default plan (build near users) | fiber cheap, electrons expensive | f04 | SVG plate | original |
| u-h-energy-abilene-arith | Abilene: tax credits -> overbuild -> negative prices -> build here | stranded power abstract | arithmetic worked; first buildings June 2024 | f04 | SVG plate | original |
| u-h-energy-abilene-sub | 200 MW + 1 GW substations; 2.1 GW campus = two Denvers | no scale anchor | largest private US substation | f04 | SVG plate | original |
| u-h-energy-abilene-cluster | One coherent cluster: one training job across all halls | campuses are separate buildings | one machine with eight halls | f04 | SVG plate | original |
| u-h-energy-abilene-work | 9,000 workers on site vs town of 120,000; 2,000 steady staff | no human scale | construction army + permanent jobs | f04 | SVG plate | original |
| u-h-energy-abilene-gas | 350 MW on-site gas plant: make power when the grid cannot | grid-only assumed | campus makes its own power | f04 | SVG plate | original |
| u-noun-gasplant | gas plant (architecture noun) | undefined | on-site generation behind the meter | f04 | SVG plate | original |
| u-h-energy-abilene-oct | Capped at 1.2 GW; 450,000 GB200s; $15B Crusoe/Blue Owl; Nvidia $150M deposit | 2.1 GW plan | 1.2 GW delivered; 43% died in grid delays | f-tab-abilene | table | original (press) |
| u-def-meter | Meter = grid interconnection boundary; across-the-meter defined | term undefined | campus as power plant that computes | f-mer-quan | mermaid | original |
| u-h-energy-quan | Quan, Texas: 3,500 workers in a town of 1,500; wind behind the meter | Abilene looks one-off | playbook repeats | f-mer-quan | mermaid | original |
| u-h-energy-usedwhere | Energy-first is now the industry default (Ohio 8 GW, neoclouds) | one company's playbook | national pattern; Nvidia as financier | f-tab-neocloud | table | original (earnings/press) |
| u-noun-neocloud | Neocloud = GPU-first cloud selling compute | term undefined | CoreWeave, Nebius, Lambda, Crusoe Cloud | f-tab-neocloud | table | original |
| u-h-bottleneck | Binding constraint moves: chips -> memory -> energized shells; labor underneath | constraint looks fixed | invest in this year's gate; last year's is sunk | f05 | SVG plate | original |
| u-def-binding | Binding constraint = the one input gating output right now | term undefined | more of everything else changes nothing | f05 | SVG plate | original |
| u-h-bottleneck-chips | 2023: H100 lead times 36-52 weeks, ~$30,000/card; eased by 2024 | chips gate everything | that phase is over | f05 | SVG plate | original |
| u-h-bottleneck-mem | HBM3E $300->$500 contract, $2,100 spot; SK hynix +250% | chips solved | constraint moved inside the machine | f05 | SVG plate | original |
| u-h-bottleneck-shells | Today: energized shells; switchgear, chillers, generation gear | chips easy to get | scarce thing is a place to turn them on | f05 | SVG plate | original |
| u-def-powershell | Power shell = energized data center ready for chips | term undefined | building + substation + cooling | f05 | SVG plate | original |
| u-h-bottleneck-labor | Labor: $4.7M/MW capitalized; $4.7B per GW | labor as footnote | bottleneck in its own right | f05 | SVG plate | original |
| u-h-bottleneck-vi | Vertical integration: energy to managed services; not chips, not models | single-layer hostage | retools when the gate moves | f05 | SVG plate | original |
| u-h-factory | Anatomy of a megawatt: $20M factory + $40M machines | $60M asserted | the walk, system by system | c4 | chapter plate | original synthesis |
| u-noun-substation | Substation: high-voltage grid power stepped down | term undefined | first stop on the power path | f06 | SVG plate | original |
| u-h-factory-pdc | Power distribution centers take 34.5 kV to equipment rows | term undefined | first stop after the substation | f06 | SVG plate | original |
| u-h-factory-xfmr | Transformers: 34.5 kV -> 480/415 V | term undefined | voltage for current at constant power | f06 | SVG plate | original |
| u-h-factory-switch | Switchgear: industrial breakers route and protect | term undefined | home panel at city scale | f06 | SVG plate | original |
| u-h-factory-ups | UPS batteries cover the seconds to generators | term undefined | a flicker never becomes an outage | f06 | SVG plate | original |
| u-h-factory-diesel | Diesel generators: last resort, core network + storage only | term undefined | backup sized to what must survive | f06 | SVG plate | original |
| u-h-factory-5nines | Five nines = 99.999%, about 5 min/yr; bought for storage + networking | term undefined | the sizing decision rule | f06 | SVG plate | original |
| u-h-factory-chillers | Chillers: cold water loop; air practical to ~15-25 kW/rack | term undefined | heat: chip -> water -> air | f07 | SVG plate | original |
| u-h-factory-cdu | CDUs bridge building water loop to rack plumbing | term undefined | last step of the chilled loop | f07 | SVG plate | original |
| u-h-factory-d2c | Direct-to-chip: 100-150 kW/rack; NVL72 ~120 kW; PUE 1.05-1.15 | air wall at ~20-50 kW | cold plate per GPU; ~55% of liquid deployments | f-tab-cooling | table | original (industry analyses) |
| u-h-factory-immersion | Immersion: 200-250+ kW/rack demonstrated; PUE 1.02; PFAS stalled two-phase | cold plates struggle ~175-200 kW | tanks + dielectric fluid; greenfield only | f-tab-cooling | table | original |
| u-h-factory-hotaisle | Hot aisle containment keeps exhaust air from supply air | term undefined | no barrier = chillers work harder | f07 | SVG plate | original |
| u-h-factory-fanwalls | Fan walls move air through the data hall | term undefined | containment sets lanes, fans drive traffic | f07 | SVG plate | original |
| u-h-factory-rpp | Remote power panels: last distribution before the racks | term undefined | tail end of the power path | f06 | SVG plate | original |
| u-h-factory-water | Water: ~1M gallons in the loop, filled once; use ~= one household/yr | "data centers drain water" | volume is not consumption | f-tab-water | table | original (session) |
| u-h-factory-steel | Steel, concrete, hands: batch plant, 24-hr pours; the $4.7M/MW line | buildings abstract | physical build made concrete | c4 | chapter plate | original synthesis |
| u-h-factory-20m | $20M/MW building stack: labor 4.7, gas 2-3, fit-out 3, electrical 3.5-4.5, mechanical 2-3, materials 1.5-2.5, soft 1-2 | $20M asserted | every layer priced | f-tab-20m | table | original (session + worked estimates) |
| u-def-opex | OpEx = money to run the asset each year | term undefined | distinguished from CapEx | f-tab-20m | table | original |
| u-def-fitout | Tenant fit-out = finishing the shell (RPPs, containment, fan walls, CDUs) | term undefined | defined | f-tab-20m | table | original |
| u-h-factory-racknet | NVLink in-rack (72 GPUs); InfiniBand/RoCE between racks | term undefined | two networks; where the $4M goes | f-tab-40m | table | original |
| u-def-nvlink | NVLink = Nvidia GPU-to-GPU interconnect | term undefined | defined | f-tab-40m | table | original |
| u-def-ib-roce | InfiniBand / RoCE / RDMA = back-end fabric between racks | term undefined | defined | f-tab-40m | table | original |
| u-h-factory-40m | $40M/MW machine stack: GPUs 30, network 4, CPU+storage 3, fit-out 3, deploy 1 | $40M asserted | every layer priced; sums ~$41M | f-tab-40m | table | original (session) |
| u-h-factory-cpu | CPU shortage from agentic loops: sequential CPU orchestration work | CPU line unexplained | inference pulls CPUs as training pulled GPUs | f-tab-40m | table | original |
| u-h-factory-spark | Crusoe Spark: 500 kW air / 2 MW liquid modular units; 30-50% savings | gigawatt-or-nothing | factory-built, weeks to deploy, incremental | f-tab-spark | table | original (session) |
| u-h-payback | Does $60M/MW earn? Three numbers and one chart | cost known, return unknown | the payback question framed | f09 | SVG plate | original |
| u-h-payback-rev | Revenue math: $60M capex, $15M/yr, $1-2M opex -> ~4-yr payback | no return math | 60/15 = about 4 years, revenue basis | f-code-payback | ASCII | original |
| u-def-rma | RMA = formal process for returning failed hardware | term undefined | defined | f-code-payback | ASCII | original |
| u-h-payback-h100 | H100 chart: prices fell, then agent demand drove them past debut | new chips kill old | old compute regained value | f08 | SVG plate | session slide |
| u-h-payback-spot | Spot pricing = live rental price vs contract rate | term undefined | the rebound is the live market | f08 | SVG plate | session slide |
| u-h-payback-depr | Depreciation: $60k server, 6-yr life -> $10k/yr; books use 5-6 yrs | term undefined | useful life is the swing variable | f09 | SVG plate | original |
| u-h-payback-cloud | Crusoe Cloud abstracts the chip: buy tokens, not chip models | hardware identity matters | Zoom analogy; longer-life bet | f09 | SVG plate | original |
| u-def-k8s | Kubernetes = open-source container scheduling across fleets | term undefined | defined | f09 | SVG plate | original |
| u-h-payback-services | Services uplift: +$5-15M/MW/yr -> $30M optimistic -> 2-yr payback | 4-year payback | rent + margin halves payback | f09 | SVG plate | original |
| u-h-payback-commodity | Three-way verdict: old commoditizes, newest premiums, scale moats | "is compute a commodity?" | priced per horizon; Nvidia margin 80%->~60% | f-tab-commodity | table | original |
| u-def-grossmargin | Gross margin = (revenue - COGS) / revenue | term undefined | defined | f-tab-commodity | table | original |
| u-h-payback-breaks | Payback sensitivity: $30M->2yr, $15M->4yr, $10M->6yr, $7.5M->8yr | base case only | rows slide with token prices | f-tab-breaks | table | original |
| u-h-bears | Falsifier: token revenue below ~$7.5M/yr -> 8-yr payback breaks thesis | bull case only | the sharp falsifier stated | f-tab-breaks | table | original |
| u-h-bears-3yr | Research Affiliates: ~3-yr economic life; $125B net of $650B at 2-yr life; Burry $176B | books say 5-6 years | replacement treadmill quantified | f-tab-lifelength | table | original (paper/press) |
| u-h-bears-volt | Voltage ladder: 345 kV (soon 765 kV) down to 900 V DC at the rack | power path static | every rung is old tech | f-ascii-volt | ASCII | original |
| u-h-bears-sst | Solid-state transformers: power electronics replace iron and copper | century-old ladder | cost of stepping down falls | f-ascii-volt | ASCII | original |
| u-h-bears-incumb | Incumbents (Eaton, Schneider): near-term partners, long-term disrupted | term undefined | partner for build, bet disruptors for decade | f-ascii-volt | ASCII | original |
| u-h-bears-oss | Open source takes share: compresses token pricing power | revenue assumed | payback rows slide down | f-tab-breaks | table | original |
| u-h-bears-starcloud | Starcloud: H100s in orbit; Starcloud-2 slipped to 2027 rideshares; 88k FCC filing | term undefined | orbital compute timeline | f-tab-space | table | original (press) |
| u-def-blackwell | Blackwell B200 = Nvidia's Blackwell-generation flagship AI chip | term undefined | defined | f-tab-space | table | original |
| u-h-bears-vanish | Vanishes vs stays in space: concrete/permitting/fiber/power go; heat/ops/launch/failure stay | space as escape | removes scarcest inputs, keeps hardest problems | f-tab-space | table | original |
| u-h-bears-suncatcher | Suncatcher: Trillium TPUs, 81-satellite formation; prototype Oct 1 2026 | term undefined | Google's in-house silicon bets the orbital round | f-tab-space | table | original (Google) |
| u-def-tpu | TPU = Google's Tensor Processing Unit; Trillium = the generation | term undefined | defined | f-tab-space | table | original |
| u-h-bears-launch | Launch scarcity: Falcon 9 rideshares booked past 2028; $250M at $2.3B | term undefined | not material in 5-10 years | f-tab-space | table | original (press) |
| u-h-bears-advice | Student advice: mountaineer thinking; infinite growth loop; learning speed compounds | term undefined | invest in learning rate, not syllabus | f-tab-advice | table | original |
| u-h-coveragemap | Coverage map: every session claim -> section + file line | claims scattered | full traceability table | f-tab-coveragemap | table | original |
| u-h-recap | 12-point recap of the whole lesson | lesson as sequence | one-screen consolidation | c1, c2, c3, c4 | chapter plates | original synthesis |
| u-h-qa | 8 interview Q&A blocks with follow-ups | n/a (G6 assessment unit) | figures inherited from their sections | n/a - G6 unit | Q&A | original |
| u-h-godeeper | Go deeper: 2 nocookie embeds + 9 verified links | n/a (media law unit) | video + links | n/a - media unit | video/links | session/press |
| u-h-official | Official sources and further reading with caveats | n/a (sourcing unit) | sources + honesty caveats | n/a - sourcing unit | links | session/course site |
| u-h-coverage | Coverage and sourcing note (no transcript; Oct 2026 updates marked) | n/a (sourcing unit) | sourcing ground rules | n/a - sourcing unit | prose | session |

## Medium-ladder justification for new figures
- f06 power path: table no (claim is a staged transformation, not a value comparison); equation no; ASCII borderline (6 stages + UPS/diesel/five-nines exceed a clean trace; stage order is positional from the campus walk); mermaid could order the stages but cannot carry the before/after voltage states with the named center rule; SVG is the first medium that fully passes. Lesson plate per the change-unit rule.
- f07 cooling loop: table no; equation no; ASCII linear trace would misrepresent the recirculation, which is the claim's core; SVG passes first.
- f08 H100 curve: table no (exact levels not in source, shape is the claim); equation no; ASCII sparkline too crude for three labeled regimes; SVG passes first.
- f09 services uplift: a table could compare the values, but the unit is a state change (4yr -> 2yr via the named rule "add the managed services layer"); the spec prescribes the lesson plate for state changes. SVG.
- f-tab-cooling: comparison of values by rack-power band -> table passes first. Added as a table, not a plate.
- f-ascii-volt: one trace, before/after lines, 6 lines -> ASCII passes first. Added as ascii block, not a plate.
- c1-c4: chapter plates are mandated extras by the spec (dense, end of concept); SVG keeps text crisp and matches the site's plate system. No generated images were needed anywhere: zero media.generate_image calls, zero API keys touched.

## Prose problems noticed but NOT touched (for the coordinator)
1. F1's in-SVG subtitle still says "Bar widths scale to the session's per-megawatt stack" only in the md caption (fixed there); the SVG's own subtitle ("Five inputs make AI. Spending concentrates on the last three.") is fine.
2. The lesson references "Project: Stanford Frontier AI." in every plate caption; harmless.
3. u-h-bears-starcloud / suncatcher / launch map to f-tab-space (the vanishes/stays table) rather than dedicated plates; a dedicated orbital plate would fail the four tests (no reusable symbol, mostly a list of dated facts), so the table mapping is the honest call.
4. (RESOLVED in fix round 1, 2026-10-06) The digital-labor webp was flagged for the figure auditor's judgment; it failed F2 (medium ladder) and was redrawn as a flat SVG lesson plate. The person-glyph concern is moot: the new plate uses no glyphs, only labeled boxes, lines, and bars.
5. In C4, the "worked estimate" muted note sits under the electrical row only, though mechanical/materials/soft rows are also worked estimates per the lesson table. Cosmetic; the lesson table itself carries the per-row marks.
