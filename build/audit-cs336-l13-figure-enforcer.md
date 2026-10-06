# FIGURE ENFORCER audit — cs336/l13-training-data
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions and the builder-stats figure count. Content gate
# issues found while enforcing are listed under "Prose problems" for the
# coordinator; they were NOT edited.

## Figure inventory

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f13-pipeline | assets/l13-pipeline.svg | SVG (kept; font stack fixed) | Pre, mid, post: large low-quality first, small high-quality last |
| f-tab-locks | inline table (NEW) | table | Two secrecy locks: competitive (lost moat) vs legal (witness list) |
| f13-crawl | assets/l13-crawl.svg | SVG (kept; font stack fixed) | Five walls shrink the crawlable web |
| f13-copyright | assets/l13-copyright.svg | SVG (kept; font stack fixed) | License or fair use: the two legal doors |
| f-tab-layers | inline table (NEW) | table | Access, copying, training: three independent layers; piracy fails at layer 2 |
| f13-commoncrawl | assets/l13-commoncrawl.svg | SVG (kept; font stack + #D9E6F5 fixed) | WARC raw, WET lossy; extraction tooling changes the tokens |
| f13-pockets | assets/l13-quality-pockets.svg | SVG (kept; font stack + #D9E6F5 fixed) | Wikipedia, GitHub, arXiv: dense sources, each with its own access pattern |
| f13-history | assets/l13-dataset-history.svg | SVG (kept; font stack + #D9E6F5 fixed) | From BookCorpus to Nemotron: the filtering arms race |
| f-tab-arms | inline table (NEW) | table | Each generation filters harder and discloses less |
| f13-rules | assets/l13-rules-vs-classifiers.svg | SVG (kept; font stack fixed) | Two filtering schools; the funnel from 240T to 3T is the game |
| f13-dclm | assets/l13-dclm-funnel.svg (NEW) | SVG lesson plate | 240T in, 3T out; the kept 1.4% beats the full pool |
| f-tab-gen-disc | inline table (NEW) | table | Generative (model T) vs discriminative (model the T/R boundary) |
| f13-stack | assets/l13-stack-commonpile.svg | SVG (kept; font stack fixed) | The Stack for code, Common Pile for the risk-averse |
| f-tab-tiers | inline table (NEW) | table | Open and audited, license-only, synthetic, secret: four tiers |
| c13-copyright | assets/l13-chap-copyright.svg (NEW) | SVG chapter plate | Three layers decide independently; caution costs the frontier |
| c13-funnel | assets/l13-chap-funnel.svg (NEW) | SVG chapter plate | Architecture is public, the recipe is not: the funnel is the moat |
| c13-sources | assets/l13-chap-sources.svg (NEW) | SVG chapter plate | Quality pockets: the web is not uniform, each pocket earns its handling |
| f-tab-mapback | existing Mapping-back table | table | Pain to fix, per section |

## Fixes applied to existing figures
- Bulk fix (all 8 existing l13 SVGs): font-family `system-ui,-apple-system,'Segoe UI',sans-serif` replaced with the spec stack `Anthropic Sans,Inter,'Source Sans 3','IBM Plex Sans',sans-serif`.
- Bulk fix (l13-commoncrawl, l13-dataset-history, l13-pipeline, l13-quality-pockets): off-palette `#D9E6F5` replaced with spec count-box `#E7F1F8` (rect fills only; no text affected).
- 4 broken webp refs replaced per the medium ladder: copyright-layers -> inline table (comparison), dataset-timeline -> inline table (comparison of generations), dclm-funnel -> SVG plate (count/merge), data-landscape -> inline table (four tiers).
- 50 superseded media-generation webp/json files for l13-l18 deleted (backed up to /tmp/webp-backup). Zero generated stills remain in these six lessons.

## Numbers verified by code (python3)
- DCLM funnel: 240 x 0.014 = 3.36T kept; 240 - 3.36 = 236.64T dropped (plate says "about 3T", "237T of junk deleted").

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | The problem: the paper says nothing about the data | data assumed public | secrecy has two locks | f-tab-locks | table | original |
| u-h-locks | Why the secrecy has two locks | secrecy unexplained | competitive lock vs legal lock | f-tab-locks | table | original |
| u-h-stages | The three-stage shape of modern training data | one undifferentiated corpus | pre, mid, post stages | f13-pipeline | SVG | Stanford |
| u-h-funnel-read | Read the stages as a funnel | stages as timeline | funnel: wide to narrow | f13-pipeline | SVG | original |
| u-h-base-gone | Why base checkpoints are disappearing | base checkpoint expected | mid-training dissolved the label | f13-pipeline | SVG | original |
| u-h-first-attempt | First attempt: train on the entire internet | "the entire internet" | crawlable subset only | f13-crawl | SVG | Stanford |
| u-h-crawler | What a crawler actually does | crawler undefined | link-graph walker defined | f13-crawl | SVG | original |
| u-h-walls | The five walls | web looks crawlable | deep web, auth, robots.txt, anti-bot, terms | f13-crawl | SVG | Stanford |
| u-h-piracy | The easy path is piracy | walls block everything | shadow libraries bypass all five | f13-crawl | SVG | Stanford |
| u-h-robots | Work the robots.txt collapse | crawlers assumed welcome | about half restrict by mid-2023 | f13-crawl | SVG | original (Consent in Crisis) |
| u-h-copyright-break | Where the easy path breaks: copyright | easy path assumed | everything is copyrighted | f13-copyright | SVG | Stanford |
| u-h-everything | Why everything is copyrighted | copyright needs registration | fixed-in-medium is enough, 75 years | f13-copyright | SVG | Stanford |
| u-h-doors | The two legal doors | one way to use works | license (CC) or fair use s.107 | f13-copyright | SVG | Stanford |
| u-h-fairuse-econ | Fair use is economics, not string matching | fair use as plagiarism test | four factors as economics test | f13-copyright | SVG | Stanford |
| u-h-authors | The Authors Guild precedent, and its limits | snippets ruling cited as blanket | 11 years, snippets only, not training | f13-copyright | SVG | Stanford |
| u-h-layers | The 2025 picture: three layers | one legal question | access, copying, training decide independently | f-tab-layers | table | Stanford |
| u-h-layers-priced | The three copyright layers, priced | layers abstract | $1.5B says the copying layer is the crime | f-tab-layers | table | Stanford |
| u-h-anthropic | Work the Anthropic arithmetic | $1.5B unpriced | about $3,000 per book at layer 2 | f-tab-layers | table | Stanford |
| u-h-tos | The terms-of-service layer | copyright only | contract law sits on top | f-tab-layers | table | Stanford |
| u-h-keyq | The key question | raw downloads are junk | what if the filtering is the model? | c13-funnel | chapter plate | original |
| u-h-funnel-beats | Why the funnel beats the architecture | architecture decides | same arch, different funnel, different model | c13-funnel | chapter plate | original |
| u-h-cc | Common Crawl: the open raw material | crawl undefined | monthly since 2007, 3-5B pages/dump, ~300B total | f13-commoncrawl | SVG | Stanford |
| u-h-warc | WARC versus WET | formats undefined | WARC raw HTTP, WET lossy text | f13-commoncrawl | SVG | Stanford |
| u-h-extract | The extraction tool matters | extraction assumed neutral | Trafilatura/Resiliparse beat stock WET | f13-commoncrawl | SVG | Stanford |
| u-h-gory | What the gory details cost | crawl as graph traversal | refresh, dedup, mirrors, dynamic URLs are the product | f13-commoncrawl | SVG | Stanford |
| u-h-pockets | Quality pockets: dense sources, special handling | web uniform | Wikipedia, GitHub, arXiv each special | f13-pockets | SVG | Stanford |
| u-h-dumptime | The dump-timing attack, worked | dumps trusted | edit before dump, revert after: poison in | f13-pockets | SVG | Stanford |
| u-h-permissive | Why permissive licenses only | code assumed free | copyleft risks license obligations | f13-pockets | SVG | Stanford |
| u-h-code-reason | Code as a reasoning corpus | code for coding only | for-loops teach iteration: math/logic gains | f13-pockets | SVG | original |
| u-h-latex | Why LaTeX source beats the PDF | PDF assumed fine | equations as text, not pixels | f13-pockets | SVG | original |
| u-h-decade | A decade of datasets: the filtering arms race | datasets as sizes | filter generations, secrecy growing | f13-history | SVG | Stanford |
| u-h-arms-read | Read the timeline as an arms race | timeline as list | harder filters, smaller keeps, more secrecy | f-tab-arms | table | original |
| u-h-one-idea | One idea per generation | generations blur | BERT to Nemotron, one insight each | f-tab-arms | table | Stanford |
| u-h-books3 | The Books3 watershed, in detail | disclosure assumed safe | naming the corpus drew the lawsuits | c13-copyright | chapter plate | Stanford |
| u-h-rules-class | Rules vs classifiers: the funnel is the lever | one filter type | rules (control) vs classifiers (learning) | f13-rules | SVG | Stanford |
| u-h-rules-mech | The rules school, mechanized | rules undefined | C4 heuristics, auditable, cannot learn | f13-rules | SVG | original |
| u-h-class-mech | The classifier school, mechanized | classifiers undefined | learn "good" from a positive set | f13-rules | SVG | original |
| u-h-combine | Why the schools combine | one school wins | rules-first coarse, classifiers fine | f13-rules | SVG | original |
| u-h-dclm-work | Work the DCLM funnel | 240T in, 1.4% kept | 3.36T out, beats the full pool | f13-dclm | SVG | paper (DataComp-LM) |
| u-h-fasttext | The fastText mechanics | fastText undefined | linear classifier on bag-of-ngrams | f13-dclm | SVG | original |
| u-h-kenlm | KenLM and the generative alternative | one filtering recipe | generative vs discriminative schools | f-tab-gen-disc | table | Stanford |
| u-h-stack | Code data and license-only data | web only | Stack (3TB permissive code), Common Pile (8TB license-only) | f13-stack | SVG | Stanford |
| u-h-issues | Issues and PRs as process data | code as artifact | issues/PRs teach the workflow | f13-stack | SVG | original |
| u-h-commonpile | Work the Common Pile result | license-only assumed fine | 8TB works: 2023-level, not Qwen | f13-stack | SVG | Stanford |
| u-h-laundering | Data laundering versus license laundering | "clean" assumed | two different taints, same defense | f13-stack | SVG | original |
| u-h-usedwhere | What is used where (who trains on what) | datasets unmapped | FineWeb/DCLM/Nemotron/Stack/Common Pile/frontier mapped | f-tab-tiers | table | original |
| u-h-econ | The economics of data: why data is the moat | inputs look equal | data is the only proprietary input | c13-funnel | chapter plate | original |
| u-h-three-inputs | The three inputs, priced | compute/architecture/data | compute priced, architecture free, data hidden | c13-funnel | chapter plate | original |
| u-h-fineweb-edu | FineWeb-Edu and the educational filter | Wikipedia-likeness assumed best | educational value beats encyclopedia shape | f-tab-tiers | table | original |
| u-h-multilingual | Multilingual data and FineWeb-2 | English assumed | per-language funnels, weaker filters | f-tab-tiers | table | original |
| u-h-longctx | Long-context data is a separate pocket | one funnel | length is the feature for mid-training | f13-pockets | SVG | original |
| u-h-unique | Unique tokens versus repeated tokens | sizes comparable | ask how many are unique | c13-funnel | chapter plate | original |
| u-h-wall | The data-wall debate | wall or no wall | wall real for best text, soft elsewhere | c13-funnel | chapter plate | original |
| u-h-pile22 | The Pile's 22 sources | Pile as one blob | 22 hand-picked domains | f13-history | SVG | Stanford |
| u-h-redpajama | RedPajama and the LLaMA replication | mixture secret | open replication of the mixture | f-tab-tiers | table | original |
| u-h-dolma | Dolma and the fully-open pipeline | open data assumed | data plus code plus model, 3T | f-tab-tiers | table | original |
| u-h-multicorp | Multilingual web corpora | English-only funnel | MADLAD-400, CulturaX, HPLT | f-tab-tiers | table | original |
| u-h-langid | Language ID at scale | language assumed known | 176-language classifier, threshold drops | f-tab-tiers | table | original |
| u-h-phi | The phi result (textbooks are all you need) | scale assumed | small excellent text beats large average | f-tab-tiers | table | original |
| u-h-cosmopedia | Cosmopedia and synthetic textbooks at scale | synthetic assumed unbounded | reasoning supply, not knowledge supply | f-tab-tiers | table | original |
| u-h-perdomain | Quality is per-domain | universal quality | funnel's positive set encodes the domain | f-tab-gen-disc | table | original |
| u-h-ccindex | The Common Crawl index | dumps only | columnar index: query, fetch records | f13-commoncrawl | SVG | original |
| u-h-wet-traf | WET versus Trafilatura, worked | extraction equivalent | same page, different tokens, different model | f13-commoncrawl | SVG | original |
| u-h-funnel-cost | The compute cost of the funnel | funnel assumed free | fastText cheap, LLM-judge expensive | c13-funnel | chapter plate | original |
| u-h-dpi | The Data Provenance Initiative | licenses assumed | per-work license auditing | c13-copyright | chapter plate | original |
| u-h-eu | Opt-out and the EU exception | US law only | EU text-and-data-mining exception plus opt-out | c13-copyright | chapter plate | original |
| u-h-curriculum | Ordering and curriculum | order matters | mixed evidence; decay phase is the exception | c13-funnel | chapter plate | original |
| u-h-mapback | Mapping back: what each idea fixes | pains unmapped | ten pains mapped to fixes | f-tab-mapback | table | original |
| u-h-price | The honest price | funnel free | law evolving, sizes mix unique/repeated | c13-copyright | chapter plate | original |
| u-h-recap | Recap: the whole lesson on one screen | story scattered | ten steps, each answering the one before | f-tab-mapback | table | original |

Blank cells: zero.

## Remap notes (fix round, 2026-10-06, honest mappings)
- `u-h-piracy` remapped `c13-copyright` -> `f13-crawl`. The old mapping
  was wrong: the chapter plate's claim (three layers decide
  independently) belongs to the copyright section, ~8 subchapters away.
  `f13-crawl` is the proximal figure (used by the adjacent
  `u-h-walls` and `u-h-robots`): it shows the five walls the piracy
  path bypasses, and the shadow-library bypass itself is named in the
  subchapter's prose under the same figure.
- `u-h-pile22` kept on `f13-history` with a note. The timeline shows
  the Pile's 2020 position and its watershed role (Books3) but cannot
  illustrate the 22 domains; the domain enumeration is prose-only and
  no proximal figure shows composition. Not blank: the figure cell
  still carries the Pile's story arc.

## Prose problems noticed, NOT touched (for the coordinator)
1. INTERNAL ARITHMETIC INCONSISTENCY (l13, "work the Anthropic arithmetic" subchapter and the "2025 picture" section): the prose says "$1.5 billion over about 7 million books: roughly $3,000 per book." 1.5e9 / 7e6 = $214 per book, not $3,000. $1.5B / $3,000 = 500,000 books. The two quoted numbers cannot both be right. The chapter plate (c13-copyright) repeats the lesson's "$1.5B, about $3,000 per book" without the 7M figure and flags the inconsistency on the plate. Builder/auditor should resolve which number the lecture actually stated.
2. Builder-stats figure count was already wrong before my pass ("7 SVG refs + 5 generated plates" = 12 claimed, but the file had 8 SVG refs + 4 webp refs = 12); updated to the true post-enforcement count: 15 (12 SVG plates: 8 kept + 1 new lesson + 3 new chapter; 3 new inline tables).
3. The "75 years" copyright duration is the lecture's telling and is self-hedged in prose ("the exact term varies by work type and date").
