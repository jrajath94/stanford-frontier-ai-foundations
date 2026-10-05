---
page_id: cs336-labs-index
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 90
nav: "CS336 · Labs"
title: "CS336 Labs: Interview Prep"
summary: "Hands-on exercises drawn from the five course assignments, framed for frontier-lab interviews."
sources:
  - tag: assignment
    label: "CS336 assignments 1-5 (PDFs ship with this system)"
---

These labs distill the five CS336 assignments into interview-sized exercises. Each one targets a skill that frontier labs actually test: implement from scratch, reason about systems, and quantify tradeoffs. Do them with a timer. Explain your reasoning out loud.

<div class="cards">
<a class="card" href="lab1-tokenizer-transformer.html">
<span class="kicker">Lab 1</span>
<h3>Tokenizer and Transformer from Scratch</h3>
<p>BPE training and encoding, attention, the full forward pass. The classic "implement it" interview.</p>
</a>
<a class="card" href="lab2-systems.html">
<span class="kicker">Lab 2</span>
<h3>Systems: Kernels and Parallelism</h3>
<p>Roofline analysis, MFU estimation, choosing a parallelism strategy. The systems-design interview.</p>
</a>
<a class="card" href="lab3-scaling.html">
<span class="kicker">Lab 3</span>
<h3>Scaling Laws</h3>
<p>Fit a scaling law from training curves, predict the compute-optimal model. The research-taste interview.</p>
</a>
<a class="card" href="lab4-data.html">
<span class="kicker">Lab 4</span>
<h3>Data Pipelines</h3>
<p>MinHash deduplication, filtering decisions. The practical ML interview.</p>
</a>
<a class="card" href="lab5-post-training.html">
<span class="kicker">Lab 5</span>
<h3>Post-Training: SFT, DPO, RLVR</h3>
<p>Implement the DPO loss, derive the policy gradient. The alignment interview.</p>
</a>
</div>

## How to use these

1. Read the linked lessons first. The labs assume that knowledge.
2. Write code from memory, not by copying the lesson. Struggle is the point.
3. Time yourself: 45 minutes per lab, matching interview length.
4. After each lab, write down what you got wrong. That list is your real study guide.
