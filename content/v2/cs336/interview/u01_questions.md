# U01 interview bank , questions

Closed-book. Answer keys are in `u01_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. What is the difference between a character, a code point, and a byte
in UTF-8?
B2. Why does BPE start from bytes rather than characters?
B3. What does the merge order of a BPE tokenizer determine at encode time?
B4. What is pretokenization, and why do merges never cross its boundaries?
B5. State the three roundtrip invariants.
B6. Name two ways a training/inference tokenizer mismatch can occur and
one check that catches each.

## Deep ladders (2 x 5)

L1. BPE training mechanics.
- L1.1 Define pair count on a weighted corpus.
- L1.2 Toy: compute the first two merges of {"ab": 3, "bc": 2} by hand.
- L1.3 Derive why greedy left-to-right application of merges in rank
  order is the correct encoding procedure.
- L1.4 Implement one merge-rewrite pass, state its complexity.
- L1.5 Compare BPE with WordPiece, debug an encoder that applies merges
  in discovery order after a re-sort, critique the word-boundary
  assumption, propose the code-versus-prose merge experiment.

L2. Tokenizer as production artifact.
- L2.1 Define the tokenizer config tuple.
- L2.2 Toy: show how a regex change shifts ids for "Testing 123".
- L2.3 Justify why ids are the model's true input alphabet.
- L2.4 Implement the four-class mismatch checker, state what each check
  compares.
- L2.5 Compare bundled versus registry storage, debug a passing hash
  with failing behavior, critique versioning-by-convention, propose the
  perplexity-under-mismatch experiment.

## Analytical exercises (2)

E1. A corpus has word counts {"the": 100, "then": 40, "their": 10}.
Compute the first BPE merge and its count. Then compute the fertility
(tokens per word, counting the end-of-word token) before and after that
single merge.
E2. A byte-level model and a BPE model share a 4096-token context window.
The BPE averages 1.3 tokens per English word, bytes average 4.5 per word
on the same text. How many English words fit in each window? What does
this imply for comparing the two models' "long context" claims?

## Implementation/debug task (1)

D1. The streaming encoder below emits wrong ids on some chunk splits.
Find the bug, fix it, and state the invariant the fix restores.

```
class BuggyStream:
    def __init__(self, merges):
        self.merges = merges
        self.buf = ""
        self.out = []
    def feed(self, chunk):
        self.buf += chunk
        words = self.buf.split(" ")
        for w in words[:-1]:
            self.out.extend(encode_bpe(w, self.merges))
        self.buf = words[-1]
    def flush(self):
        self.out.extend(encode_bpe(self.buf, self.merges))
        self.buf = ""
```

## Changed-constraint scenarios (2)

S1. The deployment adds a new language whose script was absent from BPE
training. Token counts per word triple. You cannot retrain the model.
Name two mitigations and the cost of each.
S2. Latency budget forces tokenization onto a thread with 1 ms per
request. The profile says the regex stage dominates. Name two changes
that keep ids identical and one change that changes ids but stays safe,
with the tradeoff of each.

## Research critique (1)

R1. A paper claims a new tokenizer "reduces tokens per word by 20
percent on English with no quality loss" but reports no multilingual
numbers and no downstream eval. List three methodological gaps and the
experiment that would close each.
