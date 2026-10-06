---
page_id: cs229-l01
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 1
nav: "L01 · Introduction"
title: "Lecture 1: What Machine Learning Is"
summary: "Two classic definitions of machine learning, the three learning paradigms, and the map of the whole course."
date: "2026-04-06"
instructor: "Tengyu Ma"
offering: "Spring 2026"
duration: "36:47"
video_id: DATnpGoGhM8
video_title: "Lecture 1: Introduction"
video_caption: "Original lecture. Tengyu Ma defines machine learning, surveys the three paradigms, and maps the course."
concepts: [machine-learning, supervised-learning, unsupervised-learning, reinforcement-learning, Samuel, Mitchell]
sources:
  - tag: video
    label: "Lecture 1 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=DATnpGoGhM8
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: a filter that survives Monday

In 2004, a spam filter at a large email company had a bad Monday. The
weekend's filters worked fine. By Monday noon, inboxes were full of
junk. The spammers had changed one word: "free" became "fr33". Every
hand-written rule that matched the word "free" went blind at once.

This is the job machine learning was invented for. A **task** is work
you want a computer to do: sort spam from real mail, price a house,
transcribe speech. The old way to do the task is explicit
programming: a human writes the rules. "If the subject contains
'free' and the sender is unknown, mark as spam." It works until the
world changes, which it does every Monday.

The first attempt at spam filtering was exactly this: rule lists.
Build one by hand. Watch it fail. Count the cost. A team of five
engineers wrote about 1,000 rules. Each rule took roughly 30 minutes
to write, test, and deploy. That is 500 engineer-hours. The rules held
for about two weeks. Spammers probe the filter, learn which words it
checks, and invent new ones. "Free" becomes "fr33" becomes "f.r.e.e"
becomes "complimentary". Each wave forces a new round of rule
writing. The arithmetic is brutal: the humans add rules at 10 per
day. The spammers generate evasions at 100 per day. Hand-written
rules lose by an order of magnitude, forever.

![Three learning paradigms](assets/svg/l01-paradigms.svg "The three paradigms of machine learning. Supervised learns from labeled examples, unsupervised finds structure in unlabeled data, reinforcement learns from trial and error. Source: original plate for Stanford Frontier AI.")

Here is the key question. What if the machine writes the rules
itself, from examples? Nobody writes a rule about "fr33". Instead,
you show the machine 10,000 emails that humans already labeled spam
or not spam. The machine finds the patterns on its own. When
spammers switch to "f.r.e.e", you do not rewrite code. You feed the
machine this week's labeled emails and it adjusts. The examples do
the teaching. The human never enumerates the rules.

That sentence is the whole field. **Machine learning** is the study
of algorithms that improve at a task from experience instead of from
hand-written instructions. Two classic definitions make this precise,
and both still hold after decades.

Arthur Samuel wrote the first one in 1959, before the term "software"
was common. He said machine learning is "the field of study that
gives computers the ability to learn without being explicitly
programmed." The key phrase is "without being explicitly programmed".
Nobody programmed the "fr33" rule. The machine learned it from data.

Tom Mitchell wrote the second one in 1997, and it is sharper because
it names the three moving parts. "A computer program is said to learn
from experience E with respect to some class of tasks T and
performance measure P, if its performance at tasks T, as measured by
P, improves with experience E."

Unpack it with the spam filter. The task T is "sort email into spam
and not spam". The experience E is 10,000 labeled emails. The
performance measure P is accuracy: the fraction of emails sorted
correctly. Learning happened if accuracy on new mail rises as the
machine sees more labeled examples. Mitchell's definition turns a
vague hope into a testable claim. Measure P before and after E. If P
went up, the machine learned.

Notice how broad "experience" is. It covers human-labeled data, but
also web text, sensor readings, synthetic data a model generates for
itself, and the sequence of rewards and punishments in a game. The
lecture stresses this breadth because modern models drink from all of
these sources at once.

## The three paradigms

Every learning algorithm in this course fits one of three setups.
The setups differ in what the experience looks like.

**Supervised learning** is learning from labeled examples. Each
experience is a pair: the input and the correct answer. For spam:
("Win fr33 money now!!!", spam). For house prices: (3 bedrooms, 2
baths, 1,800 sq ft. Sold for $812,000). The machine's job is to
predict the answer for new inputs it has never seen. Most of this
course is supervised learning. Lectures 2 through 8 build it from
zero.

**Unsupervised learning** is learning from unlabeled data. Nobody
tells the machine the right answer. The machine finds structure on
its own: which items group together, which patterns repeat. Give it
one million emails with no labels and it might discover that
"pharmacy" emails form one cluster and "invoice" emails form
another. The answers are not given, so the machine invents its own
categories. Lectures 9 and 10 cover this: clustering with k-means and
Gaussian mixtures, then EM and PCA.

**Reinforcement learning** is learning from trial and error. The
machine acts in an environment and gets rewards or punishments back.
Nobody shows it the right move. It plays thousands of games, wins
some, loses some, and shifts toward the moves that won. The
experience is a history of actions and rewards. Lectures 16 and 17
cover this, with language models as the flagship application.

Which setup fits a job? Ask what experience you have. Labeled pairs
means supervised. Raw data with no answers means unsupervised. A
world you can act in and get scored means reinforcement.

## Why the old way broke, in numbers

The rule-list failure deserves one more look, because it explains why
the field exists. Suppose you maintain N rules. Spammers test each
rule with probes. Each probe costs them about 1,000 emails. A wave
of evasion arrives roughly every 14 days. Each wave breaks about 5
percent of your rules and needs about 50 new rules.

Your team writes a rule in 30 minutes. Fifty rules cost 25
engineer-hours per wave, every two weeks, forever. Worse, the rules
interact: rule 612 ("mark as spam if sender unknown AND subject has
numbers") starts flagging real receipts after you add rule 663. The
false-positive rate climbs with the rule count. You are patching a
system whose complexity grows every wave while the attacker pays
nothing to probe it.

The learning approach inverts this. The machine retrains on this
week's labeled mail. No new code. The labeling is the only human
work, and one labeler tags about 2,000 emails per day. Ten thousand
labels cost five labeler-days and produce a filter that adapts to
"fr33", "f.r.e.e", and whatever comes next, because the pattern is
re-learned from fresh examples each time.

## The map of the course

This course teaches the technology that trains models, not the
technology that uses them. The lecture is explicit: you will not
learn how to prompt Claude Code. You will learn what happens inside
training, from first principles.

The order is deliberate. Lectures 2 through 6 build the classical
core: linear regression, logistic regression, generative models,
model selection. Lectures 7 and 8 build neural networks and the
algorithm that trains them, backpropagation. Lectures 9 and 10 turn
to unsupervised learning. Lectures 11 through 15 cover modern deep
learning: diffusion models, foundation models, contrastive learning,
transformers, and how large models are trained and adapted.
Lectures 16 and 17 close with reinforcement learning and its role in
training reasoning models.

![Course roadmap](assets/svg/l01-roadmap.svg "The CS229 course map. Classical supervised learning feeds neural networks and backpropagation, then unsupervised learning, modern deep learning, and reinforcement learning. Source: original plate for Stanford Frontier AI.")

Every concept in this system builds on this lesson. **Supervised
learning** is the setup of lectures 2 through 8. **Loss** is the
performance measure P made into a number the machine can minimize.
Lecture 2 builds it from zero. **Overfitting** is what happens when P
rises on the training experience but falls on new data. Lecture 6
dissects it. **Gradient descent** is the workhorse that improves P.
Lecture 2 derives it.

## The honest price

Machine learning buys adaptability and pays in three currencies.
First, data. The machine needs thousands of labeled examples where
the rule list needed one clever engineer. Second, compute. Training
scans those examples again and again, millions of arithmetic
operations. Third, trust. A learned model is a black box: nobody
wrote its rules, so nobody can read them back. When it fails, it
fails in ways no human predicted. The rest of this course is about
making each of these three costs as small as possible while keeping
the adaptability.

> [!QA]
> Q: What is the difference between the Samuel and Mitchell definitions of machine learning?
> A: Samuel (1959) says machine learning gives computers the ability to learn without being explicitly programmed. The key contrast is with hand-written rules. Mitchell (1997) says a program learns from experience E on tasks T under performance measure P when its performance at T, measured by P, improves with E. Mitchell's version is testable: measure P before and after E. Use Samuel for the intuition and Mitchell when you need to check whether learning actually happened.
> Follow-up: Give a concrete T, E, P for house-price prediction.
> A: T is "predict the sale price of a house from its features". E is 10,000 past sales with known prices. P is mean squared error on new sales. Learning happened if the error drops as the machine sees more past sales.

> [!QA]
> Q: How do I tell which of the three paradigms a problem belongs to?
> A: Look at the experience you have. If each experience is an input paired with its correct answer, it is supervised: spam labels, house prices. If the data has no answers and you want structure, it is unsupervised: clustering customers, compressing images. If the machine acts and gets rewards back, it is reinforcement: game playing, robot control. The same task can fit two paradigms with different experience: email sorted by humans is supervised, raw email clustered by topic is unsupervised.
> Follow-up: Is training a chatbot supervised or reinforcement learning?
> A: Both, in sequence. The base model trains supervised on next-word prediction: each experience is text with the correct next word. Then post-training uses reinforcement: the model generates answers and gets rewards for helpful, correct responses. Lectures 14 through 17 trace exactly this pipeline.

> [!QA]
> Q: Why did hand-written rules lose to learning for spam?
> A: The attacker moves faster than the rule writer. Spammers probe the filter with cheap test emails and invent a new evasion roughly every two weeks, while each rule costs a human 30 minutes to write and test. Rules also interact: adding rule 663 breaks rule 612, so false positives climb with the rule count. Learning inverts the cost: relabel this week's mail, retrain, and the new evasions are handled with no new code.
> Follow-up: When do hand-written rules still win?
> A: When the world does not change and the rules are few. Tax brackets, unit conversions, and chess move legality are stable and exact. Rules win on precision and auditability there. Learning wins where the world adapts or the pattern is too complex to write down.

## Recap: the whole lesson on one screen

1. **The job.** Sort spam when spammers change tactics every two
   weeks. Hand-written rules cannot keep up.
2. **The first attempt.** Rule lists: 1,000 rules, 30 minutes each,
   500 engineer-hours. Broken by "fr33" in one wave.
3. **Where it breaks.** Attackers generate evasions at 100 per day.
   Humans add rules at 10 per day. Rules interact and false
   positives climb.
4. **The key question.** What if the machine writes its own rules
   from labeled examples?
5. **The new idea.** Learn from experience. Samuel: without explicit
   programming. Mitchell: T, E, P, and P must improve with E.
6. **The three paradigms.** Labeled pairs (supervised), raw data
   (unsupervised), actions and rewards (reinforcement).
7. **The honest price.** Data, compute, and trust. Learned models
   adapt but cannot be read back like rule lists.
8. **The map.** Lectures 2 to 8 build supervised learning and neural
   nets. Lectures 9 to 10 cover unsupervised learning. Lectures 11
   to 17 cover modern deep learning and reinforcement.

## Official sources and further reading

**Official:**
- Lecture 1 video, Stanford Online YouTube:
  https://www.youtube.com/watch?v=DATnpGoGhM8 — Tengyu Ma's
  introduction, definitions, and course map.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the rigorous
  companion to the lectures.

**Caveats from these sources.** The lecture says the definitions from
1959 and 1997 "still kind of hold", but the *use* of models changed:
in the old days you built data pipelines and trained per task. Now
you often prompt a pre-trained model. The course teaches the training
technology underneath, not prompting. The Mitchell definition on the
lecture slide carries a small wording typo. The standard textbook form
is given in this lesson.

## Connections to the other courses

- **CS229 L02:** builds supervised learning from zero: the loss, the
  gradient, the fit. Read it next.
- **CS229 L09-L10:** the unsupervised paradigm: clustering and
  dimensionality reduction.
- **CS229 L16-L17:** the reinforcement paradigm: MDPs, policy
  gradient, PPO, and RL for language models.
- **CS224N:** the same three paradigms applied to language, with
  next-token prediction as the flagship supervised task.
- **CS336:** how the supervised training loop actually runs at
  scale on modern hardware.
