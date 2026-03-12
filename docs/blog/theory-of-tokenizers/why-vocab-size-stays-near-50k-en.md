---
title: "Why Vocabulary Size Stays Near 50k"
date: 2026-03-09T12:10:00-08:00
summary: "From marginal compression gain, tail sparsity, and output-layer estimation cost to a view of vocabulary size as a threshold for which fragments deserve their own parameter rows."
tags: ["tokenizer", "vocabulary", "LLM"]
---

# Why Vocabulary Size Stays Near 50k

<BlogPostLocaleSwitch current-locale="en" zh-path="/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k" en-path="/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k-en" />

If a tokenizer is a budgeted explicit codebook, then vocabulary size is not just another hyperparameter. Adding a new token is a real modeling decision: does this string fragment deserve its own embedding row, its own output class, and its own independent statistical identity?

That is the professional way to phrase the vocabulary question. A token does not merely shorten the sequence a bit. It also means:

- the fragment will now be treated as atomic by the model;
- it occupies a parameter row on the input side;
- it occupies a class in the output distribution;
- it stops sharing some of its statistics with neighboring subword pieces.

So vocabulary cannot grow without bound. A fragment deserves promotion only when its **compression gain is large enough, its reuse frequency is high enough, and its statistics are stable enough to justify an independent parameter row**. That is why modern LLM vocabularies so often land in a medium range such as `30k` to `100k`: not because of folklore, but because those conditions are jointly shaped by Zipfian data and finite training budgets [1-7].

> Core view: the right question is not "can a larger vocabulary make sequences shorter?" The right question is "which fragments are worth owning an independent parameter row?" As vocabulary expands, the truly high-yield and highly reusable fragments are absorbed early. What remains lies deeper in the Zipfian tail, where extra compression becomes small while sample sparsity, output competition, and parameter inefficiency keep worsening. That is why the best vocabulary usually stops in a moderate range instead of freely expanding toward the millions [1-7].


## 1. What does a new token actually buy?

Suppose a candidate fragment $g$ appears with frequency $f_g$ in the corpus. If promoting it to a token saves on average $\Delta T_g$ sequence positions relative to a finer-grained decomposition, then its direct benefit is roughly proportional to

$$
f_g\,\Delta T_g.
$$

The more often it appears, and the more positions it saves each time, the more attractive it looks.

But that is only the benefit side. A new token also creates three kinds of cost.

### Parameter cost

It needs its own embedding row, and often its own output-side geometry or parameters coupled to the output head [6].

### Classification cost

Next-token prediction is a classification problem over $V$ categories. Enlarging the vocabulary makes normalization and class competition in the output layer harder.

### Statistical cost

Most importantly, promoting a fragment to an independent token cuts off some parameter sharing with the smaller pieces that compose it. If the fragment is rare, its dedicated parameters become undertrained islands.

Putting those terms together, a rough incremental criterion looks like

$$
\Delta \mathcal{J}(g)
\approx
- \alpha\, f_g \Delta T_g
+ \beta
+ \frac{\gamma}{\sqrt{f_g}},
$$

where:

- the first term is the gain from shorter sequences and less local modeling burden;
- the second term is the fixed system cost of one more class and one more parameter row;
- the third term is a crude penalty for the fact that independent parameters become harder to estimate when frequency is low.

This is not a precise fitted law. It is a structural decision rule: **a fragment should become a token only if it is both common enough and useful enough to carry its own statistical identity.**

## 2. Zipf's law makes vocabulary expansion lose value quickly

Vocabulary growth enters diminishing returns not because engineers are conservative, but because language itself is Zipfian. A small head contains the reusable high-frequency structure; the tail contains vast numbers of low-frequency forms [1].

That gives vocabulary expansion a sharply uneven return profile:

- early additions tend to be high-frequency local patterns, so both $f_g$ and $\Delta T_g$ are large;
- later additions come from deeper in the tail, so $f_g$ falls rapidly;
- very deep in the tail, even long fragments often appear in too few contexts to be worth memorizing as atomic units.

So vocabulary growth does not harvest uniform gains. It quickly absorbs the valuable head and then enters a broad region of forms that are representable but not worth granting independent status.

Put differently, **language contains many observable strings, but far fewer strings that deserve independent statistical identity**. That is the real point behind an empirical number like `50k`. The relevant upper bound is not "how many different words exist?" It is "how many local forms are frequent, reusable, compressive, and estimable enough that the model should reserve a dedicated row for them?"

## 3. The real bottleneck is sample efficiency, not just compression

This is where many discussions of vocabulary size stay too shallow. A token is not a free codeword. It is a set of parameters that must be estimated repeatedly by gradient updates. When frequency is too low, an independent token suffers in at least three ways.

### Embeddings become unstable

Rare tokens receive too few contextual updates, so their vectors are more likely to reflect noise than robust reusable structure.

### Output probabilities become harder to calibrate

The language-model head must distribute probability mass not only over frequent items but also across many tail classes. As the tail grows, more mass is spent competing among poorly estimated rare categories [7].

### Compositional generalization weakens

If a rare string is hard-coded as one token, the model loses the opportunity to share statistics through more frequent subword pieces. For long-tail forms, composition from reusable fragments is often more reliable than an isolated rare token [2-4].

So the key question is not "can this fragment be encoded as one token?" It is "does the corpus provide enough evidence for this fragment to exist as an atomic model object?" This is exactly why subword tokenization usually outperforms large brittle whole-word vocabularies: it pushes long-tail statistics back toward higher-frequency pieces [2][3].

## 4. Why does the optimum often land in the tens of thousands?

It is important to be explicit here: `50k` is not a theoretical constant. Different corpora, scripts, tasks, and training budgets can move the optimum substantially.

But it is not an arbitrary number either. In many monolingual or near-monolingual LLM settings, a vocabulary in the tens of thousands often already covers:

- the highest-frequency function words and common whole words;
- many highly reusable stems, affixes, and spelling chunks;
- enough digits, punctuation patterns, whitespace conventions, and format artifacts;
- without pushing too far into the region where tail sparsity dominates.

A stricter way to say it is this: on large general-purpose corpora, the number of local fragments that are simultaneously frequent, compressive, reusable, and stably estimable usually does not grow without bound. It tends to saturate in a moderate region.

That is why many mature systems keep converging to a similar scale. They are not all copying the same superstition. They are responding to the same Zipfian head-tail structure, the same output-layer burden, and the same finite-sample estimation problem.

![Illustration of the vocabulary-size trade-off](./vocab-size-tradeoff.svg)

*Figure 1. Early vocabulary growth quickly absorbs high-yield head patterns. Deeper in the tail, each extra token yields less compression while class and sparsity costs keep accumulating, so overall utility is more likely to peak in a moderate regime.*

## 5. Why is too small a vocabulary also bad?

If we only remember that large vocabularies create tail sparsity, we drift into the opposite mistake. A very small vocabulary pushes too much local structure back onto the sequence axis. Then:

- sequences become longer, increasing attention and cache cost;
- the model must spend more layers rebuilding lexical chunks before higher-level semantics;
- prediction proceeds in more steps, each step carrying less local structure.

ByT5 makes this trade explicit: byte-level modeling gains openness and robustness, but pays for it with longer sequences and higher total systems cost [5]. So a small vocabulary is not inherently purer or more principled. It simply postpones coarse-graining into the network.

## 6. So what is the real answer to the vocabulary-size question?

The most accurate answer is not "50k is best." It is:

> A fragment belongs in the vocabulary only if its marginal compression benefit reliably covers the statistical and computational cost of giving it an independent parameter row and an independent output class.

Once phrased that way, several intuitions unify immediately:

- whole-word vocabularies fail because too many word forms are too rare to support independent parameters;
- subword vocabularies do not need to grow without bound because deeper-tail fragments save less and learn worse;
- the tens-of-thousands regime keeps reappearing because the most valuable local patterns are mostly absorbed there, while further growth mostly purchases expensive tail sparsity.

So the essence of vocabulary size is not geometric capacity and not a single-minded compression objective. It is a **statistical decision jointly governed by compression gain, sample efficiency, and output-layer complexity**.

## 7. Closing

Vocabulary size stabilizes in a medium range not because tokenizer training is lazy, but because the number of local patterns that truly deserve independent identity is much smaller than the number of distinct strings we can observe. Head patterns are valuable. Tail patterns are expensive. Subword vocabularies sit in between.

Put compactly, **vocabulary size determines how much local structure the model is willing to upgrade into atomic statistical objects**. Push that explicit codebook smaller and the natural next question appears: if we stop prepaying almost everything, what burden gets pushed back into the model by character-level modeling?

Continue reading: [Why Character-Level Tokenizers Rarely Win](/blog/theory-of-tokenizers/why-character-level-rarely-wins-en).

## References

[1] PIANTADOSI S T. Zipf's Word Frequency Law in Natural Language: A Critical Review and Future Directions[J]. *Psychonomic Bulletin & Review*, 2014, 21(5): 1112-1130. DOI: [10.3758/s13423-014-0585-6](https://doi.org/10.3758/s13423-014-0585-6).

[2] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[3] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).

[4] KUDO T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates[C]// *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Melbourne, Australia: Association for Computational Linguistics, 2018: 66-75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007).

[5] XUE L, BARUA A, CONSTANT N, et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 291-306. DOI: [10.1162/tacl_a_00461](https://doi.org/10.1162/tacl_a_00461).

[6] PRESS O, WOLF L. Using the Output Embedding to Improve Language Models[C]// *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*. Valencia, Spain: Association for Computational Linguistics, 2017: 157-163. URL: [https://aclanthology.org/E17-2025/](https://aclanthology.org/E17-2025/).

[7] KOBAYASHI G, KURIBAYASHI T, YOKOI S, et al. Transformer Language Models Handle Word Frequency in Prediction Head[C]// *Findings of the Association for Computational Linguistics: ACL 2023*. Toronto, Canada: Association for Computational Linguistics, 2023: 4523-4535. DOI: [10.18653/v1/2023.findings-acl.276](https://doi.org/10.18653/v1/2023.findings-acl.276).
