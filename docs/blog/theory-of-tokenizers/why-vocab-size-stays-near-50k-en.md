---
title: "Why Vocabulary Size Stays Near 50k"
date: 2026-03-09T12:10:00-08:00
summary: "Using diminishing sequence-length gains, long-tail sparsity, and output-layer cost to explain why LLM vocabularies often settle in a moderate range."
tags: ["tokenizer", "vocabulary", "LLM"]
---

# Why Vocabulary Size Stays Near 50k

<BlogPostLocaleSwitch current-locale="en" zh-path="/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k" en-path="/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k-en" />

If a tokenizer can promote high-frequency local structure into its codebook, an apparently natural conclusion is that the larger the vocabulary, the better: a larger vocabulary should always make sequences shorter. Reality is not like that. Although vocabularies differ across language models, monolingual or near-monolingual systems often settle in a moderate range rather than expanding without bound [1-6].

This is not an empirical coincidence. It is an interior optimum at the system level. Enlarging the vocabulary does shorten sequences, but the gain is controlled by a Zipfian long tail and quickly shows diminishing returns. Meanwhile vocabulary parameters, output-head cost, and long-tail sparsity keep increasing. Vocabulary size should therefore not be thought of as a knob where "bigger is more complete." It is a balance point between compression benefit and modeling cost.

> Core view: as vocabulary size $V$ grows, average sequence length $T(V)$ does decrease, but the decrease quickly slows because of the long-tail distribution. By contrast, embedding parameters, output-layer classification cost, and the learning sparsity of long-tail tokens keep worsening roughly with $V$. Modern LLMs therefore tend to reach a better system trade-off around a medium-sized subword vocabulary instead of promoting every rare fragment into an independent token [1-6].


## 1. First write the problem as a cost balance

Let vocabulary size be $V$, and average sequence length be $T(V)$. For a Transformer, a rough but useful system-cost model is

$$
\mathcal{J}(V) \approx a\,T(V)^2 + b\,T(V) + c\,V,
$$

where

- $a\,T(V)^2$ represents costs such as attention that grow superlinearly with sequence length;
- $b\,T(V)$ represents length-related costs such as caching, linear layers, and data movement;
- $c\,V$ represents the scale cost of embeddings, output layers, and vocabulary parameters.

This formula is not meant to capture every engineering detail. It captures the crucial structure: **a larger vocabulary reduces sequence-length cost while increasing vocabulary-side cost.** So the question is not whether vocabulary should grow, but whether further growth is still worth paying for.

## 2. Why do sequence-shortening gains diminish so quickly?

The answer comes from the long tail of language. Zipf's law means most of the mass is concentrated in a small head of very frequent units, while the tail contains many rare fragments [1]. This produces a highly uneven return curve for vocabulary expansion:

- the early high-frequency tokens shorten sequences substantially;
- as the vocabulary grows, each newly added token corresponds to a rarer fragment;
- deep in the long tail, a new vocabulary entry may replace only a tiny amount of text.

So $T(V)$ is decreasing, but its slope cannot stay constant. Instead

$$
T'(V) < 0,
\qquad
|T'(V)| \to 0.
$$

That is the mathematical statement of diminishing marginal returns. The first part of vocabulary growth is very valuable; the later part quickly enters a regime where cost rises while benefit shrinks.

## 3. Why can a very large vocabulary hurt learning?

The tokenizer is not a free compressor. Promoting more and more rare fragments to independent tokens creates at least three costs.

### Long-tail parameter sparsity

One reason Sennrich et al. introduced subword units was exactly to represent rare words compositionally through more frequent pieces rather than assigning each rare word its own nearly untrainable vector [2]. If the vocabulary becomes too large, tail tokens receive too few updates and their embeddings become undertrained parameter islands.

### Higher output-layer and softmax cost

Next-token prediction is fundamentally a classification problem over $V$ categories. Even with fused kernels, sampling approximations, or blockwise softmax, output-head cost does not disappear just because the implementation is clever.

### Weaker compositional generalization

When a rare string is hard-coded as its own token, the model loses the chance to share statistics through more common subwords. For long-tail patterns, composition from known pieces is often more stable than learning an isolated rare token.

So an oversized vocabulary is not just "more parameters." It systematically worsens the imbalance of learning in the tail.

## 4. Why is a very small vocabulary also unacceptable?

The other extreme fails as well. If the vocabulary is too small, token granularity becomes too fine and text expands into much longer sequences. In a Transformer, that triggers a cascade:

- attention becomes more expensive;
- the KV cache grows;
- a fixed context window contains less real text;
- the model must spend more layers rebuilding lexical chunks before it can move on to higher-level semantics.

SentencePiece treats "learning the best segmentation under a fixed vocabulary budget" as the central design problem [3], which already shows that vocabulary size is not a peripheral hyperparameter. If the vocabulary keeps shrinking toward characters or bytes, the model must take over a large part of the local compression work that an explicit codebook used to provide [6].

## 5. Why does a medium-size vocabulary become the stable engineering solution?

Once the costs from both sides are combined, the conclusion becomes natural: total system cost is likely to reach its minimum in some intermediate region. Figure 1 just draws that conclusion as a curve.

![Illustration of the vocabulary-size trade-off](./vocab-size-tradeoff.svg)

*Figure 1. As vocabulary grows, sequence-length cost falls but with diminishing returns. At the same time, vocabulary parameters, output-layer cost, and tail sparsity keep increasing, so total cost is more likely to have a minimum in the middle.*

The most important thing in Figure 1 is the curvature, not the absolute values: the left side falls fast, the right side rises more steadily, so an interior minimum appears. The table below simply discretizes that curve into engineering regimes.

| Vocabulary regime | Dominant problem | Typical consequence |
| --- | --- | --- |
| too small | sequences are too long and local composition is too expensive | high compute cost, poor use of context |
| medium | frequent patterns are absorbed, long-tail patterns remain compositional | good balance among compression, generalization, and training stability |
| too large | tail tokens become sparse while output and parameter costs bloat | little extra gain, more uneven learning |

This is why systems like BERT use medium-scale WordPiece vocabularies [4], and why SentencePiece makes fixed-capacity codebooks the default interface [3]. The same system logic sits underneath them: the vocabulary must be large enough to absorb head statistics, but small enough that the tail does not drag the whole training system down.

## 6. Why is this balance point not a universal constant?

One boundary condition is essential: `30k` to `50k` is a common empirical range for many monolingual or near-monolingual settings, not a law of nature. The best vocabulary moves with the data and the task:

- multilingual models must cover more scripts and morphological variation, so they often need larger vocabularies or different byte-level strategies;
- code models face identifiers and symbol strings whose optimal granularity differs from natural language;
- domain-specific corpora with concentrated statistics can sometimes cover the head with smaller vocabularies;
- byte-level setups deliberately trade some openness and robustness for longer sequences [6].

So the strict conclusion is not "every model should use 50k tokens." It is: **given a corpus, architecture, and budget, vocabulary size usually has an intermediate optimum region.**

## 7. How does model scale move this balance point?

There is a finer systems question here: as model size, context length, and implementation efficiency improve, does the optimum move? Yes, but it usually does not disappear.

Larger models can learn long-tail tokens more easily, so part of the sparsity penalty from a large vocabulary is softened. Longer contexts make every saved token more valuable, so sequence compression becomes more important. Better softmax kernels, distributed parallelism, and caching implementations can also reduce some vocabulary-side engineering cost.

But none of this changes the shape of the trade-off. As vocabulary keeps growing, the new gains still come from deeper and deeper tail fragments, while tail sparsity, parameter growth, and deployment overhead remain real. Scaling usually moves the best region; it does not eliminate the existence of a best region.

## 8. Closing

Vocabulary size often settles in a moderate range not because engineers guessed a convenient default, but because the costs on the two sides follow very different growth laws: sequence-shortening gains diminish quickly, while vocabulary-side costs rise much more steadily. Once tokenization is understood as codebook design, that conclusion is almost unavoidable.

More compactly, **vocabulary size is not a local hyperparameter. It is a central decision about how a language-model system allocates compression, parameters, and compositional generalization.** Pushed to the extreme, the character-level route becomes the clearest stress test for this trade-off: if we keep shrinking the explicit codebook, what exactly does the system lose?


## References

[1] PIANTADOSI S T. Zipf's Word Frequency Law in Natural Language: A Critical Review and Future Directions[J]. *Psychonomic Bulletin & Review*, 2014, 21(5): 1112-1130. DOI: [10.3758/s13423-014-0585-6](https://doi.org/10.3758/s13423-014-0585-6).

[2] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[3] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).

[4] DEVLIN J, CHANG M-W, LEE K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]// *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. Minneapolis, Minnesota: Association for Computational Linguistics, 2019: 4171-4186. DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423).

[5] KUDO T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates[C]// *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Melbourne, Australia: Association for Computational Linguistics, 2018: 66-75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007).

[6] XUE L, BARUA A, CONSTANT N, et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 291-306. DOI: [10.1162/tacl_a_00461](https://doi.org/10.1162/tacl_a_00461).
