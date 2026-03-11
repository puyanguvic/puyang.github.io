---
title: "Why Character-Level Tokenizers Rarely Win"
date: 2026-03-09T12:20:00-08:00
summary: "Using sequence length, optimization path length, and internal compression compensation to explain why character-level schemes rarely become the default."
tags: ["tokenizer", "character-level", "optimization"]
---

# Why Character-Level Tokenizers Rarely Win

<BlogPostLocaleSwitch current-locale="en" zh-path="/blog/theory-of-tokenizers/why-character-level-rarely-wins" en-path="/blog/theory-of-tokenizers/why-character-level-rarely-wins-en" />

If we look only at representational uniformity, character-level or byte-level modeling is almost unimpeachable: there is no OOV problem, no hand-designed vocabulary boundary, and new words, new scripts, and noisy spelling are handled more naturally. The problem is that representational elegance does not automatically translate into system efficiency. The reason mainstream large models have stayed with subword tokenization for so long is not that character-level methods cannot express language. It is that they usually lose on total compute and optimization cost [1-5].

A stricter statement is that character-level modeling does not eliminate the need for compression. It simply refuses to do the compression explicitly outside the model, so the network must reconstruct it internally on a longer sequence.

> Core view: character-level or token-free methods do offer open vocabulary, noise robustness, and cross-lingual uniformity, but they usually require longer sequences, longer paths to semantic formation, and extra architectural compensation. The most successful token-free models almost always reintroduce downsampling, local chunking, or latent subword bias explicitly or implicitly. Compression has not disappeared; it has moved from the external tokenizer into the model itself [1-5].


## 1. Why does the character-level route look so attractive?

ByT5 summarized the most commonly cited advantages of token-free modeling [1]:

- it is naturally open to any language and any new word;
- it is more robust to spelling noise, character perturbations, and nonstandard text;
- it reduces tokenizer training, deployment, and compatibility burden;
- it avoids precommitting to which strings deserve to be atomic tokens.

All of those advantages are real. In multilingual settings, noisy text, or environments where input format changes often, character-level uniformity is genuinely appealing. The problem is that a unified input format does not mean the later model learns high-level semantics more easily.

## 2. First cost: longer sequences amplify everything

Characters and bytes are the finest-grained written units, and one subword token usually corresponds to several of them. So the same text typically expands into a longer sequence in token-free systems. ByT5 explicitly reports that the token-free route pays substantial training and inference costs for that increase in sequence length [1].

That longer sequence propagates through almost every Transformer cost:

- attention becomes heavier;
- the KV cache grows;
- a fixed context window covers less real text;
- the model needs more layers before it can rebuild low-level local patterns into stable mid-level units.

So character-level modeling is not "no compression." It is deferred compression performed later and more expensively inside the network.

## 3. Second cost: the path to semantics becomes longer

From an optimization viewpoint, the deeper difficulty of character-level modeling is not just slower computation. Semantic structure has to emerge through a longer compositional chain.

With subword tokenization, the model sees units that already carry some semantic density: frequent stems, affixes, and common spelling chunks. In a character-level setup, the bottom-layer inputs themselves carry little stable meaning. The model must first learn:

1. which characters should bind into local fragments;
2. which fragments should form lexically stable mid-level units;
3. how those units should then enter higher semantic and contextual relations.

So useful task gradients must travel through a longer composition path before they can shape the lowest-level representation. Early training is therefore more easily captured by surface spelling statistics and short-range repetition, and slower to build compact mid-level semantic structure.

## 4. Third cost: the standard Transformer is not biased toward raw character streams

The inductive bias of the standard Transformer is better suited to building global dependencies between units that already carry some semantic density than to recovering local block structure from a very long raw character stream first and only then moving upward. In other words, standard Transformers are naturally good at:

- relating medium-granularity units to one another;
- repeatedly rewriting representations through stacked attention layers;
- computing long-range structure once local chunks already exist.

What they are not naturally best at is:

- discovering lexical chunks from very long character sequences;
- doing that chunk recovery while remaining computationally efficient;
- solving "learn compression first, then learn semantics" in the same architecture without extra help.

That is why a subword tokenizer, though nominally just preprocessing, ends up handling exactly the kind of low-level work a standard Transformer is least efficient at.

## 5. How do modern token-free models compensate?

The most revealing fact is not that character-level methods can work. It is that the best token-free models almost always reintroduce compression **inside** the model. Figure 1 makes that system logic explicit.

![Why token-free models still reintroduce compression](./character-level-compute-tradeoff.svg)

*Figure 1. To become practical, token-free methods usually add some internal compression mechanism: downsampling, local chunking, or latent subwords. Compression has not vanished; it has moved from the explicit tokenizer into the network.*

The key point of Figure 1 is not one particular compensation module. It is the direction of responsibility transfer: compression moves from the external codebook into internal architecture. The examples below all occupy different places in that design space.

### ByT5: accept the cost of longer sequences

ByT5 showed that a standard Transformer can work directly on byte sequences and gains clear advantages in robustness to noise [1]. But it also made clear that the token-free route pays in sequence length, training FLOPs, and inference speed. ByT5 does not prove the tokenizer is unnecessary. It proves that token-free modeling can be worthwhile if one is willing to pay a higher systems cost.

### CANINE: explicit downsampling

The key innovation of CANINE is not just "feed characters directly." It is the introduction of downsampling, which compresses the very long character sequence into a shorter intermediate representation before the deep Transformer processes it [2]. That is essentially moving part of the tokenizer's job into the network.

### Charformer: learn latent subwords

Charformer goes one step further and learns latent subword structure inside the model. Its GBST module enumerates candidate blocks and learns weights over them, producing a form of end-to-end latent subword tokenization [3]. So even after abandoning the explicit tokenizer, the model still tends to rediscover some middle-granularity unit.

The shared theme of modern token-free work is therefore not "compression is gone." It is "compression has become an internal learnable module."

## 6. So where do character-level schemes usually lose?

Putting the previous sections together, character-level systems are usually pressured along three dimensions at once:

- worse computational efficiency because the sequence is longer;
- longer optimization paths because mid-level structure must be rebuilt internally;
- weaker architectural match because the standard Transformer is not the ideal raw-character composer.

So the issue is not lack of theoretical expressive power. The issue is that **under current mainstream architectures and budgets, the total systems cost is usually higher**. That is why subword tokenizers continue to occupy the default position: they pre-compress some local structure outside the model, allowing the Transformer to spend its depth on the global relational modeling it does best [4][5].

## 7. When is the character-level route actually worth using?

None of the above means character-level modeling is never a good choice. In some settings, its advantages become very concrete.

- When the input is noisy, spelling varies wildly, or OCR, social media, and user-generated text contain many nonstandard forms, character-level robustness becomes much more valuable [1].
- When the task must cover many scripts, many novel words, or a large amount of previously unseen surface forms, open vocabulary matters more.
- When the research goal is to reduce tokenizer technical debt, unify multilingual pipelines, or test whether models can recover their own mid-level structure, the token-free route also has methodological value.

So the real question is not "is character-level modeling correct?" It is "in this task, are openness and robustness worth paying for with longer sequences and higher optimization cost?" Once that trade becomes favorable, character-level modeling can be the right choice.

## 8. Closing

What is most often overestimated about the character-level route is the feeling that removing the tokenizer removes the engineering compromise around lexical structure. In reality, compression and middle-structure discovery must happen somewhere no matter what. The only difference is whether they are done by an explicit codebook outside the model or rediscovered by a deeper and more expensive network inside the model.

Compressed into one sentence: **character-level methods usually lose not on representation, but on division of labor.** When an explicit subword tokenizer can provide stable compression more cheaply, a fully character-level route struggles to win on total efficiency. That is precisely why the most successful token-free models usually end up reinventing some form of implicit tokenizer.

## References

[1] XUE L, BARUA A, CONSTANT N, et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 291-306. DOI: [10.1162/tacl_a_00461](https://doi.org/10.1162/tacl_a_00461).

[2] CLARK J H, GARRETTE D, TURC I, et al. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 73-91. DOI: [10.1162/tacl_a_00448](https://doi.org/10.1162/tacl_a_00448).

[3] TAY Y, TRAN V Q, RUDER S, et al. Charformer: Fast Character Transformers via Gradient-based Subword Tokenization[C]// *International Conference on Learning Representations*. 2022. URL: [https://research.google/pubs/charformer-fast-character-transformers-via-gradient-based-subword-tokenization/](https://research.google/pubs/charformer-fast-character-transformers-via-gradient-based-subword-tokenization/).

[4] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[5] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).
