---
title: "Why Character-Level Tokenizers Rarely Win"
date: 2026-03-09T12:20:00-08:00
summary: "From delayed local entropy compression, scale mismatch in compute allocation, and implicit tokenization inside the network to a sharper account of why character-level modeling is rarely the default."
tags: ["tokenizer", "character-level", "optimization"]
---

# Why Character-Level Tokenizers Rarely Win

<BlogPostLocaleSwitch current-locale="en" zh-path="/blog/theory-of-tokenizers/why-character-level-rarely-wins" en-path="/blog/theory-of-tokenizers/why-character-level-rarely-wins-en" />

The appeal of character-level or byte-level modeling is obvious. It looks clean: no OOV problem, no handcrafted vocabulary boundary, and no need to keep maintaining tokenizer assets as languages and new words shift over time [1-5].

But that intuition misses the decisive layer of the problem. Once the external tokenizer is removed, the need for local compression does not disappear. Spelling chunks, morphological pieces, punctuation patterns, whitespace conventions, and other short-range regularities are still there. They simply stop being handled by an explicit codebook and must instead be rediscovered by the network over a much longer sequence.

That is the deeper cost of character-level modeling. It is not merely "finer input." It is the decision to postpone **local entropy folding and mid-level structure discovery** into a deeper and more expensive part of the system.

> Core view: character-level or token-free methods usually lose not on expressive power, but on hierarchical division of labor. They remove the external tokenizer without removing local redundancy in language, so the model must first reconstruct spelling blocks, subword-like units, and lexical chunks before it can spend its depth on higher semantics and long-range dependency. The strongest token-free architectures almost always reintroduce downsampling, local chunking, or latent subword modules, which shows that what disappears is not tokenization itself, but only its explicit outsourced form [1-5].


## 1. What layer of structure does the character-level route actually remove?

The first clarification is conceptual. Character-level modeling does not make language more atomic. It merely moves the model's input units back to characters or bytes. But useful linguistic structure is not flat. It is layered:

- characters and bytes carry writing constraints;
- subword and lexical chunks carry a large amount of reusable local structure;
- words and phrases carry higher semantic and combinatorial structure.

The role of an explicit subword tokenizer is to freeze part of that stable, high-frequency middle layer into discrete units before the main network starts. Character-level modeling refuses to do that. So the input becomes more uniform, but also less organized: the model must reinvent the middle layer from a raw character stream.

What disappears, then, is not noise or arbitrary engineering. What disappears is a **front-loaded coarse-graining of multiscale structure**. Without it, the network must solve "which characters belong together?" before it can efficiently move on to genuine semantic modeling.

## 2. The deepest problem is not sequence length alone, but scale mismatch in compute

The usual criticism is that character-level inputs create longer sequences. That is true, but not deep enough. The more important issue is that the system ends up using an expensive global model to do work that should have been handled by a cheap local mechanism.

Standard Transformers are good at:

- relating units that already have some semantic density;
- rewriting contextual representations through stacked attention and MLP blocks;
- spending computation on long-range structure, semantic composition, and reasoning chains.

They are not especially good at:

- repeatedly recovering spelling chunks from very long character streams;
- doing that recovery while remaining computationally efficient at the global scale;
- solving "learn compression first, then learn semantics" inside the same deep stack without extra help.

This is a real scale mismatch. Spelling blocks and lexical fragments are mostly short-range structure. They are much better handled by a front-end mechanism that absorbs them cheaply. Character-level modeling instead asks the global network to rediscover them on every training example. The result is not greater end-to-end purity. It is **high-cost computation being spent on low-level entropy compression**.

## 3. The path to semantic structure becomes longer too

Longer sequences are only the surface symptom. What really slows optimization is that the representational path becomes longer.

With subword tokenization, the model starts from units that are already somewhat coarse-grained. They may not be perfect linguistic atoms, but they usually absorb a meaningful amount of orthographic and morphological regularity. As a result, gradients can act earlier on units that already have mid-level statistical significance.

In a character-level setup, the model often has to learn, in sequence:

1. which characters bind into stable local chunks;
2. which local chunks deserve to function as reusable units;
3. how those units enter word-level and semantic relations;
4. how high-level contextual objectives should constrain those middle units.

That longer chain means useful semantic gradients must pass through more low-level compositional steps before they reshape the bottom of the system. Early training is therefore more easily captured by surface orthography and short-range repetition, and slower to build robust lexical and semantic abstractions.

So the problem is not just more FLOPs. It is that **semantics emerges through a longer representational path**.

## 4. Why is character-level modeling really learning the tokenizer back?

This is exactly what Figure 1 makes visible.

![Why token-free models still reintroduce compression](./character-level-compute-tradeoff.svg)

*Figure 1. To become practical, token-free methods usually add some internal compression mechanism: downsampling, local chunking, or latent subwords. Compression has not vanished; it has moved from the explicit tokenizer into the network.*

The key lesson is not that one compensation module is universally best. It is that once a character-level system is pushed toward practicality, some middle-granularity structure almost always gets rebuilt inside the model. Otherwise the network remains trapped under very long sequences and low-density inputs.

In that sense, the token-free route does not abolish tokenization. It changes tokenization from an external static codebook into an internal learned mechanism.

## 5. Why do ByT5, CANINE, and Charformer all reintroduce hierarchy?

This point is not hidden. It is visible in the design of the major token-free papers themselves.

### ByT5: accept the cost, do not eliminate it

ByT5 showed clearly that byte-level pretraining can work and can improve robustness to noise [1]. But it also showed the cost just as clearly: longer sequences, higher training FLOPs, and slower inference. ByT5 does not prove that tokenizers are unnecessary. It proves that token-free modeling can be worthwhile when one is willing to pay a higher total systems cost.

### CANINE: downsample before deep processing

The key move in CANINE is not merely "feed characters directly." It is explicit downsampling: compress the long character stream into a shorter intermediate representation before the deeper Transformer stack processes it [2]. That is already very close to moving part of the tokenizer's job inside the network.

### Charformer: learn latent subwords in the model

Charformer goes further by learning subword-like structure directly inside the architecture. Its GBST module enumerates candidate blocks and learns how to weight them, effectively constructing a latent subword tokenizer inside the network [3]. That is not a rejection of tokenization logic. It is a reinvention of it.

So the strongest shared lesson of token-free work is not "compression is gone." It is **middle-granularity structure has to appear somewhere**.

## 6. Where do character-level schemes usually lose?

Putting the pieces together, character-level modeling is usually disadvantaged along three axes at once.

### Worse division of compute

It pushes local compression into a deeper and more expensive part of the stack.

### Slower statistical accumulation

The model must first assemble stable chunks before it can accumulate reusable statistics over them, so middle-layer structure forms more slowly.

### Weaker architectural match

Standard Transformers are better at operating on units with some semantic density than at serving as raw-character compressors for long stretches of training.

That is why the theoretical elegance of character-level modeling is so easy to overestimate. A unified input interface is elegant, but elegance is not the same thing as an efficient systems decomposition. Under mainstream architectures and budgets, an explicit subword tokenizer is often just a cheaper way of doing work the model would otherwise have to do anyway [4][5].

## 7. When is character-level modeling actually worth it?

None of this means character-level modeling lacks real value. In some settings its advantages are concrete.

- If the input is noisy, spelling varies heavily, or OCR and social-media text contain many nonstandard forms, character-level robustness becomes much more valuable [1].
- If the task must span many scripts, many novel words, or frequent unseen surface forms, open vocabulary matters more.
- If the research goal is to reduce tokenizer technical debt, test whether models can recover their own middle layer, or unify the input interface, token-free modeling also has clear methodological value.

So the real question is not "is character-level modeling correct?" It is: **in this task, are openness and robustness worth paying for with worse hierarchical division of labor?**

## 8. Closing

The most misleading thing about the character-level route is the feeling that removing the tokenizer removes the engineering compromise around lexical structure. In reality, only the external explicit codebook is removed. Local compression, middle-structure discovery, and multiscale organization do not disappear at all.

Put in one sentence, **character-level methods usually lose not on representation, but on where computation is forced to happen**. When an explicit subword tokenizer can prepay the head of local structure at very low cost, asking a deep Transformer to rediscover the same structure over long character streams is usually not the default optimum.

## References

[1] XUE L, BARUA A, CONSTANT N, et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 291-306. DOI: [10.1162/tacl_a_00461](https://doi.org/10.1162/tacl_a_00461).

[2] CLARK J H, GARRETTE D, TURC I, et al. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 73-91. DOI: [10.1162/tacl_a_00448](https://doi.org/10.1162/tacl_a_00448).

[3] TAY Y, TRAN V Q, RUDER S, et al. Charformer: Fast Character Transformers via Gradient-based Subword Tokenization[C]// *International Conference on Learning Representations*. 2022. URL: [https://research.google/pubs/charformer-fast-character-transformers-via-gradient-based-subword-tokenization/](https://research.google/pubs/charformer-fast-character-transformers-via-gradient-based-subword-tokenization/).

[4] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[5] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).
