---
title: "Character-Level Tokenizer 的理论优势与工程局限"
date: 2026-03-09T12:20:00-08:00
summary: "从序列长度、优化路径和 modern token-free 模型的结构补偿出发，解释为什么 character-level 方法通常难以成为主流。"
tags: ["tokenizer", "character-level", "optimization"]
---

# Character-Level Tokenizer 的理论优势与工程局限

如果只从表示纯度看，character-level tokenizer 的确很诱人。它不需要维护复杂词表，不会出现 OOV 问题，也不需要在“这个词应不应该拆开”上做额外工程决策。所有文本都可以统一地写成字符或字节序列，看起来既彻底又优雅。

但现实中的主流大模型并没有大规模转向这种方案。原因并不在于 character-level 不能表达语言，而在于它通常不能在**整体系统效率**上赢过 subword 方案。

更准确地说：

> character-level 的问题不是表示能力不足，而是它把太多本可由 tokenizer 预先完成的局部压缩与组合工作，转移给了后续 Transformer 自己去学。

> 核心结论：token-free 或 character/byte-level 方法确实带来开放词表、鲁棒性和跨语言统一性等优点，但它们往往需要更长序列、更长优化路径以及额外的结构补偿。现代最成功的 token-free 模型几乎都会显式或隐式地重新引入下采样、局部块化或潜在 subword 归纳偏置，这恰恰反过来说明：压缩并没有消失，只是从显式 tokenizer 转移到了模型内部 [1-5]。

## 1. 为什么 character-level 看起来很“对”？

ByT5 总结过 token-free 模型最常被强调的几个优势 [1]：

- 可以天然处理任何语言和任何新词；
- 对拼写噪声、字符扰动和非标准文本更鲁棒；
- 减少预处理流水线和 tokenizer 相关技术债；
- 避免固定词表对新领域和新脚本的硬边界限制。

这些优点都是真实的。尤其在多语言、噪声文本、拼写变化频繁或输入规范不稳定的场景里，character/byte-level 的统一表示确实很有吸引力。

问题在于，**统一表示**不等于**高效学习**。字符或字节只是最细粒度的书写单位，而不是最适合 Transformer 直接进行高层语义建模的单位。

## 2. 第一个问题：序列太长，所有代价都会被放大

character-level 的最直接代价，是序列长度膨胀。一个 subword token 往往对应多个字符或字节，因此相同文本在 token-free 方案下通常会展开为更长的序列。ByT5 明确把这一点列为 token-free 模型必须面对的核心 tradeoff：更长序列会带来更高训练 FLOPs 和更慢推理速度 [1]。

这会连锁影响几乎所有 Transformer 成本：

- attention 计算更重；
- KV cache 更大；
- 固定上下文窗口能装下的真实文本更少；
- 模型需要花更多层和更多路径，先把低层局部块重新组合出稳定语义单元。

也就是说，character-level 并不是“不压缩”；它只是把“压缩的负担”从显式 tokenizer 挪到了后续网络。

## 3. 第二个问题：优化路径更长，语义形成更晚

从优化角度看，character-level 的更大问题并不是单纯更慢，而是**语义形成路径更长**。

在 subword 方案中，模型一开始就能看到具有一定语义密度的单位，例如高频词根、前后缀或常见拼写块。相反，在 character-level 方案中，底层输入本身几乎不携带稳定语义，模型必须先学会：

1. 哪些字符应当绑定成局部片段；
2. 哪些片段进一步构成稳定词法单元；
3. 这些单元如何再与上下文和任务语义关联。

因此，character-level 的难点不只是“长”，而是“高层任务的有效梯度需要穿过更长的组合链条才能影响底层表示”。这会使训练早期更容易被表面模式占据，例如拼写共现、局部重复和短程正则，而较难更快形成紧凑的中层语义结构。

## 4. 第三个问题：Transformer 并不天然偏爱原始字符流

标准 Transformer 最擅长的，是在一组已有一定语义密度的离散单位之间做全局依赖建模；它并不是最理想的底层字符组合器。换句话说，Transformer 的 inductive bias 更适合：

- 在中间粒度单元之间做关系建模；
- 在多层中反复重写表示；
- 利用注意力在已有单元之间建立依赖。

它不天然擅长的是：

- 先从极长原始字符流里恢复局部块结构；
- 再把这些块压缩成更高层的词法或语义单元；
- 最后才开始真正的上下文关系建模。

这也是为什么 subword tokenizer 虽然看似只是前处理，实际上却在帮助 Transformer 做一件它并不最擅长的事：提前把高频局部组合打包成更高语义密度的输入单位。

## 5. 现代 token-free 模型到底是怎么变强的？

这一点最能说明问题。最近几年真正有竞争力的 token-free 模型，并不是“直接把原始字符扔给标准 Transformer 就赢了”，而是几乎都会引入额外结构来**重新做压缩**。

![token-free 模型为何仍会重新引入压缩](./character-level-compute-tradeoff.svg)

*图 1. token-free 方法想要变得实用，通常都需要在模型内部重新引入某种压缩机制：要么做下采样，要么做局部块化，要么学习潜在 subword。压缩并没有消失，只是从显式 tokenizer 转移到了网络内部。*

### ByT5：保留标准 Transformer，但接受更长序列代价

ByT5 证明，标准 Transformer 在少量修改下也可以直接处理字节序列，而且在鲁棒性和噪声敏感任务上有明显优势 [1]。但它同时也明确报告了 token-free 方案在参数、训练 FLOPs 和推理速度上的实打实 tradeoff。换句话说，ByT5 的结论不是“tokenizer 没必要”，而是“如果你愿意支付更长序列的代价，token-free 方案可以很有吸引力”。

### CANINE：显式下采样

CANINE 的关键设计不是“纯字符”本身，而是**downsampling**。Clark 等人明确指出，为了让字符级输入在计算上可行，模型必须通过下采样把超长字符序列压缩成更短中间表示，再交给深层 Transformer 编码 [2]。这其实就是在模型内部重新引入压缩层。

### Charformer：学习潜在 subword

Charformer 则更直白。Tay 等人提出的 GBST（gradient-based subword tokenization）模块，会在模型内部枚举候选字符块并学习它们的打分，从而形成一种端到端的**潜在 subword tokenization** [3]。这等于告诉我们：即便不想使用显式 tokenizer，模型最后也常常需要自己重新学出某种中层块结构。

从系统角度看，可以把这几条路线概括如下：

| 方法 | 关键补偿机制 | 它说明了什么 |
| --- | --- | --- |
| ByT5 [1] | 接受更长序列与更高计算代价 | token-free 可行，但不是免费 |
| CANINE [2] | 下采样压缩字符序列 | 实用字符级模型仍然需要显式压缩 |
| Charformer [3] | 学习潜在 subword 块 | 成功的 token-free 往往会重建中间粒度单元 |

这张表最值得注意的一点是：**现代 token-free 的成功，不是取消压缩，而是把压缩迁移到了模型内部。**

## 6. 所以 character-level 真正输在哪里？

把前面几部分合起来看，character-level 方案通常会在三个维度上同时吃亏：

- 计算效率：序列更长，attention 和缓存成本都更高。
- 优化效率：模型必须先恢复局部组合，再做高层语义建模，学习路径更长。
- 架构匹配：标准 Transformer 更适合直接消费中间粒度单元，而不是从原始字符开始完成全部压缩。

因此，character-level 的问题不是“不正确”，而是“在当前主流架构和预算下，整体系统代价通常更高”。这也是为什么 subword tokenizer 仍然长期占据默认位置：它在模型之外先做掉了一部分局部压缩，让 Transformer 能更早开始自己真正擅长的全局关系建模 [4][5]。

## 7. 结语

如果要把本文压缩成一句话，我会写：

> character-level 最终往往赢不了，不是因为它不能表示语言，而是因为压缩和中间结构发现这件事无论如何都得做，而显式 subword tokenizer 往往是更便宜、更稳定的做法。

这也解释了一个看似矛盾、其实很重要的事实：最成功的 token-free 模型，往往都在想办法把某种“隐式 tokenizer”重新放回模型内部。无论这个机制叫下采样、局部块化还是 latent subword，它都在承担同一类系统职责。

如果把这篇与前两篇[Tokenization 的压缩本质](/blog/theory-of-tokenizers/what-tokenization-does)和[LLM 词表规模的自然平衡点](/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k)一起看，整个逻辑就闭合了：Tokenizer 不是随手加上的工程壳，而是大模型系统如何分配压缩、优化和计算负担的核心设计点。

## 参考文献

[1] XUE L, BARUA A, CONSTANT N, et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 291-306. DOI: [10.1162/tacl_a_00461](https://doi.org/10.1162/tacl_a_00461).

[2] CLARK J H, GARRETTE D, TURC I, et al. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 73-91. DOI: [10.1162/tacl_a_00448](https://doi.org/10.1162/tacl_a_00448).

[3] TAY Y, TRAN V Q, RUDER S, et al. Charformer: Fast Character Transformers via Gradient-based Subword Tokenization[C]// *International Conference on Learning Representations*. 2022. Available: [https://research.google/pubs/charformer-fast-character-transformers-via-gradient-based-subword-tokenization/](https://research.google/pubs/charformer-fast-character-transformers-via-gradient-based-subword-tokenization/).

[4] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[5] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).
