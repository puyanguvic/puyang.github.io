---
title: "Tokenization 的压缩本质"
date: 2026-03-09T12:00:00-08:00
summary: "从平均描述长度、Zipf 分布与 subword 码本设计出发，解释为什么 tokenization 本质上是一种神经网络友好的压缩。"
tags: ["tokenizer", "compression", "LLM"]
---

# Tokenization 的压缩本质

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/theory-of-tokenizers/what-tokenization-does" en-path="/blog/theory-of-tokenizers/what-tokenization-does-en" />

把 tokenizer 仅仅理解为“把文本切成 token 的前处理工具”，会遗漏它在整个语言模型系统中的真正角色。不同 tokenizer 会显著改变序列长度、词表规模、长尾词表示方式以及训练时梯度传播的难度；如果它只是一个中性的分词器，这些系统性差异就很难解释 [1][3-5]。

更准确的理解是：tokenization 首先是一个有限可逆码本的设计问题。它的目标不是追求语言学上最自然的切分，而是在可恢复文本信息的前提下，把原始字符串重编码成更短、更稳定、也更适合神经网络处理的离散序列。

> 核心结论：tokenizer 的首要作用不是“识别词”，而是把语言中的高频局部结构外包给一个显式码本，从而缩短序列、减少统计冗余，并把更多模型容量留给真正的上下文建模；subword 方法之所以长期主导现代 LLM，正是因为它在压缩效率、开放词表与优化稳定性之间实现了最有效的折中 [1-5]。

在“Tokenizer 的理论”系列中，本文先把 tokenization 定义为码本压缩问题；下一篇 [LLM 词表规模的自然平衡点](/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k) 再讨论这套码本为什么不会无限扩张。

## 1. tokenizer 应被定义为怎样的对象？

设原始文本为字符串 $x$，tokenizer 输出 token 序列

$$
\tau(x) = (t_1,\dots,t_m),
$$

并要求存在确定的反向映射 $\gamma$，使得

$$
\gamma(t_1)\gamma(t_2)\cdots\gamma(t_m) = x
$$

或至少在规范化后的文本空间中可恢复。于是，tokenizer 的本质并不是任意切分，而是在构造一个**有限词表上的可逆变长编码系统**。

若记输出序列长度为

$$
T_\tau(x) = m,
$$

则 tokenizer 的一个粗略目标可写成

$$
\min_\tau \ \mathbb{E}_{x \sim \mathcal{D}}[T_\tau(x)],
$$

同时满足以下约束：

- 编码与解码必须确定；
- 词表容量有限；
- 长尾输入仍然可表示；
- 输出序列应当对后续神经网络友好。

这一定义已经说明，tokenizer 从一开始就是码本设计，而不是纯语言学对象。

## 2. 为什么自然语言本身适合被压缩？

压缩成立的前提，不是 tokenizer 足够聪明，而是语言分布本身高度不均匀。Zipf 规律告诉我们，极少数高频单位覆盖了语料中的大部分质量，而其余大量低频单位形成漫长尾部 [2]。这意味着文本中存在大量可复用模式：

- 高频功能词会反复出现；
- 常见词根、前后缀和子词片段会跨词复用；
- 标点模式、空白模式和字节块会在大规模语料中持续重复；
- 许多长尾词可以由更高频的局部结构组合得到。

从 Shannon 的角度看，这正是可压缩源的典型特征：当符号分布存在明显偏斜时，平均描述长度可以通过非均匀编码显著降低 [1]。tokenizer 本质上就是在利用这种分布偏斜，把高频局部模式提升为码本中的显式条目。

## 3. BPE 与 unigram LM 分别在做什么？

现代 LLM tokenizer 的主流方法并不直接从“完整词”出发，而是从字符、字节或更小片段出发，学习一个中间粒度的码本。其中最典型的是 BPE 与 unigram language model。

### BPE：贪心地把高收益片段写进码本

Sennrich 等人将 BPE 引入神经机器翻译后，subword tokenization 迅速成为主流 [3]。BPE 的核心过程是：

1. 从字符或字节级词表开始；
2. 统计相邻片段对的出现频率；
3. 合并最有收益的高频片段对；
4. 重复直到达到目标词表规模。

从压缩视角看，BPE 每一步都在问同一个问题：如果把某个高频局部模式提升为独立码字，语料平均编码长度是否会下降？

### Unigram LM：把分段本身视为概率模型

Kudo 的 unigram LM 则把文本分段看成在固定词表上的概率分配问题 [5]。与贪心合并不同，它显式评估不同分段方案的概率，并通过近似最大似然保留更有价值的子词单元。SentencePiece 进一步把这种思路系统化，使得 tokenizer 可以直接在原始文本上训练，并把词表容量当成一级约束 [4]。

因此，不论是 BPE 还是 unigram LM，核心都不是“语言应该怎样切”，而是“在固定码本预算下，哪些片段最值得被单独记住”。图 1 可以把这一过程压缩成一个统一视角。

![tokenization 作为码本压缩的示意图](./tokenization-compression-codebook.svg)

*图 1. tokenizer 更准确的角色是：从原始字符流中提取高频可复用片段，构成有限码本，再用该码本把文本重编码成更短、更稳定的离散序列。*

图 1 所强调的，不是“切分边界长什么样”，而是“哪些局部结构被提前写进了显式码本”。这也是为什么 tokenizer 会同时影响压缩效率和归纳偏置。

## 4. 为什么 subword 会成为现代 LLM 的稳定解？

不同粒度的 tokenization，本质上是在不同位置切分“显式码本”和“神经网络内部组合”之间的工作边界。

| 粒度 | 主要优势 | 主要代价 |
| --- | --- | --- |
| word-level | 序列短，单 token 语义密度高 | OOV 严重，长尾与新词脆弱 |
| character/byte-level | 开放词表，跨语言表示统一 | 序列过长，局部组合负担重 |
| subword | 兼顾压缩、开放性与可学习性 | 需要额外训练码本，切分不总是语言学自然 |

subword 的长期优势，恰恰来自它的不极端。它做了三件关键的系统工作：

- 把大量高频局部结构提前吸收到码本中，显著缩短序列；
- 保留开放词表能力，使长尾词仍可由更细片段组合表示；
- 为 embedding 学习提供中间粒度单位，让模型不必总从字符级重新恢复常见词法结构。

因此，subword 不是“最语言学正确”的切法，而是最合适的系统分工方案之一。

## 5. tokenizer 究竟压缩了什么？

这里需要一个严格区分。tokenizer 并不是单独优化比特级压缩率；它压缩的是整个神经网络系统的输入表示复杂度。至少有三类对象被同时压缩。

### 序列长度

高频片段被收编为 token 后，同样文本对应的序列会更短。对 Transformer 而言，这会直接改变 attention、缓存和上下文窗口的有效利用率。

### 统计冗余

若某个局部模式反复出现，把它固化为 token 相当于把重复结构存进显式码本，而不是要求模型每次都从字符流中重新发现。

### 学习难度

当高频模式稳定地以相同 token 形式出现时，embedding 与后续层更容易为其学出可迁移表示。否则，模型必须先解决低层组合问题，再进入真正的语义建模。

因此，tokenization 不只是压缩文本，更是在压缩模型需要显式搜索和反复学习的局部结构空间。

## 6. tokenizer 还会注入哪些归纳偏置？

把 tokenizer 看成压缩系统之后，还必须再往前走一步：它并不是中性的压缩器。任何码本一旦被固定下来，就会顺带规定模型优先共享哪些统计规律、忽略哪些局部边界，以及把哪些字符串片段当成“天然可复用单位”。

这意味着 tokenizer 至少会注入三类归纳偏置：

- 分段偏置。模型会优先在 token 边界之内共享统计，而跨边界模式需要更高层去恢复。
- 词法偏置。某些语言中的形态边界与 subword 边界可能对齐，也可能严重错位；这会直接影响长尾词与复杂词形的学习难度。
- 规范化偏置。大小写、空格、标点、字节标准化与 Unicode 处理方式，都会改变模型最终看到的离散符号系统。

因此，tokenizer 的角色不能只被说成“压缩了多少字符”。它同时也在规定：模型将以什么粒度看待语言中的重复结构。压缩与归纳偏置在这里是同一件事的两面。

## 7. 结语

tokenization 的核心，不是把文本“切开”这么简单，而是决定哪些局部结构应由显式码本承担，哪些结构应留给神经网络在上下文中继续组合。看到这一点后，tokenizer 就不再是外围预处理，而是整个 LLM 系统如何分配压缩、泛化与计算负担的核心设计点。

归结起来，**tokenizer 首先是一套神经网络友好的压缩系统。** 一旦把它看成码本设计，词表容量为什么不会无限扩张、而会落在某个中等规模的自然平衡点，也就成为下一步必须回答的问题。

继续阅读：[LLM 词表规模的自然平衡点](/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k)。

## 参考文献

[1] SHANNON C E. A Mathematical Theory of Communication[J]. *Bell System Technical Journal*, 1948, 27(3): 379-423; 27(4): 623-656. URL: [https://www.mpi.nl/publications/item2383162/mathematical-theory-communication](https://www.mpi.nl/publications/item2383162/mathematical-theory-communication).

[2] PIANTADOSI S T. Zipf's Word Frequency Law in Natural Language: A Critical Review and Future Directions[J]. *Psychonomic Bulletin & Review*, 2014, 21(5): 1112-1130. DOI: [10.3758/s13423-014-0585-6](https://doi.org/10.3758/s13423-014-0585-6).

[3] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[4] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).

[5] KUDO T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates[C]// *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Melbourne, Australia: Association for Computational Linguistics, 2018: 66-75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007).
