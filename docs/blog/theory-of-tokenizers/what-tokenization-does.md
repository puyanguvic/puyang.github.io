---
title: "Tokenization 的压缩本质"
date: 2026-03-09T12:00:00-08:00
summary: "从平均描述长度、Zipfian 频率分布与 subword 码本设计出发，解释为什么 tokenization 本质上是一种神经网络友好的压缩。"
tags: ["tokenizer", "compression", "LLM"]
---

# Tokenization 的压缩本质

很多人在第一次接触大模型时，会把 tokenizer 看成一个前处理工具：它把字符串切成 token，然后模型再去处理这些 token。这个描述并不算错，但它解释力很弱。因为如果 tokenizer 只是“切词器”，我们就很难解释：为什么不同 tokenizer 会显著影响序列长度、训练效率、上下文利用率，甚至影响模型对长尾词和跨语言文本的处理方式。

更准确的理解是：

> tokenization 的本质不是分词，而是为神经网络设计一套可逆、可学习、可压缩的离散码本。

这里“压缩”不是指传统压缩器那种单纯追求最短比特串的目标，而是指：在尽量保留可恢复文本信息的前提下，把原始字符串重新编码成更短、更稳定、更适合下游表示学习的符号序列 [1][3-5]。

> 核心结论：tokenizer 的首要作用，是把语言中高频、可复用的局部结构外包给一个显式码本，从而减少序列长度、降低统计冗余，并把更多计算预算留给真正的上下文建模；subword 方法之所以长期主导现代 LLM，并不是因为它最“语言学正确”，而是因为它在压缩效率、开放词表和优化稳定性之间取得了最好的折中 [1-5]。

## 1. 先给一个更严格的定义：tokenizer 是可逆码本

设原始文本是一个字符串 $x$，tokenizer 输出一个 token 序列

$$
\tau(x) = (t_1, t_2, \dots, t_m),
$$

并要求存在确定的反向映射，使得

$$
\gamma(t_1)\gamma(t_2)\cdots\gamma(t_m) = x
$$

或至少在规范化后的文本空间中可恢复。这里的 $\gamma$ 可以理解为“token 到字符串片段”的码本映射。于是，tokenizer 做的并不是任意切分，而是在构造一个**有限词表上的可逆变长编码系统**。

如果把 token 序列长度记为 $T_\tau(x)=m$，那么 tokenizer 的一个粗略目标可以写成：

$$
\min_\tau \ \mathbb{E}_{x \sim \mathcal{D}} \big[T_\tau(x)\big]
$$

同时满足：

- 编码与解码可确定；
- 词表规模有限；
- 长尾输入仍然可表示；
- 输出序列对神经网络是可优化的。

这说明 tokenizer 从一开始就不是纯语言学对象，而是一个**码本设计问题**。

## 2. 为什么语言天然适合被压缩？

压缩之所以可能，不是因为 tokenizer 足够聪明，而是因为语言分布本来就高度不均匀。Piantadosi 对 Zipf 规律的综述指出，自然语言中的词频分布稳定地呈现强长尾结构：极少数高频单位覆盖大量文本质量，而绝大多数单位落在低频尾部 [2]。

这意味着文本中存在大量可复用模式：

- 高频功能词会反复出现；
- 高频词根、前后缀和语素片段会跨词复用；
- 常见拼写块、字节模式和标点结构会反复出现；
- 很多长尾词可以由更高频的子结构组合而成。

这正是 Shannon 意义下“可压缩源”的典型特征：如果一个符号源具有显著的统计偏斜，那么平均描述长度就可以通过非均匀编码显著降低 [1]。Tokenizer 本质上就是在利用这种偏斜。

## 3. BPE 与 unigram LM：两种典型的码本构造方式

现代 LLM tokenizer 的主流做法并不是直接从“词”出发，而是从字符、字节或最小片段出发，逐步学习一个中间粒度的词表。最常见的两类方法分别是 BPE 和 unigram language model。

### BPE：逐步合并最有收益的高频片段

Sennrich 等人把 BPE 引入神经机器翻译后，这一方法成为 subword tokenization 的标准基线 [3]。它的核心思想非常简单：

1. 从字符或字节级词表开始；
2. 统计语料中最频繁的相邻片段对；
3. 把收益最高的片段对合并为新 token；
4. 重复直到达到目标词表规模。

从压缩角度看，BPE 的每一步都在问同一个问题：

> 如果把这个高频局部模式提升为显式码字，语料的平均编码长度会不会下降？

### Unigram LM：直接把分词看成词表上的概率模型

Kudo 提出的 unigram LM 则更进一步，把分词本身建模为“给定固定词表后，对文本所有可能分段的概率分配”，再通过近似最大似然学习词表 [5]。和 BPE 相比，它不是贪心合并，而是显式地在“哪些片段值得保留”为独立 token 这个问题上做概率建模。

Kudo 与 Richardson 的 SentencePiece 又把这类做法系统化了，使 subword 训练可以直接从原始句子出发，并把词表大小作为训练前就指定的超参数 [4]。这一步很关键，因为它把 tokenizer 更明确地变成了**固定容量码本**的设计问题。

![tokenization 作为码本压缩的示意图](./tokenization-compression-codebook.svg)

*图 1. tokenizer 更准确的角色是：从原始字符流中抽取高频可复用片段，构成一个有限码本，再用该码本把文本重编码成更短、更稳定的 token 序列。*

## 4. 为什么 subword 会成为现代 LLM 的默认选择？

subword tokenizer 的成功，并不是偶然经验，而是因为它恰好落在一个非常有效的中间层级上。

| 粒度 | 优点 | 缺点 |
| --- | --- | --- |
| word-level | 序列短，单 token 语义强 | OOV 严重，长尾和新词脆弱 |
| character/byte-level | 完全开放词表，表示最统一 | 序列过长，语义形成路径太长 |
| subword | 兼顾压缩、开放词表与可学习性 | 需要训练一个额外码本，切分并不总是语言学自然 |

从系统设计角度看，subword 有四个非常直接的优势 [3-5]：

- 它吸收了大部分高频局部结构，显著缩短序列长度。
- 它仍然保留开放词表能力，长尾词与新词可以拆分表示。
- 它为 embedding 学习提供了更稳定的中间单元，不必把一切都留给字符级组合。
- 它与 Transformer 的计算结构更匹配，因为模型可以更早在较高语义密度的单位上开展上下文建模。

因此，subword 的主导地位不是因为它“最纯粹”，恰恰相反，是因为它最不极端。

## 5. tokenizer 到底压缩了什么？

这里需要一个技术上的澄清：tokenizer 不是在独立地最小化比特长度，它是在为整个神经网络系统压缩输入表示。它真正压缩的至少有三件事。

### 序列长度

高频模式被显式吸收入词表后，同样文本会对应更短的 token 序列。这直接影响上下文窗口的可用内容，也直接影响 Transformer 的计算成本。

### 统计冗余

把高频片段固化为 token，相当于把重复出现的局部结构显式存进码本，而不是要求模型每次都从字符流中重新发现它们。

### 学习难度

如果某个模式总是以相近 token 形式出现，embedding 和后续层就更容易为它学出稳定表示；否则，模型必须先解决低层组合问题，再去学习更高层语义。

因此，tokenizer 不只是在压缩文本，也是在压缩模型的搜索空间与优化负担。

## 6. 结语

如果要把本文压缩成一句话，我会写：

> tokenization 的本质，是把自然语言中的高频局部结构外包给一个显式码本，从而把原始字符串重新编码成更适合神经网络处理的压缩符号系统。

这也解释了为什么 tokenizer 不是一个无关紧要的前处理细节。它会直接影响：

- 序列有多长；
- 词表有多大；
- 长尾词如何被表示；
- 模型把多少计算花在“先恢复局部结构”而不是“直接建模上下文”上。

如果把这个结论放回整个系列中看，它与前文讨论的[Embedding 空间中的语义线性结构](/blog/representation-space-of-large-models/semantic-linearity)和[LLM Embedding 的球面编码视角](/blog/representation-space-of-large-models/spherical-coding)是一致的：Tokenizer 决定了模型看到哪些离散单位，而这些单位随后又决定了 embedding 空间将如何组织、压缩和泛化。

下一篇文章，我们会继续往工程设计层面走，讨论为什么现代 LLM 的词表大小通常不会无限增大，而往往停在一个中等规模的稳定区间。

## 参考文献

[1] SHANNON C E. A Mathematical Theory of Communication[J]. *Bell System Technical Journal*, 1948, 27(3): 379-423; 27(4): 623-656. Available: [https://www.mpi.nl/publications/item2383162/mathematical-theory-communication](https://www.mpi.nl/publications/item2383162/mathematical-theory-communication).

[2] PIANTADOSI S T. Zipf's Word Frequency Law in Natural Language: A Critical Review and Future Directions[J]. *Psychonomic Bulletin & Review*, 2014, 21(5): 1112-1130. DOI: [10.3758/s13423-014-0585-6](https://doi.org/10.3758/s13423-014-0585-6).

[3] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[4] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).

[5] KUDO T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates[C]// *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Melbourne, Australia: Association for Computational Linguistics, 2018: 66-75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007).
