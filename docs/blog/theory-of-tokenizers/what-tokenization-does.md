---
title: "Tokenization 的压缩本质"
date: 2026-03-09T12:00:00-08:00
summary: "从可逆重编码、局部互信息粗粒化与神经计算预算出发，解释为什么 tokenization 本质上是面向序列模型的源重参数化。"
tags: ["tokenizer", "compression", "LLM"]
---

# Tokenization 的压缩本质

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/theory-of-tokenizers/what-tokenization-does" en-path="/blog/theory-of-tokenizers/what-tokenization-does-en" />

把 tokenizer 理解成“把文本切成 token 的前处理工具”，会把它降格成一个外围实现细节。真正更贴切的说法是：tokenizer 决定了语言模型究竟以什么离散单位观察文本，也决定了哪些局部规律被提前写进显式符号系统，哪些规律必须留给深层网络继续学习。

这件事之所以重要，是因为 tokenization 并不只是改变字符串的外观。它会同时改变序列长度、统计共享方式、长尾词的可组合性，以及模型必须显式完成多少低层局部压缩工作 [1][3-5]。如果把这些影响放在一起看，tokenizer 的角色就不再是“切分”，而是**对原始文本源做一次面向神经网络的重参数化**。

> 核心观点：tokenization 的本质不是“识别词”，而是对字符串源做一次受词表预算约束的可逆粗粒化。它把高频、强局部依赖的结构提前收编进显式码本，把原本需要跨多个时间步建模的低层冗余压缩到 token 身份中，从而缩短预测链条、提高统计共享效率，并把更多计算预算留给真正的上下文建模 [1-5]。


## 1. tokenizer 首先是一种可逆重编码

设原始文本由字符或字节序列

$$
x = (c_1,\dots,c_n)
$$

构成，tokenizer 输出 token 序列

$$
\tau(x) = (t_1,\dots,t_m),
$$

并存在确定的解码映射 $\gamma$，使得

$$
\gamma(t_1)\gamma(t_2)\cdots\gamma(t_m)=x
$$

或至少在规范化后的文本空间中可恢复。这样一来，tokenization 就不是任意切分，而是一个**有限词表上的可逆变长重编码**。

这个定义带来一个常被忽略的关键点：如果编码是确定且可逆的，那么 tokenizer 并没有凭空消灭信息。Shannon 意义下源的总信息量并不会因为“换了一套符号”而神奇减少 [1]。真正发生变化的是：

- 模型要做多少个离散预测步骤；
- 每一步的不确定性被定义在什么粒度上；
- 哪些局部相关性已经被编码进 token 身份，哪些仍暴露在序列维度上。

换句话说，tokenizer 改变的不是信息守恒本身，而是**信息在“token 身份”与“token 间依赖”之间的分配方式**。

从系统设计角度，一个更贴切的目标不是单独最小化字符级比特数，而是最小化某个模型族上的总训练负担。可以把这一点粗略写成

$$
\min_{\tau}\ 
\mathcal{L}_{\mathcal{F}}(\tau;\mathcal{D})
\;+\;
\lambda\,\mathbb{E}_{x\sim\mathcal{D}}[T_{\tau}(x)]
\;+\;
\beta\,|V_{\tau}|,
$$

其中：

- $\mathcal{L}_{\mathcal{F}}$ 表示给定模型族 $\mathcal{F}$ 在这种 tokenization 下的建模难度；
- $T_{\tau}(x)$ 是 token 序列长度；
- $|V_{\tau}|$ 是词表大小。

这已经说明，tokenizer 从来不是中性的预处理，而是和模型能力、计算预算共同决定的一层离散接口。

## 2. tokenization 真正利用的是局部互信息

自然语言之所以可被 tokenizer 有效压缩，不是因为 tokenizer “理解了意义”，而是因为语言分布本身高度非均匀。Zipf 规律说明，头部少数模式覆盖了大量频率质量，长尾则由大量罕见形式组成 [2]。但更关键的一点是：这些高频模式并不是孤立符号，而常常体现为**局部共现非常强的字符块或子词块**。

如果两个相邻片段 $u,v$ 经常一起出现，那么它们之间的统计依赖就很强。用互信息语言说，这意味着知道 $u$ 之后，对 $v$ 的不确定性会显著下降。把这样的模式收编成单个 token，等价于把原本分布在多个时间步中的局部可预测性，提前折叠进一个离散符号。

这一步并没有打破 Shannon 极限。它做的是另一件更工程化、但同样关键的事：

- 把一部分短程冗余从“序列中的重复预测”迁移到“词表中的显式条目”；
- 把若干低级预测步骤合并成一次更高粒度的预测；
- 让模型更少地处理拼写级、词法级的重复结构。

因此，tokenization 不是在创造信息优势，而是在做**局部互信息的粗粒化**。高频前后缀、常见拼写块、功能词、空白和标点模式之所以值得成为 token，不是因为它们“像词”，而是因为它们携带了可以被提前结算的局部统计结构。

## 3. BPE 与 unigram LM 本质上都在学习“哪些局部结构值得单独记住”

一旦把 tokenizer 理解为受预算约束的粗粒化，BPE 和 unigram LM 的差异就容易放到同一框架里。

### BPE：贪心地吸收高收益局部模式

Sennrich 等人把 BPE 引入神经机器翻译后，subword tokenization 成为现代 NLP 的默认选择之一 [3]。BPE 的基本操作是不断合并高频相邻片段对。看起来它像是在“造词”，其实它更接近在问：

> 如果把某个反复共同出现的局部模式提升为独立码字，整个语料的序列长度与建模负担会下降多少？

BPE 的贪心性并不妨碍它抓住最值钱的头部模式，因为真正有巨大收益的局部结构，本来就集中在高频区域。

### Unigram LM：直接把分段看成概率优化问题

Kudo 的 unigram language model 更进一步，把一段字符串的不同切分方案显式看成概率对象 [5]。SentencePiece 则把这种思路做成通用工具：直接在原始文本上训练词表，并把词表容量当成一级约束 [4]。

与 BPE 相比，unigram LM 不再只靠贪心合并局部对，而是直接评估“整个切分体系”的质量。但它的目标并没有变。它仍在学习：**在固定词表预算下，哪些子串最值得拥有独立身份，哪些应继续作为组合结构存在。**

因此，两条路线虽然算法细节不同，真正共享的对象都是同一个：有限码本下的局部结构选择问题。

![tokenization 作为码本粗粒化的示意图](./tokenization-compression-codebook.svg)

*图 1. tokenizer 的关键不是寻找“最自然的词边界”，而是把高频、可复用、局部依赖强的结构吸收到显式码本里，减少模型反复处理的短程冗余。*

## 4. tokenizer 对神经网络最深的影响，不是压缩率，而是统计效率

如果只说 tokenizer 会缩短序列，仍然太浅。更重要的地方在于，它改变了模型参数看到统计规律的方式。

### 把重复上下文汇聚到同一参数行

当一个高频片段被稳定编码成单个 token 时，与之相关的大量上下文梯度会汇聚到同一行 embedding 和同一输出类别上。这样，模型可以更快学出这个片段的稳定表示。反过来，如果它总是拆成若干字符，模型就必须通过更长的组合链条，间接恢复同一个局部模式。

### 把“已解决的短程结构”从序列维度移除

一旦某些常见局部模式被并入 token 身份，这些模式在序列中就不再需要被逐步预测。可以把它理解成：**intra-token 的依赖被提前结算了，Transformer 只需处理 inter-token 的依赖。**

这件事对标准 Transformer 尤其关键。Transformer 擅长在已经具有一定语义密度的离散单元之间建模全局关系，却并不特别擅长以高代价从极长原始字符流中反复恢复低层块结构。

### 决定哪些规律会被优先共享

任何固定词表都在定义一个“默认共享坐标系”：

- token 内部的结构被视为已知块；
- token 之间的结构需要由上下文模型显式学习；
- 能够稳定对齐到 token 边界的规律，更容易得到参数共享与统计累积。

因此，tokenizer 真正注入的并不只是压缩比，而是整个模型的低层归纳偏置。

## 5. 为什么 subword 会长期成为稳定解？

从这个视角看，word-level 与 character-level 其实分别走向了两个极端。

| 粒度 | 吸收进显式码本的结构 | 主要问题 |
| --- | --- | --- |
| word-level | 太多，连长尾词都想整体记住 | OOV 严重，长尾统计无法共享 |
| character/byte-level | 太少，几乎不预付局部结构 | 序列过长，低层压缩被推迟到网络内部 |
| subword | 吸收头部局部模式，保留尾部组合性 | 仍有分段偏置，但系统平衡更好 |

subword 的优势并不在于它最符合语言学边界，而在于它更接近语言的多尺度结构：

- 高频局部块值得被单独记住；
- 长尾形式仍应允许由更高频片段组合出来；
- 模型应尽早看到有一定语义密度的单位，但不必为每个罕见词分配独立参数行。

因此，subword 本质上是一个中尺度 coarse-graining：既不把词表做成脆弱的整词字典，也不把所有局部结构都留给网络从字符级重新发明。

## 6. 结语

tokenization 的深层作用，不是把文本“切开”，而是决定哪些局部规律由显式离散码本承担，哪些规律必须留给神经网络在序列中继续建模。只要把这一点说清楚，tokenizer 就不再是语言学附属物，而是整个 LLM 输入接口的第一层表示理论。

更紧凑地说，**tokenizer 是对字符串源做预算受限的可逆粗粒化。** 它不改变信息守恒，却会显著改变信息以何种粒度流入模型、在哪一层被压缩，以及模型需要为低层冗余支付多少显式计算。

继续阅读：[LLM 词表规模的自然平衡点](/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k)。

## 参考文献

[1] SHANNON C E. A Mathematical Theory of Communication[J]. *Bell System Technical Journal*, 1948, 27(3): 379-423; 27(4): 623-656. URL: [https://www.mpi.nl/publications/item2383162/mathematical-theory-communication](https://www.mpi.nl/publications/item2383162/mathematical-theory-communication).

[2] PIANTADOSI S T. Zipf's Word Frequency Law in Natural Language: A Critical Review and Future Directions[J]. *Psychonomic Bulletin & Review*, 2014, 21(5): 1112-1130. DOI: [10.3758/s13423-014-0585-6](https://doi.org/10.3758/s13423-014-0585-6).

[3] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[4] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).

[5] KUDO T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates[C]// *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Melbourne, Australia: Association for Computational Linguistics, 2018: 66-75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007).
