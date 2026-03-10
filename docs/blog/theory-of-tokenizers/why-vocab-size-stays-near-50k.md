---
title: "LLM 词表规模的自然平衡点"
date: 2026-03-09T12:10:00-08:00
summary: "从序列长度收益递减、长尾稀疏和 softmax 成本的共同权衡出发，解释为什么 LLM 词表规模常停在一个中等区间。"
tags: ["tokenizer", "vocabulary", "LLM"]
---

# LLM 词表规模的自然平衡点

看不同的大模型，你会发现一个反复出现的工程现象：词表大小虽然并不完全一致，但很多系统都会停在一个中等区间，英语或近单语场景里常见的就是几万级，经验上常落在 `30k` 到 `50k` 附近，而不会无限增大或无限缩小。这个现象并不神秘。词表规模本来就不是一个可以独立优化的按钮，它同时影响输入序列长度、embedding 稀疏性、输出层分类成本以及长尾词的可学习性。

因此，一个更专业的说法是：

> 词表大小不是“越大越好”或“越小越统一”的单变量问题，而是压缩收益与模型成本之间的平衡点问题。

> 核心结论：随着词表规模 $V$ 增加，平均序列长度 $T(V)$ 确实会下降，但这种下降受 Zipfian 长尾分布支配而迅速出现边际递减；与此同时，embedding 参数、输出层成本和长尾 token 的学习稀疏性会近似随 $V$ 线性恶化。因此，现代 LLM 往往在一个中等词表区间达到更好的整体系统最优，而不是把所有高频和长尾片段都硬塞进独立 token [1-6]。

## 1. 先把问题写成一个 tradeoff

设 tokenizer 词表规模为 $V$，平均 token 序列长度为 $T(V)$。对 Transformer 而言，一个粗略但有用的系统成本模型可以写成

$$
\mathcal{J}(V)
\approx
a \, T(V)^2 + b \, T(V) + c \, V,
$$

其中：

- $aT(V)^2$ 对应 attention 这类随序列长度超线性增长的成本；
- $bT(V)$ 对应缓存、数据传输和线性层上的长度成本；
- $cV$ 对应词表带来的 embedding、输出层和参数规模开销。

这个式子没有假装给出严格闭式解，但它抓住了真正关键的结构：**扩大词表会减少序列长度，却会线性增加词表侧成本**。是否继续增大 $V$，本质上取决于“减少一个 token 的收益”是否仍然大于“再新增一个词表项的代价”。

## 2. 为什么序列长度收益会迅速递减？

问题的根源在于自然语言的长尾分布。Piantadosi 对 Zipf 规律的综述表明，语言中的频率质量主要集中在头部少量单位，而尾部极长 [1]。这意味着词表扩张并不是均匀有益的：

- 早期加入的高频 token 会显著缩短序列；
- 继续增大词表时，新增 token 覆盖的往往是越来越稀有的片段；
- 到尾部时，很多新增词表项只是在替换极少出现的局部字符串。

从压缩角度看，$T(V)$ 的确是单调下降的，但其斜率会越来越小，也就是

$$
T'(V) < 0,
\qquad
|T'(V)| \to 0.
$$

这就是边际递减的数学表达。也正因如此，词表扩张的前期收益很大，后期收益却会迅速趋缓。

## 3. 为什么词表太大也会出问题？

序列短一些当然有利，但词表不是免费的。词表一旦扩大，至少会引入三类代价。

### 长尾 embedding 稀疏

Sennrich 等人引入 subword 的一个核心动机，就是让长尾词由较高频片段组合来表示，而不是为每个罕见词硬分配一个独立参数 [2]。如果词表过大，尾部会出现大量训练次数极低的 token。它们很难学出稳定 embedding，最终只是参数上的“孤岛”。

### 输出层和采样成本增加

词表规模越大，下一 token 预测的分类空间就越大。即使工程上采用分块 softmax、采样近似或 fused kernel，这个方向的成本也不会凭空消失。

### 泛化能力可能反而下降

当某个低频字符串被硬编码成专门 token 时，模型可能失去通过更高频 subword 组合来共享统计规律的机会。对长尾词而言，“拆成常见片段”往往比“训练一个罕见独立 token”更稳健。

因此，过大的词表不只是更占参数；它还会系统性加重尾部稀疏问题。

## 4. 为什么词表太小也不行？

另一边，词表也绝不能太小。词表过小意味着 token 粒度过细，文本会被切成更长序列。对于 Transformer，这会带来一整串连锁反应：

- attention 计算更贵；
- KV cache 更大；
- 固定上下文窗口能容纳的真实文本更少；
- 模型必须花更多层和更多路径，先把局部字符块重新组合成熟悉的词法单元。

Kudo 与 Richardson 的 SentencePiece 明确把“预先设定词表容量，再让算法在这个容量约束下做最优分段”作为设计中心 [3]。这本身就说明：词表大小不是后验装饰，而是 tokenizer 训练时的一级约束。

如果词表持续缩小，最终就会逼近字节级或字符级建模。ByT5 的结果恰恰说明，完全去掉子词词表虽然可以提升鲁棒性和开放性，但也会带来更长序列和更重的训练/推理代价 [6]。这说明“词表尽量小”同样不是免费的。

## 5. 为什么中等规模会成为稳定工程解？

把前面几部分合在一起，就得到一个非常清楚的结论：词表规模必须停在一个中间区域，使得两侧代价都不过分。

![词表规模的权衡曲线示意图](./vocab-size-tradeoff.svg)

*图 1. 词表增大时，序列长度成本会下降，但边际收益递减；与此同时，词表参数、输出层和长尾稀疏代价近似线性增加。系统总成本因此更可能在一个中等区间出现低点。*

从工程视角看，这个中间区间通常具备以下特征：

| 词表区间 | 主要问题 | 系统表现 |
| --- | --- | --- |
| 太小 | 序列过长，字符/字节组合负担重 | 计算贵，上下文利用率低 |
| 中等 | 高频模式已被吸收，长尾仍可分解 | 压缩、泛化和可优化性较平衡 |
| 太大 | 长尾 token 稀疏，输出层与词表参数膨胀 | 额外收益变小，学习不均衡 |

这也是为什么 Devlin 等人的 BERT 这类英语模型采用了中等规模的 WordPiece 词表 [4]，而像 SentencePiece 这样的工具也把“给定一个固定容量、在容量内寻找最好分段”视为标准工作方式 [3]。二者背后都是同一个设计逻辑：**词表需要足够大，吸收头部统计结构；但也必须足够小，避免尾部和输出层拖垮整体系统。**

## 6. 为什么不同任务会偏离这个平衡点？

当然，这个平衡点不是普适常数。它会随任务和语料改变而移动。

- 多语言模型需要覆盖更多脚本与形态变化，往往需要更大词表或不同字节策略。
- 代码模型面对标识符、符号和长字符串模式时，最佳词表会与自然语言不同。
- 高度垂直的领域模型若语料更集中，有时可以用更小词表吸收主要模式。
- 字节级方案会把部分问题转移到更细粒度序列处理上，从而换取开放性和鲁棒性 [6]。

因此，更准确的说法不是“30k–50k 永远正确”，而是：**现代单语或近单语 LLM 往往会在一个中等规模 subword 词表附近遇到自然的收益递减点。**

## 7. 结语

如果要把本文压缩成一句话，我会写：

> 词表规模之所以常停在一个中等区间，不是经验巧合，而是因为序列压缩收益递减得很快，而词表侧成本上升得很稳。

这也说明，词表大小不是 tokenizer 的局部超参数，而是整个模型系统设计的一部分。它决定了模型如何在两个方向上分配负担：

- 是把更多结构预先交给显式码本；
- 还是把更多组合工作留给神经网络在更长序列上自己学习。

如果把这篇与上一篇[Tokenization 的压缩本质](/blog/theory-of-tokenizers/what-tokenization-does)一起看，逻辑会更完整：前者说明 tokenizer 本质上是在做码本压缩，这一篇则说明码本容量为什么不能无限扩张，而必须停在一个更合理的工程平衡点。

下一篇文章，我们会继续这个问题的另一端，讨论为什么看似最统一、最“彻底”的 character-level tokenizer，在现代 Transformer 体系里通常仍然赢不过 subword。

## 参考文献

[1] PIANTADOSI S T. Zipf's Word Frequency Law in Natural Language: A Critical Review and Future Directions[J]. *Psychonomic Bulletin & Review*, 2014, 21(5): 1112-1130. DOI: [10.3758/s13423-014-0585-6](https://doi.org/10.3758/s13423-014-0585-6).

[2] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[3] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).

[4] DEVLIN J, CHANG M-W, LEE K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]// *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. Minneapolis, Minnesota: Association for Computational Linguistics, 2019: 4171-4186. DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423).

[5] KUDO T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates[C]// *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Melbourne, Australia: Association for Computational Linguistics, 2018: 66-75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007).

[6] XUE L, BARUA A, CONSTANT N, et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 291-306. DOI: [10.1162/tacl_a_00461](https://doi.org/10.1162/tacl_a_00461).
