---
title: "LLM 词表规模的自然平衡点"
date: 2026-03-09T12:10:00-08:00
summary: "从边际压缩收益、尾部统计稀疏与输出层估计成本出发，解释为什么值得拥有独立参数行的 token 数量通常只落在一个中等区间。"
tags: ["tokenizer", "vocabulary", "LLM"]
---

# LLM 词表规模的自然平衡点

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k" en-path="/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k-en" />

如果 tokenizer 是一套预算受限的显式码本，那么“词表该多大”就不能再被理解成普通超参数搜索。每向词表里加入一个新 token，本质上都是在问：这个字符串片段是否值得拥有自己的 embedding 行、自己的输出类别，以及自己的独立统计身份？

这才是词表规模问题真正专业的写法。因为一个 token 并不只是让序列变短一点，它还意味着：

- 这个片段今后会被模型当成原子单元处理；
- 它会在输入侧占据一行参数；
- 它会在输出侧占据一个 softmax 类别；
- 它的出现次数会不再和相邻子词共享统计。

因此，词表不可能无限膨胀。一个片段只有在**压缩收益足够大、复用频率足够高、统计估计足够稳定**时，才值得被升级成独立 token。现代 LLM 词表之所以经常落在 `30k` 到 `100k` 这一中等区间，背后并不是经验巧合，而是这三个条件在 Zipf 长尾语料上的共同结果 [1-7]。

> 核心观点：决定词表规模的关键，不是“能否继续缩短序列”，而是“新增片段是否值得拥有独立参数行”。随着词表扩张，真正高收益且高复用的片段很快被吸收完，剩余候选大多落入 Zipf 长尾，既难以进一步显著压缩序列，又因样本稀疏而难以学稳；与此同时，输出层维度和频率不均衡仍持续恶化。因此，最佳词表通常只会停在一个中等规模区间，而不会向百万级自由膨胀 [1-7]。


## 1. 一个新 token 究竟买来了什么？

设某个候选片段 $g$ 在语料中出现频率为 $f_g$，如果把它从若干更小单位合并成一个新 token，平均能节省 $\Delta T_g$ 个序列位置。那么它带来的直接收益大致与

$$
f_g \,\Delta T_g
$$

成正比：出现越频繁、每次节省的位置越多，它越值得被收编进词表。

但这还只是收益侧。新增一个 token 还会引入三类代价。

### 参数代价

它需要自己的 embedding 行，通常也需要自己的输出头参数或与输出头耦合的几何位置 [6]。

### 分类代价

下一 token 预测是一个大小为 $V$ 的分类问题。每扩大一次词表，softmax 的归一化与类别竞争都会变得更复杂。

### 统计代价

最关键也最容易被忽略的是：把一个片段升级为独立 token，等于切断了它与组成子词之间的参数共享。如果这个片段本身出现很少，那么它的专属参数就会变成训练不足的孤岛。

把这些量放在一起，一个粗略但有解释力的增量判断可以写成

$$
\Delta \mathcal{J}(g)
\approx
- \alpha\, f_g \Delta T_g
+ \beta
+ \frac{\gamma}{\sqrt{f_g}},
$$

其中：

- 第一项代表序列缩短与局部建模负担下降带来的收益；
- 第二项代表新增类别和参数的固定系统成本；
- 第三项代表稀疏样本下独立参数更难学稳的统计惩罚。

这不是一条精确拟合式，但它准确抓住了决策结构：**只有当一个片段既足够常见、又确实带来可观压缩收益时，它才值得成为独立 token。**

## 2. Zipf 长尾会让“继续加词表”很快失去意义

词表扩张之所以很快进入收益递减区，原因不是工程师保守，而是语言分布本身的形状。Zipf 规律意味着，头部少数模式覆盖了大量概率质量，而尾部则由海量低频形式组成 [1]。

这对 tokenizer 有一个直接后果：

- 词表扩张早期，新增 token 往往对应高频局部模式，$f_g$ 大，$\Delta T_g$ 也大；
- 继续扩张后，真正还能新增的候选越来越尾部，$f_g$ 明显下降；
- 到更深尾部时，即使某些片段本身较长，它们也往往只在极少上下文中出现，难以形成稳定复用。

因此，词表扩张不是均匀获利，而是先吃掉头部最值钱的模式，再迅速进入一大片“可表示但不值得单独记住”的区域。

换句话说，**语言里当然存在海量不同字符串，但值得拥有独立统计身份的字符串远少于所有可见字符串。** 这一点对于理解 `50k` 一类经验规模尤其关键。词表的自然上限并不是“世界上有多少种词”，而是“在给定训练预算下，有多少种局部形式既高频又足够可复用，值得模型单独为它们留出一行参数”。

## 3. 真正限制词表的，不只是压缩率，而是样本效率

这正是许多对词表规模的讨论最容易失焦的地方。一个 token 不是免费的码字，而是一组必须通过梯度反复估计的参数。出现次数过低时，独立 token 往往有三种问题。

### embedding 学不稳

低频 token 收到的上下文更新太少，其向量更容易受噪声主导，而不是学出稳定可迁移的语义。

### 输出概率难校准

语言模型输出头不仅要给高频词分配概率，也要处理长尾类别之间的竞争。稀有类别太多时，模型更容易把概率质量浪费在大量训练不足的尾部行上 [7]。

### 组合泛化被削弱

如果某个罕见字符串被硬编码成单个 token，它就失去了通过更高频子词共享统计的机会。对长尾词而言，能够复用高频片段常比拥有一个训练不足的独立 token 更划算 [2-4]。

因此，词表设计的关键条件并不是“这个片段能不能被单独编码”，而是“这个片段有没有足够多的数据支持它作为原子单元存在”。这也是为什么 subword 方案长期优于大而脆弱的整词词表：它把长尾词的统计重新汇聚回高频片段上 [2][3]。

## 4. 为什么常见最优区间会落在几十 k，而不是几百或几百万？

这里需要非常明确地说一句：`50k` 不是理论常数。不同语料、脚本系统、任务形式和训练预算，都可能把最优点推向别处。

但它也绝不是随便来的经验数。对许多单语或近单语 LLM，几十 k 规模之所以反复出现，是因为这个区间通常已经覆盖了：

- 最高频的功能词和常见整词；
- 大量高复用词根、前后缀和稳定拼写块；
- 足够多的数字、标点、空白和格式模式；
- 又没有把词表推进到大面积尾部稀疏区。

可以换一个更严格的说法：在 web 级别或通用语料上，真正同时满足“高频”“可压缩”“可复用”“可稳定估计”四个条件的局部片段，数量通常并不会无限增长，而更可能在一个中等规模范围内就基本被收编完。

这也是为什么很多成熟系统最终都停在一个相似量级：不是因为所有人都迷信某个整数，而是因为他们面对的是同一类 Zipf 头尾结构、同一类 softmax 代价，以及同一类有限样本估计问题。

![词表规模的权衡曲线示意图](./vocab-size-tradeoff.svg)

*图 1. 词表扩张早期会迅速吸收高收益头部模式，但越往尾部，新增 token 的压缩收益越小，而类别与统计稀疏成本继续累积，因此总收益更可能在中等区间见顶。*

## 5. 词表太小为什么同样会出问题？

如果只看到“大词表有尾部稀疏”，又会走到另一个极端。词表太小时，系统会把本该由显式码本承担的大量局部结构重新暴露在序列维度上。这会导致：

- 序列更长，attention 与 cache 成本上升；
- 模型必须花更多层先恢复词法块，再进入更高层语义；
- 输出预测步骤变多，每一步只携带更少的局部结构。

ByT5 已经说明，byte-level 路线在开放词表与噪声鲁棒性上确有优势，但代价就是更长序列与更高系统成本 [5]。因此，小词表也不是天然更“纯粹”或更“理论正确”，它只是把粗粒化工作推迟到了模型内部。

## 6. 所以词表规模问题的真正答案是什么？

最准确的答案不是“50k 最好”，而是：

> 一个片段只有在它的边际压缩收益，能够长期覆盖它作为独立参数行和独立类别所带来的统计与计算代价时，才值得进入词表。

一旦用这个标准衡量，很多直觉就自然统一了：

- 为什么整词词表会被长尾拖垮：因为大量词型不够频繁，撑不起独立参数。
- 为什么 subword 词表不必无限做大：因为越往尾部，能节省的位置和能稳定学到的统计都快速减少。
- 为什么几十 k 的量级反复出现：因为最有价值的局部模式通常在这个区间内就基本覆盖，而继续扩张主要是在购买越来越昂贵的尾部稀疏性。

因此，词表规模的本质，不是几何容量问题，也不是“压缩得越狠越好”的单指标问题，而是一个**压缩收益、样本效率与输出层复杂度共同决定的统计决策**。

## 7. 结语

词表之所以常停在中等规模，不是因为 tokenizer 训练偷懒，而是因为“值得拥有独立身份的局部模式”本来就比“可观察到的不同字符串”少得多。头部模式很值钱，尾部模式很昂贵，而 subword 词表恰好停在这两者之间。

更紧凑地说，**词表大小决定的不是码本有多豪华，而是模型愿意把多少局部结构升级为原子统计对象。** 一旦继续缩小这个显式码本，问题就会转向下一篇：如果几乎什么都不预付，character-level 路线究竟会把怎样的负担推回模型内部？

继续阅读：[Character-Level Tokenizer 的理论优势与工程局限](/blog/theory-of-tokenizers/why-character-level-rarely-wins)。

## 参考文献

[1] PIANTADOSI S T. Zipf's Word Frequency Law in Natural Language: A Critical Review and Future Directions[J]. *Psychonomic Bulletin & Review*, 2014, 21(5): 1112-1130. DOI: [10.3758/s13423-014-0585-6](https://doi.org/10.3758/s13423-014-0585-6).

[2] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[3] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).

[4] KUDO T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates[C]// *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Melbourne, Australia: Association for Computational Linguistics, 2018: 66-75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007).

[5] XUE L, BARUA A, CONSTANT N, et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 291-306. DOI: [10.1162/tacl_a_00461](https://doi.org/10.1162/tacl_a_00461).

[6] PRESS O, WOLF L. Using the Output Embedding to Improve Language Models[C]// *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*. Valencia, Spain: Association for Computational Linguistics, 2017: 157-163. URL: [https://aclanthology.org/E17-2025/](https://aclanthology.org/E17-2025/).

[7] KOBAYASHI G, KURIBAYASHI T, YOKOI S, et al. Transformer Language Models Handle Word Frequency in Prediction Head[C]// *Findings of the Association for Computational Linguistics: ACL 2023*. Toronto, Canada: Association for Computational Linguistics, 2023: 4523-4535. DOI: [10.18653/v1/2023.findings-acl.276](https://doi.org/10.18653/v1/2023.findings-acl.276).
