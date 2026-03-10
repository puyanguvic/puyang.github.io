---
title: "Embedding 空间中的语义线性结构"
date: 2026-03-09T11:00:00-08:00
summary: "从类比任务、PMI 因子分解与语义子空间出发，解释 embedding 空间中的语义关系为何常呈现局部近似线性。"
tags: ["LLM", "embeddings", "representation learning"]
---

# Embedding 空间中的语义线性结构

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/representation-space-of-large-models/semantic-linearity" en-path="/blog/representation-space-of-large-models/semantic-linearity-en" />

语义线性之所以长期吸引研究者，并不是因为 `king - man + woman ≈ queen` 这类例子本身足够醒目，而是因为它揭示了一个更深层的问题：为什么分布式表示会反复把某些关系压缩成可复用的方向偏移？如果这种现象只是偶然，那么它不值得被系统讨论；如果它反复出现，就必须追问它对应的统计机制是什么 [1-9]。

更稳妥的表述是，embedding 中的线性并不是自然语言满足某种全局欧氏公理，而是训练目标在压缩重复共现结构时产生的一种低复杂度几何编码。它在静态词向量、输入 embedding 和输出 embedding 中最清晰，在强上下文化后的高层隐藏状态里则会明显减弱。

> 核心结论：embedding 中的语义线性通常应理解为局部稳定的关系偏移，而不是全局完美的语义轴；它之所以出现，是因为共现统计、低秩因子分解与参数共享共同鼓励模型把重复关系编码为方向结构，而这种结构在上下文化表示中会因语境依赖而显著弱化 [1-9]。

在“大模型的表示空间”系列中，本文先回答重复语义关系为什么会被写成局部稳定的方向偏移；下一篇 [LLM Embedding 的球面编码视角](/blog/representation-space-of-large-models/spherical-coding) 会把这种方向结构从局部关系推广到整个词表的高维码本。

## 1. 线性现象真正声称了什么？

设四个词的表示分别为 $v_a,v_b,v_c,v_d$。所谓类比关系成立，通常不是说某个词有一个绝对“性别坐标”或“时态坐标”，而是说两对词之间存在相似的位移：

$$
v_b - v_a \approx v_d - v_c.
$$

`king - man + woman ≈ queen` 只是这个模式的一个具体实例 [1][2]。真正被测试的是如下关系一致性：

$$
v_{\text{queen}} - v_{\text{king}}
\approx
v_{\text{woman}} - v_{\text{man}}.
$$

因此，类比现象的解释对象不是单个词向量，而是“关系向量”是否在多个词对上可复用。只要同一种变换在许多词对上产生近似平行的位移，最近邻搜索就可能把

$$
v_b - v_a + v_c
$$

映射到正确答案附近。

这一表述已经暗示了一个边界：语义线性要成立，必须先存在可重复的关系类型。若关系高度依赖上下文、词类或话语环境，它就很难被单一方向稳定编码。图 1 可以帮助把“局部可复用位移”与“全局固定语义轴”这两种看法区分开来。

![语义线性的局部结构示意图](./semantic-linearity-local-geometry.svg)

*图 1. 语义线性更适合被理解为“局部稳定的关系偏移”。在词项级表示中，一些重复关系会形成可复用方向；在强上下文化表示中，同一词项的位置会随语境明显漂移。*

因此，图 1 的重点并不是几条漂亮的平行线，而是“同一种关系只在有限邻域内保持稳定”。这也解释了为什么类比现象既真实存在，又不能被夸大成全局公理。

## 2. 为什么共现学习会自然产生位移结构？

要回答这个问题，不能停留在类比题本身，而必须回到词向量训练目标。Levy 与 Goldberg 证明，Skip-gram with Negative Sampling 可以被理解为对 shifted PMI 矩阵的隐式因子分解 [4]；Arora 等人的工作则进一步说明，词向量与 PMI 结构之间存在稳定联系 [5]。这一类结果可以概括为

$$
v_w^\top u_c \approx \operatorname{PMI}(w,c) + b_w + b_c,
$$

其中 $u_c$ 是上下文向量，$b_w,b_c$ 是偏置项。

对任意两个词 $a,b$，两边相减得到

$$
(v_b - v_a)^\top u_c
\approx
\log \frac{p(c \mid b)}{p(c \mid a)}.
$$

这条式子是理解语义线性的关键。它说明向量差分并不是任意几何操作，而是在近似表示“词 $b$ 相比词 $a$，对上下文 $c$ 的条件几率发生了多大变化”。如果两组词对 $(a,b)$ 与 $(c,d)$ 在大量上下文方向上呈现相似的对数几率变化，那么

$$
v_b - v_a \approx v_d - v_c
$$

就是一种自然的低维压缩结果，而不是偶然对齐。

## 3. 为什么模型偏好把关系写成“方向”？

仅有共现结构还不够，还需要解释为什么训练后的低维空间会把它们压缩成线性偏移。原因至少有三层。

- 许多语义或句法关系在语料中反复出现，因此它们对应的上下文变化具有统计重复性。
- 低维 embedding 无法逐个记住所有词对关系，最经济的做法是让一类重复变换共享同一方向模式。
- 下游读取本来就依赖内积、线性映射和向量加法，因此把关系编码成方向能最大化复用现有计算原语。

从表示学习角度看，这是一种典型的低秩压缩策略。模型不是在为每个关系实例单独分配参数，而是在寻找能跨许多词对复用的共享位移。只要这种共享足够广泛，线性结构就会在词向量空间里显现出来。

因此，线性并不是语义的先验真相，而是分布式表示在有限维度中压缩重复关系时的最优近似之一。

## 4. 为什么真实语义更接近“子空间”而不是“单一轴”？

线性现象最常见的误解，是把它想象成若干全局、精确、互相独立的语义坐标轴。实际情况远比这复杂。多义词会把多个语义模式叠加到同一个词向量中，不同关系之间也会相互耦合。Arora 等人在 2018 年的工作表明，词义结构更适合被理解为若干低维方向成分的组合，而不是单轴模型 [6]。

因此，更专业的说法应当是：

- 一类关系常对应局部稳定的方向簇或低维子空间；
- 这些方向在相近词类内部更可迁移，跨语域或跨词类时更容易失真；
- 关系越抽象、越依赖上下文，其线性可复用性通常越弱。

这也解释了为什么有些类比任务表现稳定，而有些一旦离开固定词表就迅速失效。语义线性常常是局部几何，而不是全局坐标系。

## 5. 为什么上下文化表示中的线性会减弱？

线性规律最清楚的地方，通常是静态词向量、输入 embedding 或输出 embedding 这类词项级表示 [7][8]。原因不难理解：这些表示主要压缩的是词类型的全局共现统计，因此更容易保留稳定的关系偏移。

一旦进入高层上下文化表示，同一个词会随语法位置、指代关系、话题与搭配环境发生明显漂移。Ethayarajh 的分析表明，BERT、ELMo 与 GPT-2 的高层表示具有明显的上下文依赖与各向异性；同一词的静态表示只能解释其中很小一部分方差 [7]。因此，在上下文化空间中，单一关系向量很难继续解释所有实例。

可以把不同层级的可见线性粗略概括如下：

| 表示层级 | 线性规律的典型强度 | 主要原因 |
| --- | --- | --- |
| 静态词向量 | 高 | 直接压缩词类型共现统计 |
| 输入/输出 embedding | 中到高 | 仍保持明显词项级几何，且与预测头耦合 [8] |
| 中高层上下文化状态 | 中到低 | 语境、位置与任务相关因素持续重写表示 [7] |

因此，语义线性并没有在大模型中完全消失，但它主要保留在更接近“词典层”的那部分空间里。

## 6. 哪些关系最容易破坏线性？

一旦把语义线性理解为“重复关系的低维压缩”，就可以更清楚地看到它的失效边界。最容易被单一方向刻画的，通常是近似平移型关系：同一种变换可以在多个词项上重复作用，而且上下文比值变化相对稳定。相反，下列关系往往更容易破坏单向量位移的假设：

- 一对多或多对一关系，例如同一词在不同语境下对应多个语义角色；
- 强层级关系或集合关系，它们更像分叉结构而不是平移结构；
- 严重依赖句法位置或篇章角色的关系，同一词项在不同位置会触发不同变换；
- 多义词驱动的关系，因为同一个静态向量必须同时承载多个不相容方向 [6][7]。

这也是为什么早期类比基准更容易在形态变化、性别对照、时态变化等较“干净”的关系上得到漂亮结果，而在篇章语义、语用功能或高度语境化的关系上难以维持同样的线性稳定性。线性并没有错，它只是只负责那一部分真正具有可重复变换结构的关系。

## 7. 群与对称性视角为什么仍然有用？

如果把“单数到复数”“现在时到过去时”理解为可重复作用在多个词上的近似变换，那么 embedding 学习可以被视为在数据中寻找共享变换结构。形式上，若某种操作 $g$ 在局部邻域中满足

$$
v_{g \cdot x} \approx T_g v_x,
$$

那么对 $T_g$ 做一阶近似，就会得到线性偏移的解释。这与等变表示学习里的思想是一致的：共享变换结构有助于降低样本复杂度并提升泛化 [9]。

但这里必须保持克制。自然语言并不是严格的群作用系统，语义变换也不满足全局封闭性或精确等变性。群论视角更适合作为分析框架，而不是对语言本体的严格公理化。

## 8. 结语

embedding 中的语义线性并不意味着语言被完全欧氏化了。它更接近这样一个事实：当模型必须在有限维空间里压缩大规模共现统计时，最稳定、最可复用的关系往往会被写成方向结构。于是，类比现象之所以重要，不是因为它展示了一个漂亮技巧，而是因为它暴露出表示学习的一个根本偏好：共享关系优先于逐例记忆，局部线性优先于零散特判。

可以把这一结论压缩成一句话：**语义方向不是语言的先验公理，而是分布式表示对重复关系做低维压缩时形成的几何结果。** 把这一步再向外推广，整个词表就会更像一个受语义约束的高维码本。

继续阅读：[LLM Embedding 的球面编码视角](/blog/representation-space-of-large-models/spherical-coding)。

## 参考文献

[1] MIKOLOV T, YIH W-T, ZWEIG G. Linguistic Regularities in Continuous Space Word Representations[C]// *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. Atlanta, Georgia: Association for Computational Linguistics, 2013: 746-751. URL: [https://aclanthology.org/N13-1090/](https://aclanthology.org/N13-1090/).

[2] MIKOLOV T, SUTSKEVER I, CHEN K, et al. Distributed Representations of Words and Phrases and their Compositionality[C]// *Advances in Neural Information Processing Systems 26*. Red Hook, NY: Curran Associates, 2013: 3111-3119. URL: [https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality).

[3] PENNINGTON J, SOCHER R, MANNING C D. GloVe: Global Vectors for Word Representation[C]// *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Doha, Qatar: Association for Computational Linguistics, 2014: 1532-1543. DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162).

[4] LEVY O, GOLDBERG Y. Neural Word Embedding as Implicit Matrix Factorization[C]// *Advances in Neural Information Processing Systems 27*. Red Hook, NY: Curran Associates, 2014: 2177-2185. URL: [https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization).

[5] ARORA S, LI Y, LIANG Y, et al. A Latent Variable Model Approach to PMI-based Word Embeddings[J]. *Transactions of the Association for Computational Linguistics*, 2016, 4: 385-399. DOI: [10.1162/tacl_a_00106](https://doi.org/10.1162/tacl_a_00106).

[6] ARORA S, LI Y, LIANG Y, et al. Linear Algebraic Structure of Word Senses, with Applications to Polysemy[J]. *Transactions of the Association for Computational Linguistics*, 2018, 6: 483-495. DOI: [10.1162/tacl_a_00034](https://doi.org/10.1162/tacl_a_00034).

[7] ETHAYARAJH K. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings[C]// *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Hong Kong, China: Association for Computational Linguistics, 2019: 55-65. DOI: [10.18653/v1/D19-1006](https://doi.org/10.18653/v1/D19-1006).

[8] PRESS O, WOLF L. Using the Output Embedding to Improve Language Models[C]// *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*. Valencia, Spain: Association for Computational Linguistics, 2017: 157-163. URL: [https://aclanthology.org/E17-2025/](https://aclanthology.org/E17-2025/).

[9] KONDOR R, TRIVEDI S. On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups[C]// *Proceedings of the 35th International Conference on Machine Learning*. PMLR, 2018: 2747-2755. URL: [https://proceedings.mlr.press/v80/kondor18a.html](https://proceedings.mlr.press/v80/kondor18a.html).
