---
title: "Embedding 空间中的语义线性结构"
date: 2026-03-09T11:00:00-08:00
summary: "从类比任务、PMI 因子分解与语义子空间出发，讨论 embedding 空间中的语义关系为何常呈现近似线性。"
tags: ["LLM", "embeddings", "representation learning"]
---

# Embedding 空间中的语义线性结构

在表示学习里，一个持续吸引研究者的现象是：某些语义关系能够稳定地表现为向量差分，即

$$
v_b - v_a \approx v_d - v_c.
$$

最经典的例子当然是

$$
\text{king} - \text{man} + \text{woman} \approx \text{queen}.
$$

但如果把这个现象只理解为“词向量会做算术”，其实仍然太浅。更重要的问题是：为什么训练目标会反复诱导出这种结构？为什么某些关系能被压缩为可复用的方向，而另一些却不能？以及，这种线性究竟是全局规律，还是局部近似？

> 核心结论：embedding 空间中的语义线性，通常不是语言本身满足了全局线性公理，而是分布式训练目标、共享共现统计与低秩压缩共同诱导出的局部近似；它在静态词向量以及 LLM 的输入/输出 embedding 中最明显，在强上下文化后的高层表示中则会显著减弱 [1-9]。

## 1. 类比现象真正说明了什么？

Mikolov 等人在词类比任务中观察到，许多语义与句法关系都可以被“关系向量”近似表达：如果两个词对共享同一种关系，那么它们的差分往往会彼此接近 [1][2]。GloVe 则进一步把这种现象提升为建模目标的一部分，明确讨论了为什么某些“意义方向”会在向量空间中浮现出来 [3]。

更严格地说，类比任务测试的不是单个词的绝对位置，而是下面这个**关系一致性假设**：

$$
v_{\text{queen}} - v_{\text{king}}
\approx
v_{\text{woman}} - v_{\text{man}}.
$$

如果同一类关系在许多词对上都共享相似的位移，那么最近邻搜索就会把

$$
v_b - v_a + v_c
$$

映射到一个合理答案附近。于是，`king - man + woman` 能工作，并不是因为空间里存在一条精确的“女性轴”，而是因为“男性到女性”的变换在若干相关词对上表现出可复用的几何模式 [1-3]。

![语义线性的局部结构示意图](./semantic-linearity-local-geometry.svg)

*图 1. 语义线性更适合理解为“局部稳定的关系偏移”：在静态 embedding 中，一些常见关系会形成可复用方向；而在更强的上下文化表示中，同一词项会随语境发生明显漂移。*

## 2. 为什么共现学习会鼓励线性偏移？

理解这一点的关键，不是从类比题本身出发，而是回到 embedding 的训练目标。Levy 与 Goldberg 证明，Skip-gram with Negative Sampling 可以被理解为一种对 shifted PMI 矩阵的隐式因子分解 [4]；Arora 等人则给出了一条更系统的概率论解释，说明词向量与 PMI 结构之间存在稳定联系 [5]。据此，一个更稳妥的解释是：

$$
v_w^\top u_c \approx \operatorname{PMI}(w,c) + b_w + b_c.
$$

于是对任意词对 $(a,b)$，有

$$
(v_b - v_a)^\top u_c
\approx
\log \frac{p(c \mid b)}{p(c \mid a)}.
$$

这条式子非常关键。它意味着词向量差分并不只是几何游戏；它对应的是**两个词对上下文分布比值的差异**。如果两对词共享相近的上下文比值轮廓，那么它们的差分就会在许多上下文方向上产生相似投影，因此自然会表现为相近的位移 [3-5]。

从这个角度看，线性结构出现至少有三层原因：

- 许多语义关系会在语料中重复出现，从而产生相似的上下文统计变换。
- 低维 embedding 被迫压缩高维共现结构，最经济的方式往往是把重复关系编码成共享方向，而不是为每个词对单独记忆一个例外。
- 线性偏移便于复用。对下游模型而言，加法、内积和线性映射都是最廉价、最稳定的计算单元。

因此，线性并不是“语义天然是直线”的证据，而更像是训练目标在高维空间里找到的一种低复杂度参数化方式。

## 3. 语义“方向”更接近子空间，而不是单一轴

实际 embedding 空间远没有类比题展示得那么整齐。首先，不同关系之间会耦合；其次，多义词会把多个语义模式叠加到同一个词向量中。Arora 等人在 TACL 2018 的工作中证明，词义结构常常更适合被理解为若干低维方向成分的线性叠加，而不是“一个词对应一条完美语义轴” [6]。

因此，对 gender、tense、plural 这些经典例子，一个更专业的表述应当是：

- 它们往往对应**局部稳定的方向簇**或低维子空间；
- 这些方向在相近词类内部更可迁移，在跨词类、跨语域时更容易失真；
- 关系越抽象、越依赖语境，其线性可复用性通常越弱。

这也解释了为什么有些类比题表现很好，而有些只在特定词集上成立。线性结构并不是全局坐标系中的一条轴，而更像是一个在局部邻域内足够稳定的切向模式。

## 4. 群与对称性视角：为什么它仍然有解释力？

如果把“单数到复数”“现在时到过去时”“男性到女性”理解为一类可重复变换，那么 embedding 学习可以被视为在数据中寻找近似共享的作用规则。形式上，若某种语义操作 $g$ 在局部邻域内可以近似写成

$$
v_{g \cdot x} \approx T_g v_x,
$$

那么对 $T_g$ 做一阶近似就会得到线性偏移。这个视角与现代等变表示学习中的思想是一致的：共享变换结构可以显著降低样本复杂度，并提高泛化能力 [9]。

当然，这里必须保持严格：语言并不是一个干净的群作用系统，语义变换也并不满足全局封闭性、可逆性与精确等变性。把群表示拿来理解 embedding，更准确地说是一种**分析框架**，而不是关于自然语言的严格公理化声明。

## 5. 为什么在线性最强的地方，恰好也是最“词典式”的地方？

在线性规律最容易被观察到的，往往不是 LLM 的高层上下文化隐藏状态，而是更接近词类型表征的那部分空间，例如输入 embedding、输出 embedding，以及经过特殊归一化后的句向量空间 [7][8]。原因并不神秘：

- 输入/输出 embedding 更接近“词项级”统计汇总，因此更容易保留稳定的类型关系；
- 输出层常与输入 embedding 共享或紧密耦合，使得词表几何同时服务于表示与预测 [8]；
- 一旦进入高层上下文化阶段，同一个词会因句法位置、指代、话题和搭配而发生显著漂移，单一全局方向就很难继续解释全部变化 [7]。

Ethayarajh 的研究表明，BERT、ELMo 与 GPT-2 的高层表示既明显各向异性，又高度依赖上下文；对同一词而言，静态 embedding 只能解释其上下文化表示中很小的一部分方差 [7]。这意味着：

> 语义线性并没有在大模型里消失，但它更像是底层词汇几何的性质，而不是所有层都共享的一条全局定律。

从工程上看，可以把不同表示层级的“线性可复用性”粗略总结如下：

| 表示层级 | 线性规律的可见度 | 主要原因 |
| --- | --- | --- |
| 静态词向量 | 高 | 词类型共现统计被直接压缩进固定向量 |
| LLM 输入/输出 embedding | 中到高 | 词表几何同时服务于词项表示与 softmax 预测 [8] |
| 中高层上下文化隐藏状态 | 中到低 | 同一词项会随语境、句法位置与话语功能发生明显漂移 [7] |

## 6. 结语

embedding 空间中的语义线性，并不意味着语言被“欧氏化”了；它意味着模型在压缩大规模共现统计时，倾向于把最稳定、最可复用的关系编码成方向结构。类比现象之所以重要，不是因为它展示了一个漂亮的例子，而是因为它揭示出表示学习的一个基本偏好：**共享关系优先于孤立记忆，局部线性优先于逐点特判**。

因此，更准确的总结应当是：

> 语义关系之所以常表现为向量方向，不是因为语义本身天然线性，而是因为线性结构是分布式表示在有限维度里压缩重复关系时最经济的结果。

如果把这个结论放回整个系列中看，它与前文讨论的[高维向量近似正交的几何机制](/blog/high-dimensional-space-and-machine-learning/orthogonality)和[Embedding 向量的超球面分布及其成因](/blog/high-dimensional-space-and-machine-learning/hypersphere)是连在一起的：高维几何先提供方向容量与球壳约束，线性语义再把其中一部分容量组织成可复用的关系偏移。

下一篇文章，我们会把视角从“语义方向”推进到“球面码本”，讨论为什么 LLM 的 token embedding 更适合被理解为受语义约束的高维球面编码系统。

## 参考文献

[1] MIKOLOV T, YIH W-T, ZWEIG G. Linguistic Regularities in Continuous Space Word Representations[C]// *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. Atlanta, Georgia: Association for Computational Linguistics, 2013: 746-751. Available: [https://aclanthology.org/N13-1090/](https://aclanthology.org/N13-1090/).

[2] MIKOLOV T, SUTSKEVER I, CHEN K, et al. Distributed Representations of Words and Phrases and their Compositionality[C]// *Advances in Neural Information Processing Systems 26*. Red Hook, NY: Curran Associates, 2013: 3111-3119. Available: [https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality).

[3] PENNINGTON J, SOCHER R, MANNING C D. GloVe: Global Vectors for Word Representation[C]// *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Doha, Qatar: Association for Computational Linguistics, 2014: 1532-1543. DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162).

[4] LEVY O, GOLDBERG Y. Neural Word Embedding as Implicit Matrix Factorization[C]// *Advances in Neural Information Processing Systems 27*. Red Hook, NY: Curran Associates, 2014: 2177-2185. Available: [https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization).

[5] ARORA S, LI Y, LIANG Y, et al. A Latent Variable Model Approach to PMI-based Word Embeddings[J]. *Transactions of the Association for Computational Linguistics*, 2016, 4: 385-399. DOI: [10.1162/tacl_a_00106](https://doi.org/10.1162/tacl_a_00106).

[6] ARORA S, LI Y, LIANG Y, et al. Linear Algebraic Structure of Word Senses, with Applications to Polysemy[J]. *Transactions of the Association for Computational Linguistics*, 2018, 6: 483-495. DOI: [10.1162/tacl_a_00034](https://doi.org/10.1162/tacl_a_00034).

[7] ETHAYARAJH K. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings[C]// *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Hong Kong, China: Association for Computational Linguistics, 2019: 55-65. DOI: [10.18653/v1/D19-1006](https://doi.org/10.18653/v1/D19-1006).

[8] PRESS O, WOLF L. Using the Output Embedding to Improve Language Models[C]// *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*. Valencia, Spain: Association for Computational Linguistics, 2017: 157-163. Available: [https://aclanthology.org/E17-2025/](https://aclanthology.org/E17-2025/).

[9] KONDOR R, TRIVEDI S. On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups[C]// *Proceedings of the 35th International Conference on Machine Learning*. PMLR, 2018: 2747-2755. Available: [https://proceedings.mlr.press/v80/kondor18a.html](https://proceedings.mlr.press/v80/kondor18a.html).
