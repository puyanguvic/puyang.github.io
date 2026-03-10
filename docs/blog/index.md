---
title: Blog
description: Notes and short posts.
---

# Blog

按专题折叠浏览文章：

::: details Mar 9, 2026 · 高维空间与机器学习
这个系列讨论高维几何如何影响机器学习中的表示、距离、角度与检索。

1. [高维空间中的距离集中与度量失效](/blog/high-dimensional-space-and-machine-learning/distance-breakdown)
2. [高维向量近似正交的几何机制](/blog/high-dimensional-space-and-machine-learning/orthogonality)
3. [Embedding 向量的超球面分布及其成因](/blog/high-dimensional-space-and-machine-learning/hypersphere)
:::

::: details Mar 9, 2026 · 大模型的表示空间
这个系列讨论大模型 embedding 的几何组织方式：为什么某些语义关系会呈现近似线性，为什么词表可以被看作高维球面上的受约束码本，以及这些几何性质如何共同支撑检索、预测与泛化。

1. [Embedding 空间中的语义线性结构](/blog/representation-space-of-large-models/semantic-linearity)
   从类比任务、PMI 因子分解与语义子空间出发，讨论语义关系为何常表现为局部稳定的方向偏移。
2. [LLM Embedding 的球面编码视角](/blog/representation-space-of-large-models/spherical-coding)
   从球面码、Welch 下界与高维角度集中出发，解释为什么 `4096` 维空间足以承载 `100k` 级词表。
:::

::: details Mar 9, 2026 · Transformer的几何结构
这个系列讨论 Transformer 的核心计算如何用几何语言来理解：attention 如何生成上下文相关的软坐标并重建表示，multi-head 机制又为什么等价于并行构造多个上下文坐标系。

1. [Transformer Attention 的几何本质](/blog/geometry-of-transformers/what-attention-does)
   从 query-key 匹配、概率单纯形与 value 重建出发，解释 attention 为什么不是简单加权平均。
2. [Multi-Head Attention 的必要性与表达优势](/blog/geometry-of-transformers/why-multi-head-matters)
   从单头注意力的几何瓶颈出发，说明多头机制如何把异质关系分散到多个并行视角中处理。
:::

::: details Mar 9, 2026 · Tokenizer的理论
这个系列讨论 tokenizer 的理论基础与工程折中：tokenization 为什么本质上是码本压缩，词表规模为什么会停在一个中等区间，以及 token-free 方案为何常常又在模型内部重新引入压缩。

1. [Tokenization 的压缩本质](/blog/theory-of-tokenizers/what-tokenization-does)
   从平均描述长度、Zipfian 频率分布与 subword 码本设计出发，解释 tokenizer 为什么首先是压缩系统。
2. [LLM 词表规模的自然平衡点](/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k)
   从序列长度收益递减、长尾稀疏与 softmax 成本的权衡出发，说明词表为什么不会无限扩张。
3. [Character-Level Tokenizer 的理论优势与工程局限](/blog/theory-of-tokenizers/why-character-level-rarely-wins)
   从序列长度、优化路径和 modern token-free 模型的结构补偿出发，解释为什么纯字符方案通常难成主流。
:::
