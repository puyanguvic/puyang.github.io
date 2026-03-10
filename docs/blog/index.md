---
title: Blog
description: 关于高维几何、表示空间、Transformer 与 tokenizer 的系列长文。
---

# Blog

按专题浏览。本页收录的文章已经按系列长文统一修订；每组都依次处理问题定义、机制分析、边界条件与应用含义，适合连续阅读。

::: details Mar 9, 2026 · 高维空间与机器学习
本组文章从最基础的高维几何出发，说明为什么机器学习不能直接把原始欧氏空间当成语义空间来用。主线是：距离在高维中先失去分辨率，方向结构随后成为更稳定的几何信号，而训练后的表示又进一步被压到近似球面上。

1. [高维空间中的距离集中与度量失效](/blog/high-dimensional-space-and-machine-learning/distance-breakdown)
   先形式化“距离失效”究竟失效在哪里，再从薄壳现象与极值收缩解释为什么最近邻与最远邻会逐渐难以区分。
2. [高维向量近似正交的几何机制](/blog/high-dimensional-space-and-machine-learning/orthogonality)
   说明在范数已集中之后，角度为何比长度更稳定，以及高维球面为什么能容纳大量彼此低相关的方向。
3. [Embedding 向量的超球面分布及其成因](/blog/high-dimensional-space-and-machine-learning/hypersphere)
   将前两篇与现代表示学习目标结合起来，解释 embedding 为什么常呈现近似球壳分布，以及余弦相似度为何更自然。
:::

::: details Mar 9, 2026 · 大模型的表示空间
本组文章讨论 LLM 词表与 embedding 空间的组织原则。核心问题不是“向量会不会做算术”，而是训练目标如何把重复关系压缩为稳定方向，以及高维词表为什么更适合被理解为一个受语义约束的球面码本。

1. [Embedding 空间中的语义线性结构](/blog/representation-space-of-large-models/semantic-linearity)
   从类比现象回到 PMI 因子分解，解释语义线性何以出现、为何只在局部稳定，以及为什么它在上下文化表示中会减弱。
2. [LLM Embedding 的球面编码视角](/blog/representation-space-of-large-models/spherical-coding)
   从球面码、coherence 与 Welch 下界出发，说明有限维 embedding 为什么足以容纳巨型词表，以及这种视角能解释什么、不能解释什么。
:::

::: details Mar 9, 2026 · Transformer 的几何结构
本组文章把 Transformer 的核心算子写回几何语言。attention 不是“简单加权平均”，而是由 query-key 几何诱导出的软坐标系统；多头注意力也不是重复运算，而是并行构造多套不同的上下文坐标系。

1. [Transformer Attention 的几何本质](/blog/geometry-of-transformers/what-attention-does)
   从双线性匹配、概率单纯形与重心重建出发，解释 attention 如何完成上下文相关的读取与重写。
2. [Multi-Head Attention 的必要性与表达优势](/blog/geometry-of-transformers/why-multi-head-matters)
   从单头注意力的几何瓶颈出发，说明为什么关系解耦、内容解耦与并行计算都需要多头结构。
:::

::: details Mar 9, 2026 · Tokenizer 的理论
本组文章讨论 tokenizer 在整个 LLM 系统中的角色。论证主线是：tokenization 首先是码本压缩问题；词表规模因此存在自然平衡点；而所谓 token-free 路线并没有取消压缩，只是把压缩移入了模型内部。

1. [Tokenization 的压缩本质](/blog/theory-of-tokenizers/what-tokenization-does)
   说明 tokenizer 为什么应被理解为有限可逆码本，以及它如何利用语言分布的统计偏斜来缩短序列并降低学习负担。
2. [LLM 词表规模的自然平衡点](/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k)
   从序列长度收益递减、长尾稀疏和输出层成本三方面解释词表为什么不会无限扩张。
3. [Character-Level Tokenizer 的理论优势与工程局限](/blog/theory-of-tokenizers/why-character-level-rarely-wins)
   说明字符级方案为何在表示上统一、在系统上却常常更贵，以及为什么成功的 token-free 模型往往仍会重新引入内部压缩。
:::
