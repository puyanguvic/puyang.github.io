---
title: "Transformer Attention 的几何本质"
date: 2026-03-09T11:30:00-08:00
summary: "把 attention 理解为 query-key 几何诱导出的软坐标系统，以及在 value 空间中的内容自适应重建。"
tags: ["Transformer", "attention", "geometry of representation"]
---

# Transformer Attention 的几何本质

Transformer 的核心计算常被讲成一串工程步骤：先算 `Q, K, V`，再做 `softmax`，最后加权求和。这种解释当然没有错，但它不够抓本质。真正重要的问题是：为什么这样一个看似简单的运算，能够在每一层里完成上下文检索、长程依赖建模和表示更新？

更精确的说法是：

> attention 的本质，不是“给 token 分配权重”，而是由 query-key 几何诱导出一组软坐标，再在 value 空间中做内容自适应重建。

这句话和“子空间投影 + 表示重建”的直觉接近，但比它更严格。因为标准 softmax attention 并不是正交投影；它更接近一种**内容相关的核回归 / 重心重建** [1-4]。

> 核心结论：在 Transformer 中，query 决定“当前表示应当沿哪种关系读取上下文”，key 决定“上下文中有哪些可被读取的方向”，而 softmax 则把匹配分数变成概率单纯形上的软坐标；最终输出是这些坐标在 value 空间中的加权重建，而不是某个 token 表示的简单复制 [1-7]。

## 1. 先从公式开始：attention 是一个行随机算子

标准 scaled dot-product attention 写成矩阵形式是

$$
\mathrm{Attn}(Q, K, V)
=
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

设

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right),
$$

则有

$$
O = AV.
$$

这里的 $A \in \mathbb{R}^{n \times n}$ 是一个**行随机矩阵**：每一行都非负、且和为 `1`。如果只看第 $i$ 个 token，对应的输出为

$$
o_i = \sum_{j=1}^n \alpha_{ij} v_j,
\qquad
\alpha_i \in \Delta^{n-1},
$$

其中 $\Delta^{n-1}$ 是 $(n-1)$ 维概率单纯形。这个式子已经透露出 attention 的几何结构：对每个 query 而言，softmax 会给出一组位于单纯形上的软坐标，输出则是这些坐标在 value 向量上的重心组合 [1][2]。

从角色上看，三组向量并不对称：

| 对象 | 几何角色 | 功能解释 |
| --- | --- | --- |
| query | 读取方向 / 测试函数 | 决定当前 token 需要按什么关系去看上下文 |
| key | 可匹配方向 / 索引 | 决定上下文中哪些位置会被当前 query 激活 |
| value | 被重建内容 | 决定一旦某位置被选中，真正传回来的表示是什么 |

## 2. Query-Key 内积到底在测什么？

对单个位置 $i$ 与 $j$，attention logits 为

$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}
=
\frac{x_i^\top W_Q^\top W_K x_j}{\sqrt{d_k}}.
$$

这个式子非常重要，因为它说明 attention 比“直接比较两个 token 是否相似”更一般。模型并不是在原始表示空间里计算欧氏距离或余弦相似度，而是在一个由 $W_Q^\top W_K$ 定义的**可学习双线性形式**下比较两个位置。于是：

- query 不是内容本身，而是“当前应读取哪种关系”的探针；
- key 不是内容本身，而是“当前这个位置在哪些关系下可被读取”的索引；
- 同一个 token 在不同 head、不同层里，会对应完全不同的匹配几何。

因此，attention 的第一步并不是“找最像我的 token”，而是：

> 在当前层、当前 head 所定义的关系度量下，哪些上下文位置与我最对齐？

这也是为什么 attention 可以同时服务句法依赖、指代关系、局部搭配、篇章回溯与任务相关读取。它不是一个固定度量，而是一族由参数和上下文共同决定的动态匹配规则 [1][5]。

## 3. 为什么它更像“软投影”，而不是“硬检索”？

一旦得到了 $\alpha_i$，输出

$$
o_i = \sum_j \alpha_{ij} v_j
$$

就落在当前 value 集合的凸包里。也就是说，对固定的 `V` 而言，attention 输出不是任意向量，而是上下文 value 的一个**重心重建**。这正是 attention 比“top-1 检索”更强的地方：它允许模型从多个位置同时取信息，并且可微地控制组合比例。

![attention 的软坐标与重建示意图](./attention-soft-coordinates.svg)

*图 1. attention 更精确的几何描述是：query 在 key 几何上产生一组软坐标，softmax 把这些坐标限制在概率单纯形内，最后输出是 value 空间中的重心重建。它功能上接近软投影，但并不是正交投影。*

这时再回看“子空间投影”这个比喻，就能更严格地区分：

- 它**像**投影，因为 query 的确在通过 key 几何选择一组相关方向；
- 它又**不同于**正交投影，因为 softmax 引入了非线性归一化，且 value 空间与 key 空间可以解耦。

所以，标准 attention 的更稳妥表述应当是：**由 query-key 相容性诱导出的软坐标系统，再在 value 空间上做内容自适应重建**。

## 4. 为什么 attention 比固定卷积或固定窗口更灵活？

Vaswani 等人最早强调过 self-attention 的两个关键优势：任意两位置之间的路径长度短，以及权重完全依赖内容而不是固定位置 [1]。从几何上看，这等价于说 attention 具备三个很重要的性质。

### 全局感受野

每个 query 默认都可以访问整个上下文，而不是只访问局部窗口。因此，长距离依赖不需要通过多步递推才能传递。

### 内容自适应权重

卷积核在不同位置共享同一组权重，而 attention 的权重会随 query-key 匹配实时变化，因此读取规则是输入相关的。

### 可退化为更简单的局部算子

attention 并不只会做“全局查找”。Cordonnier 等人证明，多头 self-attention 在合适参数化下可以表达卷积层 [7]。这说明 attention 包含卷积作为一个特例，而不是卷积对 attention 的近似替代。

也正因为如此，attention 更像一个**可学习的动态几何算子**：需要局部时，它可以像局部算子；需要跨句、跨段时，它又可以立刻切到全局读取模式。

## 5. 为什么不能把 attention 权重直接当成解释？

这里必须保持技术上的克制。说“attention 给出了软坐标”，并不等于说“attention 权重就是模型解释”。Brunner 等人在 ICLR 2020 里指出，当序列长度超过 head 维度时，attention 权重本身并不具有良好的可识别性；不同参数配置可能产生相同输出，却对应不同的注意力分布 [5]。

这意味着：

- 注意力分布是模型内部的一部分计算状态；
- 它确实反映了某种读取路径；
- 但它未必是唯一的、可直接人类解释的因果说明。

此外，attention 也不是 Transformer 的全部。Geva 等人表明，FFN 在很大程度上可以被理解为位置上的 key-value memory；attention 负责路由上下文，FFN 则负责在每个位置上做更强的模式选择与内容写入 [6]。因此，真正的 Transformer 表示更新更接近：

1. 用 attention 读取相关上下文。
2. 通过残差把原表示保留下来。
3. 用 FFN 在当前位置做进一步变换与记忆检索。

也就是说，attention 提供的是**上下文路由与混合**，不是完整的语义计算闭环。

## 6. 这种机制有多强，又有什么边界？

从正面看，Transformer 的 attention 模块具有很强的表达力。Yun 等人证明，带位置编码的 Transformer 可以普适逼近连续的序列到序列函数 [3]。这说明 attention 并不是一个“只能做加权平均”的弱算子，而是更大表示系统中的关键上下文映射模块。

但从反面看，纯 self-attention 也不是没有边界。Hahn 在 TACL 2020 中指出，如果层数或头数不随输入长度增长，纯 self-attention 在某些形式语言和层级结构上存在严格限制 [4]。这提醒我们两个事实：

- attention 很强，但它的能力依赖于层数、头数、位置编码与后续非线性模块；
- 单层 attention 的几何意义很清楚，但模型整体能力来自多层堆叠、残差连接和 FFN 的共同作用。

因此，最准确的说法不是“attention 单独完成理解”，而是“attention 为 Transformer 提供了可内容化的上下文坐标系统，使后续层能够在全局依赖之上继续计算”。

## 7. 结语

如果要把本文压缩成一句话，我会写：

> attention 的本质，是由 query-key 几何生成软坐标，再在 value 空间做内容相关重建。

这比“加权平均”更准确，也比“简单检索”更有解释力。它说明 Transformer 的关键不只是看到了哪些 token，而是为每个 token 动态建立了一套“如何读取上下文”的局部坐标系。

如果把这个结论放回整个系列中看，它与前文讨论的[Embedding 空间中的语义线性结构](/blog/representation-space-of-large-models/semantic-linearity)和[LLM Embedding 的球面编码视角](/blog/representation-space-of-large-models/spherical-coding)是一致的：前者讨论 token 表示在静态空间中如何组织，attention 则负责在这些表示之上动态生成上下文相关的几何读取规则。

下一篇文章，我们会继续这个视角，讨论为什么 multi-head attention 不是简单地“多做几遍 attention”，而是在并行构造多个不同的语义坐标系。

## 参考文献

[1] VASWANI A, SHAZEER N, PARMAR N, et al. Attention Is All You Need[C]// *Advances in Neural Information Processing Systems 30*. Red Hook, NY: Curran Associates, 2017. Available: [https://papers.nips.cc/paper/7181-attention-is-all-you-need](https://papers.nips.cc/paper/7181-attention-is-all-you-need).

[2] LEE J, LEE Y, KIM J, et al. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks[C]// *Proceedings of the 36th International Conference on Machine Learning*. PMLR, 2019: 3744-3753. Available: [https://proceedings.mlr.press/v97/lee19d.html](https://proceedings.mlr.press/v97/lee19d.html).

[3] YUN C, BHOJANAPALLI S, RAWAT A S, et al. Are Transformers Universal Approximators of Sequence-to-Sequence Functions?[C]// *International Conference on Learning Representations*. 2020. Available: [https://openreview.net/forum?id=ByxRM0Ntvr](https://openreview.net/forum?id=ByxRM0Ntvr).

[4] HAHN M. Theoretical Limitations of Self-Attention in Neural Sequence Models[J]. *Transactions of the Association for Computational Linguistics*, 2020, 8: 156-171. DOI: [10.1162/tacl_a_00306](https://doi.org/10.1162/tacl_a_00306).

[5] BRUNNER G, LIU Y, PASCUAL D, et al. On Identifiability in Transformers[C]// *International Conference on Learning Representations*. 2020. Available: [https://research.google/pubs/on-identifiability-in-transformers/](https://research.google/pubs/on-identifiability-in-transformers/).

[6] GEVA M, SCHUSTER R, BERANT J, et al. Transformer Feed-Forward Layers Are Key-Value Memories[C]// *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, 2021: 5484-5495. DOI: [10.18653/v1/2021.emnlp-main.446](https://doi.org/10.18653/v1/2021.emnlp-main.446).

[7] CORDONNIER J-B, LOUKAS A, JAGGI M. On the Relationship between Self-Attention and Convolutional Layers[C]// *International Conference on Learning Representations*. 2020. Available: [https://openreview.net/forum?id=zoPf7R-2wZr](https://openreview.net/forum?id=zoPf7R-2wZr).
