---
title: "Multi-Head Attention 的必要性与表达优势"
date: 2026-03-09T11:40:00-08:00
summary: "从单头注意力的几何瓶颈出发，解释 multi-head attention 为什么等价于并行构造多个上下文坐标系。"
tags: ["Transformer", "multi-head attention", "representation geometry"]
---

# Multi-Head Attention 的必要性与表达优势

如果说 attention 的本质，是由 query-key 几何生成软坐标，再在 value 空间做重建，那么接下来的自然问题就是：为什么还需要 **multi-head**？为什么不能只保留一个更宽的 head，把所有事情都交给同一套注意力结构？

Vaswani 等人引入 multi-head attention 时给出的直觉是：模型需要“从不同表示子空间联合关注信息” [1]。这个直觉是对的，但还可以说得更严格一些：

> multi-head attention 的关键价值，不是简单重复同一运算，而是并行定义多套不同的匹配几何、软坐标系统与内容重建通道。

换句话说，单头注意力给模型的是一张上下文坐标图；多头注意力给模型的是多张并行坐标图。它们共同构成一个更高维、更可解耦的上下文读取系统。

> 核心结论：每个 attention head 都对应一套独立的 query-key 度量和 value 映射，因此 multi-head attention 本质上是在并行构造多个“上下文坐标系”。它的作用不只是增加容量，而是把不同关系类型分散到不同几何视角中处理，再通过拼接与线性映射重新整合 [1-6]。

## 1. 从公式上看，多头到底多了什么？

标准 multi-head attention 可写为

$$
\mathrm{MHA}(X)
=
\mathrm{Concat}(H_1, \dots, H_m) W_O,
$$

其中第 $h$ 个 head 为

$$
H_h = A_h V_h,
\qquad
A_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right),
$$

且

$$
Q_h = XW_Q^{(h)}, \qquad
K_h = XW_K^{(h)}, \qquad
V_h = XW_V^{(h)}.
$$

这组公式最值得注意的，不是“有很多个 head”，而是每个 head 都有自己的一套：

- query-key 匹配度量；
- 注意力分布 $A_h$；
- value 内容映射；
- 最终重建结果 $H_h$。

因此，多头并不只是把同一个注意力矩阵复制多份；它是在并行学习多种不同的“谁该被读取、被读出来之后该提供什么”的规则。

## 2. 单头注意力的真正瓶颈是什么？

单头 attention 并不是不能工作，而是它必须让所有关系共享同一套坐标系统。对固定 query 而言，单头只能产生一组权重

$$
\alpha_i \in \Delta^{n-1},
$$

并据此做一次 value 重建。这会带来三个直接限制。

### 只有一种匹配几何

单头只能通过一套双线性形式 $W_Q^\top W_K$ 来定义“相关性”。而自然语言中的相关性并不唯一：句法依赖、共指关系、位置偏移、篇章回溯和任务相关读取，往往需要完全不同的度量规则。

### 只有一组软坐标

对于同一个 token，单头只能给出一张注意力分布图。如果当前 token 同时需要“主语是谁”“代词指谁”“局部修饰语是什么”这三类信息，那么所有需求都必须在同一组坐标中竞争。

### 只有一条内容通道

因为单头只有一组 value 映射，它无法自然地区分“我关注这个位置是为了拿句法角色”还是“我关注这个位置是为了拿语义属性”。匹配与内容读取容易耦合在一起。

因此，单头的根本局限不是参数太少，而是：**它把太多异质关系压进了同一套几何读取机制中**。

## 3. 多头为什么可以被看成多个上下文坐标系？

一旦引入多头，第 $i$ 个 token 在每个 head 上都会得到一组独立的软坐标

$$
\alpha_i^{(h)} \in \Delta^{n-1}.
$$

这意味着同一个位置不再只有一种“看上下文”的方式，而是同时拥有多种局部坐标展开。更具体地说，每个 head 都在回答一个稍有不同的问题：

- 从这个视角看，哪些 token 与当前 query 最相关？
- 这种相关性应按什么关系定义？
- 一旦选中这些位置，应当取回哪一类内容表示？

![multi-head attention 的并行坐标系示意图](./multi-head-coordinate-systems.svg)

*图 1. 单头注意力只能在一套匹配几何上生成一组软坐标；多头注意力则会并行生成多组坐标，并在不同 value 通道上完成多个重建，再由输出映射整合。*

因此，把 multi-head attention 理解为“多个语义坐标系”是有技术含义的，而不是修辞：每个 head 的确都定义了独立的匹配度量、独立的单纯形坐标，以及独立的内容重建路径。

## 4. 为什么这会带来更强的表达能力？

多头带来的提升，至少来自三层机制。

### 关系解耦

不同 head 可以分别学习不同关系类型。有的 head 更像位置算子，有的更像句法选择器，有的更像篇章回溯器。Clark 等人和 Voita 等人都观察到，部分 head 会稳定呈现出特定模式，例如关注分隔符、固定偏移、句法依赖或共指结构 [2][3]。

### 内容解耦

即使多个 head 都关注同一位置，由于 $W_V^{(h)}$ 不同，它们也可以从同一 token 中提取不同方面的内容。一个 head 可能取出结构线索，另一个取出语义属性，第三个则取出任务相关的预测提示。

### 计算并行化

Weiss 等人的 RASP 视角说明，Transformer 的许多序列计算本质上可以被分解成若干并行的 selection / aggregation 步骤，而 head 数直接影响这类并行组合能否在少层数内完成 [6]。从这个角度看，多头不只是“表达更多”，而是“允许更多关系同时被计算”。

因此，多头机制的价值不只是容量增大，而是让模型有条件把复杂任务分散到多条并行的几何路径上执行。

## 5. 实证上，多头真会学出分工吗？

答案是：会，但不是每个 head 都同样重要。

Voita 等人在 ACL 2019 中发现，一小部分 head 会承担稳定、重要且常带有语言学可解释性的功能，而很多其余 head 可以被大规模剪枝，性能仅轻微下降 [3]。Michel 等人在 NeurIPS 2019 中也给出类似结论：大量 attention heads 在测试时可以被移除，但某些层和某些 head 对性能明显更关键 [4]。

这说明一个容易被误解的事实：

> multi-head 的价值，并不要求每个 head 都不可替代；它只要求模型拥有把关系分散到多个通道中的自由度。

换句话说，冗余并不等于无用。多头机制可以提供更宽松的优化空间，让重要 head 形成稳定分工，同时允许其他 head 作为备份、近似或训练过程中的辅助通道存在。

从工程上看，可以把经验事实粗略总结如下：

| 观察 | 含义 |
| --- | --- |
| 一些 head 表现出稳定功能模式 [2][3] | 多头确实会产生一定程度的分工 |
| 大量 head 可被剪枝 [3][4] | 分工存在冗余，并非每个 head 都关键 |
| 关键 head 通常集中在特定层或特定关系类型 [3][4] | multi-head 的价值是结构性的，而不是平均分布的 |

## 6. 为什么头数也不是越多越好？

如果保持模型总宽度不变，head 数增加意味着单个 head 的维度 $d_h$ 下降。于是会出现一个经典权衡：

- 头太少：关系不够解耦，许多异质读取需求被迫挤在同一套几何中。
- 头太多：每个 head 的通道过窄，容易产生冗余或表达不足。

这也是为什么 Michel 等人的结果并不否定 multi-head，反而说明了一个更精确的结论：**多头是必要的，但具体多少头有效，取决于模型宽度、层数、任务与训练过程** [4]。

此外，Cordonnier 等人证明，多头 self-attention 在合适参数化下可表达卷积层 [5]。这一点也能解释为什么“多个头”比“一个很宽的头”更有结构优势：前者天然支持多种局部或全局模式并行存在，后者则必须把这些模式揉进同一套权重系统中。

## 7. 结语

如果要把本文压缩成一句话，我会写：

> multi-head attention 的核心价值，是为模型并行提供多套上下文坐标系，而不是简单重复同一注意力运算。

这也解释了为什么 Transformer 能在同一层里同时处理句法依赖、位置模式、共指关系、篇章回溯和任务相关读取：它不是靠单一大型注意力矩阵“一次性看懂一切”，而是靠多个 head 从不同视角同时解析上下文，再把这些局部几何读数整合成新的 token 表示。

如果把这篇与前文的[Transformer Attention 的几何本质](/blog/geometry-of-transformers/what-attention-does)一起看，逻辑会更完整：单个 head 已经是一套“软坐标 + 重建”机制，而 multi-head 则把这种机制并行复制到多种不同的匹配几何中。两者结合，才构成了 Transformer 最核心的上下文表示引擎。

## 参考文献

[1] VASWANI A, SHAZEER N, PARMAR N, et al. Attention Is All You Need[C]// *Advances in Neural Information Processing Systems 30*. Red Hook, NY: Curran Associates, 2017. Available: [https://papers.nips.cc/paper/7181-attention-is-all-you-need](https://papers.nips.cc/paper/7181-attention-is-all-you-need).

[2] CLARK K, KHANDELWAL U, LEVY O, et al. What Does BERT Look At? An Analysis of BERT's Attention[C]// *Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*. Florence, Italy: Association for Computational Linguistics, 2019: 276-286. DOI: [10.18653/v1/W19-4828](https://doi.org/10.18653/v1/W19-4828).

[3] VOITA E, TALBOT D, MOISEEV F, et al. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned[C]// *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Florence, Italy: Association for Computational Linguistics, 2019: 5797-5808. DOI: [10.18653/v1/P19-1580](https://doi.org/10.18653/v1/P19-1580).

[4] MICHEL P, LEVY O, NEUBIG G. Are Sixteen Heads Really Better than One?[C]// *Advances in Neural Information Processing Systems 32*. Red Hook, NY: Curran Associates, 2019. Available: [https://papers.nips.cc/paper/9551-are-sixteen-heads-really-better-than-one](https://papers.nips.cc/paper/9551-are-sixteen-heads-really-better-than-one).

[5] CORDONNIER J-B, LOUKAS A, JAGGI M. On the Relationship between Self-Attention and Convolutional Layers[C]// *International Conference on Learning Representations*. 2020. Available: [https://openreview.net/forum?id=zoPf7R-2wZr](https://openreview.net/forum?id=zoPf7R-2wZr).

[6] WEISS G, GOLDBERG Y, YAHAV E. Thinking Like Transformers[C]// *Proceedings of the 38th International Conference on Machine Learning*. PMLR, 2021: 11080-11090. Available: [https://proceedings.mlr.press/v139/weiss21a.html](https://proceedings.mlr.press/v139/weiss21a.html).
