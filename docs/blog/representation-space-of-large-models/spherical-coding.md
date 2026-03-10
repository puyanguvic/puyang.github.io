---
title: "LLM Embedding 的球面编码视角"
date: 2026-03-09T11:10:00-08:00
summary: "从球面码、Welch 下界与高维角度集中出发，解释为什么 LLM 的 token embedding 可以被视为受语义约束的高维码本。"
tags: ["LLM", "embeddings", "spherical code"]
---

# LLM Embedding 的球面编码视角

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/representation-space-of-large-models/spherical-coding" en-path="/blog/representation-space-of-large-models/spherical-coding-en" />

一张大模型词表的 embedding 矩阵通常包含数万到数十万个 token，而表示维度却常常只有几千。若沿用低维线性代数的直觉，很容易误以为这样的系统必然“拥挤”。但真正决定容量的并不是维度是否大于词表规模，而是归一化后方向之间能否保持足够低的相关性 [1-9]。

这正是球面编码视角的出发点。只要把每个 token embedding 归一化，整个词表就可以被看成单位球面上的一个点集，也就是一个受语义、频率和预测目标约束的高维码本。这样一来，问题就从“一词能否分到一维”转化为“在给定维度下，多少个方向仍可被可靠区分”。

> 核心结论：把归一化后的 token embedding 视为球面码，并不是修辞，而是一种相当有力的一阶模型。它解释了三件关键事实：表示主要通过方向关系被读取，词表容量主要由角间隔与 coherence 决定，而训练目标则会在这个高维码本上进一步压入语义、频率和预测兼容性的结构 [1-9]。

在“大模型的表示空间”系列中，本文承接上一篇 [Embedding 空间中的语义线性结构](/blog/representation-space-of-large-models/semantic-linearity)，把“局部关系会写成方向”进一步推广到“整个词表可以看成受语义约束的球面码本”；若继续追问这些静态几何如何进入上下文计算，可接着读 [Transformer Attention 的几何本质](/blog/geometry-of-transformers/what-attention-does)。

## 1. 从词表矩阵到球面码本

设 embedding 矩阵为

$$
E \in \mathbb{R}^{V \times d},
$$

其中 $V$ 是词表大小，$d$ 是表示维度。若将每一行归一化为

$$
\hat e_i = \frac{e_i}{\|e_i\|},
$$

则词表可以被写成

$$
C = \{\hat e_1,\dots,\hat e_V\} \subset \mathbb{S}^{d-1}.
$$

在编码理论中，这样的集合就叫做 spherical code [1][2]。衡量它是否“排得开”的两个基本量是最小角间隔

$$
\theta_{\min}(C) = \min_{i \neq j} \arccos(\hat e_i^\top \hat e_j)
$$

以及最大相关性

$$
\mu(C) = \max_{i \neq j} |\hat e_i^\top \hat e_j|.
$$

最小角间隔越大，最大相关性越小，码字就越容易区分。于是，一旦 embedding 主要通过内积或余弦被读取，词表几何就天然可以改写成球面码问题。

## 2. 为什么这一视角特别适合语言模型？

语言模型的很多关键计算天然偏向方向读出。输入 embedding、输出层打分、归一化后的相似度比较以及许多检索操作，本质上都依赖内积结构。Press 与 Wolf 的分析还指出，输出 embedding 并不是预测头的附属部件，而是语言模型整体质量的重要组成部分；在权重共享或紧耦合设置下，输入与输出几何会相互影响 [5]。

与此同时，前文已经说明，高维表示天然倾向于范数集中，而训练过程又常把有用自由度进一步推向方向结构 [4][6][7]。因此，虽然真实 LLM embedding 不一定被显式训练成理想球面码，它们的有效几何往往越来越接近“长度受控、方向主导、内积可读”的码本形态。图 1 能把这个问题重写得更清楚。

![token embedding 的球面码本视角示意图](./spherical-coding-codebook.svg)

*图 1. 把归一化后的词表看成球面码之后，核心问题不再是“维度够不够分”，而是“角间隔与最大互相关是否仍然足够可用”。*

图 1 真正完成的转换，是把“对象数量大于维度数”这一线性代数直觉，改写成“角间隔能否维持”这一高维几何问题。后文关于 Welch 下界的讨论，正是对这幅图的定量化。

## 3. 为什么几千维足以容纳几十万 token？

这正是球面编码视角最直接的解释力所在。对于单位向量集合，经典的 Welch 下界给出 [2]

$$
\mu(C)^2 \ge \frac{V-d}{d(V-1)}.
$$

若取 $V = 100000$、$d = 4096$，则

$$
\mu(C) \ge \sqrt{\frac{100000-4096}{4096 \cdot 99999}} \approx 0.0153.
$$

对应的角度下界约为

$$
\arccos(0.0153) \approx 89.1^\circ.
$$

这个数值非常关键。它意味着：即便在最理想的均匀排布下，`100k` 个单位向量放进 `4096` 维空间，也只要求最坏情况下的两两余弦相似度不低于约 `0.0153`。而对高维随机单位向量而言，典型内积波动本来就约为

$$
\operatorname{Std}(\hat x^\top \hat y) \approx \frac{1}{\sqrt{d}} = \frac{1}{64} \approx 0.0156.
$$

这与 Welch 下界几乎处于同一量级 [2][3]。因此，从纯几何容量看，`4096` 维承载 `100k` 级词表并不勉强；真正困难的不是“能否放下”，而是“如何在放下的同时保留语义、频率与预测结构”。

用同样的量级估算，可以得到：

| 词表规模 $V$ | 维度 $d$ | Welch 下界 $\mu_{\min}$ | 对应角度下界 |
| --- | --- | --- | --- |
| `50k` | `4096` | `0.01497` | `89.14^\circ` |
| `100k` | `4096` | `0.01530` | `89.12^\circ` |
| `200k` | `4096` | `0.01546` | `89.11^\circ` |

当 $V \gg d$ 时，控制量级实际上已经非常接近 $1/\sqrt{d}$。这说明真正限制词表设计的，不是简单的“几何塞不下”，而是更上层的统计与优化约束。

## 4. 真实词表为什么不会是理想均匀码？

理想球面码追求的是整体均匀展开，而真实语言模型追求的是任务相关的非均匀组织。高频功能词、内容词、语义邻近词、形态变化词和特殊符号，不可能在球面上保持完全对称。训练目标会主动偏离几何最优，去换取更好的预测性。

这至少表现为三种结构偏差。

- 频率偏差。高频 token 往往在输出头里承担特殊角色，几何位置和偏置项会受到频率分布影响 [8]。
- 语义局部性。语义相近 token 会在局部形成簇，而不是被均匀打散到整个球面。
- 任务可读性。模型需要让注意力、线性层和 softmax 都能高效读取这些向量，因此几何组织必须服务于后续计算，而不只是追求最大角间隔。

因此，LLM embedding 更准确的说法不是“理想球面码”，而是“受语义、频率和预测头共同约束的球面码本近似”。

## 5. 容量充足为什么不等于训练容易？

Welch 下界给出的是一个容量约束，而不是一条优化保证。它说明在给定维度与词表规模下，最坏情况下的互相关至少要有多大，却并不说明真实训练过程一定能逼近这种理想排布。换句话说，球面码理论回答的是“几何上能否放下”，而不是“梯度下降能否轻松学出”。

真实词表几何之所以更难，至少有三层原因：

- 词频极不均匀，高频 token 会反复收到梯度，长尾 token 则可能长期训练不足；
- 输入 embedding、输出 embedding 与 softmax 头往往彼此耦合，码本必须同时服务表示与预测 [5][8]；
- 语义局部性要求部分 token 明确形成簇，这会主动牺牲某些全局角间隔。

因此，一个空间在几何上完全“放得下”几十万 token，并不意味着真实模型就能轻松学出一个近似最优码本。球面编码视角最有价值的地方，不是替代优化理论，而是把“容量是否足够”和“训练是否容易”这两个问题清楚分开。

## 6. 球面码视角能解释什么，不能解释什么？

这一视角至少解释了三个核心事实。

- 为什么 cosine similarity 常比未经归一化的欧氏距离更自然：如果表示主要活在球面上，角度就是首要变量。
- 为什么巨型词表不会自动把 embedding 空间挤爆：高维球面的方向容量远大于低维直觉。
- 为什么全局可分与局部相近可以并存：球面码要求的是整体低相关，而不是处处等距。

但它也有明确边界。

- 它不能单独解释上下文化语义，因为真实隐藏状态会随上下文与层深持续重写 [6][7]。
- 它不能保证各向同性，预训练语言模型常常保留明显的频率偏置与各向异性热点 [6-8]。
- 它也不要求所有向量严格单位范数；更准确的说法是，在很多关键操作中，归一化之后的方向几何提供了更稳健的一阶近似。

因此，球面编码视角最好的用法，不是把它当成完整真相，而是把它当成理解词表几何的主导模型。

## 7. 结语

一旦把 embedding 表看成高维码本，许多分散的现象就被统一起来了：近似正交解释容量，球壳分布解释为什么方向比长度更重要，Welch 下界解释几千维为何足以承载巨型词表，而训练目标则解释为什么真实码本必然带有频率与语义结构。

归结起来，**embedding 不是一张松散的参数表，而是一张被训练出来的高维语义码本。** 球面编码视角当然不足以穷尽 LLM 表示的全部复杂性，但它抓住了词表几何最硬的约束。

上一篇：[Embedding 空间中的语义线性结构](/blog/representation-space-of-large-models/semantic-linearity)。延伸阅读：[Transformer Attention 的几何本质](/blog/geometry-of-transformers/what-attention-does)。

## 参考文献

[1] CONWAY J H, SLOANE N J A. *Sphere Packings, Lattices and Groups*[M]. 3rd ed. New York: Springer, 1999. DOI: [10.1007/978-1-4757-6568-7](https://doi.org/10.1007/978-1-4757-6568-7).

[2] DATTA S, HOWARD S D, COCHRAN D. Geometry of the Welch Bounds[J]. *Linear Algebra and its Applications*, 2012, 437(10): 2455-2470. DOI: [10.1016/j.laa.2012.05.036](https://doi.org/10.1016/j.laa.2012.05.036).

[3] CAI T T, FAN J, JIANG T. Distributions of Angles in Random Packing on Spheres[J]. *Journal of Machine Learning Research*, 2013, 14(57): 1837-1864. URL: [https://jmlr.org/papers/v14/cai13a.html](https://jmlr.org/papers/v14/cai13a.html).

[4] WANG T, ISOLA P. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere[C]// *Proceedings of the 37th International Conference on Machine Learning*. PMLR, 2020: 9929-9939. URL: [https://proceedings.mlr.press/v119/wang20k.html](https://proceedings.mlr.press/v119/wang20k.html).

[5] PRESS O, WOLF L. Using the Output Embedding to Improve Language Models[C]// *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*. Valencia, Spain: Association for Computational Linguistics, 2017: 157-163. URL: [https://aclanthology.org/E17-2025/](https://aclanthology.org/E17-2025/).

[6] LI B, ZHOU H, HE J, et al. On the Sentence Embeddings from Pre-trained Language Models[C]// *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Online: Association for Computational Linguistics, 2020: 9119-9130. DOI: [10.18653/v1/2020.emnlp-main.733](https://doi.org/10.18653/v1/2020.emnlp-main.733).

[7] GAO T, YAO X, CHEN D. SimCSE: Simple Contrastive Learning of Sentence Embeddings[C]// *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, 2021: 6894-6910. DOI: [10.18653/v1/2021.emnlp-main.552](https://doi.org/10.18653/v1/2021.emnlp-main.552).

[8] KOBAYASHI G, KURIBAYASHI T, YOKOI S, et al. Transformer Language Models Handle Word Frequency in Prediction Head[C]// *Findings of the Association for Computational Linguistics: ACL 2023*. Toronto, Canada: Association for Computational Linguistics, 2023: 4523-4535. DOI: [10.18653/v1/2023.findings-acl.276](https://doi.org/10.18653/v1/2023.findings-acl.276).

[9] DUAN Y, LU J, ZHOU J. UniformFace: Learning Deep Equidistributed Representation for Face Recognition[C]// *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2019: 3415-3424. URL: [https://openaccess.thecvf.com/content_CVPR_2019/html/Duan_UniformFace_Learning_Deep_Equidistributed_Representation_for_Face_Recognition_CVPR_2019_paper.html](https://openaccess.thecvf.com/content_CVPR_2019/html/Duan_UniformFace_Learning_Deep_Equidistributed_Representation_for_Face_Recognition_CVPR_2019_paper.html).
