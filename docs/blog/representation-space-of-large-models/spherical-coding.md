---
title: "LLM Embedding 的球面编码视角"
date: 2026-03-09T11:10:00-08:00
summary: "从球面码、Welch 下界与高维角度集中出发，解释为什么 LLM token embedding 可以被视为受语义约束的高维码本。"
tags: ["LLM", "embeddings", "spherical code"]
---

# LLM Embedding 的球面编码视角

如果把大模型的 token embedding 仅仅理解为“一张词表对应一堆向量”，我们会错过一个更有解释力的问题：为什么一个只有 `d=4096` 左右维度的空间，能够稳定地承载 `50k`、`100k` 甚至更大规模的词表，同时又保留足够强的可分性、相似性与可计算性？

一个更专业的视角来自编码理论与球面几何：把 embedding 表理解为一个**受语义约束的高维码本**。在这个视角下，每个 token 对应一个码字，模型的任务不是给每个 token 分配一根独立坐标轴，而是在高维球面上组织一个既可分、又可比较、还能被下游线性算子高效读取的点集。

> 核心结论：把归一化后的 token embedding 看作球面码（spherical code）并不是一个修辞比喻，而是一个相当严谨的近似模型。它抓住了三个关键事实：表示主要由方向关系读取，容量主要由角间隔而非“独占维度”决定，而训练目标则在这个高维码本上进一步压入语义与频率结构 [1-9]。

## 1. 什么是 spherical code？

在编码理论里，一个球面码可以写成

$$
C = \{c_1, c_2, \dots, c_N\} \subset \mathbb{S}^{d-1},
$$

也就是单位球面上的 $N$ 个点。通常关心的两个量是最小角间隔

$$
\theta_{\min}(C) = \min_{i \ne j} \arccos(c_i^\top c_j),
$$

以及最大相关性

$$
\mu(C) = \max_{i \ne j} |c_i^\top c_j|.
$$

二者是等价的：$\theta_{\min}$ 越大，$\mu$ 越小，码字越容易区分 [1][2]。因此，球面码的基本问题并不是“如何让每个点占一个维度”，而是“在固定维度下，如何让大量点尽可能均匀地分布在球面上”。

这与 token embedding 的实际角色高度相似。若把 embedding 矩阵记为

$$
E \in \mathbb{R}^{V \times d},
$$

并把每一行归一化为

$$
\hat e_i = \frac{e_i}{\|e_i\|},
$$

那么词表就变成了单位球面上的一组点。此时，比较 token 是否相近，本质上就变成了比较 $\hat e_i^\top \hat e_j$ 或它对应的夹角。也就是说，**一旦范数波动相对稳定，embedding 表就自然近似成一个球面码本**。

## 2. 为什么这个视角对语言模型特别合适？

首先，现代语言模型的大量关键计算本来就偏向角度读出。无论是相似度检索、归一化后的 embedding 比较，还是 softmax 头对词表的逐项评分，本质上都依赖内积结构。Press 与 Wolf 还指出，输出 embedding 不只是预测头的附属参数，而是语言模型质量的重要组成部分；在权重共享或紧耦合设置下，输入/输出词表几何会被同时用于表示与解码 [5]。

其次，高维表示学习普遍会削弱径向自由度、强化方向结构。Wang 与 Isola 在 ICML 2020 中把“局部对齐 + 全局均匀”刻画为球面表示学习的核心张力 [4]；Li 等人与 Gao 等人则分别从 BERT-flow 和 SimCSE 出发，给出 NLP 里的直接证据：未经处理的预训练表示往往存在各向异性，而适当的归一化、流变换或对比学习会显著改善其球面均匀性与语义可读性 [6][7]。

更谨慎地说，从 [4-7] 可以推得一个很稳妥的判断：语言模型 embedding 不一定被显式训练成“理想球面码”，但其有效几何往往越来越接近“长度受控、方向主导、内积可读”的高维码本。

![token embedding 的球面码本视角示意图](./spherical-coding-codebook.svg)

*图 1. 把归一化后的词表看作球面码后，问题会从“维度够不够分”转为“角间隔是否足够大”。对大模型而言，真正关键的是高维球面上的方向容量，而不是一词一维。*

## 3. 为什么 `4096` 维足以容纳 `100k` token？

这是球面编码视角最有力的一点。很多人会误以为：如果只有 `4096` 维，就不可能稳定表示 `100k` 个 token。这个直觉来自低维线性代数，而不是来自高维几何。

对于单位向量集合，经典的 Welch 下界给出 [2]

$$
\mu(C)^2 \ge \frac{N-d}{d(N-1)}.
$$

把 $N = 100000$、$d = 4096$ 代入，有

$$
\mu(C) \ge \sqrt{\frac{100000-4096}{4096 \cdot 99999}} \approx 0.0153.
$$

这意味着什么？它意味着即便在“最理想的均匀排布”下，最坏情况下的两两余弦相似度也只需要大于约 `0.0153`。换成角度，就是大约

$$
\arccos(0.0153) \approx 89.1^\circ.
$$

这个数值非常说明问题：`100k` 个码字放进 `4096` 维空间，并不要求它们彼此远离到不可思议的程度；只要大多数方向接近正交即可。而对高维随机单位向量而言，典型内积波动本来就只有

$$
\operatorname{Std}(\hat x^\top \hat y) \approx \frac{1}{\sqrt{d}} = \frac{1}{64} \approx 0.0156,
$$

这和上面的 Welch 量级几乎一致 [2][3]。换句话说：

> 从纯几何容量上看，`4096` 维承载 `100k` 级词表并不勉强，反而正处在一个非常自然的量级区间。

真正困难的，从来不是“把 token 塞进去”，而是“在塞进去的同时还保留语义、频率、句法和预测兼容性”。

如果把词表规模继续放大，量级关系也并不会突然崩坏：

| 词表规模 $N$ | 维度 $d$ | Welch 下界 $\mu_{\min}$ | 对应角度下界 |
| --- | --- | --- | --- |
| `50k` | `4096` | `0.01497` | `89.14°` |
| `100k` | `4096` | `0.01530` | `89.12°` |
| `200k` | `4096` | `0.01546` | `89.11°` |

这个表最值得注意的地方是：当 $N \gg d$ 时，约束的主量级已经接近 $1/\sqrt{d}$，因此真正决定词表几何上限的，往往不是“能否放下”，而是“是否还能保留训练目标想要的结构” [2][3]。

## 4. embedding 容量到底来自哪里？

如果继续追问，容量至少来自三层相互耦合的机制。

### 高维球面的方向容量

这是最底层的几何来源。高维球面允许大量近似正交方向共存，随机点对的角度会自然集中到 `90` 度附近 [3]。因此，词表规模远大于维度，并不构成理论障碍。

### 训练目标塑造了“非均匀但有组织”的码本

理想球面码追求的是尽量均匀展开；语言模型并不追求绝对均匀，而是追求**任务相关的均匀性**。高频 token、语义邻近 token、功能词与内容词，在空间中的位置不会完全对称。Kobayashi 等人对 Transformer prediction head 的分析表明，词频信息会在输出头中形成稳定偏置与几何效应 [8]。因此，真实的 LLM 码本不是“纯几何最优码”，而是“几何容量 + 语料统计 + 预测目标”共同决定的折中解。

### 下游读取方式奖励角度友好的组织

只要模型大量使用内积、归一化和 softmax 读取词表，方向结构就比绝对长度更稳定、更容易被共享。这个判断在视觉表征中也有直接工程先例：UniformFace 明确把类别中心视为超球面上的均匀分布点集，并通过均匀化约束提升整体判别力 [9]。这当然不是语言模型的直接证据，但它说明“把大规模类别表理解为球面码本”在现代深度学习里是成熟而有效的设计思想。

## 5. 球面编码视角能解释什么，又不能解释什么？

这个视角至少解释了三件非常关键的事。

- 为什么 cosine similarity 往往比未经归一化的欧氏距离更自然：如果表示主要活在球面上，那么角度就是首要变量。
- 为什么巨型词表不会自动把 embedding 空间挤爆：高维球面的容量远比低维直觉大得多。
- 为什么“全局可分”与“局部语义相近”可以同时成立：球面码要求的是整体低相关，而不是处处等距，因此完全允许局部聚类与全局展开并存。

但它同样有边界。

- 它不能单独解释上下文化语义。真实 LLM 隐状态会随上下文、层深与任务读出方式发生显著变化 [6][7]。
- 它不能保证码本各向同性。预训练语言模型天然会出现频率偏置、语法偏置与各向异性热点 [6-8]。
- 它也不是说 embedding 必须严格单位范数。更准确的说法是：在很多关键操作里，把长度因素压缩掉之后，球面几何成了更稳健的近似。

因此，“球面码”最好的用法不是把它当成完整真相，而是把它当成**描述词表几何的一阶模型**。

## 6. 结语

把 LLM embedding 看成球面码本，有一个很直接的收益：我们不再用“每个 token 需要一个独立维度”这种低维直觉去理解词表，而是改用“在高维球面上组织一个受约束的方向系统”去理解它。

这会把很多零散现象串起来：近似正交、球壳分布、cosine 检索、巨型词表容量、输出头几何、频率偏置，它们并不是互不相关的小技巧，而是同一类高维码本结构在不同模块中的投影。

如果与前文的[高维向量近似正交的几何机制](/blog/high-dimensional-space-and-machine-learning/orthogonality)和[Embedding 向量的超球面分布及其成因](/blog/high-dimensional-space-and-machine-learning/hypersphere)一起阅读，这个结论会更完整：先是高维概率把表示压到球壳并把方向推向近似正交，随后训练目标再把这片高容量方向场塑造成带有词频、语义和预测约束的实际码本。

如果要把本文压缩成一句话，我会写：

> embedding 不是简单的参数表，而是一张被训练出来的高维语义码本；球面编码视角抓住了它最核心的几何约束。

## 参考文献

[1] CONWAY J H, SLOANE N J A. *Sphere Packings, Lattices and Groups*[M]. 3rd ed. New York: Springer, 1999. DOI: [10.1007/978-1-4757-6568-7](https://doi.org/10.1007/978-1-4757-6568-7).

[2] DATTA S, HOWARD S D, COCHRAN D. Geometry of the Welch Bounds[J]. *Linear Algebra and its Applications*, 2012, 437(10): 2455-2470. DOI: [10.1016/j.laa.2012.05.036](https://doi.org/10.1016/j.laa.2012.05.036).

[3] CAI T T, FAN J, JIANG T. Distributions of Angles in Random Packing on Spheres[J]. *Journal of Machine Learning Research*, 2013, 14(57): 1837-1864. Available: [https://jmlr.org/papers/v14/cai13a.html](https://jmlr.org/papers/v14/cai13a.html).

[4] WANG T, ISOLA P. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere[C]// *Proceedings of the 37th International Conference on Machine Learning*. PMLR, 2020: 9929-9939. Available: [https://proceedings.mlr.press/v119/wang20k.html](https://proceedings.mlr.press/v119/wang20k.html).

[5] PRESS O, WOLF L. Using the Output Embedding to Improve Language Models[C]// *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*. Valencia, Spain: Association for Computational Linguistics, 2017: 157-163. Available: [https://aclanthology.org/E17-2025/](https://aclanthology.org/E17-2025/).

[6] LI B, ZHOU H, HE J, et al. On the Sentence Embeddings from Pre-trained Language Models[C]// *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Online: Association for Computational Linguistics, 2020: 9119-9130. DOI: [10.18653/v1/2020.emnlp-main.733](https://doi.org/10.18653/v1/2020.emnlp-main.733).

[7] GAO T, YAO X, CHEN D. SimCSE: Simple Contrastive Learning of Sentence Embeddings[C]// *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, 2021: 6894-6910. DOI: [10.18653/v1/2021.emnlp-main.552](https://doi.org/10.18653/v1/2021.emnlp-main.552).

[8] KOBAYASHI G, KURIBAYASHI T, YOKOI S, et al. Transformer Language Models Handle Word Frequency in Prediction Head[C]// *Findings of the Association for Computational Linguistics: ACL 2023*. Toronto, Canada: Association for Computational Linguistics, 2023: 4523-4535. DOI: [10.18653/v1/2023.findings-acl.276](https://doi.org/10.18653/v1/2023.findings-acl.276).

[9] DUAN Y, LU J, ZHOU J. UniformFace: Learning Deep Equidistributed Representation for Face Recognition[C]// *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2019: 3415-3424. Available: [https://openaccess.thecvf.com/content_CVPR_2019/html/Duan_UniformFace_Learning_Deep_Equidistributed_Representation_for_Face_Recognition_CVPR_2019_paper.html](https://openaccess.thecvf.com/content_CVPR_2019/html/Duan_UniformFace_Learning_Deep_Equidistributed_Representation_for_Face_Recognition_CVPR_2019_paper.html).
