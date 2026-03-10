---
title: "Embedding 向量的超球面分布及其成因"
date: 2026-03-09T10:20:00-08:00
summary: "从范数集中、归一化训练目标与角度度量出发，解释为什么 embedding 常常表现为近似球壳分布。"
tags: ["machine learning", "embeddings", "hypersphere geometry"]
---

# Embedding 向量的超球面分布及其成因

在许多现代表示学习系统里，embedding 向量并不是均匀散落在整个欧氏空间中，而是更像贴着某个高维球壳分布。这个现象并非偶然。高维概率本身会让范数集中到一个稳定尺度附近，而归一化层、对比学习目标与角度间隔损失又会进一步压缩径向自由度，使得语义差异更多地编码在方向上，而不是编码在长度上 [1-5]。

> 核心结论：embedding 之所以常常适合用 cosine similarity，不只是经验上“好用”，而是因为训练后的表示在几何上已经越来越接近一个高维超球面：范数相近，方向承载主要语义，角度比长度更稳定 [1-5]。

## 1. 为什么向量范数会先集中到 $\sqrt{d}$ 量级？

如果 $x \in \mathbb{R}^d$ 的各维经过中心化且方差尺度相近，那么

$$
\|x\|^2 = \sum_{i=1}^d x_i^2.
$$

当每一维都贡献有限能量时，平方范数的期望会随维度线性增长，从而得到

$$
\mathbb{E}\|x\|^2 \approx d,
\qquad
\|x\| \approx \sqrt{d}.
$$

更重要的是，范数的相对波动会随着 `d` 的上升而缩小。对高斯或更一般的次高斯模型，范数集中是标准结论 [1]。这意味着即使样本方向差异很大，它们的长度也往往落在相近区间内。换句话说，向量首先会被高维统计压到一层薄壳上，而不是自由散布在从 `0` 到任意大的整个半径范围里。

这一步并不要求模型已经学会语义结构。它只是说明：在高维里，“长度几乎一样”本身就是一种天然倾向。

## 2. 为什么训练会进一步强化球壳结构？

真实的 embedding 并不只是随机向量。训练过程通常还会进一步削弱径向波动，并把判别信号转移到角度上。

- 归一化机制会稳定特征尺度，使不同样本的范数保持在较窄范围。
- 对比学习往往同时追求局部对齐与全局均匀展开，这会把表示推向更接近球面的组织方式 [2]。
- 一类经典的人脸识别方法直接在单位超球面上学习判别边界，例如 NormFace、SphereFace 与 ArcFace 都明确把角度间隔作为核心优化对象 [3-5]。

因此，球壳分布不是单一机制的产物，而是高维统计、网络归一化和训练目标共同塑形的结果：高维先让范数集中，优化再把“有用的自由度”更多地保留给方向。

![embedding 的球壳几何示意图](./hypersphere-embedding-geometry.svg)

*图 1. 当 $\|x\|$ 与 $\|y\|$ 都稳定在半径 $r$ 附近时，样本差异主要体现为夹角；这也是 cosine similarity 在 embedding 检索中更自然的原因。*

## 3. 为什么 cosine similarity 会成为更自然的度量？

对两个向量 `x` 与 `y`，有

$$
\cos \theta = \frac{x^\top y}{\|x\| \|y\|},
$$

以及

$$
\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2x^\top y.
$$

如果 $\|x\|$ 与 $\|y\|$ 都稳定在同一尺度 $r$ 附近，那么上式可近似写成

$$
\|x-y\|^2 \approx 2r^2 (1 - \cos \theta).
$$

此时，欧氏距离与 cosine similarity 本质上在读取同一套角度信息，只不过前者仍然会把残余的范数波动一起算进去，后者则显式地把注意力集中到方向关系上 [2-5]。这就是为什么在文本检索、语义匹配、推荐系统和向量数据库中，cosine similarity 往往比未经归一化的欧氏距离更稳健。

从几何上说，使用 cosine similarity 等价于先把向量投影到单位球面，再比较它们在球面上的相对位置。对于已经接近球壳分布的 embedding，这种度量方式更贴近其真实几何。

## 4. 范数真的完全不重要吗？

答案是否定的。说 embedding“近似位于球面上”，并不意味着范数彻底失去意义。实际系统里，范数有时仍可能携带一些附加信息，例如样本置信度、频率、显著性，或者某些任务相关的强弱信号。

更准确的表述应该是：在很多现代表示学习场景中，**方向比范数更稳定、更可迁移，也更接近我们想利用的语义结构**。范数可能仍然有用，但它通常不是最主要、最通用的判别来源。

## 5. 一个更统一的理解：embedding 空间接近超球面流形

把前面三篇文章连起来看，会得到一条连贯的几何链条：

- 高维随机向量的范数先集中到薄壳上。
- 球壳上的随机方向再趋向彼此近似正交。
- 训练过程进一步把语义组织到方向结构里，并弱化径向噪声。

因此，embedding 空间虽然仍然嵌在 `R^d` 中，但一个更贴切的近似模型往往是：它被训练成了一个具有局部语义结构的超球面流形。模型并没有充分使用整个欧氏空间的所有自由度，而是把真正有意义的表示压缩到了一个更稳定、更低熵的几何对象上。

这也是为什么理解 embedding，不能只盯着某一个坐标值，而要把它看成一个空间问题：范数分布、角度分布、局部邻域与球面上的相对位置，都会直接影响检索、分类和泛化。

## 6. 结语

embedding 常常呈现超球面分布，并不是工程实现中的偶然副作用，而是高维概率规律与现代训练目标共同作用的结果。范数集中让球壳成为自然舞台，归一化与对比目标让方向成为主要载体，而 cosine similarity 则恰好对应了这种几何结构最自然的读取方式。

如果要用一句话总结本文，那就是：embedding 适合用 cosine，不只是因为它好算，而是因为它本来就更像球面上的几何。

## 参考文献

[1] VERSHYNIN R. *High-Dimensional Probability: An Introduction with Applications in Data Science*[M]. Cambridge: Cambridge University Press, 2018. DOI: [10.1017/9781108231596](https://doi.org/10.1017/9781108231596).

[2] WANG T, ISOLA P. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere[C]// *Proceedings of the 37th International Conference on Machine Learning*. PMLR, 2020: 9929-9939. Available: [https://proceedings.mlr.press/v119/wang20k.html](https://proceedings.mlr.press/v119/wang20k.html).

[3] WANG F, XIANG X, CHENG J, et al. NormFace: L2 Hypersphere Embedding for Face Verification[C]// *Proceedings of the 25th ACM International Conference on Multimedia*. New York: ACM, 2017: 1041-1049. DOI: [10.1145/3123266.3123359](https://doi.org/10.1145/3123266.3123359).

[4] LIU W, WEN Y, YU Z, et al. SphereFace: Deep Hypersphere Embedding for Face Recognition[C]// *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017: 6738-6746. DOI: [10.1109/CVPR.2017.713](https://doi.org/10.1109/CVPR.2017.713).

[5] DENG J, GUO J, XUE N, et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition[C]// *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2019: 4690-4699. DOI: [10.1109/CVPR.2019.00482](https://doi.org/10.1109/CVPR.2019.00482).
