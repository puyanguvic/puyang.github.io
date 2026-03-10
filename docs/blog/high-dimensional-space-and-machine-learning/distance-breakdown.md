---
title: "高维空间中的距离集中与度量失效"
date: 2026-03-09T10:00:00-08:00
summary: "从薄壳现象、集中不等式与最近邻理论出发，解释高维空间里欧氏距离为何逐渐失去判别力。"
tags: ["machine learning", "high-dimensional geometry", "representation learning"]
---

# 高维空间中的距离集中与度量失效

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/high-dimensional-space-and-machine-learning/distance-breakdown" en-path="/blog/high-dimensional-space-and-machine-learning/distance-breakdown-en" />

高维几何给机器学习带来的第一个冲击，不是“空间更大，所以结构更丰富”，而恰恰相反：在相当宽的一类随机模型下，样本之间的欧氏距离会被压缩到越来越窄的区间中。于是，距离并不会消失，但它作为排序与判别信号的分辨率会系统性下降 [1-5]。

更严格地说，问题不在于某两个样本的距离能否被计算出来，而在于整个样本集里的距离层次是否仍然足够展开。只要最近邻与最远邻的相对差距持续收缩，任何依赖“远近排序”的算法都会变得更脆弱。这正是高维文献中所谓 distance concentration 或 relative contrast collapse 的核心含义 [1][2]。

> 核心结论：在各向同性高维模型下，样本先集中到半径约为 $\sqrt{d}$ 的薄壳上，随后样本对距离再集中到 $\sqrt{2d}$ 附近；在样本数没有指数级增长的情况下，最近邻与最远邻的相对差距会随维度上升而收缩，欧氏距离因而逐步失去判别力 [1-5]。

在“高维空间与机器学习”系列中，本文先回答原始欧氏距离为什么会失去分辨率；下一篇 [高维向量近似正交的几何机制](/blog/high-dimensional-space-and-machine-learning/orthogonality) 会把焦点从长度转向方向。

## 1. 距离“失效”到底指什么？

设查询点为 $q$，数据库样本为 $x_1,\dots,x_n$，记

$$
D_i = \|q - x_i\|, \qquad
D_{\min} = \min_i D_i, \qquad
D_{\max} = \max_i D_i.
$$

高维近邻搜索理论关心的不是单个距离值，而是极值之间的相对分离度，例如

$$
\mathrm{RC}(q) = \frac{D_{\max} - D_{\min}}{D_{\min}}.
$$

当 $\mathrm{RC}(q)$ 很大时，“最近”和“最远”之间存在清楚的几何层级；当它趋近于零时，距离虽然仍然数值可算，却几乎不再提供稳定的相对排序 [1][2]。因此，所谓距离失效并不意味着

$$
D_1 = D_2 = \cdots = D_n,
$$

而是意味着这些距离被压缩在同一典型尺度附近，以至于噪声、尺度扰动或无关维度都足以改变排序结果。

这一定义很重要，因为它把“欧氏距离还能不能算”与“欧氏距离还能不能可靠地区分样本”区分开了。机器学习真正依赖的是后者。

## 2. 第一步：样本范数先集中到同一薄壳

先看最简单的高斯模型。设

$$
x = (x_1,\dots,x_d), \qquad x_i \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,1).
$$

则

$$
\|x\|^2 = \sum_{i=1}^d x_i^2 \sim \chi_d^2,
$$

从而

$$
\mathbb{E}\|x\|^2 = d, \qquad
\mathrm{Var}(\|x\|^2) = 2d.
$$

这说明典型半径随维度增长为 $\sqrt{d}$。真正关键的是相对波动：对高斯向量，范数作为 `1`-Lipschitz 函数满足标准集中不等式 [3-5]

$$
\mathbb{P}\!\left(\left|\|x\| - \mathbb{E}\|x\|\right| \ge t\right)
\le 2 e^{-t^2/2}.
$$

于是有

$$
\|x\| = \sqrt{d} + O_P(1).
$$

也就是说，绝对波动停留在常数量级，而相对波动按 $1/\sqrt{d}$ 衰减。维度越高，样本越不像“均匀填满整个球体”，而越像“集中贴在某个窄球壳上”。这就是薄壳现象，它是后续一切距离退化结论的起点 [3][4]。图 1 把这一点直观化：关键不是球半径变大，而是相对厚度快速变薄。

![高维薄壳现象示意图](./distance-breakdown-thin-shell.svg)

*图 1. 在低维时，概率质量分布在较宽的半径区间中；在高维时，大部分样本被压缩到半径约为 $\sqrt{d}$ 的薄壳附近。*

图 1 对应的真正结论是，径向自由度在高维中很快变得稀缺；一旦这一点成立，后续距离分布就不可能再像低维那样充分展开。

## 3. 第二步：样本对距离也集中到单一尺度

如果 $x,y \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,I_d)$，则

$$
x-y \sim \mathcal{N}(0,2I_d),
\qquad
\|x-y\|^2 \sim 2\chi_d^2.
$$

因此

$$
\mathbb{E}\|x-y\|^2 = 2d, \qquad
\mathrm{Var}(\|x-y\|^2) = 8d.
$$

对平方距离而言，相对波动量级为

$$
\frac{\sqrt{\mathrm{Var}(\|x-y\|^2)}}{\mathbb{E}\|x-y\|^2}
= \Theta\!\left(\frac{1}{\sqrt{d}}\right).
$$

经过 delta method 或等价的集中估计，可以得到

$$
\|x-y\| = \sqrt{2d} + O_P(1).
$$

这一结论的含义非常直接：高维并没有让样本对的“可能距离”更加分散，反而让绝大多数距离都拥挤在 $\sqrt{2d}$ 附近的窄带内 [3-5]。薄壳现象约束了点到原点的半径，距离集中则进一步约束了点与点之间的相对位置。图 2 可以把这种“宽分布变窄分布”的变化直接翻译成检索语义。

![高维距离集中示意图](./distance-breakdown-concentration.svg)

*图 2. 低维时，样本对距离分布较宽；高维时，距离分布收缩到更窄区间，最近邻与最远邻之间的可分性随之下降。*

图 2 中最值得注意的并不是均值位置，而是分布宽度的收缩。一旦宽度显著缩小，极值排序就只能依赖很小的数值差，算法对噪声和扰动也就更敏感。

## 4. 第三步：极值之间的相对差距开始塌缩

一旦单个距离已经集中到某个典型尺度，最近邻与最远邻的差别就只能来自极值波动，而极值波动通常只比单样本波动多出一个 $\sqrt{\log n}$ 量级。启发式地，可以写成

$$
D_{\max} - D_{\min} = O_P(\sqrt{\log n}),
\qquad
D_{\min} = \Theta_P(\sqrt{d}),
$$

从而

$$
\frac{D_{\max} - D_{\min}}{D_{\min}}
= O_P\!\left(\sqrt{\frac{\log n}{d}}\right).
$$

因此，只要样本规模 $n$ 没有随维度 $d$ 以 $e^{cd}$ 的速度指数增长，相对对比度就会持续下降。这正是 Beyer 等人与 Aggarwal 等人讨论“最近邻逐渐失去意义”时的理论背景 [1][2]。

这一点值得单独强调。高维下的失败不是由“最近邻太远”造成的，而是由“所有邻居都差不多远”造成的。前者还能靠绝对阈值修补，后者则直接伤害了排序本身。

## 5. 这对机器学习意味着什么？

一旦距离分辨率下降，多个经典算法都会受到直接影响。

- 对 `kNN` 而言，邻居排序会更敏感。无关维度、尺度偏差或轻微噪声都可能改变最近邻集合。
- 对聚类而言，簇内与簇间距离的分离度下降，基于欧氏球形假设的方法更容易拟合采样波动而不是真实结构。
- 对向量检索而言，若直接在未经学习的原始特征上做欧氏近邻搜索，性能常常不稳；有效系统往往需要先学习表示，再选择更匹配该表示几何的度量。

这里最容易被误读的一点是：距离退化不等于几何方法失效。更准确的说法是，在缺乏结构假设时，原始欧氏距离难以直接充当语义相似性的代理。如果数据本身位于低维流形上，或者已经经过标准化、降维、度量学习与表示学习，距离仍然可以重新变得有用。

## 6. 这种退化是否只属于欧氏距离？

一个自然反应是：既然欧氏距离会退化，是否只要换一种距离就能解决问题？答案通常是否定的。Aggarwal 等人的分析表明，不同的 $L_p$ 距离在高维中的退化速度确实可能不同，较小的 $p$ 有时会保留稍好的对比度，但这并不改变更大的事实：一旦数据缺乏明显的低维结构、而坐标又近似独立地向各维扩散，许多基于范数的距离都会围绕其典型值集中 [2]。

因此，度量选择当然重要，但它更像是在改变退化的快慢和有限样本行为，而不是凭空取消维数灾难。也正因为如此，实践中真正有效的改进通常不是“从欧氏距离机械切换到另一种原始距离”，而是：

- 先对特征做标准化、白化或降维，削弱无关维度；
- 再用 cosine、Mahalanobis 或学习到的度量读取表示；
- 更进一步，直接学习一个让几何与任务目标一致的表示空间。

这里的分界线，不在于“选哪种公式当距离”，而在于“当前度量是否已经适配数据真实结构”。如果度量仍建立在未经学习的原始坐标系上，那么换范数通常只能缓解问题，难以根治问题。

## 7. 为什么下一步必须转向角度与表示学习？

一旦半径差异被薄壳现象压缩，真正剩下的自由度主要就来自方向结构。也正因为如此，现代高维表示学习通常不会把“离原点多远”当作主要信息来源，而会更关注归一化后的角度关系、局部子空间和由训练目标塑造出来的低相关方向系统。

这并不意味着几何方法就此失效；它只是迫使我们承认，原始坐标系里的几何往往不是任务需要的几何。表示学习的意义正在于此：学习一个新的空间，使“近”和“远”重新获得稳定、可解释的统计含义。

## 8. 结语

高维空间最反直觉的地方，不是它让点彼此更远，而是它让大多数距离同时变得差不多。薄壳现象先压缩范数，距离集中再压缩样本对尺度，最终最近邻与最远邻之间的相对差距也被拖向零。

结论并不是“欧氏距离从此不能用”，而是：**未经学习的原始高维欧氏空间，通常不足以直接承载语义相似性。** 一旦半径与距离同时开始退化，问题就会自然转向方向关系。

继续阅读：[高维向量近似正交的几何机制](/blog/high-dimensional-space-and-machine-learning/orthogonality)。

## 参考文献

[1] BEYER K S, GOLDSTEIN J, RAMAKRISHNAN R, et al. When Is "Nearest Neighbor" Meaningful?[C]//BEERI C, BUNEMAN P, eds. *Database Theory - ICDT'99*. Berlin, Heidelberg: Springer, 1999: 217-235. DOI: [10.1007/3-540-49257-7_15](https://doi.org/10.1007/3-540-49257-7_15).

[2] AGGARWAL C C, HINNEBURG A, KEIM D A. On the Surprising Behavior of Distance Metrics in High Dimensional Space[C]//VAN DEN BUSSCHE J, VIANU V, eds. *Database Theory - ICDT 2001*. Berlin, Heidelberg: Springer, 2001: 420-434. DOI: [10.1007/3-540-44503-X_27](https://doi.org/10.1007/3-540-44503-X_27).

[3] LEDOUX M. *The Concentration of Measure Phenomenon*[M]. Providence, RI: American Mathematical Society, 2001. DOI: [10.1090/surv/089](https://doi.org/10.1090/surv/089).

[4] VERSHYNIN R. *High-Dimensional Probability: An Introduction with Applications in Data Science*[M]. Cambridge: Cambridge University Press, 2018. DOI: [10.1017/9781108231596](https://doi.org/10.1017/9781108231596).

[5] BOUCHERON S, LUGOSI G, MASSART P. *Concentration Inequalities: A Nonasymptotic Theory of Independence*[M]. Oxford: Oxford University Press, 2013. DOI: [10.1093/acprof:oso/9780199535255.001.0001](https://doi.org/10.1093/acprof:oso/9780199535255.001.0001).
