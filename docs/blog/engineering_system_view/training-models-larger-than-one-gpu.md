---
title: "当模型大到一张 GPU 装不下时，我们是怎么把它训练出来的？"
date: 2026-03-10T12:00:00-07:00
summary: "从参数、梯度、优化器状态与激活的内存构成出发，解释大模型训练为什么会一步步走向 ZeRO/FSDP、张量并行、流水线并行与通信优化的组合。"
tags: ["大模型训练", "Scale Up", "多卡", "工程实践"]
---

# 当模型大到一张 GPU 装不下时，我们是怎么把它训练出来的？

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/engineering_system_view/training-models-larger-than-one-gpu" en-path="/blog/engineering_system_view/training-models-larger-than-one-gpu-en" />

当模型仍能完整落在单卡上时，训练主要表现为优化问题；一旦跨过单卡边界，它首先变成系统问题。此时需要重新定义的，不再只是学习率、batch size 或收敛稳定性，而是训练状态如何放置、单层计算如何拆分、同步是否能够被计算隐藏。

因此，“大模型是怎么训练出来的”最准确的回答，不应从某个框架名词开始，而应从瓶颈迁移路径开始。工程上真正发生的是：瓶颈先从单卡显存转向模型状态放置，随后转向算子拆分、流水调度和跨卡通信。ZeRO/FSDP、张量并行、流水线并行、混合精度、checkpointing 与 FlashAttention，并不是彼此可替换的技巧列表，而是在不同约束下依次接管系统不同层面的机制 [1-6]。

> 核心结论：当模型超过单卡能力后，训练系统首先要解决的不是“多加几张卡”，而是如何重写单卡假设。工程上需要分别回答三件事：模型状态如何分布式存放，单层计算如何跨设备执行，通信开销如何被控制或隐藏。所谓 scale up，本质上是在不断把瓶颈从模型层推向系统层。

在“工程和系统视角”系列中，本文从训练系统的演化路径出发，回答当模型大到一张 GPU 装不下时，工程上到底发生了什么；若想回到专题入口，可从 [Blog](/blog/) 继续按系列浏览。

![大模型训练中的瓶颈迁移与并行轴示意图](./training-system-parallelism-map.svg)

*图 1. 当单卡假设失效后，训练系统会沿“状态存储、算子执行、网络深度、激活内存、暴露通信”这条链路不断迁移瓶颈。不同并行与优化技术并不是同义替换，而是在不同层上接管系统。*

## 1. 单卡边界究竟卡在哪里？

单卡显存里要放的，并不只有参数本身。对训练而言，更准确的记法是：

$$
M_{\text{device}}
\approx
M_{\text{param}}
+
M_{\text{grad}}
+
M_{\text{optim}}
+
M_{\text{act}}.
$$

这里四项分别对应参数、梯度、优化器状态和激活。真正让系统先撞墙的，往往不是参数这一项，而是后面三项的叠加。

- 如果用 Adam 一类优化器，`m`、`v` 等状态会显著放大非激活内存；
- 如果还保留 master weights，优化器相关内存通常会大于参数本体；
- 如果序列长度继续拉长，激活与 attention 工作区又会快速上升。

以常见的训练 recipe 为例，如果参数和梯度使用 `BF16`，而 Adam 的一阶、二阶矩以及可能存在的 master weights 使用 `FP32`，那么**仅参数相关的常驻状态**就常常已经来到每参数十几 bytes 的量级；实现不同，数字会在一个范围内波动，但数量级不会差太多。也就是说，在没有任何分片之前，百亿到千亿参数模型的“静态状态成本”本身就足以压垮单卡。

因此，大模型训练遇到的第一类硬约束，不是表达能力或优化理论，而是训练状态是否物理可容纳。这也是为什么很多作业不是先在 loss 上失败，而是先在 `CUDA out of memory` 上失败。

## 2. 更大的 GPU 和数据并行，为什么都不够？

升级到更大的 GPU 通常是第一步，但它本质上只是在推迟问题。单卡内存边界是刚性的，模型、上下文长度和训练状态只要继续增长，总会再次撞到同一堵墙。

接下来很多人自然会转向多卡，最常见的是数据并行。数据并行的作用非常重要，但要说清它到底解决什么、没有解决什么。

它真正解决的是：

- 把一个 step 的样本分到多张卡上，提高吞吐；
- 在保持同一模型副本的前提下并行计算，加快 wall-clock 训练速度。

它没有解决的是：

- 每张卡上的完整模型副本依然存在；
- 参数、梯度、优化器状态的静态占用并没有按卡数等比例下降。

更严格地说，在固定 global batch 的前提下，数据并行可以因为 local batch 变小而减轻每卡激活压力；但**模型状态内存并没有因此被打散**。所以数据并行首先是吞吐扩展手段，不是突破单卡模型容量的根本办法。

这里必须区分两类完全不同的问题：

- `scale out compute`：同一个模型更快地训练；
- `break single-device capacity`：同一个模型终于能被放下。

前者主要靠数据并行，后者则需要下一类技术。

## 3. 第一次真正跨过单卡边界：状态分片

一旦你发现单卡放不下的根源是“每张卡都在重复保存几乎相同的状态”，问题就会从多卡训练转向状态分片。ZeRO 的核心思想正是去掉这种冗余：Stage 1 分片优化器状态，Stage 2 再分片梯度，Stage 3 进一步分片参数本身 [1]。从系统设计上看，PyTorch 的 FSDP full sharding 与这一思路属于同一谱系。

这一步的本质不是“多卡一起训”，而是“多卡一起存”。如果有 $N$ 张卡参与同一个 sharding group，那么被分片的那部分模型状态，其每卡常驻内存会从“每卡一整份”变成近似“每卡一份的 $1/N$”。于是模型规模才真正开始突破单卡显存上限。

但收益不是白来的，因为你牺牲的是“本地立即可用”：

- 参数可能需要在层执行前 `all-gather`；
- 梯度在反向后需要 `reduce-scatter` 或等价同步；
- 优化器更新后，参数分片还要重新分发或重建视图。

所以状态分片的本质代价，是把显存约束改写成通信约束。单卡容量问题被解除后，系统性能开始更强地依赖网络带宽、同步时序与通信重叠能力。也正因如此，ZeRO/FSDP 首先是容量解锁器，而不是免费的加速器。

## 4. 当单层都太大时：张量并行

状态分片解决的是“模型状态如何存”，但不自动解决“一个算子如何算”。模型继续变大后，你会遇到下一堵墙：不是整网放不下，而是某个 `Linear`、某个 `QKV` projection、某个 FFN 大矩阵本身就已经不适合由单张卡独立完成。

这时需要的是张量并行，也就是把算子本身按张量维度拆开，让多张卡共同执行一次层内计算。Megatron-LM 让这种 row-parallel / column-parallel 的分解在 Transformer 中变得标准化 [2]。它解决的是单层算子的容量和算力边界，而不是样本维度上的吞吐扩展。

这类并行有一个非常重要的工程特征：通信粒度更细，频率更高。因为一次层计算被拆给了多张卡，每层前后都可能出现 `all-reduce`、`all-gather` 或类似集体通信。于是张量并行通常更适合放在高带宽、低时延的互连域里，例如同一台机器内的 NVLink/NVSwitch，而不适合跨慢网络做过细拆分。

所以，张量并行不是“多份模型一起训练”，而是“同一个模型的一层，被分布式执行”。这也是为什么它在系统行为上比数据并行更像是分布式线性代数，而不是简单的样本切分。

## 5. 当模型太深时：流水线并行

如果模型不仅宽，而且很深，那么即使单层可以被执行，整条网络依然可能无法被同一组设备完整承载。此时问题不再只是算子尺寸，而是网络深度和激活生命周期。流水线并行的回答是：既然单个设备组装不下整条网络，那就按层把网络切成多个 stage，再用 micro-batch 把这些 stage 组织成流水线 [3]。

它解决了两个问题：

- 不同设备只负责部分层，降低单设备承载的深度；
- 通过多 micro-batch 重叠执行，让前后阶段同时工作。

但流水线并行并不只是“把层分一分”这么简单。真正的工程难点主要在三个地方：

- bubble：流水线填充和排空阶段存在天然空转；
- load balance：不同 stage 的计算量不均，会直接拖慢全链路；
- activation scheduling：前向与反向的穿插方式决定了内存占用和延迟，1F1B、PipeDream 一类调度正是围绕这个问题展开 [3][9]。

因此，流水线并行首先是调度问题。模型虽然被撑起来了，但系统此时真正优化的已是 stage 划分、micro-batch 数量与 1F1B 一类执行策略，而不再是简单的“复制模型后并行计算”。

## 6. 真正把系统推远的，往往是几类正交优化

当状态分片、张量并行和流水线并行都到位之后，工程优化会进入一个更细的层面：不是再找一个新的“大招”，而是持续压缩存储、带宽和访存成本。这里最关键的几类方法通常是正交叠加的。

### 混合精度

混合精度的本质是重新分配数值表示预算。`FP16` 能显著节省显存和带宽，但动态范围更窄，常需要 loss scaling；`BF16` 保留了与 `FP32` 相同量级的指数范围，因此在大模型训练里通常更稳，也更常成为默认选择；`FP8` 进一步压缩成本，但仍强依赖硬件、kernel 与训练 recipe [4]。

### Activation Checkpointing 与选择性重算

Checkpointing 的核心不是“省一点显存”，而是显式放弃部分中间激活的常驻，把它们改成在反向时重算。它是典型的“用额外计算换更低内存”的做法，也是训练系统中最常见、最有效的激活内存管理手段之一 [5]。在更成熟的大模型训练栈里，工程实践往往不会对整层做一刀切的全量重算，而是结合选择性重算，只对最值得重算的中间量回放，从而避免把显存节约全部变成额外 FLOPs [8]。

### Sequence Parallelism

当瓶颈主要来自激活而不是参数时，系统还会继续引入 sequence parallelism 一类额外轴。它的出发点是：既然张量并行已经把部分算子拆到了多张卡上，那么某些非张量并行层上重复保留完整序列激活也是一种浪费。把这部分状态沿序列维进一步切开，可以继续降低激活内存，并减少“为了省内存而不得不做的大规模重算” [8]。这类方法尤其常见于长上下文训练，因为那里先撞墙的往往不是参数量，而是序列相关激活和注意力工作区。

### FlashAttention 与 IO-aware kernel

FlashAttention 这类优化并没有把 dense exact attention 的算术复杂度从二次降成线性；更准确地说，它通过 tiling 和 IO-aware 设计，避免在高带宽显存里显式物化完整的 $N \times N$ attention 矩阵，从而降低 HBM 读写和工作区压力 [6]。对长上下文训练而言，这种“减少显式中间张量和访存成本”的优化非常关键。

这些方法之所以重要，是因为它们分别作用在不同层面：

- 混合精度压缩表示成本；
- checkpointing 压缩激活驻留时间；
- IO-aware kernel 压缩访存和临时工作区。

单独看，它们都不像“决定性架构变化”；叠加起来，却往往决定一个训练作业到底能不能落地。

## 7. 训练做到后面，真正的瓶颈常常是暴露出来的通信

模型一旦进入多维并行，step time 就不再主要由算子 FLOPS 决定，而更多由“有多少通信没有被隐藏掉”决定。一个更贴近系统现实的抽象是：

$$
T_{\text{step}}
\approx
T_{\text{compute}}
+
T_{\text{comm, exposed}}
+
T_{\text{bubble}}
+
T_{\text{misc}}.
$$

这里最重要的不是总通信量，而是**暴露通信量**。如果通信能与计算重叠，它对 wall-clock 的伤害会小很多；如果不能，它就会直接变成 step time。

不同并行方式暴露出来的通信也不同：

- 数据并行主要是梯度同步；
- 状态分片主要是参数 `all-gather` 与梯度 `reduce-scatter`；
- 张量并行主要是层内集体通信；
- 流水线并行主要是 stage 之间的点对点激活传输。

这也是为什么真正的大模型训练系统不仅选择并行方式，还会反复调优通信重叠、拓扑映射、bucket 大小、梯度累积和 stage 放置。GPU 数量本身并不等于系统能力；只有当通信被合理组织后，设备规模才会转化为有效吞吐。

## 8. 真实的大模型训练，更像一套混合系统而不是单一方案

工程实践里，很少只靠一种并行方式把问题做完。更常见的现实是：

`数据并行 × ZeRO/FSDP × 张量并行 × 流水线并行 + BF16/FP16 + Checkpointing + FlashAttention`

其中每一项都在回答不同问题：

- 数据并行负责吞吐扩展；
- ZeRO/FSDP 负责消除模型状态冗余；
- 张量并行负责拆开单层大算子；
- 流水线并行负责拆开网络深度；
- 混合精度、checkpointing 和高效 kernel 继续压缩显存与带宽压力。

更工程化地看，大规模 dense Transformer 的进程拓扑通常可以写成

$$
\text{world size}
=
DP \times TP \times PP,
$$

其中 `DP` 是数据并行，`TP` 是张量并行，`PP` 是流水线并行。如果再引入 sequence/context parallelism 或 MoE 中的 expert parallelism，就会变成更高维的分解 [7][8]。这时真正重要的不是名词数量，而是**这些轴如何映射到物理拓扑**：

- `TP` 往往最依赖低时延、高带宽互连，因此通常尽量收在单机或单个高速交换域内；
- `PP` 关心相邻 stage 的点对点带宽和负载均衡；
- `DP` 或 `FSDP/ZeRO` 常作为更外层的扩展轴，去复用相对更弱的跨机网络。

因此，所谓“大模型训练栈”并不是功能清单，而是一张瓶颈映射表。模型一旦变大，系统设计几乎必然走向混合并行，因为没有单一技术能同时解决容量、吞吐、深度和通信四类约束。

把这件事写成更形式化的训练系统目标，可以近似看成：

$$
\min_{\text{parallel plan}} \ T_{\text{step}}
\quad
\text{s.t.}
\quad
M_{\text{device}} \le M_{\text{budget}},
\quad
T_{\text{comm, exposed}} \text{ 尽可能小},
\quad
T_{\text{bubble}} \text{ 可控}.
$$

这里的 `parallel plan` 包括并行轴分解、拓扑映射、重算策略与 kernel 选择。换言之，真实工程题目不是“要不要某种并行”，而是在既定硬件拓扑上求一个满足内存约束、暴露通信最小且吞吐尽可能高的系统分解。

为了把这些方法之间的边界讲得更清楚，可以把它们压缩成表 1。

| 技术 | 主要解决什么 | 不直接解决什么 | 主要代价 |
| --- | --- | --- | --- |
| 数据并行 | 吞吐扩展、样本维并行 | 单卡模型状态容量 | 梯度同步 |
| ZeRO / FSDP | 参数、梯度、优化器状态冗余 | 单层大算子本身太大 | 更频繁的参数/梯度通信 |
| 张量并行 | 单层算子容量与算力边界 | 网络深度过大 | 层内集体通信 |
| 流水线并行 | 网络深度与分阶段承载 | 层内单算子过大 | bubble、stage imbalance、调度复杂度 |
| Checkpointing / Selective Recomputation | 激活常驻内存 | 模型状态冗余 | 额外重算 |
| Sequence Parallelism | 序列相关激活冗余 | 参数状态内存 | 额外并行轴和相关通信 |
| FlashAttention / IO-aware kernel | attention 访存和工作区压力 | 参数分片或网络深度问题 | kernel 依赖与实现复杂度 |
| 通信重叠 / 拓扑映射 | 暴露通信量 | 算法本身的容量边界 | 调参复杂度、拓扑耦合 |

## 9. Scale up 不是无限扩张，而是把问题一路推向系统层

直觉上，好像只要继续加机器，就能无限 scale up。但现实里有几个边界不会消失：

- 时延有物理下界，不能靠“再多加一点卡”消除；
- 同步效率会随着规模增长持续下降；
- 故障概率、重启成本和 straggler 风险会越来越显著；
- 预算、功耗和网络基础设施都是硬上限。

所以更接近现实的说法不是“scale up 可以无限长大”，而是：**我们在不断把瓶颈从模型层推向系统层。** 一开始是显存容量，随后是状态放置，再往后是算子拆分、流水调度和通信隐藏。模型是否还能继续做大，最终取决于整个训练系统能否承受这些复杂度。

## 10. 结语

当模型大到一张 GPU 装不下时，训练问题就不再只是“这个网络结构好不好”，而是在反复回答三个更本质的工程问题：

- 模型状态该如何分布式存放？
- 单层与整网计算该如何拆开执行？
- 通信代价还能否被系统承受和隐藏？

这也是为什么大模型训练几乎不存在银弹。真正把模型训练出来的，不是某个单独框架名词，而是一整套围绕内存、计算与通信持续重写系统边界的工程过程。

## 参考文献

[1] RAJBHANDARI S, RASLEY J, RUWASE O, et al. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models[J]. *arXiv preprint arXiv:1910.02054*, 2020. DOI: [10.48550/arXiv.1910.02054](https://doi.org/10.48550/arXiv.1910.02054).

[2] SHOEYBI M, PATWARY M, PURI R, et al. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism[J]. *arXiv preprint arXiv:1909.08053*, 2019. DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053).

[3] HUANG Y, CHENG Y, CHEN D, et al. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism[J]. *arXiv preprint arXiv:1811.06965*, 2018. DOI: [10.48550/arXiv.1811.06965](https://doi.org/10.48550/arXiv.1811.06965).

[4] MICIKEVICIUS P, NARANG S, ALBEN J, et al. Mixed Precision Training[J]. *arXiv preprint arXiv:1710.03740*, 2017. DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740).

[5] CHEN T, XU B, ZHANG C, et al. Training Deep Nets with Sublinear Memory Cost[J]. *arXiv preprint arXiv:1604.06174*, 2016. DOI: [10.48550/arXiv.1604.06174](https://doi.org/10.48550/arXiv.1604.06174).

[6] DAO T, FU D Y, ERMON S, et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness[J]. *arXiv preprint arXiv:2205.14135*, 2022. DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135).

[7] NARAYANAN D, SHOEYBI M, CASPER J, et al. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM[C]// *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*. New York: ACM, 2021. DOI: [10.1145/3458817.3476209](https://doi.org/10.1145/3458817.3476209).

[8] KORTHIKANTI V A, CASPER J, LYM S, et al. Reducing Activation Recomputation in Large Transformer Models[J]. *arXiv preprint arXiv:2205.05198*, 2022. DOI: [10.48550/arXiv.2205.05198](https://doi.org/10.48550/arXiv.2205.05198).

[9] NARAYANAN D, HARLAP A, PHAN N, et al. PipeDream: Generalized Pipeline Parallelism for DNN Training[C]// *Proceedings of the 27th ACM Symposium on Operating Systems Principles*. New York: ACM, 2019. DOI: [10.1145/3341301.3359646](https://doi.org/10.1145/3341301.3359646).
