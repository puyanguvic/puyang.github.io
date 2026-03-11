---
title: "当上下文不断膨胀时，我们到底在优化什么？"
date: 2026-03-11T10:00:00-08:00
summary: "将 Agent 的 context 重新理解为运行时工作集。本文主张，context engineering 的核心不是扩充窗口，而是设计一套能在效果、成本、时延、保真度、时效性与可控性之间稳定权衡的信息选择与状态管理机制。"
tags: ["AI Agent", "LLM Systems", "Context Engineering", "RAG", "Memory"]
---

# 当上下文不断膨胀时，我们到底在优化什么？

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/engineering_system_view/what-are-we-optimizing-in-agent-context" en-path="/blog/engineering_system_view/what-are-we-optimizing-in-agent-context-en" />

最近，像 OpenClaw 这样的开源 AI Agent 框架持续走热。很多人第一次直观地感受到：当一个模型能够调用工具、浏览网页、运行代码，并沿着任务链条逐步完成任务时，系统的复杂度不再只来自模型本身，而来自模型在每一步究竟“看到了什么”。

表面上看，Agent 系统是在增加 tools、skills 和 workflow。但在系统层面，真正困难的部分并不是工具数量，而是上下文如何被构造、筛选和呈现给模型。工具越多、状态越多、执行链越长，这个问题就越尖锐。

当大模型只做单轮问答时，prompt 看起来像一段输入文本；到了 AI Agent 系统里，它更接近模型在当前 step 上可见的**运行时工作集**。用户目标、历史对话、检索文档、工具返回、执行轨迹、反思记录以及任务状态，都要在一次调用前被组织成一个可供模型消费的上下文。

问题也因此变了。Agent 的上限不再只由模型参数、推理技巧或工具数量决定，同样取决于系统在这一刻究竟让模型看见什么、不让它看见什么，以及用什么形式让它看见。一旦这层构造失控，即使 context window 更大，系统也可能在无关信息里失焦、在关键位置上失真，或者把本该结构化保存的状态重新压回自然语言通道 [1-3]。

> 核心观点：在 Agent 里，context engineering 不是扩充 prompt 的技巧，而是运行时信息管理。真正要优化的不是“能塞多少 token”，而是“在给定预算内，什么信息应该进入当前工作集，什么应该被压缩，什么应该留在外部系统按需取回”。

![Agent context as a runtime working set](./agent-context-working-set.svg)

*图 1. 在 Agent 系统中，prompt 不是输入文本的简单累加，而是控制器从多路状态中选择、压缩并渲染出的当前工作集。*

## 1. Agent 的 context，本质上是运行时工作集

如果把 Agent 在时刻 $t$ 的一次模型调用写成更贴近系统现实的抽象，可以近似记为

$$
C_t
=
\mathrm{render}(u_t, H_t, R_t, O_t, M_t, s_t; B_t),
$$

其中 $u_t$ 是当前用户请求或当前子任务目标，$H_t$ 是历史对话，$R_t$ 是检索得到的外部文档，$O_t$ 是工具调用或环境交互产生的 observation，$M_t$ 是从长期记忆中取回的经验或摘要，$s_t$ 是结构化任务状态，$B_t$ 则代表当前 step 可用的 token、时延和成本预算。

这个写法的重要性不在公式本身，而在它迫使我们承认一件事：模型接收到的从来不是“状态本身”，而只是状态在当前调用上的一个视图。换句话说，prompt 并不等于系统拥有的信息；它只是控制器在预算约束下，对异构状态进行筛选、排序、压缩和模板化后生成的一份渲染结果。传统软件把状态分散在内存、缓存、日志和数据库里，LLM Agent 则常常把这些东西统一序列化成 token，经由同一条输入通道送进模型。

一旦从这个角度看，context window 的角色就很清楚了。它更像一块昂贵而有限的工作内存，而不是“尽量多装内容”的容器。真正困难的环节并不在 `LLM inference`，而在推理之前的 `context construction`：哪些历史需要保留，哪些外部信息必须检索，哪些工具返回应该原样保留，哪些状态更适合留在外部系统，只在需要时按任务视图被渲染进 prompt [4-8]。因此，工程上真正值得优化的对象，不是 prompt 的文案，而是构造 prompt 的那套 policy。

如果把这一点再说得更严格些，Agent 在每个 step 上面对的并不是“信息越多越好”，而是“什么信息最能改善下一步决策”。系统真正想求的，更接近下面这个问题：

$$
\max_{C_t} \Pr(a_t^\star \mid C_t)
\quad
\text{s.t.}
\quad
|C_t| \le B_t,
\quad
\mathrm{latency}(C_t) \le \tau_t
$$

这里 $a_t^\star$ 可以理解为当前 step 的正确回答、正确动作或正确工具调用。这个写法揭示了 context engineering 的核心：它优化的不是长度本身，而是工作集对下一步决策质量的边际贡献。

## 2. 真正要优化的，不是窗口大小，而是工作集质量

只要把 context 视为工作集，长上下文为什么不是根本解法就很容易解释。更大的 window 当然有价值，它推迟了容量瓶颈，使系统有机会在一次调用中看到更长历史、更大文档和更多中间结果；但它并不会自动完成信息管理。标准 Transformer 的计算与显存开销会随序列长度快速上升 [1]，而且“能容纳更多 token”从来不等于“能稳定利用更多 token”。`Lost in the Middle` 的结果已经相当明确：即便模型名义上支持长上下文，关键信息处在输入中部时，模型对它的利用仍可能显著退化 [2]。

这意味着问题并不只是容量不够，而是有效信息密度、位置偏置和注意力分配都在约束最终效果。很多时候，盲目扩窗只是在给系统更多机会塞进噪声。LongLLMLingua、LLMLingua 和 Selective Context 的经验恰恰说明，在不少长上下文任务里，对 prompt 做有针对性的压缩或裁剪，不仅能够降低成本，反而可能提升效果 [3][9][10]。这不是悖论，而是工作集管理的正常结果：更多信息不自动变成更多有效信息。

因此，context optimization 从一开始就是一个多目标约束问题。工程上更贴近现实的表述是：

$$
\max_{C_t \in \mathcal{F}_t}
\Big(
Q_t(C_t),
F_t(C_t),
R_t(C_t),
S_t(C_t),
-L_t(C_t),
-K_t(C_t)
\Big),
$$

其中 $Q_t$ 表示当前 step 的任务质量，$F_t$ 表示 freshness，$R_t$ 表示 fidelity，$S_t$ 表示稳定性与可控性，$L_t$ 和 $K_t$ 分别表示时延与成本。这里没有哪个维度是可有可无的。只追求质量而不管成本，策略无法部署；只追求压缩率而不管 fidelity，系统会在关键细节上持续失真；只追求“记住更多”，又会把过时状态带回当前决策。

这也是为什么“当前 step 真的需要什么”必须始终先于“系统总共知道什么”。知识密集型问答更在意 freshness、证据相关性与来源可追溯性；代码 Agent 和数据分析 Agent 往往更在意工具输出的 fidelity、执行状态的一致性以及错误信息的原样保留；长周期陪伴式 Agent 则更依赖 memory policy、信息过期和跨 step 的稳定性。不同工作负载给这些目标赋予的权重不同，因此不存在脱离任务分布与服务预算的“全局最优 context”。只有面向某一类任务、某一类预算和某一级别服务目标的更优 policy。

从这个角度看，context engineering 的专业性恰恰体现在它拒绝把所有状态都翻译成自然语言。很多系统今天仍然默认让 prompt 同时承载说明文、知识文本、执行记录和控制状态，这种做法的优点是接口统一，缺点则是把语义内容和系统状态都挤进同一条昂贵通道里。只要某些状态本可以结构化表达，却仍被冗长地改写成文本，系统就在用最高的成本购买最低的可控性。

## 3. 检索、压缩、记忆与结构化状态，各自在解决不同问题

把主流方法放在一起看，最容易出现的误判，就是把它们当成几项可以彼此替换的技巧。其实它们回答的是不同层面的问题，因此更准确的理解方式应当是“分层分工”，而不是“技巧清单”。

检索回答的是知识规模与窗口长度之间的失配。RAG 的价值并不只是“补知识”，更在于把大部分知识留在外部非参数存储里，只在当前问题需要时按需取回 [4]。它真正完成的是解耦：知识库可以继续增长，而 prompt 不必同步膨胀。但 RAG 从来没有消灭选择问题，它只是把选择前移到了 retrieval 阶段。query 怎么构造、chunk 如何切分、top-k 取多少、多个来源如何融合，这些仍然决定了最终进入工作集的是什么。

压缩回答的是信息密度与预算之间的失配。LLMLingua、Selective Context 和 LongLLMLingua 之所以实用，是因为它们承认一个事实：很多长 prompt 里真正昂贵的不是必要信息，而是冗余表达 [3][9][10]。但压缩不是无害清洗，而是明确的有损编码。背景说明、闲聊历史和低风险描述通常适合压缩；代码、合同条款、报错栈、关键配置和工具返回里的精确字段，则往往不适合被轻易改写。系统一旦用摘要替代原文，也就同时承担了细节不可逆丢失的风险 [11]。

记忆架构回答的是跨 step 持久性与当前可见性之间的矛盾。从 Transformer-XL 到 Memorizing Transformers，再到 Agent 层面的 episodic memory 和 reflexion，一条反复出现的设计线索是：不要让所有历史都常驻在当前上下文里，而是把它们分层放置，在需要时再取回 [5][6][12][13]。这一步真正困难的地方不在“有没有 memory store”，而在 memory policy：什么信息值得写入，什么信息只保留索引，什么信息应该过期，以及当多条记忆彼此冲突时，系统如何决定信任哪一条。

结构化状态与工具轨迹管理回答的，则是另一类经常被低估的问题：并不是所有执行痕迹都应以自然语言全文贴进 prompt。ReAct 和 Toolformer 都说明，Agent 的关键上下文不仅来自知识文本，也来自执行过程本身 [7][8]。但把完整工具调用记录、原始返回结果和长日志逐轮追加到上下文里，几乎总是扩展性最差的做法。更合理的系统通常会把原始 artifact 保留在外部存储中，在 prompt 里只保留当前 step 真正需要的关键字段、结构化摘要，以及必要时可重新抓取原文的引用。

因此，所谓更成熟的 context architecture，本质上不是“更会写 prompt”，而是逐步把系统拆成几层：外部知识和执行结果留在外部，检索负责召回，压缩负责控制密度，记忆负责跨 step 保留经验，结构化状态负责承载那些根本不值得翻译成自然语言的控制信息，而最终送进模型的 prompt 只是这一整套机制在当前时刻的渲染结果。

## 4. Context policy 的难点，在于它始终是一个持续失效、持续权衡的系统问题

真正困难的地方，从来不是今天有没有长窗口、检索、压缩或记忆模块，而是这些机制一旦同时进入系统，失效模式也会同时出现。工程上最常见的失败并不玄妙，无非四类：该进工作集的信息没有进来，这是 `omission`；进入 prompt 的内容看似相关但已经过时，这是 `staleness`；摘要或裁剪破坏了关键细节，这是 `distortion`；相关信息虽然在上下文里，却被大量噪声淹没，这是 `interference`。这四类失效往往互相转化。为了避免 omission 而盲目加长上下文，常常会引入 interference；为了压缩成本而过度摘要，又会引入 distortion；为了提高复用而长期保留记忆，则可能在稍后演化成 staleness。

这也是为什么 context policy 至今仍主要依赖 heuristic。最近窗口、top-k retrieval、相似度阈值、salience score、摘要长度预算，这些工程规则都有效，但大多数并没有严格回答“哪段信息对当前 step 的边际价值最高”。现实系统往往只能在经验上知道两件事：信息太少会坏，信息太多也会坏；而最难的部分恰恰是中间那条可部署的边界。

评估一个 context policy 时，最容易犯的错误，是只拿一个 benchmark 或一个 accuracy 数字来下结论。更合理的做法至少分三层。第一层是能力边界，要用 LongBench、RULER、∞Bench 一类长上下文评测确认模型和基础策略是否真的具备最低限度的长程处理能力 [14-16]。但这一步只是在检查“模型看不看得见”，而不是在检查“系统会不会用得对”；HELMET 已经表明，许多流行的 synthetic long-context 任务并不能稳定预测真实下游表现 [17]。第二层是系统指标，必须把任务质量与 `p50/p95 latency`、token 成本、检索成本、压缩率和保真度损失一起报告，因为任何脱离代价的“更优”都没有部署意义。第三层才是 Agent 场景真正关心的跨 step 行为结果：多步任务成功率、工具调用正确率、状态一致性，以及在固定预算下的单位成本任务完成率。

只有这三层同时成立，某种 context policy 才值得被称为更优。否则，我们很容易把“测试集上的 prompt 技巧”误认为“系统层面的设计进步”。从这个意义上说，context engineering 之所以顽固，不是因为大家还没想到更多技巧，而是因为它本来就是一个工作负载相关、预算相关、服务目标相关的系统问题。它没有脱离场景的标准答案，只有持续靠近 Pareto 前沿的工程选择。

## 结语

如果把 prompt 只理解为“对模型说的话”，context engineering 很容易被误写成 prompt 技巧的延长线；但只要把 context 看成 Agent 在当前 step 上的运行时工作集，问题就会立刻收敛到它真正的系统形态。更大的 context window 只能缓解容量约束，不能替代信息管理。真正决定 Agent 质量的，是信息如何被选择、压缩、分层、组织和淘汰。

因此，未来更强的 Agent 未必只是拥有更大的窗口，更可能是拥有更好的工作集管理系统。届时最关键的能力，也许不再是把多少 token 塞进模型，而是系统能否在效果、成本、时延、保真度、时效性与可控性之间持续取得更优平衡，并把对当前决策最有价值的信息稳定地放到模型眼前。

## 参考文献

[1] VASWANI A, SHAZEER N, PARMAR N, et al. Attention Is All You Need[C]// *Advances in Neural Information Processing Systems 30*. Red Hook, NY: Curran Associates, 2017. URL: [https://papers.nips.cc/paper/7181-attention-is-all-you-need](https://papers.nips.cc/paper/7181-attention-is-all-you-need).

[2] LIU N F, LIN K, HEWITT J, et al. Lost in the Middle: How Language Models Use Long Contexts[J]. *Transactions of the Association for Computational Linguistics*, 2024, 12: 157-173. DOI: [10.1162/tacl_a_00638](https://doi.org/10.1162/tacl_a_00638).

[3] JIANG H, WU Q, LUO X, et al. LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression[C]// *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Bangkok, Thailand: Association for Computational Linguistics, 2024: 1658-1677. DOI: [10.18653/v1/2024.acl-long.91](https://doi.org/10.18653/v1/2024.acl-long.91).

[4] LEWIS P, PEREZ E, PIKTUS A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks[C]// *Advances in Neural Information Processing Systems 33*. Red Hook, NY: Curran Associates, 2020. URL: [https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).

[5] DAI Z, YANG Z, YANG Y, et al. Transformer-XL: Attentive Language Models beyond a Fixed-Length Context[C]// *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Florence, Italy: Association for Computational Linguistics, 2019: 2978-2988. DOI: [10.18653/v1/P19-1285](https://doi.org/10.18653/v1/P19-1285).

[6] WU Y, RABE M N, HUTCHINS D, et al. Memorizing Transformers[C]// *The Tenth International Conference on Learning Representations*. OpenReview.net, 2022. URL: [https://openreview.net/forum?id=TrjbxzRcnf-](https://openreview.net/forum?id=TrjbxzRcnf-).

[7] YAO S, ZHAO J, YU D, et al. ReAct: Synergizing Reasoning and Acting in Language Models[C]// *The Eleventh International Conference on Learning Representations*. OpenReview.net, 2023. URL: [https://openreview.net/forum?id=WE_vluYUL-X](https://openreview.net/forum?id=WE_vluYUL-X).

[8] SCHICK T, DWIVEDI-YU J, DESSI R, et al. Toolformer: Language Models Can Teach Themselves to Use Tools[C]// *Advances in Neural Information Processing Systems 36*. Red Hook, NY: Curran Associates, 2023. URL: [https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html).

[9] JIANG H, WU Q, LIN C-Y, et al. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models[C]// *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. Singapore: Association for Computational Linguistics, 2023: 13358-13376. DOI: [10.18653/v1/2023.emnlp-main.825](https://doi.org/10.18653/v1/2023.emnlp-main.825).

[10] LI Y, DONG B, GUERIN F, et al. Compressing Context to Enhance Inference Efficiency of Large Language Models[C]// *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. Singapore: Association for Computational Linguistics, 2023: 6342-6353. DOI: [10.18653/v1/2023.emnlp-main.391](https://doi.org/10.18653/v1/2023.emnlp-main.391).

[11] DAI Y, LIAN J, HUANG Y, et al. Pretraining Context Compressor for Large Language Models with Embedding-Based Memory[C]// *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Vienna, Austria: Association for Computational Linguistics, 2025: 28715-28732. DOI: [10.18653/v1/2025.acl-long.1394](https://doi.org/10.18653/v1/2025.acl-long.1394).

[12] PARK J S, O'BRIEN J C, CAI C, et al. Generative Agents: Interactive Simulacra of Human Behavior[C]// *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*. New York, NY, USA: ACM, 2023. DOI: [10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763).

[13] SHINN N, CASSANO F, GOPINATH A, et al. Reflexion: Language Agents with Verbal Reinforcement Learning[C]// *Advances in Neural Information Processing Systems 36*. Red Hook, NY: Curran Associates, 2023. URL: [https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html).

[14] BAI Y, LV X, ZHANG J, et al. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding[C]// *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Bangkok, Thailand: Association for Computational Linguistics, 2024: 3119-3137. DOI: [10.18653/v1/2024.acl-long.172](https://doi.org/10.18653/v1/2024.acl-long.172).

[15] HSIEH C-P, SUN S, KRIMAN S, et al. RULER: What's the Real Context Size of Your Long-Context Language Models?[J]. *arXiv preprint arXiv:2404.06654*, 2024. DOI: [10.48550/arXiv.2404.06654](https://doi.org/10.48550/arXiv.2404.06654).

[16] ZHANG X, CHEN Y, HU S, et al. ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens[C]// *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Bangkok, Thailand: Association for Computational Linguistics, 2024: 15262-15277. DOI: [10.18653/v1/2024.acl-long.814](https://doi.org/10.18653/v1/2024.acl-long.814).

[17] YEN H, GAO T, HOU M, et al. HELMET: How to Evaluate Long-context Models Effectively and Thoroughly[C]// *The Thirteenth International Conference on Learning Representations*. OpenReview.net, 2025. URL: [https://proceedings.iclr.cc/paper_files/paper/2025/hash/f5332c8273d02729730a9c24dec2135e-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f5332c8273d02729730a9c24dec2135e-Abstract-Conference.html).
