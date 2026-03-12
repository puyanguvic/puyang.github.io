---
title: "Character-Level Tokenizer 的理论优势与工程局限"
date: 2026-03-09T12:20:00-08:00
summary: "从局部熵压缩被推迟到网络内部、计算尺度错配与隐式分词补偿出发，解释为什么 character-level 路线通常难成默认解。"
tags: ["tokenizer", "character-level", "optimization"]
---

# Character-Level Tokenizer 的理论优势与工程局限

<BlogPostLocaleSwitch current-locale="zh" zh-path="/blog/theory-of-tokenizers/why-character-level-rarely-wins" en-path="/blog/theory-of-tokenizers/why-character-level-rarely-wins-en" />

character-level 或 byte-level 路线最吸引人的地方，是它看起来非常“干净”：没有 OOV，没有人为词边界，不需要为多语言和新词反复维护 tokenizer 资产。只要从表达完备性出发，这条路线几乎无懈可击 [1-5]。

但这套直觉遗漏了真正决定系统优劣的那一层。tokenizer 被拿掉之后，语言中的局部压缩需求并没有消失。拼写块、形态片段、常见词法模式、空白和标点结构仍然存在；只是它们不再由外部显式码本承担，而必须由模型自己在更长序列上重新恢复。

这就是 character-level 路线最深的代价。它不是简单地“输入更细了”，而是把本该在廉价前端完成的**局部熵折叠与中层结构发现**，推迟给了更深、更贵的网络。

> 核心观点：character-level 或 token-free 方法输的通常不是表达能力，而是层级分工。它们取消了外部 tokenizer，却没有取消语言中的局部冗余；结果是模型必须先在极长序列上重新学出拼写块、子词块和词法单元，再谈高层语义与长程依赖。现代最成功的 token-free 架构几乎都会重新引入下采样、局部块化或潜在 subword 模块，这说明被取消的并不是 tokenization 本身，而只是它的显式外包形式 [1-5]。


## 1. character-level 真正取消的，是哪一层结构？

先把问题说得更准确一些。character-level 并没有让语言变得“更原子”，它只是把模型输入的原子单位退回到字符或字节。可自然语言的有效结构从来不是平的，而是明显分层的：

- 字符和字节承载书写约束；
- 子词和词法块承载大量局部可复用模式；
- 词与短语承载更高层语义和组合关系。

显式 subword tokenizer 的作用，正是把其中一部分最稳定、最常见的局部结构提前固化为离散单元。character-level 路线拒绝做这一步，于是模型看到的输入虽然更统一，却也更“未整理”：它必须自己从低级字符流里重新发明中层块结构。

所以 character-level 取消的不是工程噪声，而是**一种多尺度结构的前置粗粒化**。没有这层粗粒化，网络就必须先解决“哪些字符应被绑定在一起”这个问题，才能高效进入真正的语义建模。

## 2. 最大的问题不是更长序列，而是计算尺度错配

字符级方案最常被提到的缺点是序列更长。这当然是真的，但还不够深。更本质的问题在于：它会让系统拿一个昂贵的全局建模器，去做本来应该由廉价局部模块完成的工作。

标准 Transformer 最擅长的事是：

- 在已有一定语义密度的离散单元之间建立关系；
- 用多层 attention 和 FFN 反复重写上下文表示；
- 把算力花在跨位置依赖、语义组合和推理链条上。

它不特别擅长的事则是：

- 从极长字符流中高频重复地恢复拼写块；
- 在恢复局部块的同时仍保持全局计算高效；
- 把“先学压缩”与“再学语义”混在同一个深层堆栈里同时做好。

这可以被视为一种明确的尺度错配。拼写和词法块大多是短程规律，本来更适合被前端局部机制便宜地吸收；character-level 却要求全局网络反复在每个训练样本上重新发现它们。结果并不是“更端到端”，而是**把高价计算浪费在低层熵压缩上**。

## 3. 语义形成路径也会因此被拉长

序列更长只是表层，真正拖慢优化的是表征形成链条的变长。

在 subword 方案里，模型一开始就能看到一定程度上已经粗粒化的单位。哪怕这些单位不完美，它们通常也已经吸收了相当一部分正字法和形态学稳定结构。因此，梯度可以更早直接作用到具有中层统计意义的单元上。

而在 character-level 方案里，模型通常必须依次学会：

1. 哪些字符会稳定组成局部块；
2. 哪些局部块值得被当成复用单元；
3. 这些复用单元如何再进入词级和语义级关系；
4. 高层上下文任务如何反过来约束这些中层单元。

这条链条更长，意味着有效语义梯度必须穿过更多低层组合步骤，才能回到输入侧。训练早期，模型更容易首先学会表面拼写统计，而不是迅速形成高质量的词法和语义抽象。

因此，character-level 的问题不只是 FLOPs 更多，而是**语义需要经过更长的表示路径才能稳定显现**。

## 4. 为什么说 character-level 本质上是在把 tokenizer 学回去？

这正是图 1 想说明的核心。

![token-free 模型为何仍会重新引入压缩](./character-level-compute-tradeoff.svg)

*图 1. token-free 方法想要变得实用，通常都要在网络内部重新引入某种压缩机制：下采样、局部块化或潜在 subword。压缩并没有消失，只是从显式 tokenizer 转移到了模型内部。*

图 1 最重要的结论不是“某个补偿模块更好”，而是：只要想把字符级系统做得实用，模型几乎总要在内部恢复一层中间粒度结构。否则，极长序列和低语义密度输入会持续拖累整个网络。

从这个意义上说，token-free 路线真正做的不是取消 tokenization，而是把 tokenization 从外部静态码本，改成内部可学习模块。

## 5. ByT5、CANINE 和 Charformer 为什么都在重新引入层级？

这一点在主流 token-free 工作里几乎是公开写在方法里的。

### ByT5：接受代价，但没有消灭代价

ByT5 证明了基于字节的预训练模型完全可以工作，而且在噪声鲁棒性方面有清楚优势 [1]。但它也清楚展示了代价：更长序列、更高训练 FLOPs，以及更重的推理开销。ByT5 的意义不是“tokenizer 多余”，而是“如果你愿意为统一输入接口和鲁棒性买单，token-free 可以成立”。

### CANINE：先下采样，再交给深层网络

CANINE 的关键创新并不是“直接输入字符”，而是引入显式下采样，把很长的字符流先压成更短的中间表示，再交给后续深层 Transformer [2]。这其实已经非常接近“把 tokenizer 的局部压缩职责移入模型内部”。

### Charformer：在网络中学习潜在 subword

Charformer 更进一步，直接在模型里学习基于字符块的潜在分词结构。它的 GBST 模块通过枚举候选块并学习权重，显式构造出一种 latent subword tokenization [3]。这不是对 tokenizer 逻辑的否定，恰恰是对它的重新发明。

因此，这些方法最有说服力的共同点并不是“摆脱 tokenizer”，而是都承认了同一个事实：**中间粒度结构终究必须出现。**

## 6. 所以 character-level 一般输在什么地方？

把前面几层合起来，character-level 路线通常在三个维度上同时吃亏。

### 计算分工更差

它把原本可以前置完成的局部压缩，推给了更贵的深层网络。

### 参数统计更慢

模型需要先把字符组合成稳定块，再在这些块上积累可迁移统计，因此中层表征形成更慢。

### 架构匹配更弱

标准 Transformer 更适合处理已经具备一定语义密度的离散单元，而不是长时间充当字符压缩器。

这也是为什么 character-level 的“理论统一性”很容易被高估。统一输入接口当然很优雅，但优雅并不等于更优的系统分工。对于主流预算和架构而言，显式 subword tokenizer 往往只是更便宜地做掉了模型迟早也要做的那部分工作 [4][5]。

## 7. 哪些场景下 character-level 反而值得选？

上面的分析并不意味着 character-level 没有真实价值。恰恰相反，在某些任务里，它的优势非常具体。

- 输入噪声重、拼写变化大，或 OCR、社交媒体文本中存在大量非标准写法时，character-level 往往更鲁棒 [1]。
- 需要覆盖多脚本、多语言、新词频繁涌现的场景时，开放词表可以显著减少 tokenizer 维护成本。
- 当研究目标本身就是减少 tokenizer 技术债、测试模型能否自行恢复中层结构，或统一输入接口时，token-free 路线也有明确方法论意义。

因此，问题不该被问成“character-level 对不对”，而应被问成：**在这个任务里，统一性和鲁棒性是否值得我们为更差的层级分工付费？**

## 8. 结语

character-level 路线最大的误导性，在于它看起来取消了 tokenizer，于是似乎也取消了所有词法层面的工程折中。实际上，被取消的只是外部显式码本；局部压缩、中间结构发现和多尺度表示组织这三件事，一件都没有消失。

更紧凑地说，**character-level 方法通常输的不是表示能力，而是计算该在哪一层发生。** 当显式 subword tokenizer 已经能用极低成本把头部局部结构预付掉时，让深层 Transformer 在长字符流上重复发明这些结构，通常不会成为默认最优解。

## 参考文献

[1] XUE L, BARUA A, CONSTANT N, et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 291-306. DOI: [10.1162/tacl_a_00461](https://doi.org/10.1162/tacl_a_00461).

[2] CLARK J H, GARRETTE D, TURC I, et al. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation[J]. *Transactions of the Association for Computational Linguistics*, 2022, 10: 73-91. DOI: [10.1162/tacl_a_00448](https://doi.org/10.1162/tacl_a_00448).

[3] TAY Y, TRAN V Q, RUDER S, et al. Charformer: Fast Character Transformers via Gradient-based Subword Tokenization[C]// *International Conference on Learning Representations*. 2022. URL: [https://research.google/pubs/charformer-fast-character-transformers-via-gradient-based-subword-tokenization/](https://research.google/pubs/charformer-fast-character-transformers-via-gradient-based-subword-tokenization/).

[4] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[5] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).
