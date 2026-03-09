---
title: "Tokenizer的理论"
description: "从压缩、熵和工程约束角度理解 tokenizer 的设计。"
---

# Tokenizer的理论

这个系列讨论 tokenizer 的理论基础与工程折中：为什么 tokenization 可以理解为压缩，为什么词表大小会集中在某个范围，以及为什么 character-level 方法虽然看起来更纯粹却并不流行。

## 文章目录

1. [Tokenizer 到底在做什么？](/blog/theory-of-tokenizers/what-tokenization-does)
2. [为什么 LLM 的词表大小总在 50k 左右？](/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k)
3. [为什么 character-level tokenizer 理论最优却几乎不用？](/blog/theory-of-tokenizers/why-character-level-rarely-wins)
