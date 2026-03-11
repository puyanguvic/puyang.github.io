---
title: "Why Multi-Head Attention Matters"
date: 2026-03-09T11:40:00-08:00
summary: "Explaining why multi-head attention substantially improves expressivity by starting from the geometric bottlenecks of single-head attention and parallel selection."
tags: ["Transformer", "multi-head attention", "representation geometry"]
---

# Why Multi-Head Attention Matters

<BlogPostLocaleSwitch current-locale="en" zh-path="/blog/geometry-of-transformers/why-multi-head-matters" en-path="/blog/geometry-of-transformers/why-multi-head-matters-en" />

If a single attention head can already generate soft coordinates through query-key matching and reconstruct content in value space, then a natural question follows: why do we need multi-head attention at all? Why not keep one wider head and let the same mechanism handle every relation? This cannot be answered by saying "it works better in practice." We need to return to the geometry of the operator itself [1-6].

The true limitation of a single head is not merely that it has too few parameters. It is that it can offer only one matching geometry, one set of soft coordinates, and one content channel for each position. Natural language dependencies are obviously not single-type objects: positional offsets, syntax, coreference, discourse, and task-specific cues often require different reading rules. Multi-head attention matters because it provides several context coordinate systems in parallel inside the same layer.

> Core claim: each attention head defines its own query-key metric, softmax coordinate system, and value mapping. So the essence of multi-head attention is not repeating the same operation, but constructing several distinct context-geometry maps in parallel. What it improves is not just parameter count, but relation disentangling, content disentangling, and parallel computation [1-6].


## 1. What does multi-head add, exactly?

Standard multi-head attention is written as

$$
\operatorname{MHA}(X)
=
\operatorname{Concat}(H_1,\dots,H_m)W_O,
$$

where the $h$-th head is

$$
H_h = A_h V_h,
\qquad
A_h = \operatorname{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right),
$$

with

$$
Q_h = XW_Q^{(h)}, \qquad
K_h = XW_K^{(h)}, \qquad
V_h = XW_V^{(h)}.
$$

The crucial point in these equations is that every head has its own

- matching metric $W_Q^{(h)\top}W_K^{(h)}$;
- attention distribution $A_h$;
- content map $W_V^{(h)}$;
- reconstructed output $H_h$.

So multi-head attention does not copy the same attention map several times. It learns several parallel rules for deciding who should be read and what kind of information should be extracted.

## 2. Why is one wider single head not equivalent?

It is natural to ask: if total width is fixed, why not replace several small heads with one wider head? The key difference is that a single head, however wide, still produces only **one** attention matrix. Multi-head attention produces several independent attention matrices.

More formally, if total channel width is fixed at $m d_h$, a single wide head still generates one weight vector

$$
\alpha_i \in \Delta^{n-1}
$$

for each position $i$, so every retrieved feature must be mixed under the same coordinate system. By contrast, multi-head attention generates

$$
\alpha_i^{(1)},\dots,\alpha_i^{(m)},
$$

which allows the same position to attend differently for different subproblems. One head may focus on local syntax, another on long-range coreference, and a third on positional regularities. These choices do not have to compete inside one probability distribution.

More importantly, this gap cannot be erased by a simple linear reparameterization. Softmax is applied independently inside each head. In general, "concatenate first and normalize once" cannot reproduce "normalize separately and then concatenate." Multi-head attention therefore does not just slice channels into smaller pieces. It introduces several independent nonlinear selection mechanisms.

## 3. Where is the bottleneck of single-head attention?

For a fixed query position $i$, a single head can generate only one soft coordinate vector

$$
\alpha_i \in \Delta^{n-1},
$$

and use it to reconstruct one value mixture. This creates three structural limitations.

### One matching geometry

A single head can define only one bilinear notion of relevance. But linguistic relevance is not unique; the same position may need to read context according to syntax, reference, position, or discourse structure.

### One coordinate system

A single head provides only one attention distribution. If the current token needs to locate its subject, resolve a pronoun, and extract a local modifier, these demands all compete inside one simplex.

### One content channel

Even if the correct position is selected, a single head can read it only through one $W_V$. So "attend here for syntactic reasons" and "attend here for semantic reasons" are still forced through the same output path.

So the real limitation of single-head attention is not that it cannot model any relation. It is that **too many heterogeneous relations are compressed into one geometric reading mechanism.**

## 4. Why should multi-head attention be understood as multiple context coordinate systems?

Once multiple heads are introduced, the same position $i$ receives independent coordinate vectors

$$
\alpha_i^{(h)} \in \Delta^{n-1}
$$

for different heads. The model therefore no longer has only one way to read context. It has several local coordinate charts in parallel. Each head answers slightly different questions:

- under this view, which positions are relevant?
- what relation defines relevance here?
- once those positions are read, what type of content should be extracted?

Figure 1 is the clearest way to distinguish "one wider head" from "many independent heads."

![Illustration of parallel coordinate systems in multi-head attention](./multi-head-coordinate-systems.svg)

*Figure 1. A single head can generate only one soft coordinate system under one matching geometry. Multi-head attention generates several coordinate systems in parallel, reconstructs several value mixtures, and then maps them back into the output space.*

The point of Figure 1 is that multi-head attention adds not only more channels, but more independently normalized choice mechanisms. So interpreting multi-head attention as "several context coordinate systems" is not a metaphor. It is a direct geometric reading of the equations.

## 5. Why does this parallel coordinateization improve expressivity?

The gains from multi-head attention come from at least three levels.

### Relation disentangling

Different heads can learn different relation types. Analyses by Clark et al. and Voita et al. show that some heads reliably attend to separators, relative positions, syntax, or coreference patterns [2][3]. So the model really does use multiple heads to distribute relation types across channels.

### Content disentangling

Even when two heads attend to the same position, different value maps $W_V^{(h)}$ let them extract different aspects of that token. One head may prioritize structure, another semantics, and another task-specific hints.

### Parallel composition

The RASP perspective of Weiss et al. suggests that many sequence computations decompose into parallel selection-and-aggregation operations, and the number of heads directly affects how many such operations can be carried out at shallow depth [6]. So multi-head attention helps not only with cleaner relation structure, but also with parallel computational composition.

From this angle, multi-head attention is not one wider reading path. It is a set of reading paths that can be combined in parallel.

## 6. Do heads really specialize in practice?

The empirical answer is yes, but not evenly. Voita et al. found that some heads perform stable and important linguistic functions, while many others can be pruned with only mild performance loss [3]. Michel et al. reported similar results: a nontrivial fraction of heads are redundant, but certain layers and certain heads are clearly more important [4].

This means the value of multi-head attention should not be misread as "every head is indispensable." The better interpretation is:

- the model needs enough heads to provide relation separation and optimization freedom;
- training lets some of them settle into stable functions;
- others may act as backup channels, auxiliary channels, or redundant slack for optimization.

So redundancy does not refute the mechanism. It shows that multi-head attention gives the optimizer room to disentangle relations more flexibly.

## 7. Why is "more heads" not always better?

If total width is fixed, adding more heads reduces the per-head dimension $d_h$. That creates an explicit trade-off:

- too few heads, and heterogeneous relations are forced into the same geometry;
- too many heads, and each channel becomes too narrow, leading to weak expressivity or redundant fragmentation.

This is why "some heads can be pruned" does not imply "one head is enough." The right conclusion is: multi-head structure is necessary, but the effective number of heads depends on model width, depth, task, and training dynamics [4][5].

The real question is not whether multiple heads are needed, but how many parallel coordinate systems the system truly benefits from.

## 8. Closing

The main value of multi-head attention is not repeating one attention operation many times. It is that one layer can provide several different context geometries in parallel. A single head already implements "soft coordinates plus reconstruction." Multi-head attention duplicates that mechanism across distinct relational viewpoints and then integrates the results through the output projection.

At bottom, **multi-head attention is parallel coordinateization.** The Transformer can process position patterns, syntactic dependencies, coreference, and task cues in the same layer not because one giant attention matrix is all-powerful, but because several coordinate systems can cooperate and specialize at the same time.


## References

[1] VASWANI A, SHAZEER N, PARMAR N, et al. Attention Is All You Need[C]// *Advances in Neural Information Processing Systems 30*. Red Hook, NY: Curran Associates, 2017. URL: [https://papers.nips.cc/paper/7181-attention-is-all-you-need](https://papers.nips.cc/paper/7181-attention-is-all-you-need).

[2] CLARK K, KHANDELWAL U, LEVY O, et al. What Does BERT Look At? An Analysis of BERT's Attention[C]// *Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*. Florence, Italy: Association for Computational Linguistics, 2019: 276-286. DOI: [10.18653/v1/W19-4828](https://doi.org/10.18653/v1/W19-4828).

[3] VOITA E, TALBOT D, MOISEEV F, et al. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned[C]// *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Florence, Italy: Association for Computational Linguistics, 2019: 5797-5808. DOI: [10.18653/v1/P19-1580](https://doi.org/10.18653/v1/P19-1580).

[4] MICHEL P, LEVY O, NEUBIG G. Are Sixteen Heads Really Better than One?[C]// *Advances in Neural Information Processing Systems 32*. Red Hook, NY: Curran Associates, 2019. URL: [https://papers.nips.cc/paper/9551-are-sixteen-heads-really-better-than-one](https://papers.nips.cc/paper/9551-are-sixteen-heads-really-better-than-one).

[5] CORDONNIER J-B, LOUKAS A, JAGGI M. On the Relationship between Self-Attention and Convolutional Layers[C]// *International Conference on Learning Representations*. 2020. URL: [https://openreview.net/forum?id=zoPf7R-2wZr](https://openreview.net/forum?id=zoPf7R-2wZr).

[6] WEISS G, GOLDBERG Y, YAHAV E. Thinking Like Transformers[C]// *Proceedings of the 38th International Conference on Machine Learning*. PMLR, 2021: 11080-11090. URL: [https://proceedings.mlr.press/v139/weiss21a.html](https://proceedings.mlr.press/v139/weiss21a.html).
