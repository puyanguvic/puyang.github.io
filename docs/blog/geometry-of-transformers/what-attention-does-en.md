---
title: "The Geometric Core of Transformer Attention"
date: 2026-03-09T11:30:00-08:00
summary: "Starting from query-key matching, the probability simplex, and value reconstruction to explain the geometric core of attention."
tags: ["Transformer", "attention", "geometry of representation"]
---

# The Geometric Core of Transformer Attention

<BlogPostLocaleSwitch current-locale="en" zh-path="/blog/geometry-of-transformers/what-attention-does" en-path="/blog/geometry-of-transformers/what-attention-does-en" />

Attention is often described as a familiar engineering pipeline: compute `Q`, `K`, and `V`, apply `softmax`, and take a weighted sum of `V`. That description is correct at the implementation level, but if we stop there it becomes hard to explain why attention can support context retrieval, relation routing, and representation rewriting at the same time. The real issue is not the weighted average itself. It is how those weights are generated and what geometric object they constrain the output to lie in [1-7].

A more precise statement is that attention uses query-key matching to define a context-dependent set of soft coordinates, then performs a content-adaptive reconstruction in value space. In that sense, it is closer to dynamic kernel regression, barycentric reconstruction, or content-dependent projection than to simply copying one token's representation.

> Core claim: in standard self-attention, the query decides under which relation the current representation should read its context, the keys decide which positions can be activated under that relation, and softmax maps the scores into soft coordinates on a probability simplex. The final output is a barycentric reconstruction in value space, not a direct copy of any single position [1-7].

In the "Geometry of Transformers" series, this article first defines the soft-coordinate geometry of single-head attention. The next piece, [Why Multi-Head Attention Matters](/blog/geometry-of-transformers/why-multi-head-matters-en), explains why one coordinate system is not enough for the heterogeneous relations that language requires.

## 1. Reading the structure directly from the matrix formula

Standard scaled dot-product attention can be written as

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Let

$$
A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right),
$$

then the output is

$$
O = AV.
$$

Here $A \in \mathbb{R}^{n \times n}$ is a row-stochastic matrix: each row is nonnegative and sums to `1`. For position $i$,

$$
o_i = \sum_{j=1}^n \alpha_{ij} v_j,
\qquad
\alpha_i \in \Delta^{n-1},
$$

where $\Delta^{n-1}$ is the probability simplex. This already reveals the geometry of attention: each query position first generates a set of soft coordinates over the context positions, then uses those coordinates to form a barycentric combination in value space.

The attention output is therefore not an arbitrary vector. It always lies inside the convex hull of the current value set. That is the first fundamental difference between attention and a free linear layer.

## 2. What does the query-key inner product actually measure?

For positions $i,j$, the attention logit is

$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}
= \frac{x_i^\top W_Q^\top W_K x_j}{\sqrt{d_k}}.
$$

This shows that attention is not directly comparing whether two tokens are "similar" in the original representation space. It is measuring compatibility under a learnable bilinear form defined by $W_Q^\top W_K$. So:

- the query is not the content itself, but a probe representing the current reading task;
- the key is not the content itself, but an index of whether a position can be read under a certain relation;
- the matching rule is not a fixed distance, but a dynamic geometry determined by parameters and layer context.

So the first step of attention should not be described as "find the tokens most like me." It should be described as: **under the relation metric defined by this head, which context positions are most worth reading right now?**

That is why the same token can participate in very different matching patterns across layers and heads. Attention learns not one universal similarity measure, but a family of context-dependent relation metrics [1][5].

## 3. Why divide by $\sqrt{d_k}$?

The scaling term in scaled dot-product attention is often brushed aside as an implementation detail, but it plays an important numerical role. If the coordinates of $q_i$ and $k_j$ are roughly zero mean with comparable variance, then the unscaled inner product

$$
q_i^\top k_j = \sum_{\ell=1}^{d_k} q_{i\ell} k_{j\ell}
$$

has variance that typically grows linearly with $d_k$. As head width grows, logits become $O(\sqrt{d_k})$, softmax rapidly enters a near-saturated regime, a few weights approach `1`, the others approach `0`, and gradients become sharp and unstable [1].

Dividing by $\sqrt{d_k}$ rescales the logit variance back to the $O(1)$ range, so softmax remains trainable across different head widths. The scaling factor is therefore not a decorative constant. It is what turns a learnable bilinear match into an optimizable probability-coordinate system.

## 4. Why is softmax a "soft coordinate system" rather than hard retrieval?

Once the logits $s_{ij}$ are computed, softmax converts them into weights satisfying

$$
\alpha_{ij} \ge 0, \qquad \sum_j \alpha_{ij} = 1.
$$

Geometrically, that means the reading rule of query $i$ is constrained to the probability simplex. The output

$$
o_i = \sum_j \alpha_{ij} v_j
$$

is therefore a barycentric reconstruction of the value vectors.

This matters because it distinguishes attention from two common misreadings.

- It is not hard retrieval: the model does not have to copy information from a single position.
- It is not an orthogonal projection either: softmax adds nonlinear normalization, and key space need not equal value space.

So the more accurate analogy is a soft projection or content-dependent kernel regression, not a projection operator in the ordinary linear-algebra sense. Figure 1 is helpful precisely because it separates these two stages.

![Illustration of soft coordinates and reconstruction in attention](./attention-soft-coordinates.svg)

*Figure 1. Query-key matching first generates soft coordinates on the probability simplex. The attention output is then the barycentric reconstruction of the values under those coordinates.*

What Figure 1 really clarifies is that "weighted average" must be decomposed into "choose coordinates first, then reconstruct." If the first step is ignored, attention gets misread as a generic smoother rather than a relation-selection operator.

## 5. Why is attention more flexible than fixed convolution or fixed windows?

From a geometric viewpoint, the power of self-attention is not simply that it can "look far." It is that the reading rule itself is input-dependent. Compared with fixed convolution or fixed windows, it has at least three structural advantages.

### Global reach

Each query can access the whole context by default, so long-range dependencies do not have to be propagated token by token across many local steps [1].

### Content adaptivity

Convolution uses the same reading pattern at every position, while attention changes its weights with query-key matching. Where the model looks depends on the current content.

### Reduction to simpler operators

With the right parameterization, multi-head self-attention can express convolution-like structure [7]. Attention is therefore not a weak substitute for convolution, but a more general dynamic relation operator.

So the main advantage of attention is not "global averaging." It is content-dependent reading over a global range.

## 6. Why should attention weights not be identified with explanation?

Once attention is understood as a soft coordinate system, it is tempting to treat those coordinates as a direct causal explanation of the model's reasoning. That inference does not hold. Brunner et al. showed that under some conditions attention weights are not identifiable: different parameterizations can produce the same outputs while assigning different attention distributions [5].

So attention weights do reveal one reading path through the model, but they are not unique, and they are not a full human-readable explanation. Attention is also not the whole Transformer. Geva et al. argued that feed-forward layers can be interpreted as key-value memories at each position, carrying out stronger content selection and writing operations [6].

A full layer update is therefore better described as:

1. attention performs context routing and mixing;
2. residual connections preserve the original representation;
3. the FFN applies stronger nonlinear content transformation at each position.

So attention provides a context-reading mechanism, not a complete semantic reasoning loop by itself.

## 7. How powerful is attention, and where are its limits?

On the positive side, Transformers with positional encoding are highly expressive as sequence-to-sequence approximators [3]. Attention is not a weak averaging gadget. It is a key context-mapping module inside a larger representation system.

On the limiting side, pure self-attention also has theoretical boundaries. Hahn showed that if depth or head count does not grow with input length, pure self-attention is limited on some formal languages and hierarchical structures [4]. That reminds us that attention's power depends on layer depth, positional encoding, nonlinear modules, and multi-head structure, not on a single attention matrix alone.

So the most accurate statement is not "attention alone performs understanding," but: **attention gives the Transformer a content-dependent coordinate system over context, which later layers can continue to compute on.**

## 8. Closing

The essence of attention should not be compressed to "a weighted average over values." A better description is that it first uses query-key geometry to generate input-dependent soft coordinates, then uses those coordinates to perform barycentric reconstruction in value space. The crucial question in a Transformer is therefore not only which tokens were seen, but what local coordinate system for reading context was built for the current token.

In short, **attention is a content-dependent geometric read operator.** It chooses coordinates first and reconstructs second. The question of multi-head attention is then how many such coordinate systems a layer can provide in parallel. The next article addresses exactly that.

Continue reading: [Why Multi-Head Attention Matters](/blog/geometry-of-transformers/why-multi-head-matters-en).

## References

[1] VASWANI A, SHAZEER N, PARMAR N, et al. Attention Is All You Need[C]// *Advances in Neural Information Processing Systems 30*. Red Hook, NY: Curran Associates, 2017. URL: [https://papers.nips.cc/paper/7181-attention-is-all-you-need](https://papers.nips.cc/paper/7181-attention-is-all-you-need).

[2] LEE J, LEE Y, KIM J, et al. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks[C]// *Proceedings of the 36th International Conference on Machine Learning*. PMLR, 2019: 3744-3753. URL: [https://proceedings.mlr.press/v97/lee19d.html](https://proceedings.mlr.press/v97/lee19d.html).

[3] YUN C, BHOJANAPALLI S, RAWAT A S, et al. Are Transformers Universal Approximators of Sequence-to-Sequence Functions?[C]// *International Conference on Learning Representations*. 2020. URL: [https://openreview.net/forum?id=ByxRM0Ntvr](https://openreview.net/forum?id=ByxRM0Ntvr).

[4] HAHN M. Theoretical Limitations of Self-Attention in Neural Sequence Models[J]. *Transactions of the Association for Computational Linguistics*, 2020, 8: 156-171. DOI: [10.1162/tacl_a_00306](https://doi.org/10.1162/tacl_a_00306).

[5] BRUNNER G, LIU Y, PASCUAL D, et al. On Identifiability in Transformers[C]// *International Conference on Learning Representations*. 2020. URL: [https://research.google/pubs/on-identifiability-in-transformers/](https://research.google/pubs/on-identifiability-in-transformers/).

[6] GEVA M, SCHUSTER R, BERANT J, et al. Transformer Feed-Forward Layers Are Key-Value Memories[C]// *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, 2021: 5484-5495. DOI: [10.18653/v1/2021.emnlp-main.446](https://doi.org/10.18653/v1/2021.emnlp-main.446).

[7] CORDONNIER J-B, LOUKAS A, JAGGI M. On the Relationship between Self-Attention and Convolutional Layers[C]// *International Conference on Learning Representations*. 2020. URL: [https://openreview.net/forum?id=zoPf7R-2wZr](https://openreview.net/forum?id=zoPf7R-2wZr).
