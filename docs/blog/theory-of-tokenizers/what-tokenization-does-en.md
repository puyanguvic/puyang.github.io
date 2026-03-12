---
title: "Tokenization as Reversible Coarse-Graining"
date: 2026-03-09T12:00:00-08:00
summary: "From reversible recoding, coarse-graining of local mutual information, and neural compute budgets to a view of tokenization as source reparameterization for sequence models."
tags: ["tokenizer", "compression", "LLM"]
---

# Tokenization as Reversible Coarse-Graining

<BlogPostLocaleSwitch current-locale="en" zh-path="/blog/theory-of-tokenizers/what-tokenization-does" en-path="/blog/theory-of-tokenizers/what-tokenization-does-en" />

If tokenization is described only as a preprocessing step that splits text into tokens, its real role in the language-model stack disappears. A tokenizer decides what discrete units the model is allowed to see, which local regularities are written into an explicit symbol system, and which regularities must still be reconstructed by the network itself.

That matters because tokenization does not merely change the surface form of a string. It changes sequence length, statistical sharing, long-tail compositionality, and how much low-level local compression the model must perform explicitly during training [1][3-5]. Once those effects are put together, a tokenizer stops looking like a neutral segmenter and starts looking like a **reparameterization of the text source for neural sequence modeling**.

> Core view: tokenization is not primarily about "finding words." It is a reversible coarse-graining of the string source under a vocabulary budget. It promotes high-frequency, strongly locally dependent patterns into an explicit codebook, folding low-level redundancy into token identity. That shortens prediction chains, improves statistical efficiency, and leaves more compute for genuine contextual modeling [1-5].


## 1. A tokenizer is first a reversible recoding

Let raw text be a character or byte sequence

$$
x = (c_1,\dots,c_n),
$$

and let the tokenizer output a token sequence

$$
\tau(x) = (t_1,\dots,t_m),
$$

with a deterministic decoder $\gamma$ such that

$$
\gamma(t_1)\gamma(t_2)\cdots\gamma(t_m)=x
$$

or at least reconstructs the normalized text space. Tokenization is therefore not arbitrary segmentation. It is a **reversible variable-length recoding over a finite vocabulary**.

That definition clarifies a point that is often blurred: if the mapping is deterministic and reversible, a tokenizer does not destroy information in the Shannon sense [1]. What changes is something else:

- how many discrete prediction steps the model must execute;
- at what granularity uncertainty is presented to the model;
- which local dependencies are folded into token identity and which remain exposed along the sequence axis.

So tokenization does not alter information conservation itself. It alters the **distribution of information between token identity and cross-token dependence**.

From a systems viewpoint, the relevant objective is not just bit-level compression of raw strings. It is closer to minimizing total modeling burden for a model family $\mathcal{F}$:

$$
\min_{\tau}
\mathcal{L}_{\mathcal{F}}(\tau;\mathcal{D})
\;+\;
\lambda\,\mathbb{E}_{x\sim\mathcal{D}}[T_{\tau}(x)]
\;+\;
\beta\,|V_{\tau}|,
$$

where:

- $\mathcal{L}_{\mathcal{F}}$ measures modeling difficulty under that tokenization;
- $T_{\tau}(x)$ is the resulting token length;
- $|V_{\tau}|$ is vocabulary size.

Even this rough objective is enough to show that tokenization is not a peripheral linguistic choice. It is part of the interface between the source distribution, the model class, and the compute budget.

## 2. What tokenization really exploits is local mutual information

Language is compressible not because the tokenizer is magically intelligent, but because language distributions are deeply nonuniform. Zipf's law says that a small head of very frequent patterns carries a large fraction of the mass, while the tail contains many rare forms [2]. But the deeper fact is that these frequent patterns often appear as **strongly dependent local chunks** rather than isolated symbols.

If two adjacent fragments $u$ and $v$ tend to occur together, then knowing $u$ sharply reduces uncertainty about $v$. In mutual-information terms, their local dependence is high. Promoting such a pattern into a single token means that some predictability that used to be spread across multiple time steps is now folded into one discrete symbol.

That does not violate Shannon limits. It does something more practical:

- it moves some short-range redundancy from repeated sequence prediction into the explicit codebook;
- it merges several low-level prediction steps into one higher-granularity step;
- it reduces how often the model must reconstruct spelling-level or lexical microstructure from scratch.

So tokenization is not creating an information advantage. It is performing a **coarse-graining of local mutual information**. Frequent affixes, common spelling chunks, function words, whitespace conventions, and punctuation patterns become good token candidates not because they "look like words," but because they package local statistical structure that can be prepaid.

## 3. BPE and unigram LM are both learning which local structures deserve atomic status

Once tokenization is viewed as budgeted coarse-graining, the contrast between BPE and unigram LM becomes much easier to state cleanly.

### BPE: greedily absorb high-yield local patterns

After Sennrich et al. brought BPE into neural machine translation, subword tokenization became a standard choice in modern NLP [3]. The mechanics are simple: repeatedly merge the adjacent pair that yields the highest gain.

Looked at through the present lens, BPE is repeatedly asking:

> If this recurring local pattern is promoted to its own codeword, how much sequence length and modeling burden do we save?

Its greedy nature does not prevent it from finding the most valuable structure early, because the most valuable local structure is concentrated in the frequency head anyway.

### Unigram LM: score the segmentation system directly

Kudo's unigram language model treats segmentations of a string as probabilistic objects [5]. SentencePiece turned that idea into a general tokenizer-training interface over raw text, with vocabulary size as a first-class constraint [4].

Compared with BPE, unigram LM is less about local merge decisions and more about optimizing the segmentation system as a whole. But the underlying question is the same: **under a fixed vocabulary budget, which substrings are worth owning an independent identity, and which should remain compositional?**

So the algorithmic details differ, but the object being learned is the same: a finite codebook of local structures worth remembering explicitly.

![Illustration of tokenization as codebook coarse-graining](./tokenization-compression-codebook.svg)

*Figure 1. The crucial point is not the exact boundary location. It is which frequent, reusable, strongly dependent local patterns are absorbed into the explicit codebook so the model does not need to rediscover them repeatedly.*

## 4. The deepest effect on neural models is statistical efficiency, not just compression ratio

Saying that tokenization shortens sequences is true, but still too shallow. The more important effect is that it changes how parameters accumulate evidence.

### Repeated contexts collapse onto the same parameter row

When a frequent fragment is consistently encoded as one token, many gradient updates from many contexts accumulate on the same embedding row and the same output class. That makes it easier to learn a stable reusable representation. If the same fragment is always split into characters, the model must recover it indirectly through a longer compositional path.

### Solved short-range structure is removed from the sequence axis

Once some common local patterns are absorbed into token identity, they no longer need to be predicted step by step along the sequence. Put differently, **intra-token dependence is prepaid, so the Transformer can focus more on inter-token dependence**.

That matters especially for standard Transformers. They are good at modeling long-range relations between units that already carry some semantic density. They are not especially efficient at repeatedly reconstructing low-level lexical chunks from very long raw character streams.

### The tokenizer defines what gets shared first

Any fixed vocabulary defines a default sharing geometry:

- structure inside a token is treated as an already-solved chunk;
- structure across tokens must be learned by the contextual model;
- regularities that align with token boundaries accumulate evidence more easily.

So the tokenizer injects more than a compression ratio. It defines part of the model's low-level inductive bias.

## 5. Why subword tokenization became the stable operating point

From this perspective, word-level and character-level tokenization are two opposite extremes.

| Granularity | What is absorbed into the explicit codebook | Main problem |
| --- | --- | --- |
| word-level | too much, including fragile long-tail whole words | severe OOV issues, weak sharing in the tail |
| character/byte-level | too little, almost no prepaid local structure | long sequences, low-level compression is delayed into the network |
| subword | head local structure is absorbed while tail structure stays compositional | segmentation bias remains, but the system balance is far better |

Subword methods win not because they are linguistically pure, but because they better match the multiscale structure of language:

- frequent local chunks are worth remembering explicitly;
- rare forms should still be composable from more frequent pieces;
- the model should see units with some semantic density early, without allocating a separate parameter row to every rare word form.

In that sense, subword tokenization is a middle-scale coarse-graining. It avoids both extremes: the brittle whole-word dictionary and the raw-character stream that forces the network to reinvent everything internally.

## 6. Closing

The deep function of tokenization is not simply to cut text into pieces. It is to decide which local regularities should be handled by an explicit discrete codebook and which should remain the responsibility of the neural network operating over sequences. Once stated that way, tokenization is no longer a peripheral linguistic convenience. It becomes part of the first layer of representation theory for LLM inputs.

Put compactly, **a tokenizer is a budgeted reversible coarse-graining of the string source**. It does not change information conservation, but it strongly changes the granularity at which information enters the model, the stage at which local structure is compressed, and how much explicit compute the model must spend on low-level redundancy.

Continue reading: [Why Vocabulary Size Stays Near 50k](/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k-en).

## References

[1] SHANNON C E. A Mathematical Theory of Communication[J]. *Bell System Technical Journal*, 1948, 27(3): 379-423; 27(4): 623-656. URL: [https://www.mpi.nl/publications/item2383162/mathematical-theory-communication](https://www.mpi.nl/publications/item2383162/mathematical-theory-communication).

[2] PIANTADOSI S T. Zipf's Word Frequency Law in Natural Language: A Critical Review and Future Directions[J]. *Psychonomic Bulletin & Review*, 2014, 21(5): 1112-1130. DOI: [10.3758/s13423-014-0585-6](https://doi.org/10.3758/s13423-014-0585-6).

[3] SENNRICH R, HADDOW B, BIRCH A. Neural Machine Translation of Rare Words with Subword Units[C]// *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Berlin, Germany: Association for Computational Linguistics, 2016: 1715-1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162).

[4] KUDO T, RICHARDSON J. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing[C]// *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Brussels, Belgium: Association for Computational Linguistics, 2018: 66-71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012).

[5] KUDO T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates[C]// *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Melbourne, Australia: Association for Computational Linguistics, 2018: 66-75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007).
