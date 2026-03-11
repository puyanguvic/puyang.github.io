---
title: "What Are We Optimizing as Context Keeps Expanding?"
date: 2026-03-11T10:00:00-08:00
summary: "This article reframes agent context as a runtime working set and argues that context engineering is not about enlarging the window, but about building an information-selection and state-management mechanism that can balance quality, cost, latency, fidelity, freshness, and controllability in deployment."
tags: ["AI Agent", "LLM Systems", "Context Engineering", "RAG", "Memory"]
---

# What Are We Optimizing as Context Keeps Expanding?

<BlogPostLocaleSwitch current-locale="en" zh-path="/blog/engineering_system_view/what-are-we-optimizing-in-agent-context" en-path="/blog/engineering_system_view/what-are-we-optimizing-in-agent-context-en" />

When a large model is used only for single-turn question answering, the prompt looks like a piece of input text. In an AI agent system, however, it is closer to the **runtime working set** visible to the model at the current step. User goals, conversation history, retrieved documents, tool outputs, execution traces, reflections, and task state all have to be organized into a context that the model can consume for a single call.

That changes the nature of the problem. An agent's ceiling is determined not only by model parameters, reasoning style, or tool availability, but also by what the system chooses to show the model, what it withholds, and in what form it presents it. Once that construction layer loses discipline, a larger context window does not rescue the system: it can still lose focus in irrelevant material, distort critical details, or push state that should have remained structured back into the natural-language channel [1-3].

> Core claim: in an agent, context engineering is not a technique for making prompts longer. It is runtime information management. The real question is not how many tokens can be packed into the prompt, but which information should enter the live working set, which should be compressed, and which should remain outside the system until it is needed.

In the "Engineering and Systems Perspectives" series, this article places the problem back in its proper systems setting. The prompt is only the final rendering of a `context policy`, not the problem itself. To return to the topic overview, use [Blog](/blog/).

![Agent context as a runtime working set](./agent-context-working-set.svg)

*Figure 1. In an agent system, the prompt is not a simple accumulation of input text. It is the working set selected, compressed, and rendered by a controller for the current step.*

## 1. Agent Context Is Fundamentally a Runtime Working Set

If we write a single model call at time $t$ in a form that is closer to system reality, it can be approximated as

$$
C_t
=
\mathrm{render}(u_t, H_t, R_t, O_t, M_t, s_t; B_t),
$$

where $u_t$ is the current user request or subtask goal, $H_t$ is dialogue history, $R_t$ is externally retrieved evidence, $O_t$ is the observation produced by tools or environment interaction, $M_t$ is experience or summary retrieved from long-term memory, $s_t$ is structured task state, and $B_t$ is the available token, latency, and cost budget for the current step.

The importance of this notation is not the equation itself. It is that it forces us to acknowledge a basic fact: the model never receives "the state itself." It only receives a view of state for the current call. Put differently, the prompt is not the information the system has. It is a rendered result produced by a controller that filters, ranks, compresses, and templates heterogeneous state under a budget. Conventional software distributes state across memory, caches, logs, and databases. LLM agents often serialize those heterogeneous states into tokens and feed them through one common input channel.

From that perspective, the role of the context window becomes much clearer. It behaves like an expensive and limited working memory, not a container that should simply be filled up. The difficult stage is not `LLM inference` alone, but `context construction` before inference: which history should remain visible, which external information must be retrieved, which tool outputs should be preserved verbatim, and which state should stay outside the prompt and only be rendered into it when the current task view requires it [4-8]. What matters in practice is therefore not the wording of the prompt, but the policy that constructs it.

If we state that more strictly, the agent is not trying to maximize the amount of information at each step. It is trying to maximize the value of information for the next decision. What the system really wants to solve is closer to

$$
\max_{C_t} \Pr(a_t^\star \mid C_t)
\quad
\text{s.t.}
\quad
|C_t| \le B_t,
\quad
\mathrm{latency}(C_t) \le \tau_t
$$

where $a_t^\star$ can be read as the correct answer, action, or tool call for the current step. That makes the central objective explicit: context engineering does not optimize length itself. It optimizes the marginal contribution of the working set to next-step decision quality.

## 2. The Real Optimization Target Is Not Window Size, but Working-Set Quality

Once context is treated as a working set, it becomes easy to see why long context is not the fundamental solution. A larger window is valuable because it delays the capacity bottleneck and gives the system room to see longer history, larger documents, and more intermediate results in a single call. But it does not perform information management on its own. In the standard Transformer, compute and memory cost rise quickly with sequence length [1], and "being able to fit more tokens" has never meant "being able to use more tokens reliably." `Lost in the Middle` makes this point clearly: even models that nominally support long context can degrade sharply when the critical evidence sits in the middle of the prompt [2].

So the problem is not merely insufficient capacity. It is also a problem of effective information density, position bias, and attention allocation. In many cases, blindly enlarging the window only gives the system more room to insert noise. Results from LongLLMLingua, LLMLingua, and Selective Context show exactly this: for many long-context tasks, targeted compression or pruning can reduce cost and improve performance at the same time [3][9][10]. That is not a paradox. It is a normal consequence of working-set management: more information does not automatically become more useful information.

Context optimization is therefore multi-objective from the outset. A more realistic systems-level abstraction is

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

where $Q_t$ denotes step-level task quality, $F_t$ freshness, $R_t$ fidelity, $S_t$ stability and controllability, and $L_t$ and $K_t$ latency and cost. None of these dimensions can be ignored. A strategy that improves quality but explodes cost will not deploy. A strategy that aggressively compresses but damages fidelity will fail on critical details. A strategy that tries to "remember more" without control will feed stale state back into current decisions.

That is why the question "what does the current step actually need?" must always come before "what does the system know in total?" Knowledge-intensive QA systems care more about freshness, evidence relevance, and provenance. Code agents and data-analysis agents care more about the fidelity of tool output, consistency of execution state, and verbatim preservation of error messages. Long-horizon companion agents care more about memory policy, expiration, and cross-step stability. Different workloads assign different weights to these objectives, so there is no globally optimal context independent of task distribution and service budget. There are only context policies that are better for a specific workload, budget, and service objective.

This is also where the professional quality of context engineering shows up: it refuses to translate all state into natural language. Many systems still let the prompt carry prose instructions, knowledge text, execution logs, and control state at the same time. The advantage is interface uniformity. The cost is that semantic content and system state are forced through the same expensive channel. Whenever state could have remained structured but is instead rewritten into verbose text, the system is paying the highest price for the lowest controllability.

## 3. Retrieval, Compression, Memory, and Structured State Solve Different Problems

The most common mistake in discussing current methods is to treat them as interchangeable prompt tricks. They are not. They address different layers of the systems problem, so the right way to read them is as a division of labor rather than a checklist.

Retrieval addresses the mismatch between knowledge scale and window length. The value of RAG is not just that it "adds knowledge," but that it leaves most knowledge in external non-parametric storage and brings it back only when the current question needs it [4]. What it really achieves is decoupling: the knowledge base can keep growing without forcing the prompt to grow with it. But RAG does not eliminate selection. It only moves selection earlier, into retrieval itself. Query formulation, chunking, top-k, and multi-source fusion still determine what actually enters the working set.

Compression addresses the mismatch between information density and budget. LLMLingua, Selective Context, and LongLLMLingua are useful precisely because they acknowledge a basic fact: in many long prompts, what is expensive is not necessary information but redundant expression [3][9][10]. But compression is not a harmless cleanup step. It is an explicitly lossy encoding. Background explanation, casual dialogue history, and low-risk context are often good candidates for compression; code, contract clauses, stack traces, critical configuration, and exact fields in tool output often are not. Once the system replaces original text with a summary, it also accepts the risk of irreversible detail loss [11].

Memory architecture addresses the tension between cross-step persistence and immediate visibility. From Transformer-XL to Memorizing Transformers and then to episodic memory and reflexion at the agent layer, the recurring design idea is simple: not all history should remain resident in the prompt [5][6][12][13]. It should be layered and retrieved when needed. The hard part is not whether a memory store exists, but what the memory policy is: what deserves to be written, what should remain only as an index, what should expire, and what the system should trust when multiple memories conflict.

Structured state and tool-trace management address another problem that is still underestimated: not every execution trace should be pasted into the prompt as natural language. ReAct and Toolformer both show that an agent's critical context comes not only from external knowledge, but also from the execution process itself [7][8]. But appending full tool traces, raw outputs, and long logs to the prompt at every step is usually the least scalable choice. Better systems keep the full artifact outside the prompt and expose only the fields needed for the current step, a structured summary sufficient to recover execution state, and references that allow the raw result to be fetched again when needed.

So a more mature context architecture is not one that "writes prompts better." It is one that splits responsibilities cleanly: external knowledge and execution artifacts stay outside, retrieval handles recall, compression controls density, memory preserves useful cross-step experience, structured state carries control information that should never have been translated into prose, and the prompt is only the rendered result of that whole mechanism at one moment in time.

## 4. The Difficulty of Context Policy Is That It Is a Systems Problem of Persistent Failure and Trade-offs

The hard part is not whether long windows, retrieval, compression, or memory modules exist in isolation. It is that once they all enter the same system, their failure modes enter with them. In practice, the common failures are not mysterious. They usually fall into four types: the information that should have entered the working set never enters it, which is `omission`; the information in the prompt looks relevant but is already outdated, which is `staleness`; summarization or pruning damages critical detail, which is `distortion`; and relevant information is present but drowned by noise, which is `interference`. These failures transform into one another easily. Adding more context to avoid omission often creates interference. Compressing too aggressively to save cost introduces distortion. Retaining memory more aggressively to improve reuse eventually creates staleness.

That is why context policy is still driven largely by heuristics. Sliding windows, top-k retrieval, similarity thresholds, salience scores, and summary-length budgets all work in practice, but most do not answer the harder question with real precision: what is the marginal value of this specific piece of information for the current step? Most systems only know two things reliably from experience: too little information is bad, and too much information is also bad. The difficult part is the deployable boundary in between.

This is also why a context policy should never be judged by a single benchmark or a single accuracy number. A defensible evaluation needs at least three layers. First, there is the capability boundary: benchmarks such as LongBench, RULER, and ∞Bench are needed to check whether the model and the baseline policy can actually handle long-range dependencies at all [14-16]. But that only answers whether the model can "see" long context, not whether the system can use it correctly; HELMET shows that many popular synthetic long-context tests do not reliably predict downstream performance [17]. Second, there are systems metrics: task quality must be reported together with `p50/p95 latency`, token cost, retrieval cost, compression ratio, and fidelity loss, because any claim of improvement that ignores cost has no deployment meaning. Third, there is the layer agents actually care about: cross-step behavior, including multi-step task success rate, tool-call correctness, state consistency, and task completion per unit of service budget.

Only when all three layers improve together is it defensible to say that one context policy is better than another. Otherwise, it is too easy to mistake "a prompt trick that works on one test set" for "a systems improvement that generalizes." That is why context engineering remains stubbornly hard. Not because the field lacks techniques, but because the problem is inherently workload-dependent, budget-dependent, and service-dependent. It has no context-free answer, only engineering choices that move closer to the Pareto frontier for a given setting.

## Conclusion

If the prompt is understood only as "what we say to the model," then context engineering is easy to misread as an extension of prompt craft. Once context is treated as the runtime working set visible to the agent at the current step, the problem collapses back into its real systems shape. A larger context window only relaxes a capacity constraint. It does not replace information management. What actually determines agent quality is how information is selected, compressed, layered, organized, and evicted.

The stronger agents of the future may therefore be defined less by larger windows than by better working-set management systems. The key capability will not simply be putting more tokens in front of the model, but sustaining a better balance among quality, cost, latency, fidelity, freshness, and controllability while keeping the information that matters most to the current decision reliably in view.

## References

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
