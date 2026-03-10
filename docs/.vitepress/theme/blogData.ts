export type BlogLocale = "zh" | "en";

export type LocalizedText = Record<BlogLocale, string>;

export type BlogPost = {
  href: Record<BlogLocale, string>;
  title: LocalizedText;
  summary: LocalizedText;
};

export type BlogSeries = {
  date: LocalizedText;
  title: LocalizedText;
  description: LocalizedText;
  posts: BlogPost[];
};

type BlogSidebarSection = {
  text: string;
  link?: string;
  items?: Array<{
    text: string;
    link: string;
  }>;
};

export const BLOG_LOCALE_STORAGE_KEY = "blog-language";

export const blogSeries: BlogSeries[] = [
  {
    date: {
      zh: "2026年3月9日",
      en: "Mar 9, 2026"
    },
    title: {
      zh: "高维空间与机器学习",
      en: "High-Dimensional Space and Machine Learning"
    },
    description: {
      zh: "本组文章从最基础的高维几何出发，说明为什么机器学习不能直接把原始欧氏空间当成语义空间来用。主线是：距离在高维中先失去分辨率，方向结构随后成为更稳定的几何信号，而训练后的表示又进一步被压到近似球面上。",
      en: "This series starts from basic high-dimensional geometry and explains why machine learning cannot treat the raw Euclidean space as a semantic space. The main arc is that distances lose resolution first, directional structure becomes the more stable signal, and trained representations are then pushed toward an approximately spherical shell."
    },
    posts: [
      {
        href: {
          zh: "/blog/high-dimensional-space-and-machine-learning/distance-breakdown",
          en: "/blog/high-dimensional-space-and-machine-learning/distance-breakdown-en"
        },
        title: {
          zh: "高维空间中的距离集中与度量失效",
          en: "Distance Concentration and Metric Breakdown in High Dimensions"
        },
        summary: {
          zh: "先形式化“距离失效”究竟失效在哪里，再从薄壳现象与极值收缩解释为什么最近邻与最远邻会逐渐难以区分。",
          en: "It first makes precise what exactly breaks when distance stops being useful, then uses the thin-shell effect and extreme-value shrinkage to explain why nearest and farthest neighbors become harder to distinguish."
        }
      },
      {
        href: {
          zh: "/blog/high-dimensional-space-and-machine-learning/orthogonality",
          en: "/blog/high-dimensional-space-and-machine-learning/orthogonality-en"
        },
        title: {
          zh: "高维向量近似正交的几何机制",
          en: "Why High-Dimensional Vectors Become Nearly Orthogonal"
        },
        summary: {
          zh: "说明在范数已集中之后，角度为何比长度更稳定，以及高维球面为什么能容纳大量彼此低相关的方向。",
          en: "It shows why angles become more stable than lengths once norms concentrate, and why high-dimensional spheres can hold many weakly correlated directions."
        }
      },
      {
        href: {
          zh: "/blog/high-dimensional-space-and-machine-learning/hypersphere",
          en: "/blog/high-dimensional-space-and-machine-learning/hypersphere-en"
        },
        title: {
          zh: "Embedding 向量的超球面分布及其成因",
          en: "Why Embeddings Gather on an Approximate Hypersphere"
        },
        summary: {
          zh: "将前两篇与现代表示学习目标结合起来，解释 embedding 为什么常呈现近似球壳分布，以及余弦相似度为何更自然。",
          en: "It connects the first two pieces to modern representation-learning objectives and explains why embeddings often lie on an approximate shell and why cosine similarity becomes the natural metric."
        }
      }
    ]
  },
  {
    date: {
      zh: "2026年3月9日",
      en: "Mar 9, 2026"
    },
    title: {
      zh: "大模型的表示空间",
      en: "Representation Space of Large Models"
    },
    description: {
      zh: "本组文章讨论 LLM 词表与 embedding 空间的组织原则。核心问题不是“向量会不会做算术”，而是训练目标如何把重复关系压缩为稳定方向，以及高维词表为什么更适合被理解为一个受语义约束的球面码本。",
      en: "This series examines how LLM vocabularies and embedding spaces are organized. The core question is not whether vectors can do arithmetic, but how training compresses recurring relations into stable directions and why a large vocabulary is better viewed as a semantically constrained spherical codebook."
    },
    posts: [
      {
        href: {
          zh: "/blog/representation-space-of-large-models/semantic-linearity",
          en: "/blog/representation-space-of-large-models/semantic-linearity-en"
        },
        title: {
          zh: "Embedding 空间中的语义线性结构",
          en: "Semantic Linearity in Embedding Space"
        },
        summary: {
          zh: "从类比现象回到 PMI 因子分解，解释语义线性何以出现、为何只在局部稳定，以及为什么它在上下文化表示中会减弱。",
          en: "It traces the familiar analogy phenomenon back to PMI factorization, explaining why semantic linearity appears, why it is only locally stable, and why it weakens in contextual representations."
        }
      },
      {
        href: {
          zh: "/blog/representation-space-of-large-models/spherical-coding",
          en: "/blog/representation-space-of-large-models/spherical-coding-en"
        },
        title: {
          zh: "LLM Embedding 的球面编码视角",
          en: "A Spherical-Coding View of LLM Embeddings"
        },
        summary: {
          zh: "从球面码、coherence 与 Welch 下界出发，说明有限维 embedding 为什么足以容纳巨型词表，以及这种视角能解释什么、不能解释什么。",
          en: "Using spherical codes, coherence, and the Welch bound, it shows why finite-dimensional embeddings can still host very large vocabularies and clarifies what this perspective can and cannot explain."
        }
      }
    ]
  },
  {
    date: {
      zh: "2026年3月9日",
      en: "Mar 9, 2026"
    },
    title: {
      zh: "Transformer 的几何结构",
      en: "The Geometry of Transformers"
    },
    description: {
      zh: "本组文章把 Transformer 的核心算子写回几何语言。attention 不是“简单加权平均”，而是由 query-key 几何诱导出的软坐标系统；多头注意力也不是重复运算，而是并行构造多套不同的上下文坐标系。",
      en: "This series rewrites the core Transformer operators in geometric terms. Attention is not a simple weighted average, but a soft coordinate system induced by query-key geometry; multi-head attention is not repetition, but parallel construction of different contextual coordinate systems."
    },
    posts: [
      {
        href: {
          zh: "/blog/geometry-of-transformers/what-attention-does",
          en: "/blog/geometry-of-transformers/what-attention-does-en"
        },
        title: {
          zh: "Transformer Attention 的几何本质",
          en: "The Geometric Core of Transformer Attention"
        },
        summary: {
          zh: "从双线性匹配、概率单纯形与重心重建出发，解释 attention 如何完成上下文相关的读取与重写。",
          en: "Starting from bilinear matching, the probability simplex, and barycentric reconstruction, it explains how attention performs context-dependent reading and rewriting."
        }
      },
      {
        href: {
          zh: "/blog/geometry-of-transformers/why-multi-head-matters",
          en: "/blog/geometry-of-transformers/why-multi-head-matters-en"
        },
        title: {
          zh: "Multi-Head Attention 的必要性与表达优势",
          en: "Why Multi-Head Attention Matters"
        },
        summary: {
          zh: "从单头注意力的几何瓶颈出发，说明为什么关系解耦、内容解耦与并行计算都需要多头结构。",
          en: "It begins with the geometric bottlenecks of single-head attention and shows why relation disentangling, content disentangling, and parallel computation all rely on multi-head structure."
        }
      }
    ]
  },
  {
    date: {
      zh: "2026年3月9日",
      en: "Mar 9, 2026"
    },
    title: {
      zh: "Tokenizer 的理论",
      en: "A Theory of Tokenizers"
    },
    description: {
      zh: "本组文章讨论 tokenizer 在整个 LLM 系统中的角色。论证主线是：tokenization 首先是码本压缩问题；词表规模因此存在自然平衡点；而所谓 token-free 路线并没有取消压缩，只是把压缩移入了模型内部。",
      en: "This series studies the role of tokenizers in the full LLM system. The main claim is that tokenization is first a codebook-compression problem; vocabulary size therefore has a natural balance point; and so-called token-free approaches do not remove compression, but move it inside the model."
    },
    posts: [
      {
        href: {
          zh: "/blog/theory-of-tokenizers/what-tokenization-does",
          en: "/blog/theory-of-tokenizers/what-tokenization-does-en"
        },
        title: {
          zh: "Tokenization 的压缩本质",
          en: "What Tokenization Really Does"
        },
        summary: {
          zh: "说明 tokenizer 为什么应被理解为有限可逆码本，以及它如何利用语言分布的统计偏斜来缩短序列并降低学习负担。",
          en: "It explains why a tokenizer should be understood as a finite reversible codebook, and how it exploits statistical skew in language to shorten sequences and reduce the learning burden."
        }
      },
      {
        href: {
          zh: "/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k",
          en: "/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k-en"
        },
        title: {
          zh: "LLM 词表规模的自然平衡点",
          en: "Why Vocabulary Size Stays Near 50k"
        },
        summary: {
          zh: "从序列长度收益递减、长尾稀疏和输出层成本三方面解释词表为什么不会无限扩张。",
          en: "It explains from diminishing sequence-length returns, long-tail sparsity, and output-layer cost why the vocabulary does not expand without bound."
        }
      },
      {
        href: {
          zh: "/blog/theory-of-tokenizers/why-character-level-rarely-wins",
          en: "/blog/theory-of-tokenizers/why-character-level-rarely-wins-en"
        },
        title: {
          zh: "Character-Level Tokenizer 的理论优势与工程局限",
          en: "Why Character-Level Tokenizers Rarely Win"
        },
        summary: {
          zh: "说明字符级方案为何在表示上统一、在系统上却常常更贵，以及为什么成功的 token-free 模型往往仍会重新引入内部压缩。",
          en: "It shows why character-level schemes are elegant at the representation level yet often more expensive as systems, and why successful token-free models usually reintroduce internal compression."
        }
      }
    ]
  }
];

export function buildBlogSidebarSections(locale: BlogLocale): BlogSidebarSection[] {
  const overviewText = locale === "zh" ? "总览" : "Overview";

  return [
    { text: overviewText, link: "/blog/" },
    ...blogSeries.map((series) => ({
      text: series.title[locale],
      items: series.posts.map((post) => ({
        text: post.title[locale],
        link: post.href[locale]
      }))
    }))
  ];
}

export function buildThemeBlogSidebar() {
  return [
    { text: "Overview", link: "/blog/" },
    ...(["zh", "en"] as BlogLocale[]).flatMap((locale) =>
      blogSeries.map((series) => ({
        text: series.title[locale],
        collapsed: true,
        items: series.posts.map((post) => ({
          text: post.title[locale],
          link: post.href[locale]
        }))
      }))
    )
  ];
}
