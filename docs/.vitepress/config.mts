import { defineConfig } from "vitepress";

const siteUrl = process.env.CUSTOM_DOMAIN
  ? `https://${process.env.CUSTOM_DOMAIN}`
  : "https://puyang.me";

export default defineConfig({
  title: "Pu Yang",
  description: "Personal website of Pu Yang",
  lastUpdated: true,
  head: [["link", { rel: "icon", href: "/favicon.ico" }]],
  ignoreDeadLinks: [
    /^mailto:/,
    /^https?:\/\//
  ],
  sitemap: {
    hostname: siteUrl
  },
  themeConfig: {
    logo: "/image/profile.png",
    nav: [
      { text: "Home", link: "/" },
      { text: "Blog", link: "/blog/" },
      { text: "Projects", link: "/projects/" },
      { text: "Publications", link: "/publications/" }
    ],
    search: {
      provider: "local"
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/puyanguvic" }
    ],
    footer: {
      message: "Built with VitePress",
      copyright: `Copyright ${new Date().getFullYear()} Pu Yang`
    },
    sidebar: {
      "/blog/": [
        {
          text: "Blog",
          items: [
            { text: "Overview", link: "/blog/" },
            { text: "Tokenizer的理论", link: "/blog/theory-of-tokenizers/" },
            { text: "Tokenizer 到底在做什么", link: "/blog/theory-of-tokenizers/what-tokenization-does" },
            { text: "为什么词表总在 50k 左右", link: "/blog/theory-of-tokenizers/why-vocab-size-stays-near-50k" },
            { text: "为什么 character-level 很少用", link: "/blog/theory-of-tokenizers/why-character-level-rarely-wins" },
            { text: "Transformer的几何结构", link: "/blog/geometry-of-transformers/" },
            { text: "Attention 其实在做什么", link: "/blog/geometry-of-transformers/what-attention-does" },
            { text: "为什么 Multi-head 如此重要", link: "/blog/geometry-of-transformers/why-multi-head-matters" },
            { text: "大模型的表示空间", link: "/blog/representation-space-of-large-models/" },
            { text: "语义为什么是线性的", link: "/blog/representation-space-of-large-models/semantic-linearity" },
            { text: "LLM embedding 的球面编码", link: "/blog/representation-space-of-large-models/spherical-coding" },
            { text: "高维空间与机器学习", link: "/blog/high-dimensional-space-and-machine-learning/" },
            { text: "距离会失效", link: "/blog/high-dimensional-space-and-machine-learning/distance-breakdown" },
            { text: "几乎总是正交", link: "/blog/high-dimensional-space-and-machine-learning/orthogonality" },
            { text: "几乎都在球面上", link: "/blog/high-dimensional-space-and-machine-learning/hypersphere" },
            { text: "Welcome to my blog", link: "/blog/welcome-to-my-blog" }
          ]
        }
      ],
      "/projects/": [
        {
          text: "Projects",
          items: [
            { text: "Overview", link: "/projects/" },
            { text: "Network Architecture for 6G Network", link: "/projects/proj1" },
            { text: "Wireless Avionics Intra-Communications System", link: "/projects/proj2" }
          ]
        }
      ],
      "/publications/": [
        {
          text: "Publications",
          items: [
            { text: "Overview", link: "/publications/" },
            { text: "Congestion-aware delay-guaranteed scheduling and routing", link: "/publications/pub1" },
            { text: "DGR: Delay-Guaranteed Routing Protocol", link: "/publications/pub2" },
            { text: "DDR: A Deadline-Driven Routing Protocol", link: "/publications/pub3" },
            { text: "ROMAM: An Intelligent Distributed Routing Protocol Architecture", link: "/publications/pub4" },
            { text: "Mixed-Numerology Channel Division for Wireless Avionics Intracommunications", link: "/publications/pub5" }
          ]
        }
      ],
      "/news/": [
        {
          text: "News",
          items: [
            { text: "Overview", link: "/news/" },
            { text: "Launched a new website", link: "/news/2026-01-20-launched-new-site" },
            { text: "Paper accepted", link: "/news/2025-12-05-paper-accepted" }
          ]
        }
      ]
    }
  }
});
