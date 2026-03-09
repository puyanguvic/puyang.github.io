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
