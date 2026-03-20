import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitepress";

const siteUrl = process.env.CUSTOM_DOMAIN
  ? `https://${process.env.CUSTOM_DOMAIN}`
  : "https://puyang.me";

const docsDir = fileURLToPath(new URL("..", import.meta.url));
const projectsDir = path.join(docsDir, "projects");

type ProjectEntry = {
  text: string;
  link: string;
  date: number;
};

function getProjectSidebarItems(): ProjectEntry[] {
  return readdirSync(projectsDir)
    .map((name) => {
      const projectDir = path.join(projectsDir, name);

      if (!statSync(projectDir).isDirectory()) {
        return null;
      }

      const indexPath = path.join(projectDir, "index.md");

      try {
        const source = readFileSync(indexPath, "utf8");
        const title = source.match(/^title:\s*["']?(.*?)["']?\s*$/m)?.[1];
        const draft = /^draft:\s*true\s*$/m.test(source);
        const dateValue = source.match(/^date:\s*(.*?)\s*$/m)?.[1];

        if (!title || draft) {
          return null;
        }

        return {
          text: title,
          link: `/projects/${name}/`,
          date: dateValue ? Date.parse(dateValue) : 0
        };
      } catch {
        return null;
      }
    })
    .filter((entry): entry is ProjectEntry => entry !== null)
    .sort((a, b) => b.date - a.date || a.text.localeCompare(b.text));
}

const projectSidebarItems = getProjectSidebarItems();

export default defineConfig({
  title: "Pu Yang",
  description: "Personal website of Pu Yang",
  lastUpdated: true,
  head: [["link", { rel: "icon", href: "/favicon.ico" }]],
  markdown: {
    math: true
  },
  ignoreDeadLinks: [
    /^mailto:/,
    /^https?:\/\//
  ],
  sitemap: {
    hostname: siteUrl
  },
  themeConfig: {
    logo: "/image/profile.jpg",
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
          text: "Home",
          link: "/blog/"
        }
      ],
      "/projects/": [
        {
          text: "Projects",
          items: [
            { text: "Overview", link: "/projects/" },
            ...projectSidebarItems.map(({ text, link }) => ({ text, link }))
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
