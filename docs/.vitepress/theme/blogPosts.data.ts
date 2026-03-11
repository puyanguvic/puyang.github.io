import { createContentLoader } from "vitepress";
import { buildBlogSeries, type BlogPageData } from "./blogData";

type BlogFrontmatter = {
  title?: unknown;
  date?: unknown;
  summary?: unknown;
  seriesZh?: unknown;
  seriesEn?: unknown;
  seriesDescriptionZh?: unknown;
  seriesDescriptionEn?: unknown;
};

function asString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function asDateString(value: unknown) {
  if (value instanceof Date) {
    return value.toISOString();
  }

  return asString(value);
}

function normalizeUrl(url: string) {
  if (!url) {
    return "/";
  }

  return url === "/" ? url : url.replace(/\/$/, "");
}

export default createContentLoader("blog/**/*.md", {
  transform(raw) {
    const pages: BlogPageData[] = raw
      .map((page) => {
        const frontmatter = (page.frontmatter ?? {}) as BlogFrontmatter;

        return {
          url: normalizeUrl(page.url),
          title: asString(frontmatter.title),
          date: asDateString(frontmatter.date),
          summary: asString(frontmatter.summary),
          seriesZh: asString(frontmatter.seriesZh),
          seriesEn: asString(frontmatter.seriesEn),
          seriesDescriptionZh: asString(frontmatter.seriesDescriptionZh),
          seriesDescriptionEn: asString(frontmatter.seriesDescriptionEn)
        };
      })
      .filter((page) => page.url !== "/blog" && page.title);

    return buildBlogSeries(pages);
  }
});
