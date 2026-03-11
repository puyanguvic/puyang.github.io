import { createContentLoader } from "vitepress";

type PublicationFrontmatter = {
  title?: unknown;
  authors?: unknown;
  date?: unknown;
  publication?: unknown;
  summary?: unknown;
  pdf?: unknown;
};

export type PublicationItem = {
  url: string;
  title: string;
  authors: string[];
  date: string;
  year: string;
  publication: string;
  summary: string;
  pdf: string;
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

function normalizeAuthors(value: unknown) {
  if (Array.isArray(value)) {
    return value.flatMap((item) =>
      typeof item === "string"
        ? item
            .split(",")
            .map((author) => author.trim())
            .filter(Boolean)
        : []
    );
  }

  if (typeof value === "string") {
    return value
      .split(",")
      .map((author) => author.trim())
      .filter(Boolean);
  }

  return [];
}

function getYear(date: string) {
  const year = new Date(date).getUTCFullYear();
  return Number.isNaN(year) ? "" : String(year);
}

function toEpochMs(date: string) {
  const timestamp = Date.parse(date);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

export default createContentLoader("publications/*.md", {
  transform(raw) {
    return raw
      .map((page) => {
        const frontmatter = (page.frontmatter ?? {}) as PublicationFrontmatter;
        const date = asDateString(frontmatter.date);

        return {
          url: normalizeUrl(page.url),
          title: asString(frontmatter.title),
          authors: normalizeAuthors(frontmatter.authors),
          date,
          year: getYear(date),
          publication: asString(frontmatter.publication),
          summary: asString(frontmatter.summary),
          pdf: asString(frontmatter.pdf)
        } satisfies PublicationItem;
      })
      .filter((page) => page.url !== "/publications" && page.title)
      .sort((left, right) => toEpochMs(right.date) - toEpochMs(left.date) || left.title.localeCompare(right.title));
  }
});
