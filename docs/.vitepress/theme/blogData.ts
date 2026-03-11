export type BlogLocale = "zh" | "en";

export type LocalizedText = Record<BlogLocale, string>;

export type BlogPageData = {
  url: string;
  title: string;
  date: string;
  summary: string;
  seriesZh?: string;
  seriesEn?: string;
  seriesDescriptionZh?: string;
  seriesDescriptionEn?: string;
};

export type BlogPost = {
  key: string;
  date: string;
  href: LocalizedText;
  title: LocalizedText;
  summary: LocalizedText;
};

export type BlogSeries = {
  key: string;
  date: string;
  title: LocalizedText;
  description: LocalizedText;
  posts: BlogPost[];
};

type PartialLocalizedText = Partial<Record<BlogLocale, string>>;

type BlogSidebarSection = {
  text: string;
  link?: string;
  items?: Array<{
    text: string;
    link: string;
  }>;
};

type ParsedBlogUrl = {
  canonicalKey: string;
  seriesKey: string;
  locale: BlogLocale;
};

export const BLOG_LOCALE_STORAGE_KEY = "blog-language";

const GENERAL_SERIES_KEY = "general";

const LEGACY_SERIES_METADATA: Record<
  string,
  {
    title: PartialLocalizedText;
    description: PartialLocalizedText;
  }
> = {
  engineering_system_view: {
    title: {
      zh: "工程和系统视角",
      en: "Engineering and Systems Perspectives"
    },
    description: {
      zh: "本组文章从真实训练瓶颈出发，讨论大模型为什么会一步步从单卡走向参数分片、张量并行、流水线并行与通信优化。重点不是背某个框架名词，而是看系统在什么限制下被迫重新分工。",
      en: "This series starts from real training bottlenecks and explains why large-model training evolves from single-GPU execution toward state sharding, tensor parallelism, pipeline schedules, and communication-aware optimization. The point is not to memorize framework names, but to see which system constraint forces each step."
    }
  },
  "high-dimensional-space-and-machine-learning": {
    title: {
      zh: "高维空间与机器学习",
      en: "High-Dimensional Space and Machine Learning"
    },
    description: {
      zh: "本组文章从最基础的高维几何出发，说明为什么机器学习不能直接把原始欧氏空间当成语义空间来用。主线是：距离在高维中先失去分辨率，方向结构随后成为更稳定的几何信号，而训练后的表示又进一步被压到近似球面上。",
      en: "This series starts from basic high-dimensional geometry and explains why machine learning cannot treat the raw Euclidean space as a semantic space. The main arc is that distances lose resolution first, directional structure becomes the more stable signal, and trained representations are then pushed toward an approximately spherical shell."
    }
  },
  "representation-space-of-large-models": {
    title: {
      zh: "大模型的表示空间",
      en: "Representation Space of Large Models"
    },
    description: {
      zh: "本组文章讨论 LLM 词表与 embedding 空间的组织原则。核心问题不是“向量会不会做算术”，而是训练目标如何把重复关系压缩为稳定方向，以及高维词表为什么更适合被理解为一个受语义约束的球面码本。",
      en: "This series examines how LLM vocabularies and embedding spaces are organized. The core question is not whether vectors can do arithmetic, but how training compresses recurring relations into stable directions and why a large vocabulary is better viewed as a semantically constrained spherical codebook."
    }
  },
  "geometry-of-transformers": {
    title: {
      zh: "Transformer 的几何结构",
      en: "The Geometry of Transformers"
    },
    description: {
      zh: "本组文章把 Transformer 的核心算子写回几何语言。attention 不是“简单加权平均”，而是由 query-key 几何诱导出的软坐标系统；多头注意力也不是重复运算，而是并行构造多套不同的上下文坐标系。",
      en: "This series rewrites the core Transformer operators in geometric terms. Attention is not a simple weighted average, but a soft coordinate system induced by query-key geometry; multi-head attention is not repetition, but parallel construction of different contextual coordinate systems."
    }
  },
  "theory-of-tokenizers": {
    title: {
      zh: "Tokenizer 的理论",
      en: "A Theory of Tokenizers"
    },
    description: {
      zh: "本组文章讨论 tokenizer 在整个 LLM 系统中的角色。论证主线是：tokenization 首先是码本压缩问题；词表规模因此存在自然平衡点；而所谓 token-free 路线并没有取消压缩，只是把压缩移入了模型内部。",
      en: "This series studies the role of tokenizers in the full LLM system. The main claim is that tokenization is first a codebook-compression problem; vocabulary size therefore has a natural balance point; and so-called token-free approaches do not remove compression, but move it inside the model."
    }
  },
  [GENERAL_SERIES_KEY]: {
    title: {
      zh: "未分组文章",
      en: "General"
    },
    description: {
      zh: "",
      en: ""
    }
  }
};

function normalizeText(value: string | undefined) {
  return value?.trim() ?? "";
}

function fillLocalizedText(
  source: PartialLocalizedText,
  fallback: PartialLocalizedText = {},
  finalFallback = ""
): LocalizedText {
  const zh = normalizeText(source.zh) || normalizeText(fallback.zh) || normalizeText(source.en) || normalizeText(fallback.en) || finalFallback;
  const en = normalizeText(source.en) || normalizeText(fallback.en) || normalizeText(source.zh) || normalizeText(fallback.zh) || finalFallback;

  return { zh, en };
}

function toEpochMs(date: string) {
  const timestamp = Date.parse(date);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

function compareDatesAsc(left: string, right: string) {
  return toEpochMs(left) - toEpochMs(right);
}

function compareDatesDesc(left: string, right: string) {
  return toEpochMs(right) - toEpochMs(left);
}

function prettifySeriesKey(seriesKey: string) {
  return seriesKey
    .split(/[-_]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function parseBlogUrl(url: string): ParsedBlogUrl | null {
  const normalizedUrl = url.replace(/\/$/, "").replace(/\.html$/, "");

  if (!normalizedUrl || normalizedUrl === "/blog") {
    return null;
  }

  const segments = normalizedUrl.replace(/^\//, "").split("/");

  if (segments[0] !== "blog" || segments.length < 2) {
    return null;
  }

  const articleSlug = segments[segments.length - 1];
  const locale = articleSlug.endsWith("-en") ? "en" : "zh";
  const canonicalSlug = locale === "en" ? articleSlug.slice(0, -3) : articleSlug;
  const articleSegments = segments.slice(1, -1).concat(canonicalSlug);
  const seriesKey = segments.length > 2 ? segments[1] : GENERAL_SERIES_KEY;

  return {
    canonicalKey: articleSegments.join("/"),
    seriesKey,
    locale
  };
}

function buildSeriesMetadata(seriesKey: string, pages: BlogPageData[]) {
  const titlesFromFrontmatter: PartialLocalizedText = {};
  const descriptionsFromFrontmatter: PartialLocalizedText = {};

  for (const page of pages) {
    if (!titlesFromFrontmatter.zh && normalizeText(page.seriesZh)) {
      titlesFromFrontmatter.zh = page.seriesZh;
    }

    if (!titlesFromFrontmatter.en && normalizeText(page.seriesEn)) {
      titlesFromFrontmatter.en = page.seriesEn;
    }

    if (!descriptionsFromFrontmatter.zh && normalizeText(page.seriesDescriptionZh)) {
      descriptionsFromFrontmatter.zh = page.seriesDescriptionZh;
    }

    if (!descriptionsFromFrontmatter.en && normalizeText(page.seriesDescriptionEn)) {
      descriptionsFromFrontmatter.en = page.seriesDescriptionEn;
    }
  }

  const legacyMetadata = LEGACY_SERIES_METADATA[seriesKey];
  const fallbackTitle = prettifySeriesKey(seriesKey);

  return {
    title: fillLocalizedText(titlesFromFrontmatter, legacyMetadata?.title, fallbackTitle),
    description: fillLocalizedText(descriptionsFromFrontmatter, legacyMetadata?.description, "")
  };
}

export function buildBlogSeries(pages: BlogPageData[]): BlogSeries[] {
  const filteredPages = pages.filter((page) => page.url && page.title);
  const pagesBySeries = new Map<string, BlogPageData[]>();
  const postsByKey = new Map<
    string,
    {
      seriesKey: string;
      date: string;
      href: PartialLocalizedText;
      title: PartialLocalizedText;
      summary: PartialLocalizedText;
    }
  >();

  for (const page of filteredPages) {
    const parsed = parseBlogUrl(page.url);

    if (!parsed) {
      continue;
    }

    const seriesPages = pagesBySeries.get(parsed.seriesKey) ?? [];
    seriesPages.push(page);
    pagesBySeries.set(parsed.seriesKey, seriesPages);

    const existingPost = postsByKey.get(parsed.canonicalKey) ?? {
      seriesKey: parsed.seriesKey,
      date: page.date,
      href: {},
      title: {},
      summary: {}
    };

    if (!existingPost.date || compareDatesDesc(page.date, existingPost.date) < 0) {
      existingPost.date = page.date;
    }

    existingPost.href[parsed.locale] = page.url;
    existingPost.title[parsed.locale] = page.title;
    existingPost.summary[parsed.locale] = page.summary;
    postsByKey.set(parsed.canonicalKey, existingPost);
  }

  const postsBySeries = new Map<string, BlogPost[]>();

  for (const [canonicalKey, post] of postsByKey.entries()) {
    const canonicalUrl = `/blog/${canonicalKey}`;
    const seriesPosts = postsBySeries.get(post.seriesKey) ?? [];

    seriesPosts.push({
      key: canonicalKey,
      date: post.date,
      href: fillLocalizedText(post.href, {}, canonicalUrl),
      title: fillLocalizedText(post.title, {}, canonicalKey),
      summary: fillLocalizedText(post.summary, {}, "")
    });

    postsBySeries.set(post.seriesKey, seriesPosts);
  }

  const seriesList: BlogSeries[] = [];

  for (const [seriesKey, seriesPosts] of postsBySeries.entries()) {
    const seriesPages = pagesBySeries.get(seriesKey) ?? [];
    const metadata = buildSeriesMetadata(seriesKey, seriesPages);
    const sortedPosts = [...seriesPosts].sort((left, right) => compareDatesAsc(left.date, right.date));
    const latestDate = sortedPosts.reduce((currentLatest, post) => (
      compareDatesDesc(post.date, currentLatest) < 0 ? post.date : currentLatest
    ), sortedPosts[0]?.date ?? "");

    seriesList.push({
      key: seriesKey,
      date: latestDate,
      title: metadata.title,
      description: metadata.description,
      posts: sortedPosts
    });
  }

  return seriesList.sort((left, right) => compareDatesDesc(left.date, right.date));
}

export function buildBlogSidebarSections(seriesList: BlogSeries[], locale: BlogLocale): BlogSidebarSection[] {
  const overviewText = locale === "zh" ? "总览" : "Overview";

  return [
    { text: overviewText, link: "/blog/" },
    ...seriesList.map((series) => ({
      text: series.title[locale],
      items: series.posts.map((post) => ({
        text: post.title[locale],
        link: post.href[locale]
      }))
    }))
  ];
}

export function formatBlogDate(date: string, locale: BlogLocale) {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(date);

  if (!match) {
    return date;
  }

  const [, year, month, day] = match;
  const utcDate = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day)));
  const formatLocale = locale === "zh" ? "zh-CN" : "en-US";
  const formatOptions: Intl.DateTimeFormatOptions = locale === "zh"
    ? { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" }
    : { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" };

  return new Intl.DateTimeFormat(formatLocale, formatOptions).format(utcDate);
}

export function formatPostCount(count: number, locale: BlogLocale) {
  if (locale === "zh") {
    return `${count} 篇文章`;
  }

  return count === 1 ? "1 article" : `${count} articles`;
}
