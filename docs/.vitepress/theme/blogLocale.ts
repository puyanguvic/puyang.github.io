import { readonly, ref } from "vue";
import { BLOG_LOCALE_STORAGE_KEY, type BlogLocale } from "./blogData";

const locale = ref<BlogLocale>("zh");

function normalizeBlogLocale(value: string | null): BlogLocale | null {
  if (value === "zh" || value === "en") {
    return value;
  }

  return null;
}

function readStoredBlogLocale(): BlogLocale | null {
  if (typeof window === "undefined") {
    return null;
  }

  return normalizeBlogLocale(window.localStorage.getItem(BLOG_LOCALE_STORAGE_KEY));
}

function writeStoredBlogLocale(nextLocale: BlogLocale) {
  if (typeof window !== "undefined") {
    window.localStorage.setItem(BLOG_LOCALE_STORAGE_KEY, nextLocale);
  }
}

export function normalizeBlogPath(path: string) {
  if (!path) {
    return "/";
  }

  const withoutHash = path.replace(/[?#].*$/, "").replace(/\.html$/, "");

  if (withoutHash === "/") {
    return withoutHash;
  }

  return withoutHash.replace(/\/$/, "");
}

export function detectBlogLocaleFromPath(path: string): BlogLocale | null {
  const normalizedPath = normalizeBlogPath(path);

  if (!normalizedPath.startsWith("/blog")) {
    return null;
  }

  if (normalizedPath === "/blog") {
    return null;
  }

  return normalizedPath.endsWith("-en") ? "en" : "zh";
}

export function useBlogLocale() {
  function setLocale(nextLocale: BlogLocale) {
    locale.value = nextLocale;
    writeStoredBlogLocale(nextLocale);
  }

  function syncLocale(preferredLocale?: BlogLocale | null) {
    const nextLocale = preferredLocale ?? readStoredBlogLocale() ?? "zh";
    locale.value = nextLocale;
    writeStoredBlogLocale(nextLocale);
  }

  return {
    locale: readonly(locale),
    setLocale,
    syncLocale
  };
}
