<script setup lang="ts">
import { type BlogLocale as Locale } from "./blogData";
import { useBlogLocale } from "./blogLocale";

const localeOptions: Locale[] = ["zh", "en"];
const { setLocale, syncLocale } = useBlogLocale();

const props = defineProps<{
  currentLocale: Locale;
  zhPath: string;
  enPath: string;
}>();

const pageText = {
  label: {
    zh: "文章语言",
    en: "Article language"
  },
  switcher: {
    zh: "文章语言切换",
    en: "Article language switch"
  },
  localeLabel: {
    zh: {
      zh: "中文",
      en: "English"
    },
    en: {
      zh: "Chinese",
      en: "English"
    }
  }
} as const;

function getPath(locale: Locale) {
  return locale === "zh" ? props.zhPath : props.enPath;
}

function switchLocale(locale: Locale) {
  if (typeof window !== "undefined") {
    setLocale(locale);

    if (locale !== props.currentLocale) {
      window.location.assign(getPath(locale));
    }
  }
}

syncLocale(props.currentLocale);
</script>

<template>
  <div class="blog-post-locale-switch" :aria-label="pageText.switcher[currentLocale]" role="group">
    <span class="blog-post-locale-switch__label">{{ pageText.label[currentLocale] }}</span>
    <button
      v-for="option in localeOptions"
      :key="option"
      :aria-pressed="currentLocale === option"
      :class="[
        'blog-post-locale-switch__option',
        { 'blog-post-locale-switch__option--active': currentLocale === option }
      ]"
      type="button"
      @click="switchLocale(option)"
    >
      {{ pageText.localeLabel[currentLocale][option] }}
    </button>
  </div>
</template>

<style scoped>
.blog-post-locale-switch {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  margin: 0 0 1.25rem;
  padding: 0.35rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  background: var(--vp-c-bg-soft);
}

.blog-post-locale-switch__label {
  padding: 0 0.5rem 0 0.35rem;
  color: var(--vp-c-text-2);
  font-size: 0.92rem;
  font-weight: 600;
}

.blog-post-locale-switch__option {
  border: 0;
  border-radius: 999px;
  padding: 0.45rem 0.9rem;
  background: transparent;
  color: var(--vp-c-text-2);
  font: inherit;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s ease, color 0.2s ease;
}

.blog-post-locale-switch__option:hover {
  color: var(--vp-c-text-1);
}

.blog-post-locale-switch__option--active {
  background: var(--vp-c-brand-1);
  color: var(--vp-c-bg);
}

@media (max-width: 640px) {
  .blog-post-locale-switch {
    width: 100%;
    border-radius: 20px;
  }

  .blog-post-locale-switch__label {
    width: 100%;
    padding-bottom: 0.2rem;
  }

  .blog-post-locale-switch__option {
    flex: 1 1 0;
    text-align: center;
  }
}
</style>
