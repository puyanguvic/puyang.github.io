<script setup lang="ts">
import { onMounted } from "vue";
import { blogSeries, type BlogLocale as Locale } from "./blogData";
import { useBlogLocale } from "./blogLocale";

const localeOptions: Locale[] = ["zh", "en"];
const { locale, setLocale, syncLocale } = useBlogLocale();

syncLocale();

const pageText = {
  title: {
    zh: "Blog",
    en: "Blog"
  },
  lead: {
    zh: "按专题浏览。本页收录的文章已经按系列长文统一修订；每组都依次处理问题定义、机制分析、边界条件与应用含义，适合连续阅读。",
    en: "Browse by series. The posts listed here have been revised into long-form sequences; each group moves from the problem setup to mechanism analysis, boundary conditions, and practical implications."
  },
  note: {
    zh: "每篇文章都提供中英文版本；从本页进入正文时，会按当前语言直接跳转到对应版本。",
    en: "Each post is now available in Chinese and English; links from this overview open the matching version for the current language."
  },
  switcher: {
    zh: "Blog 语言切换",
    en: "Blog language switch"
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

onMounted(() => {
  syncLocale();
});
</script>

<template>
  <section class="blog-overview">
    <div class="blog-overview__switch" :aria-label="pageText.switcher[locale]" role="group">
      <button
        v-for="option in localeOptions"
        :key="option"
        :aria-pressed="locale === option"
        :class="[
          'blog-overview__switch-option',
          { 'blog-overview__switch-option--active': locale === option }
        ]"
        type="button"
        @click="setLocale(option)"
      >
        {{ pageText.localeLabel[locale][option] }}
      </button>
    </div>

    <h1>{{ pageText.title[locale] }}</h1>
    <p class="blog-overview__lead">{{ pageText.lead[locale] }}</p>
    <p class="blog-overview__note">{{ pageText.note[locale] }}</p>

    <details v-for="item in blogSeries" :key="item.title.en" class="blog-overview__series">
      <summary class="blog-overview__summary">
        <span class="blog-overview__date">{{ item.date[locale] }}</span>
        <span class="blog-overview__series-title">{{ item.title[locale] }}</span>
      </summary>

      <div class="blog-overview__series-body">
        <p>{{ item.description[locale] }}</p>
        <ol>
          <li v-for="post in item.posts" :key="post.href.en">
            <a :href="post.href[locale]">{{ post.title[locale] }}</a>
            {{ " " }}
            <span>{{ post.summary[locale] }}</span>
          </li>
        </ol>
      </div>
    </details>
  </section>
</template>

<style scoped>
.blog-overview {
  margin-top: 0.25rem;
}

.blog-overview__switch {
  display: inline-flex;
  gap: 0.35rem;
  padding: 0.28rem;
  margin-bottom: 1.2rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  background: var(--vp-c-bg-soft);
}

.blog-overview__switch-option {
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

.blog-overview__switch-option:hover {
  color: var(--vp-c-text-1);
}

.blog-overview__switch-option--active {
  background: var(--vp-c-brand-1);
  color: var(--vp-c-bg);
}

.blog-overview__lead {
  margin-bottom: 0.8rem;
}

.blog-overview__note {
  margin: 0 0 1.4rem;
  color: var(--vp-c-text-2);
  font-size: 0.96rem;
}

.blog-overview__series {
  margin: 0 0 0.95rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--vp-c-bg-soft);
  overflow: hidden;
}

.blog-overview__summary {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: baseline;
  padding: 1rem 1.1rem;
  cursor: pointer;
  list-style: none;
}

.blog-overview__summary::-webkit-details-marker {
  display: none;
}

.blog-overview__date {
  color: var(--vp-c-text-2);
  font-size: 0.95rem;
}

.blog-overview__series-title {
  color: var(--vp-c-text-1);
  font-weight: 700;
}

.blog-overview__series-body {
  padding: 0 1.1rem 1.1rem;
}

.blog-overview__series-body p {
  margin-top: 0;
}

.blog-overview__series-body ol {
  margin: 0.75rem 0 0;
  padding-left: 1.2rem;
}

.blog-overview__series-body li + li {
  margin-top: 0.7rem;
}

@media (max-width: 640px) {
  .blog-overview__switch {
    width: 100%;
    justify-content: space-between;
  }

  .blog-overview__switch-option {
    flex: 1 1 0;
    text-align: center;
  }
}
</style>
