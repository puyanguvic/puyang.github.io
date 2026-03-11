<script setup lang="ts">
import { onMounted } from "vue";
import { VPLink } from "vitepress/theme";
import { type BlogLocale as Locale, formatBlogDate, formatPostCount } from "./blogData";
import { useBlogLocale } from "./blogLocale";
import { data as blogSeries } from "./blogPosts.data";

const localeOptions: Locale[] = ["zh", "en"];
const { locale, setLocale, syncLocale } = useBlogLocale();

syncLocale();

const pageText = {
  eyebrow: {
    zh: "写在前面",
    en: "Before We Begin"
  },
  title: {
    zh: "我为什么写这个博客",
    en: "Why I Write This Blog"
  },
  lead: {
    zh: "我是工业界的 AI 研究员，在计算机行业学习和工作已十几年。这个博客不是一份目录，也不是资讯流；它更像一份持续更新的技术笔记，用来记录我对 AI、系统、模型与现实世界的理解。",
    en: "I am an AI researcher in industry, and I have spent more than a decade studying and working in computing. This blog is not meant to be a catalog or a news feed. It is a running technical notebook for how I think about AI, systems, models, and the world they are entering."
  },
  manifesto: {
    zh: "在我看来，AI 正在推动一场极其猛烈的变革。我甚至愿意把它看作第四次工业革命的一部分。它是前所未有的生产工具，值得我们去理解、去使用；但它也像一场来势很猛的洪水，把安全、社会治理、劳动结构和利益分配的问题一并推到眼前。",
    en: "In my view, AI is driving a transformation of unusual force. I would go as far as to call it part of a fourth industrial revolution. It is an extraordinarily powerful tool for production, and we need to understand it and use it. But it also arrives like a flood, bringing difficult questions about safety, governance, labor, and distribution."
  },
  sections: {
    zh: [
      {
        title: "为什么现在要认真理解 AI",
        body: "这一次的变化，不只是自动化程度更高，也不只是软件能力更强。AI 正在以越来越接近“智能”的方式进入现实世界。它改变的不只是某个行业的效率，而是人如何工作、组织如何决策、社会如何分配能力与机会。"
      },
      {
        title: "我会在这里写什么",
        body: "我关心的不只是模型怎么更强，也关心它为什么有效、在什么边界下失效，以及它最终会把工程系统和现实世界带向哪里。这里会写大模型、表示学习、高维几何、训练系统、Agent、AI 安全，以及这些问题落到实际工程之后的约束与代价。"
      },
      {
        title: "这个博客想做什么",
        body: "我创建这个博客，是为了记录工作中的技术心得、长期思考和一些必须慢慢展开的问题。我希望把复杂问题讲清楚，也希望这些文字能帮助更多人在这场 AI 风暴中更平稳地度过。欢迎交流。"
      }
    ],
    en: [
      {
        title: "Why AI Needs Serious Attention Now",
        body: "This shift is not only about stronger software or more automation. AI is entering the real world in ways that increasingly resemble intelligence. It changes not just efficiency inside one industry, but how people work, how organizations decide, and how societies distribute capability and opportunity."
      },
      {
        title: "What I Write About Here",
        body: "I am interested not only in how to make models stronger, but in why they work, where they fail, and what they do to real systems once deployed. This blog focuses on large models, representation learning, high-dimensional geometry, training systems, agents, AI safety, and the constraints and costs that appear in real engineering."
      },
      {
        title: "What This Blog Is For",
        body: "I created this blog to record technical lessons from work, long-form reasoning, and questions that need to be unfolded carefully rather than rushed. If these notes help more people move through the AI storm with steadier footing, then the blog is doing its job. Conversation is welcome."
      }
    ]
  },
  interestsTitle: {
    zh: "当前技术兴趣",
    en: "Current Technical Interests"
  },
  interests: {
    zh: [
      "大模型与表示学习",
      "高维几何",
      "训练系统与并行",
      "Agent 与工具使用",
      "AI 安全",
      "社会影响与治理"
    ],
    en: [
      "LLMs and representation learning",
      "High-dimensional geometry",
      "Training systems and parallelism",
      "Agents and tool use",
      "AI safety",
      "Social impact and governance"
    ]
  },
  catalogEyebrow: {
    zh: "自动归档",
    en: "Auto Index"
  },
  catalogTitle: {
    zh: "文章目录",
    en: "Article Catalog"
  },
  catalogLead: {
    zh: "这里的系列和文章会直接从 `docs/blog` 目录与 markdown frontmatter 自动生成。以后新增文章，不需要再手工改博客界面代码。",
    en: "The series and posts below are generated directly from the `docs/blog` directory and markdown frontmatter. New posts no longer require manual updates to the blog UI code."
  },
  empty: {
    zh: "还没有检测到可展示的文章。",
    en: "No posts were detected yet."
  },
  closing: {
    zh: "理解它，使用它，也警惕它带来的代价。这大概就是我写这个博客的全部动机。",
    en: "Understand it. Use it. Stay alert to the costs it brings. That is the whole reason this blog exists."
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

    <div class="blog-overview__hero">
      <p class="blog-overview__eyebrow">{{ pageText.eyebrow[locale] }}</p>
      <h1>{{ pageText.title[locale] }}</h1>
      <p class="blog-overview__lead">{{ pageText.lead[locale] }}</p>
      <p class="blog-overview__manifesto">{{ pageText.manifesto[locale] }}</p>
    </div>

    <div class="blog-overview__grid">
      <article
        v-for="section in pageText.sections[locale]"
        :key="section.title"
        class="blog-overview__card"
      >
        <h2>{{ section.title }}</h2>
        <p>{{ section.body }}</p>
      </article>
    </div>

    <div class="blog-overview__interests">
      <h2>{{ pageText.interestsTitle[locale] }}</h2>
      <ul class="blog-overview__tags">
        <li v-for="item in pageText.interests[locale]" :key="item">
          {{ item }}
        </li>
      </ul>
    </div>

    <section class="blog-overview__catalog">
      <div class="blog-overview__catalog-header">
        <p class="blog-overview__eyebrow">{{ pageText.catalogEyebrow[locale] }}</p>
        <h2>{{ pageText.catalogTitle[locale] }}</h2>
        <p>{{ pageText.catalogLead[locale] }}</p>
      </div>

      <div v-if="blogSeries.length" class="blog-overview__series-list">
        <article v-for="series in blogSeries" :key="series.key" class="blog-overview__series">
          <div class="blog-overview__series-head">
            <div>
              <p class="blog-overview__series-date">{{ formatBlogDate(series.date, locale) }}</p>
              <h3>{{ series.title[locale] }}</h3>
            </div>
            <p class="blog-overview__series-count">{{ formatPostCount(series.posts.length, locale) }}</p>
          </div>

          <p v-if="series.description[locale]" class="blog-overview__series-description">
            {{ series.description[locale] }}
          </p>

          <div class="blog-overview__posts">
            <VPLink
              v-for="post in series.posts"
              :key="post.key"
              class="blog-overview__post"
              :href="post.href[locale]"
            >
              <p class="blog-overview__post-date">{{ formatBlogDate(post.date, locale) }}</p>
              <h4>{{ post.title[locale] }}</h4>
              <p>{{ post.summary[locale] }}</p>
            </VPLink>
          </div>
        </article>
      </div>

      <p v-else class="blog-overview__empty">
        {{ pageText.empty[locale] }}
      </p>
    </section>

    <p class="blog-overview__closing">{{ pageText.closing[locale] }}</p>
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

.blog-overview__hero {
  position: relative;
  padding: 1.5rem 1.5rem 1.6rem;
  margin-bottom: 1.25rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 24px;
  background:
    radial-gradient(circle at top left, rgb(196 227 255 / 0.45), transparent 38%),
    linear-gradient(135deg, rgb(248 250 252 / 0.96), rgb(255 255 255 / 0.9));
  overflow: hidden;
}

.blog-overview__eyebrow {
  margin: 0 0 0.45rem;
  color: #7a4b19;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.blog-overview__lead {
  margin: 0.9rem 0 0;
  font-size: 1.02rem;
}

.blog-overview__manifesto {
  margin: 1rem 0 0;
  padding-top: 1rem;
  border-top: 1px solid rgb(15 23 42 / 0.1);
  color: var(--vp-c-text-2);
  font-size: 0.98rem;
}

.blog-overview__grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.blog-overview__card,
.blog-overview__series,
.blog-overview__interests {
  border: 1px solid var(--vp-c-divider);
  border-radius: 20px;
  box-shadow: 0 14px 34px rgb(15 23 42 / 0.04);
}

.blog-overview__card {
  padding: 1.15rem 1.15rem 1.2rem;
  background: linear-gradient(180deg, var(--vp-c-bg-soft), rgb(255 255 255 / 0.92));
}

.blog-overview__card h2 {
  margin: 0 0 0.65rem;
  font-size: 1.02rem;
}

.blog-overview__card p {
  margin: 0;
  color: var(--vp-c-text-2);
}

.blog-overview__interests {
  padding: 1.15rem 1.2rem 1.25rem;
  margin-bottom: 1.25rem;
  background: var(--vp-c-bg-soft);
}

.blog-overview__interests h2 {
  margin: 0 0 0.85rem;
  font-size: 1rem;
}

.blog-overview__tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.blog-overview__tags li {
  padding: 0.45rem 0.8rem;
  border: 1px solid rgb(15 23 42 / 0.08);
  border-radius: 999px;
  background: rgb(255 255 255 / 0.85);
  color: var(--vp-c-text-2);
  font-size: 0.92rem;
}

.blog-overview__catalog {
  margin-bottom: 1rem;
}

.blog-overview__catalog-header {
  margin-bottom: 0.9rem;
}

.blog-overview__catalog-header h2 {
  margin: 0 0 0.45rem;
  font-size: 1.25rem;
}

.blog-overview__catalog-header p:last-child {
  margin: 0;
  color: var(--vp-c-text-2);
}

.blog-overview__series-list {
  display: grid;
  gap: 1rem;
}

.blog-overview__series {
  padding: 1.2rem;
  background:
    radial-gradient(circle at top right, rgb(250 204 21 / 0.15), transparent 28%),
    linear-gradient(180deg, rgb(255 255 255 / 0.98), rgb(248 250 252 / 0.95));
}

.blog-overview__series-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}

.blog-overview__series-head h3 {
  margin: 0;
  font-size: 1.18rem;
}

.blog-overview__series-date,
.blog-overview__post-date {
  margin: 0 0 0.35rem;
  color: #8a5b28;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.blog-overview__series-count {
  margin: 0;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  background: rgb(15 23 42 / 0.06);
  color: var(--vp-c-text-2);
  font-size: 0.86rem;
  white-space: nowrap;
}

.blog-overview__series-description {
  margin: 0.85rem 0 1rem;
  color: var(--vp-c-text-2);
}

.blog-overview__posts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.85rem;
}

.blog-overview__post {
  display: block;
  padding: 1rem;
  border: 1px solid rgb(15 23 42 / 0.08);
  border-radius: 18px;
  background: rgb(255 255 255 / 0.86);
  text-decoration: none;
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    box-shadow 0.2s ease;
}

.blog-overview__post:hover {
  transform: translateY(-2px);
  border-color: rgb(234 179 8 / 0.35);
  box-shadow: 0 12px 28px rgb(15 23 42 / 0.08);
}

.blog-overview__post h4 {
  margin: 0 0 0.45rem;
  color: var(--vp-c-text-1);
  font-size: 1rem;
}

.blog-overview__post p:last-child {
  margin: 0;
  color: var(--vp-c-text-2);
}

.blog-overview__empty {
  margin: 0;
  padding: 1rem 1.1rem;
  border: 1px dashed var(--vp-c-divider);
  border-radius: 18px;
  color: var(--vp-c-text-2);
}

.blog-overview__closing {
  margin: 1rem 0 0;
  color: var(--vp-c-text-1);
  font-weight: 600;
}

@media (max-width: 960px) {
  .blog-overview__grid,
  .blog-overview__posts {
    grid-template-columns: 1fr;
  }
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

  .blog-overview__hero,
  .blog-overview__card,
  .blog-overview__interests,
  .blog-overview__series {
    padding-left: 1rem;
    padding-right: 1rem;
  }

  .blog-overview__series-head {
    flex-direction: column;
  }

  .blog-overview__series-count {
    white-space: normal;
  }
}
</style>
