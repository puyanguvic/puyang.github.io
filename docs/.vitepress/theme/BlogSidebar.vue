<script setup lang="ts">
import { useRoute } from "vitepress";
import { VPLink } from "vitepress/theme";
import { computed, watch } from "vue";
import { buildBlogSidebarSections } from "./blogData";
import { detectBlogLocaleFromPath, normalizeBlogPath, useBlogLocale } from "./blogLocale";

const route = useRoute();
const { locale, syncLocale } = useBlogLocale();

const currentPath = computed(() => normalizeBlogPath(route.path));
const isBlogRoute = computed(() => currentPath.value === "/blog" || currentPath.value.startsWith("/blog/"));

const sections = computed(() => buildBlogSidebarSections(locale.value));

watch(
  () => route.path,
  (path) => {
    const pathLocale = detectBlogLocaleFromPath(path);

    if (pathLocale) {
      syncLocale(pathLocale);
      return;
    }

    if (normalizeBlogPath(path) === "/blog") {
      syncLocale();
    }
  },
  { immediate: true }
);

function isActive(link: string) {
  return currentPath.value === normalizeBlogPath(link);
}

function hasActiveChild(items?: Array<{ text: string; link: string }>) {
  return items?.some((item) => isActive(item.link)) ?? false;
}
</script>

<template>
  <div v-if="isBlogRoute" class="blog-sidebar">
    <section
      v-for="section in sections"
      :key="section.text"
      class="blog-sidebar__section"
      :class="{ 'blog-sidebar__section--active': hasActiveChild(section.items) || (section.link && isActive(section.link)) }"
    >
      <VPLink
        v-if="section.link"
        class="blog-sidebar__heading blog-sidebar__heading--link"
        :href="section.link"
      >
        {{ section.text }}
      </VPLink>

      <h2 v-else class="blog-sidebar__heading">
        {{ section.text }}
      </h2>

      <div v-if="section.items?.length" class="blog-sidebar__items">
        <VPLink
          v-for="item in section.items"
          :key="item.link"
          class="blog-sidebar__item"
          :class="{ 'blog-sidebar__item--active': isActive(item.link) }"
          :href="item.link"
        >
          {{ item.text }}
        </VPLink>
      </div>
    </section>
  </div>
</template>

<style scoped>
:global(.VPSidebar .nav > .blog-sidebar ~ .group) {
  display: none;
}

.blog-sidebar__section {
  padding-bottom: 24px;
}

.blog-sidebar__section + .blog-sidebar__section {
  border-top: 1px solid var(--vp-c-divider);
  padding-top: 10px;
}

.blog-sidebar__heading {
  margin: 0;
  padding: 4px 0;
  line-height: 24px;
  font-size: 14px;
  font-weight: 700;
  color: var(--vp-c-text-1);
}

.blog-sidebar__heading--link {
  display: block;
  transition: color 0.25s;
}

.blog-sidebar__heading--link:hover {
  color: var(--vp-c-brand-1);
}

.blog-sidebar__items {
  border-left: 1px solid var(--vp-c-divider);
  margin-top: 4px;
  padding-left: 16px;
}

.blog-sidebar__item {
  display: block;
  padding: 4px 0;
  line-height: 24px;
  font-size: 14px;
  font-weight: 500;
  color: var(--vp-c-text-2);
  transition: color 0.25s;
}

.blog-sidebar__item:hover {
  color: var(--vp-c-brand-1);
}

.blog-sidebar__section--active .blog-sidebar__heading,
.blog-sidebar__heading--link.blog-sidebar__heading--link:hover,
.blog-sidebar__item--active {
  color: var(--vp-c-brand-1);
}

@media (min-width: 960px) {
  .blog-sidebar__section {
    width: calc(var(--vp-sidebar-width) - 64px);
  }
}
</style>
