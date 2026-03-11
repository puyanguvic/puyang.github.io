<script setup lang="ts">
import { onMounted, watch } from "vue";
import { useRoute } from "vitepress";

declare global {
  interface Window {
    goatcounter?: {
      count: (vars?: { path?: string; title?: string }) => void;
    };
  }
}

const counterUrl = (import.meta.env.VITE_GOATCOUNTER_URL || "").replace(/\/$/, "");
const route = useRoute();
let hasLoadedScript = false;
let lastCountedPath = "";

function currentPath() {
  return window.location.pathname + window.location.search;
}

function countPageView() {
  if (!counterUrl || !hasLoadedScript || !window.goatcounter) {
    return;
  }

  const path = currentPath();
  if (path === lastCountedPath) {
    return;
  }

  lastCountedPath = path;
  window.goatcounter.count({
    path,
    title: document.title
  });
}

function injectTrackingScript() {
  if (!counterUrl) {
    return;
  }

  const existing = document.querySelector<HTMLScriptElement>("script[data-goatcounter]");
  if (existing) {
    if (window.goatcounter) {
      hasLoadedScript = true;
      countPageView();
      return;
    }

    existing.addEventListener("load", () => {
      hasLoadedScript = true;
      countPageView();
    });
    return;
  }

  const script = document.createElement("script");
  script.async = true;
  script.src = "https://gc.zgo.at/count.js";
  script.dataset.goatcounter = `${counterUrl}/count`;
  script.dataset.goatcounterSettings = "{\"no_onload\":true}";
  script.addEventListener("load", () => {
    hasLoadedScript = true;
    countPageView();
  });
  document.head.appendChild(script);
}

onMounted(() => {
  injectTrackingScript();

  watch(
    () => route.path,
    () => {
      countPageView();
    },
    { immediate: true }
  );
});
</script>

<template>
  <span v-if="false" />
</template>
