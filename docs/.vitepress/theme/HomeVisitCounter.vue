<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

const counterUrl = (import.meta.env.VITE_GOATCOUNTER_URL || "").replace(/\/$/, "");
const totalVisits = ref<string | null>(null);
const isEnabled = computed(() => Boolean(counterUrl));

async function loadTotalVisits() {
  if (!counterUrl) {
    return;
  }

  try {
    const response = await fetch(`${counterUrl}/counter/TOTAL.json`);
    if (!response.ok) {
      return;
    }

    const payload = await response.json();
    totalVisits.value = payload.count ?? null;
  } catch {
    totalVisits.value = null;
  }
}

function injectTrackingScript() {
  if (!counterUrl) {
    return;
  }

  const existing = document.querySelector<HTMLScriptElement>("script[data-goatcounter]");
  if (existing) {
    return;
  }

  const script = document.createElement("script");
  script.async = true;
  script.src = "https://gc.zgo.at/count.js";
  script.dataset.goatcounter = `${counterUrl}/count`;
  document.head.appendChild(script);
}

onMounted(() => {
  injectTrackingScript();
  void loadTotalVisits();
});
</script>

<template>
  <p v-if="isEnabled" class="home-visit-counter">
    Site visits:
    <strong>{{ totalVisits ?? "Loading..." }}</strong>
  </p>
  <p v-else class="home-visit-counter home-visit-counter--muted">
    Visit counter will appear here after GoatCounter is configured.
  </p>
</template>
