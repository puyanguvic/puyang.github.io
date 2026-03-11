<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";

const counterUrl = (import.meta.env.VITE_GOATCOUNTER_URL || "").replace(/\/$/, "");
const formatter = new Intl.NumberFormat("en-US");
const totalVisits = ref<string | null>(null);
const isEnabled = computed(() => Boolean(counterUrl));
let refreshTimer: number | undefined;

async function loadTotalVisits() {
  if (!counterUrl) {
    return;
  }

  try {
    const response = await fetch(`${counterUrl}/counter/TOTAL.json?ts=${Date.now()}`, {
      cache: "no-store"
    });
    if (!response.ok) {
      return;
    }

    const payload = await response.json();
    if (typeof payload.count === "string" && payload.count.trim()) {
      totalVisits.value = payload.count;
      return;
    }

    const count = Number(payload.count);
    totalVisits.value = Number.isFinite(count) ? formatter.format(count) : null;
  } catch {
    totalVisits.value = null;
  }
}

onMounted(() => {
  void loadTotalVisits();
  refreshTimer = window.setTimeout(() => {
    void loadTotalVisits();
  }, 2500);
});

onUnmounted(() => {
  if (refreshTimer) {
    window.clearTimeout(refreshTimer);
  }
});
</script>

<template>
  <p v-if="isEnabled" class="home-visit-counter" aria-live="polite">
    <span class="home-visit-counter__label">Visits</span>
    <strong>{{ totalVisits ?? "..." }}</strong>
  </p>
</template>
