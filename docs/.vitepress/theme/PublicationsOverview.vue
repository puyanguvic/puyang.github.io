<script setup lang="ts">
import { data as publications } from "./publications.data";

function isExternalLink(url: string) {
  return url.startsWith("http://") || url.startsWith("https://");
}

function isCurrentAuthor(author: string) {
  return author.trim().toLowerCase() === "pu yang";
}
</script>

<template>
  <section class="publications-page">
    <header class="publications-page__header">
      <h1>Publications</h1>
      <p class="publications-page__intro">
        See my <a href="/cv/puyang-resume.pdf">CV</a> for a more complete list. Selected papers
        are shown in reverse chronological order.
      </p>
    </header>

    <div class="publications-list">
      <article v-for="publication in publications" :key="publication.url" class="publication-entry">
        <p class="publication-entry__year">{{ publication.year }}</p>

        <div class="publication-entry__body">
          <p class="publication-entry__title">
            <a :href="publication.url">{{ publication.title }}</a>.
          </p>

          <p class="publication-entry__authors">
            <template v-for="(author, index) in publication.authors" :key="`${publication.url}-${author}`">
              <span
                :class="{ 'publication-entry__author--self': isCurrentAuthor(author) }"
              >
                {{ author }}
              </span>
              <span v-if="index < publication.authors.length - 1">, </span>
            </template>
          </p>

          <p class="publication-entry__meta">
            <span class="publication-entry__venue">{{ publication.publication }}</span>
            <span class="publication-entry__separator">/</span>
            <a :href="publication.url">details</a>
            <template v-if="publication.pdf">
              <span class="publication-entry__separator">/</span>
              <a
                :href="publication.pdf"
                :target="isExternalLink(publication.pdf) ? '_blank' : undefined"
                :rel="isExternalLink(publication.pdf) ? 'noreferrer' : undefined"
              >
                pdf
              </a>
            </template>
            <template v-else>
              <span class="publication-entry__separator">/</span>
              <span class="publication-entry__resource publication-entry__resource--muted">
                pdf coming soon
              </span>
            </template>
          </p>
        </div>
      </article>
    </div>
  </section>
</template>
