import DefaultTheme from "vitepress/theme";
import { h } from "vue";
import BlogOverview from "./BlogOverview.vue";
import BlogPostLocaleSwitch from "./BlogPostLocaleSwitch.vue";
import BlogSidebar from "./BlogSidebar.vue";
import GoatCounterTracker from "./GoatCounterTracker.vue";
import HomeIntro from "./HomeIntro.vue";
import HomeVisitCounter from "./HomeVisitCounter.vue";
import "./custom.css";

export default {
  ...DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      "layout-bottom": () => h(GoatCounterTracker),
      "sidebar-nav-before": () => h(BlogSidebar)
    });
  },
  enhanceApp({ app }) {
    app.component("BlogOverview", BlogOverview);
    app.component("BlogPostLocaleSwitch", BlogPostLocaleSwitch);
    app.component("HomeIntro", HomeIntro);
    app.component("HomeVisitCounter", HomeVisitCounter);
  }
};
