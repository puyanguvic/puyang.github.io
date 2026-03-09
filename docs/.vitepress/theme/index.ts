import DefaultTheme from "vitepress/theme";
import HomeIntro from "./HomeIntro.vue";
import HomeVisitCounter from "./HomeVisitCounter.vue";
import "./custom.css";

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component("HomeIntro", HomeIntro);
    app.component("HomeVisitCounter", HomeVisitCounter);
  }
};
