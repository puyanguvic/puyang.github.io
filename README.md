# VitePress personal website (GitHub Pages)

This repo is a personal website built with **VitePress** and deployed to **GitHub Pages**.

## Prerequisites

- Node.js 18+
- npm 9+
- Optional for CV builds: a LaTeX install with `latexmk` and XeLaTeX

## Local development

- Install dependencies:
  - `npm install`
- Start the docs dev server:
  - `npm run dev`

## Content structure

- Homepage: `docs/index.md`
- News: `docs/news/`
- Blog: `docs/blog/`
- Projects: `docs/projects/`
- Publications: `docs/publications/`

## CV (LaTeX)

- Edit the LaTeX source in `cv/`
- Commit the generated PDF copy at `docs/public/cv/puyang-resume.pdf`
- GitHub Pages publishes that static file directly as: `/cv/puyang-resume.pdf`
- Rebuild the PDF locally only when the CV source changes (requires a LaTeX install):
  - `npm run build:cv`
- `build:cv` compiles in a temporary directory and refreshes only `docs/public/cv/puyang-resume.pdf`
- After rebuilding, commit both the LaTeX source changes and the refreshed PDF
- CI checks this rule in `.github/workflows/check-cv.yml`: if `cv/` changes, the committed PDF must also change, and the CV must still compile

## Optional visitor counter

- This site supports an optional GoatCounter counter on the homepage.
- Set the repo variable `GOATCOUNTER_URL` to your GoatCounter site URL, for example `https://example.goatcounter.com`.
- GoatCounter setup reference: https://www.goatcounter.com/help/start

## Deploy to GitHub Pages

This repo includes a GitHub Actions workflow that builds the site and publishes `./docs/.vitepress/dist` to the `gh-pages` branch on every push to `main`:

- Workflow: `.github/workflows/deploy.yml`

### Pages settings

1) In your GitHub repo: **Settings → Pages**
2) Set **Build and deployment** to **Deploy from a branch**
3) Select:
   - Branch: `gh-pages`
   - Folder: `/ (root)`

## Customize

- Site config: `docs/.vitepress/config.mts`
- Theme overrides: `docs/.vitepress/theme/`
