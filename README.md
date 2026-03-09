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
- About page: `docs/about.md`
- News: `docs/news/`
- Blog: `docs/blog/`
- Projects: `docs/projects/`
- Publications: `docs/publications/`

## CV (LaTeX)

- Edit the LaTeX source: `content/cv/puyang-resume.tex`
- `docs/public/cv.pdf` is a generated file and should not be committed
- GitHub Actions compiles the PDF on every deploy and publishes it as: `/cv.pdf`
- Optional local build (requires a LaTeX install):
  - `latexmk -xelatex -interaction=nonstopmode content/cv/puyang-resume.tex -output-directory=content/cv`
  - `node scripts/copy-cv.mjs --require`

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
