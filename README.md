# Hugo + PaperMod personal website (GitHub Pages)

This repo is a minimal personal site template built with **Hugo (extended)** and the **PaperMod** theme.

## Prerequisites

- Hugo **extended** installed locally
- Theme submodule initialized:
  - `git submodule update --init --recursive`

## Local development

- Start dev server (includes drafts):
  - `hugo server -D`

## Add content

- Add a news item:
  - `hugo new news/YYYY-MM-DD-title.md`
- Add a blog post:
  - `hugo new posts/my-post.md`

## Deploy to GitHub Pages

This repo includes a GitHub Actions workflow that builds the site and publishes `./public` to the `gh-pages` branch on every push to `main`:

- Workflow: `.github/workflows/deploy.yml`

### Pages settings

1) In your GitHub repo: **Settings → Pages**
2) Set **Build and deployment** to **Deploy from a branch**
3) Select:
   - Branch: `gh-pages`
   - Folder: `/ (root)`

### baseURL

Update `baseURL` in `hugo.yaml`:

- User/Org site repo named `<username>.github.io`:
  - `https://<username>.github.io/`
- Project site repo (for example `my-site`):
  - `https://<username>.github.io/my-site/`

## Customize

- Site config: `hugo.yaml` (profile name/subtitle, social links, buttons)
- Homepage layout override (adds “Recent News” + “Recent Blog Posts”): `layouts/index.html`

