import fs from "node:fs";
import path from "node:path";

const args = new Set(process.argv.slice(2));
const requireSource = args.has("--require");
const source = path.resolve("content/cv/puyang-resume.pdf");
const target = path.resolve("docs/public/cv.pdf");

fs.mkdirSync(path.dirname(target), { recursive: true });

if (fs.existsSync(source)) {
  fs.copyFileSync(source, target);
  console.log(`Copied ${source} -> ${target}`);
} else if (requireSource) {
  console.error(`Missing required CV PDF: ${source}`);
  process.exit(1);
} else {
  if (fs.existsSync(target)) {
    fs.rmSync(target);
  }
  console.warn(`Skipped CV copy because ${source} does not exist yet.`);
}
