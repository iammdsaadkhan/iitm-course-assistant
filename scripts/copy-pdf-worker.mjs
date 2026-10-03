import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// PDF.js requires a same-version worker file. Copy it from the locked npm
// dependency into /public during installation so the browser can load it.
const require = createRequire(import.meta.url);
const source = require.resolve("pdfjs-dist/build/pdf.worker.min.mjs");
const projectRoot = fileURLToPath(new URL("..", import.meta.url));
const destination = join(projectRoot, "public", "pdf.worker.min.mjs");

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
console.log("Copied the PDF.js worker to public/pdf.worker.min.mjs");
