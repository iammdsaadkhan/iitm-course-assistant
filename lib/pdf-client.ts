import type { PdfChunk } from "@/lib/types";

export type ExtractedPage = {
  page: number;
  text: string;
};

/** Extract selectable text from a PDF in the browser (the PDF file is not uploaded). */
export async function extractPdfPages(
  file: File,
  onProgress?: (completed: number, total: number) => void,
): Promise<ExtractedPage[]> {
  // Import PDF.js only in the browser, so it is not loaded during server rendering.
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

  const bytes = new Uint8Array(await file.arrayBuffer());
  const loadingTask = pdfjs.getDocument({ data: bytes, isEvalSupported: false });
  const pdf = await loadingTask.promise;
  const pages: ExtractedPage[] = [];

  try {
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      const content = await page.getTextContent();
      const text = content.items
        .map((item) => ("str" in item ? item.str : ""))
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();

      if (text) pages.push({ page: pageNumber, text });
      onProgress?.(pageNumber, pdf.numPages);
    }
  } finally {
    await pdf.destroy();
  }

  return pages;
}

/** Split each page into word chunks and keep page information for citations. */
export function makeChunks(
  pages: ExtractedPage[],
  fileName: string,
  chunkSize = 160,
  overlap = 30,
): PdfChunk[] {
  const chunks: PdfChunk[] = [];
  const step = chunkSize - overlap;
  let chunkNumber = 0;

  for (const page of pages) {
    const words = page.text.split(/\s+/).filter(Boolean);
    for (let start = 0; start < words.length; start += step) {
      const text = words.slice(start, start + chunkSize).join(" ");
      if (!text) break;
      chunks.push({
        id: `${fileName}-${page.page}-${chunkNumber}`,
        fileName,
        page: page.page,
        text,
      });
      chunkNumber += 1;
      if (start + chunkSize >= words.length) break;
    }
  }

  return chunks;
}
