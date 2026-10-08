export type PdfChunk = {
  id: string;
  fileName: string;
  page: number;
  text: string;
};

export type SourceHit = {
  chunk: PdfChunk;
  score: number;
};

export type IndexedDocument = {
  chunks: PdfChunk[];
  mode: "groq" | "huggingface" | "local";
  vectors?: number[][];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceHit[];
};
