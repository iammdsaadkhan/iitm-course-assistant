import type { PdfChunk, SourceHit } from "@/lib/types";

const STOP_WORDS = new Set([
  "about", "after", "again", "also", "and", "are", "because", "been", "before",
  "being", "between", "but", "can", "could", "does", "for", "from", "have", "into",
  "just", "more", "most", "not", "our", "out", "over", "should", "some", "such",
  "than", "that", "the", "their", "them", "there", "these", "they", "this", "those",
  "through", "under", "very", "was", "were", "what", "when", "where", "which", "who",
  "will", "with", "would", "your",
]);

function words(value: string): string[] {
  return (value.toLowerCase().match(/[\p{L}\p{N}]{2,}/gu) ?? [])
    .filter((word) => !STOP_WORDS.has(word));
}

/** A small keyword-overlap search used when no Hugging Face token is configured. */
export function findLocalMatches(
  question: string,
  chunks: PdfChunk[],
  topK = 3,
): SourceHit[] {
  const queryWords = [...new Set(words(question))];
  if (queryWords.length === 0) return [];

  return chunks
    .map((chunk) => {
      const chunkWords = new Set(words(chunk.text));
      const matched = queryWords.filter((word) => chunkWords.has(word)).length;
      return { chunk, score: matched / queryWords.length };
    })
    .filter((hit) => hit.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

/** Pick the most question-relevant sentences as a transparent local-mode answer. */
export function makeExtractiveAnswer(question: string, sources: SourceHit[]): string {
  if (sources.length === 0) {
    return "I could not find a matching passage. Try different words from the PDF, or enable Hugging Face AI mode for semantic search.";
  }

  const queryWords = [...new Set(words(question))];
  const candidates = sources.flatMap(({ chunk, score: sourceScore }) =>
    chunk.text
      .split(/(?<=[.!?])\s+/)
      .map((sentence) => sentence.trim())
      .filter(Boolean)
      .map((sentence) => {
        const sentenceWords = new Set(words(sentence));
        const overlap = queryWords.filter((word) => sentenceWords.has(word)).length;
        return { sentence, score: overlap / Math.max(queryWords.length, 1) + sourceScore * 0.1 };
      }),
  );

  const bestSentences = candidates
    .filter((candidate) => candidate.score > 0.1)
    .sort((a, b) => b.score - a.score)
    .slice(0, 2)
    .map((candidate) => candidate.sentence);

  if (bestSentences.length > 0) {
    return `From the document: ${bestSentences.join(" ")}`;
  }
  return `Here is the closest passage I found: “${sources[0].chunk.text.slice(0, 450)}${sources[0].chunk.text.length > 450 ? "…”" : "”"}`;
}
