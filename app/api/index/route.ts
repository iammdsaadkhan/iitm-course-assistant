import { InferenceClient } from "@huggingface/inference";
import { NextResponse } from "next/server";
import { getAIProviderConfig } from "@/lib/ai-provider";
import { readHuggingFaceEmbeddingRows } from "@/lib/embedding-vectors";
import type { PdfChunk } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 120;

const MAX_CHUNKS = 180;
const MAX_TEXT_CHARACTERS = 600_000;
const BATCH_SIZE = 16;

function readChunks(value: unknown): PdfChunk[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_CHUNKS) {
    throw new Error(`The document must create between 1 and ${MAX_CHUNKS} chunks.`);
  }

  const chunks: PdfChunk[] = [];
  let totalCharacters = 0;
  for (const item of value) {
    if (
      !item ||
      typeof item.id !== "string" ||
      typeof item.fileName !== "string" ||
      !Number.isInteger(item.page) ||
      typeof item.text !== "string" ||
      item.text.length === 0
    ) {
      throw new Error("The uploaded document text was not in the expected format.");
    }
    totalCharacters += item.text.length;
    chunks.push({
      id: item.id.slice(0, 300),
      fileName: item.fileName.slice(0, 200),
      page: item.page,
      text: item.text,
    });
  }

  if (totalCharacters > MAX_TEXT_CHARACTERS) {
    throw new Error("This document has too much extracted text. Try a shorter PDF.");
  }
  return chunks;
}

export async function POST(request: Request) {
  const config = getAIProviderConfig();
  if (!config) {
    return NextResponse.json(
      { error: "AI is not configured. Add GROQ_API_KEY or HF_TOKEN to your Vercel environment." },
      { status: 503 },
    );
  }
  if (config.provider !== "huggingface" || !config.embeddingModel) {
    return NextResponse.json(
      { error: "Groq mode uses built-in keyword retrieval and does not need an embedding request." },
      { status: 409 },
    );
  }

  try {
    const body = (await request.json()) as { chunks?: unknown };
    const chunks = readChunks(body.chunks);
    const client = new InferenceClient(config.apiKey);
    const vectors: number[][] = [];

    // Batch calls so larger documents stay within provider request limits.
    for (let start = 0; start < chunks.length; start += BATCH_SIZE) {
      const batch = chunks.slice(start, start + BATCH_SIZE);
      const output = await client.featureExtraction({
        model: config.embeddingModel,
        provider: "hf-inference",
        inputs: batch.map((chunk) => chunk.text),
        normalize: true,
        truncate: true,
      });
      vectors.push(...readHuggingFaceEmbeddingRows(output, batch.length));
    }

    return NextResponse.json({ vectors, model: config.embeddingModel, provider: config.provider });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      { error: `Could not create document embeddings with Hugging Face. ${message}` },
      { status: 502 },
    );
  }
}
