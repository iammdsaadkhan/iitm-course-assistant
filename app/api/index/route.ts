import { InferenceClient } from "@huggingface/inference";
import { NextResponse } from "next/server";
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

function averageAndNormalize(value: unknown): number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("The embedding service returned an empty vector.");
  }

  // Some providers return one pooled vector; others return vectors for each token.
  let vector: number[];
  if (typeof value[0] === "number") {
    vector = value as number[];
  } else if (Array.isArray(value[0]) && typeof value[0][0] === "number") {
    const tokenVectors = value as number[][];
    const dimension = tokenVectors[0].length;
    vector = Array.from({ length: dimension }, (_, column) =>
      tokenVectors.reduce((sum, row) => sum + (row[column] ?? 0), 0) / tokenVectors.length,
    );
  } else if (Array.isArray(value[0]) && Array.isArray(value[0][0])) {
    // Handle an extra single-input dimension by flattening one level first.
    return averageAndNormalize(value.flat() as unknown);
  } else {
    throw new Error("The embedding service returned an unexpected vector format.");
  }

  const length = Math.sqrt(vector.reduce((sum, number) => sum + number * number, 0));
  if (!Number.isFinite(length) || length === 0) {
    throw new Error("The embedding service returned an invalid vector.");
  }
  return vector.map((number) => number / length);
}

function readEmbeddingRows(output: unknown, expectedCount: number): number[][] {
  if (!Array.isArray(output)) {
    throw new Error("The embedding service returned an unexpected response.");
  }

  // A single input may be returned as a bare vector or as token vectors.
  let rows = output;
  if (expectedCount === 1 && typeof output[0] === "number") {
    rows = [output];
  } else if (
    expectedCount === 1 &&
    output.length > 1 &&
    Array.isArray(output[0]) &&
    typeof output[0][0] === "number"
  ) {
    // Wrap token-by-token output as the one input row; averageAndNormalize pools it.
    rows = [output];
  }
  if (rows.length !== expectedCount) {
    throw new Error("The embedding service returned a different number of vectors than expected.");
  }
  return rows.map(averageAndNormalize);
}

export async function POST(request: Request) {
  const token = process.env.HF_TOKEN;
  if (!token) {
    return NextResponse.json(
      { error: "Hugging Face AI is not configured. Use local mode or add HF_TOKEN." },
      { status: 503 },
    );
  }

  try {
    const body = (await request.json()) as { chunks?: unknown };
    const chunks = readChunks(body.chunks);
    const model = process.env.HF_EMBEDDING_MODEL ?? "sentence-transformers/all-MiniLM-L6-v2";
    const client = new InferenceClient(token);
    const vectors: number[][] = [];

    // Batch calls so large documents stay within provider request limits.
    for (let start = 0; start < chunks.length; start += BATCH_SIZE) {
      const batch = chunks.slice(start, start + BATCH_SIZE);
      const output = await client.featureExtraction({
        model,
        provider: "hf-inference",
        inputs: batch.map((chunk) => chunk.text),
        normalize: true,
        truncate: true,
      });
      vectors.push(...readEmbeddingRows(output, batch.length));
    }

    return NextResponse.json({ vectors, model });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      {
        error: `Could not create document embeddings. Check your Hugging Face token, model access, and inference quota. (${message})`,
      },
      { status: 502 },
    );
  }
}
