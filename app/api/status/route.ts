import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json(
    {
      aiEnabled: Boolean(process.env.HF_TOKEN),
      embeddingModel:
        process.env.HF_EMBEDDING_MODEL ?? "sentence-transformers/all-MiniLM-L6-v2",
      chatModel: process.env.HF_CHAT_MODEL ?? "openai/gpt-oss-20b:fastest",
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
