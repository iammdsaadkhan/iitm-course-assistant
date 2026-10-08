import { NextResponse } from "next/server";
import { getAIProviderConfig } from "@/lib/ai-provider";

export const runtime = "nodejs";

export async function GET() {
  const config = getAIProviderConfig();
  const provider = config?.provider ?? "local";
  const searchMethod =
    config?.provider === "huggingface" && config.embeddingModel
      ? `Semantic embeddings (${config.embeddingModel})`
      : "Keyword search (browser)";

  return NextResponse.json(
    {
      aiEnabled: config !== null,
      provider,
      searchMethod,
      chatModel: config?.chatModel ?? "Extractive answers",
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
