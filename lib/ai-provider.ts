export type AIProvider = "groq" | "huggingface";

export type AIProviderConfig = {
  provider: AIProvider;
  apiKey: string;
  embeddingModel?: string;
  chatModel: string;
};

const GROQ_API_BASE = "https://api.groq.com/openai/v1";

/**
 * Prefer Groq when both providers are configured. This lets a deployment move
 * from Hugging Face by adding GROQ_API_KEY without first removing HF_TOKEN.
 */
export function getAIProviderConfig(): AIProviderConfig | null {
  const groqApiKey = process.env.GROQ_API_KEY?.trim();
  if (groqApiKey) {
    return {
      provider: "groq",
      apiKey: groqApiKey,
      chatModel: process.env.GROQ_CHAT_MODEL?.trim() || "openai/gpt-oss-20b",
    };
  }

  const huggingFaceToken = process.env.HF_TOKEN?.trim();
  if (huggingFaceToken) {
    return {
      provider: "huggingface",
      apiKey: huggingFaceToken,
      embeddingModel:
        process.env.HF_EMBEDDING_MODEL?.trim() || "sentence-transformers/all-MiniLM-L6-v2",
      chatModel: process.env.HF_CHAT_MODEL?.trim() || "openai/gpt-oss-20b:fastest",
    };
  }

  return null;
}

function readGroqError(payload: unknown): string | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
  const error = (payload as Record<string, unknown>).error;
  if (typeof error === "string") return error;
  if (error && typeof error === "object" && !Array.isArray(error)) {
    const message = (error as Record<string, unknown>).message;
    if (typeof message === "string") return message;
  }
  return null;
}

/** Make a server-to-server request to Groq's OpenAI-compatible Chat API. */
export async function groqChatCompletion(
  config: AIProviderConfig,
  body: Record<string, unknown>,
): Promise<unknown> {
  const response = await fetch(`${GROQ_API_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
  });

  const responseText = await response.text();
  let payload: unknown;
  try {
    payload = responseText ? JSON.parse(responseText) : null;
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const detail = readGroqError(payload) ?? (responseText.slice(0, 400) || response.statusText);
    if (response.status === 401 || response.status === 403) {
      throw new Error(
        "Groq rejected the API key. Check that GROQ_API_KEY is a current Groq key in the Vercel environment, then redeploy.",
      );
    }
    if (response.status === 429) {
      throw new Error(
        "Groq's rate limit or free-plan quota was reached. Check your Groq Console limits and try again after they reset.",
      );
    }
    if (response.status === 400 || response.status === 404) {
      throw new Error(
        `Groq could not use chat model "${config.chatModel}". Check the model ID and whether it is available to your Groq account. (${detail})`,
      );
    }
    throw new Error(`Groq request failed (${response.status}): ${detail}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Groq returned an empty or invalid response. Please try again.");
  }
  return payload;
}
