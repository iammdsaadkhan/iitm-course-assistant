# Interview notes — PageWise PDF Assistant

## 30-second explanation

> I built a Vercel-ready PDF question-answering app with Next.js. PDF.js extracts selectable text in the browser, then the app makes overlapping chunks while keeping the PDF name and page number. If a server-side provider key is configured, the app retrieves relevant passages and sends them to a chat model. Groq is preferred and uses built-in keyword retrieval; Hugging Face also supports semantic embeddings. The UI shows the answer and source pages. The original PDF is never sent to the server. A keyword/extractive fallback also works without an API key.

## Follow the code

1. **`components/pdf-assistant.tsx`** — upload, document processing, chat state, and rendering.
2. **`lib/pdf-client.ts`** — `extractPdfPages` uses PDF.js in the browser; `makeChunks` splits page text and preserves page metadata.
3. **`app/api/status/route.ts`** — reports which provider is configured, without returning its key.
4. **`app/api/index/route.ts`** — validates text chunks and requests Hugging Face embeddings in small batches. The key stays on the server.
5. **`app/api/ask/route.ts`** — uses keyword matches in Groq mode or cosine similarity in Hugging Face mode to select up to four passages, then calls the configured chat model with that context.
6. **`lib/ai-provider.ts`** and **`lib/embedding-vectors.ts`** — provider configuration, Groq requests, and embedding validation/normalization.
7. **`lib/local-rag.ts`** — browser-only keyword matching and extractive answers when no provider key is set.
8. **`app/globals.css`** — responsive styles; no CSS framework or remote fonts required.

## Data flow

```text
PDF selected in browser
  -> PDF.js extracts page text
  -> overlapping chunks include { fileName, page, text }
  -> Groq mode: built-in keyword search selects passages
  -> Hugging Face mode: /api/index embeds chunks; /api/ask ranks them by cosine similarity
  -> selected passages + question go to the configured chat completion API
  -> answer and source pages appear in the UI
```

`GROQ_API_KEY` selects Groq; it takes priority if an `HF_TOKEN` is also present. If Groq is not configured, `HF_TOKEN` selects Hugging Face. Without either, extraction, keyword retrieval, and extractive output stay in the browser. That mode is a useful fallback, but it is not semantic embedding search or model-generated QA.

## Concepts to explain

**What is RAG?**  
Retrieval-augmented generation finds useful document passages first, then gives them to a language model as context. It is more grounded in the uploaded document than asking a model to answer from memory alone.

**Why chunk and overlap?**  
Models and APIs have context limits, and searching smaller units is more precise. Overlap keeps a little shared text at chunk boundaries, making it less likely that a split sentence loses context. The trade-off is some duplicated text.

**What are embeddings?**  
Embeddings are numeric vectors representing text meaning. The app normalizes vectors and uses cosine similarity to rank document passages against the question.

**Why parse PDFs in the browser?**  
It keeps the original file off the server and avoids sending a large PDF through a Vercel Function. Only capped extracted text chunks are sent to the server in AI mode.

**Why keep provider keys server-side?**
A browser-visible key can be copied and abused. `GROQ_API_KEY` and `HF_TOKEN` are read only by server routes and are never returned by `/api/status` or embedded in frontend code.

**Why is there a local mode?**  
It lets the site return matching source text before an inference key is configured. It also makes the upload flow testable without external model access, while the UI clearly identifies that mode.

## Common questions

**Q: Does the website store the PDF?**  
A: No. PDF.js reads it in the browser. The document index stays in client memory. In AI mode, extracted text and questions are sent to the app's server routes and forwarded to the configured provider, but this starter has no database or permanent document storage.

**Q: Can it read scanned PDFs?**  
A: Not currently. A scanned PDF contains page images, not selectable text. I detect when no text was extracted and tell the user. OCR such as Tesseract would be a future extension.

**Q: Why not upload the PDF to a Vercel API route?**  
A: This app does not need to store the original file. Client-side extraction avoids function request-body limits and means only text goes to the AI pipeline.

**Q: Is the similarity score a probability?**  
A: No. It is a ranking signal that says which passage is more similar to the question. It is not a calibrated confidence value.

**Q: What are the limitations?**  
A: The no-key mode is keyword-based; AI mode depends on provider/model availability and quotas; scanned PDF OCR and persistent storage are not included; and a public deployment should add authentication and durable rate limiting.

**Q: How would you scale or productionize it?**  
A: Add auth and durable rate limiting, move large indexes to a vector database, persist documents only with explicit user consent, add OCR if needed, and evaluate retrieval/answer quality with a test set. I would also add observability and retries for model-provider errors.

## Trade-offs

- **Client-side PDF extraction:** better file-size handling and the original PDF stays in-browser, but the browser still has to process the PDF.
- **In-memory index:** simple and fast for a single session, but a page refresh requires re-indexing.
- **Top four passages:** gives the model context without sending the whole document; larger top-k can add noise and tokens.
- **Provider API keys:** keep model weights off Vercel but rely on external model access, quotas, and network availability.
