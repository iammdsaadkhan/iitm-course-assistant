# PageWise — deployable PDF RAG website

A small Next.js website for uploading PDFs and asking questions about them. It is designed for Vercel. PDF text is extracted in the browser, so the original PDF is not sent to a Vercel Function or stored by this demo.

## What works

- Drag/drop or select up to 5 PDFs (25 MB total); includes a downloadable sample PDF for testing.
- Extract selectable text in the browser with PDF.js; show page-level source citations.
- Split text into overlapping chunks and retrieve relevant passages.
- **Local mode works without a secret:** keyword search and extractive answers run in the browser.
- **Groq AI mode:** with `GROQ_API_KEY`, built-in keyword search finds relevant passages and Groq generates a grounded answer. This requires only the Groq chat API key.
- **Hugging Face AI mode:** supported as a fallback if `GROQ_API_KEY` is not set and a Hugging Face inference token is configured; it uses semantic embeddings.
- Responsive desktop/mobile interface; no database or PDF storage is required.

Scanned/image-only PDFs need OCR, which is not included. The demo supports up to 180 text chunks and 600,000 extracted characters per indexing session. The index and chat are kept in browser memory; a page refresh means re-uploading and indexing.

## Run locally

Requirements: Node.js 20.16 or newer and npm.

```bash
npm install
cp .env.example .env.local
```

For Groq, create an API key in the [Groq console](https://console.groq.com/keys) and set `GROQ_API_KEY` in `.env.local`. The default answer model is `openai/gpt-oss-20b`. Groq mode uses built-in keyword retrieval, so it does not need an embedding API or a separate embedding key. If you also set `HF_TOKEN`, Groq is selected first. Groq account access, model availability, free-plan quotas, and rate limits can change; use a chat model enabled for your account.

Hugging Face is also supported: set `HF_TOKEN` to a fine-grained token with **Inference Providers** permission. Keep all provider keys private. Without either key, the website still works in local keyword-search mode.

```bash
npm run dev
```

Open the local URL printed by Next.js (usually `http://localhost:3000`). The first AI request may take longer than local mode.

## Deploy to Vercel

1. Push this folder to a GitHub repository, or put it in a repository you already have.
2. In Vercel, choose **Add New → Project** and import the repository.
3. This repository's Next.js app is at the repository root, so leave Vercel's **Root Directory** as `./` (change it only if you moved the app into a subdirectory).
4. Select the Next.js framework preset and Node.js 20.x.
5. In **Project Settings → Environment Variables**, add the following for the environment(s) you use (Production, Preview, and/or Development):

   | Name | Value |
   | --- | --- |
   | `GROQ_API_KEY` | Your private Groq API key (required for Groq mode) |
   | `GROQ_CHAT_MODEL` | `openai/gpt-oss-20b` (default; optional) |

6. Redeploy after changing environment variables. Vercel does not apply newly added variables to an already-built deployment.

The app reads `GROQ_API_KEY` on the server. **Do not name it `NEXT_PUBLIC_GROQ_API_KEY`** and do not paste the key into client-side code or Git. Groq is preferred automatically when its key is present, even if an old `HF_TOKEN` is still configured. If there is no Groq key, the app uses Hugging Face when `HF_TOKEN` is available; if neither is set, it uses local mode.

If the UI still says **Local mode**, check that the variable is spelled exactly `GROQ_API_KEY`, is added to the same Vercel project and deployment environment as the URL you are testing, and that you redeployed. If it says **Groq AI** but indexing or asking fails, the UI displays the API error; check that the key is valid and the configured models are available to your Groq account. A `429` response means a provider rate limit or quota was reached.

### Vercel PDF upload size note

Vercel Functions limit request bodies to 4.5 MB. This app avoids sending the raw PDF to a Function: PDF.js extracts text in the browser, and the app sends only capped text chunks (only matching passages in Groq mode; chunks plus embeddings in Hugging Face mode) to its API routes. Very large or very text-heavy documents are rejected with a readable message rather than failing with a Vercel 413 error.

## How the RAG flow works

```text
PDF file (browser only)
  -> PDF.js extracts page text
  -> overlapping word chunks retain file name + page number
  -> Groq mode: built-in keyword search selects relevant passages
  -> Hugging Face mode: semantic embeddings + cosine similarity select top passages
  -> configured chat model answers from those passages
  -> UI displays the answer and expandable source passages
```

Without an AI key, the flow uses a small local keyword-overlap search and displays the best matching source sentences. This fallback is useful for a no-key demo, but it is not an AI-generated answer or semantic embedding search.

## Project map

```text
app/
├── api/status/route.ts  # reports the configured provider and model names, never the key
├── api/index/route.ts   # securely creates Hugging Face semantic embeddings
├── api/ask/route.ts     # retrieves passages and calls the configured answer model
├── globals.css          # responsive, self-contained styles
├── layout.tsx           # site metadata and root layout
└── page.tsx             # page entry point
components/
└── pdf-assistant.tsx    # upload, chat, and UI state
lib/
├── ai-provider.ts      # server-side Groq/Hugging Face configuration and Groq requests
├── embedding-vectors.ts # validates and normalizes provider embeddings
├── local-rag.ts        # no-key keyword-search fallback
├── pdf-client.ts       # browser PDF extraction and chunking
└── types.ts            # shared data shapes
public/
├── pdf.worker.min.mjs  # matching PDF.js worker
└── sample-research-summary.pdf # sample file users can download and upload
scripts/
└── copy-pdf-worker.mjs  # copies the matching PDF.js worker during install
INTERVIEW_NOTES.md      # project walkthrough and common interview questions
```

## Data and safety notes

- The raw PDF is parsed in the browser and is not uploaded or stored by the app.
- In Groq mode, the full text stays in the browser for keyword matching; only matched passages and the question are sent to your Vercel API and Groq for answer generation. In Hugging Face mode, text chunks and questions are sent to your Vercel API and forwarded to Hugging Face for embeddings and answers. Do not upload sensitive documents unless that sharing is acceptable.
- Nothing is written to a database. The in-memory index is lost when the tab refreshes.
- A public demo with an AI-provider key should add authentication and durable rate limiting before wider production use; otherwise strangers may consume the key's quota. This starter intentionally keeps the app simple.
- Free provider plans have usage and rate limits, and model access can change. Check your provider's console for current limits.
- AI-generated responses can be wrong. Check the displayed source page for important information.

## Interview explanation

> I built a Vercel-ready RAG app. The browser extracts each PDF page with PDF.js, then the app chunks the text while preserving page metadata. In Groq mode, browser-side keyword search selects passages and a server-only Groq key generates a grounded answer; Hugging Face mode adds semantic embeddings and cosine-similarity retrieval. The UI shows the response alongside its source pages. The original PDF never leaves the browser, and a local keyword/extractive mode works if no inference key is configured.
