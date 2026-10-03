# PageWise — deployable PDF RAG website

A small Next.js website for uploading PDFs and asking questions about them. It is designed for Vercel. PDF text is extracted in the browser, so the original PDF is not sent to a Vercel Function or stored by this demo.

## What works

- Drag/drop or select up to 5 PDFs (25 MB total); includes a downloadable sample PDF for testing.
- Extract selectable text in the browser with PDF.js; show page-level source citations.
- Split text into overlapping chunks and retrieve relevant passages.
- **Local mode works without a secret:** keyword search and extractive answers run in the browser.
- **Hugging Face AI mode:** with a server-side Hugging Face token, the app creates semantic embeddings and generates grounded answers through Hugging Face Inference Providers.
- Responsive desktop/mobile interface; no database or PDF storage is required.

Scanned/image-only PDFs need OCR, which is not included. The demo supports up to 180 text chunks and 600,000 extracted characters per indexing session. The index and chat are kept in browser memory; a page refresh means re-uploading and indexing.

## Run locally

Requirements: Node.js 20.16 or newer and npm.

```bash
npm install
cp .env.example .env.local
```

For **Hugging Face AI mode**, edit `.env.local` and add a fine-grained Hugging Face token with **Inference Providers** permission (create one at [Hugging Face token settings](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained)). Keep the token private. Without it, the website still works in local keyword-search mode.

```bash
npm run dev
```

Open the local URL printed by Next.js (usually `http://localhost:3000`). The first AI request can take longer while an inference provider starts up. Hugging Face inference availability, quotas, and possible provider charges depend on your account and model.

## Deploy to Vercel

1. Push this folder to a GitHub repository, or put it in a repository you already have.
2. In Vercel, choose **Add New → Project** and import the repository.
3. If this is inside a larger repository, set the **Root Directory** to `pdf-assistant-vercel`.
4. Select the Next.js framework preset and Node.js 20.x.
5. To enable AI mode, add these Environment Variables in **Project Settings → Environment Variables**:

   | Name | Value |
   | --- | --- |
   | `HF_TOKEN` | Your private Hugging Face token with Inference Providers permission |
   | `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` (default) |
   | `HF_CHAT_MODEL` | `openai/gpt-oss-20b:fastest` (default) |

6. Deploy. Redeploy after changing environment variables.

Do **not** name the token `NEXT_PUBLIC_HF_TOKEN`: `HF_TOKEN` is read only by server routes and is never included in browser JavaScript. If a default model is not available to your Hugging Face account/provider, select a supported model and change the corresponding Vercel environment variable.

### Vercel PDF upload size note

Vercel Functions limit request bodies to 4.5 MB. This app avoids sending the raw PDF to a Function: PDF.js extracts text in the browser, and the app sends only capped text chunks and their embeddings to its API routes. Very large or very text-heavy documents are rejected with a readable message rather than failing with a Vercel 413 error.

## How the RAG flow works

```text
PDF file (browser only)
  -> PDF.js extracts page text
  -> overlapping word chunks retain file name + page number
  -> Hugging Face feature-extraction embeddings (AI mode)
  -> cosine similarity retrieves the top four passages
  -> Hugging Face chat model answers from those passages
  -> UI displays the answer and expandable source passages
```

Without `HF_TOKEN`, the flow uses a small local keyword-overlap search and displays the best matching source sentences. This fallback is useful for a no-key demo, but it is not an AI-generated answer or semantic embedding search.

## Project map

```text
app/
├── api/status/route.ts  # tells the browser whether the server has an HF token
├── api/index/route.ts   # securely embeds PDF text chunks
├── api/ask/route.ts     # retrieves passages and calls the answer model
├── globals.css          # responsive, self-contained styles
├── layout.tsx           # site metadata and root layout
└── page.tsx             # page entry point
components/
└── pdf-assistant.tsx    # upload, chat, and UI state
lib/
├── local-rag.ts         # no-token keyword-search fallback
├── pdf-client.ts        # browser PDF extraction and chunking
└── types.ts             # shared data shapes
public/
├── pdf.worker.min.mjs  # matching PDF.js worker
└── sample-research-summary.pdf # sample file users can download and upload
scripts/
└── copy-pdf-worker.mjs  # copies the matching PDF.js worker during install
INTERVIEW_NOTES.md      # project walkthrough and common interview questions
```

## Data and safety notes

- The raw PDF is parsed in the browser and is not uploaded or stored by the app.
- In AI mode, extracted text chunks and questions are sent to your Vercel API and forwarded to Hugging Face for model inference. Do not upload sensitive documents unless that sharing is acceptable.
- Nothing is written to a database. The in-memory index is lost when the tab refreshes.
- A public demo with a Hugging Face token should add authentication and durable rate limiting before wider production use; this starter intentionally keeps the app simple.
- AI-generated responses can be wrong. Check the displayed source page for important information.

## Interview explanation

> I built a Vercel-ready RAG app. The browser extracts each PDF page with PDF.js, then the app chunks the text while preserving page metadata. In AI mode, a server-only Hugging Face token is used to embed the chunks. For a question, the server embeds the question, ranks document chunks by cosine similarity, and gives the best passages to a chat model. The UI shows the response alongside its source pages. The original PDF never leaves the browser, and a local keyword-search mode works if no inference token is configured.
