"use client";

import { useEffect, useRef, useState, type ChangeEvent, type DragEvent, type FormEvent } from "react";
import { extractPdfPages, makeChunks } from "@/lib/pdf-client";
import { findLocalMatches, makeExtractiveAnswer } from "@/lib/local-rag";
import type { ChatMessage, IndexedDocument, PdfChunk, SourceHit } from "@/lib/types";

const MAX_FILES = 5;
const MAX_TOTAL_FILE_BYTES = 25 * 1024 * 1024;
const MAX_CHUNKS = 180;
const MAX_TEXT_CHARACTERS = 600_000;

const EXAMPLE_QUESTIONS = [
  "What is this document about?",
  "List the key findings.",
  "What dates or deadlines are mentioned?",
];

type ServiceStatus = {
  aiEnabled: boolean;
  embeddingModel: string;
  chatModel: string;
};

type Phase = "idle" | "extracting" | "embedding" | "asking";

function formatSize(size: number): string {
  return size < 1024 * 1024
    ? `${Math.max(1, Math.round(size / 1024))} KB`
    : `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function formatAnswer(data: unknown): { answer: string; sources: SourceHit[] } {
  if (!data || typeof data !== "object") throw new Error("The server returned an invalid response.");
  const value = data as { answer?: unknown; sources?: unknown };
  if (typeof value.answer !== "string") throw new Error("The server did not return an answer.");
  return {
    answer: value.answer,
    sources: Array.isArray(value.sources) ? (value.sources as SourceHit[]) : [],
  };
}

export default function PdfAssistant() {
  const fileInput = useRef<HTMLInputElement>(null);
  const chatEnd = useRef<HTMLDivElement>(null);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [indexedDocument, setIndexedDocument] = useState<IndexedDocument | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    fetch("/api/status", { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("Status check failed");
        return (await response.json()) as ServiceStatus;
      })
      .then(setServiceStatus)
      .catch(() =>
        setServiceStatus({
          aiEnabled: false,
          embeddingModel: "Local keyword search",
          chatModel: "Extractive answers",
        }),
      );
  }, []);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, phase]);

  function addFiles(incoming: File[]) {
    setError("");
    if (incoming.length === 0) return;
    const combined = [...files, ...incoming].filter(
      (file, index, all) => all.findIndex((item) => item.name === file.name && item.size === file.size) === index,
    );
    if (combined.some((file) => !file.name.toLowerCase().endsWith(".pdf"))) {
      setError("Please choose PDF files only.");
      return;
    }
    if (combined.length > MAX_FILES) {
      setError(`You can add up to ${MAX_FILES} PDFs at a time.`);
      return;
    }
    if (combined.some((file) => file.size === 0)) {
      setError("One of those files is empty. Choose a valid PDF.");
      return;
    }
    if (combined.reduce((total, file) => total + file.size, 0) > MAX_TOTAL_FILE_BYTES) {
      setError("The selected PDFs are too large together. Keep the total under 25 MB.");
      return;
    }

    setFiles(combined);
    setIndexedDocument(null);
    setMessages([]);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  }

  function removeFile(index: number) {
    const remaining = files.filter((_, fileIndex) => fileIndex !== index);
    setFiles(remaining);
    setIndexedDocument(null);
    setMessages([]);
    setError("");
  }

  async function prepareDocuments() {
    if (files.length === 0 || phase !== "idle") return;
    setError("");
    setProgress(0);
    setPhase("extracting");

    try {
      const chunks: PdfChunk[] = [];
      let completedFiles = 0;

      for (const file of files) {
        const pages = await extractPdfPages(file, (completed, total) => {
          const fileProgress = total === 0 ? 0 : completed / total;
          setProgress(Math.round(((completedFiles + fileProgress) / files.length) * 60));
        });
        if (pages.length === 0) {
          throw new Error(`No selectable text was found in “${file.name}”. Scanned PDFs need OCR.`);
        }
        chunks.push(...makeChunks(pages, file.name));
        completedFiles += 1;
      }

      const totalTextLength = chunks.reduce((sum, chunk) => sum + chunk.text.length, 0);
      if (chunks.length > MAX_CHUNKS) {
        throw new Error(
          `This PDF set creates ${chunks.length} text sections; this demo supports up to ${MAX_CHUNKS}. Try fewer or shorter PDFs.`,
        );
      }
      if (totalTextLength > MAX_TEXT_CHARACTERS) {
        throw new Error("This PDF set contains too much text for one request. Try a shorter document.");
      }

      if (serviceStatus?.aiEnabled) {
        setPhase("embedding");
        setProgress(65);
        const response = await fetch("/api/index", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chunks }),
        });
        const result = (await response.json()) as { vectors?: number[][]; error?: string };
        if (!response.ok) throw new Error(result.error ?? "Could not create document embeddings.");
        if (!Array.isArray(result.vectors) || result.vectors.length !== chunks.length) {
          throw new Error("The embedding service returned an incomplete index. Please try again.");
        }
        setIndexedDocument({ chunks, vectors: result.vectors, mode: "huggingface" });
      } else {
        // No API key is needed for this fallback. Text and search stay in the browser.
        setIndexedDocument({ chunks, mode: "local" });
      }

      setProgress(100);
      setMessages([
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: serviceStatus?.aiEnabled
            ? `Ready. I indexed ${chunks.length} text sections from ${files.length} PDF${files.length === 1 ? "" : "s"}. Ask me a question and I’ll show the page sources.`
            : `Ready. I found ${chunks.length} text sections. This deployment is in local mode, so answers use keyword matching. Add an HF_TOKEN in Vercel to enable semantic search and generated answers.`,
        },
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not process the PDF files.");
      setIndexedDocument(null);
    } finally {
      setPhase("idle");
      window.setTimeout(() => setProgress(0), 900);
    }
  }

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || !indexedDocument || phase !== "idle") return;

    setError("");
    setQuestion("");
    setPhase("asking");
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: cleanQuestion,
    };
    setMessages((current) => [...current, userMessage]);

    try {
      let answer: string;
      let sources: SourceHit[];

      if (indexedDocument.mode === "huggingface" && indexedDocument.vectors) {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: cleanQuestion,
            chunks: indexedDocument.chunks,
            vectors: indexedDocument.vectors,
          }),
        });
        const result = (await response.json()) as { answer?: string; sources?: SourceHit[]; error?: string };
        if (!response.ok) throw new Error(result.error ?? "The AI service could not answer this question.");
        ({ answer, sources } = formatAnswer(result));
      } else {
        sources = findLocalMatches(cleanQuestion, indexedDocument.chunks);
        answer = makeExtractiveAnswer(cleanQuestion, sources);
      }

      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "assistant", content: answer, sources },
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong. Please try again.");
    } finally {
      setPhase("idle");
    }
  }

  const busy = phase !== "idle";
  const hasIndex = indexedDocument !== null;

  return (
    <main className="site-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="PDF Assistant home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none">
              <path d="M8 4.5h10l6 6V27a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 6 27V6a1.5 1.5 0 0 1 2-1.5Z" stroke="currentColor" strokeWidth="1.8" />
              <path d="M18 5v6h6M10 17h10M10 21h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              <path d="m22.5 15.8.7 1.5 1.6.7-1.6.6-.7 1.6-.6-1.6-1.6-.6 1.6-.7.6-1.5Z" fill="currentColor" />
            </svg>
          </span>
          <span className="brand-name">Page<span>Wise</span></span>
        </a>
        <div className="topbar-right">
          <span className={`mode-pill ${serviceStatus?.aiEnabled ? "mode-pill-ai" : "mode-pill-local"}`}>
            <span className="status-dot" />
            {serviceStatus === null ? "Checking setup" : serviceStatus.aiEnabled ? "Hugging Face AI" : "Local mode"}
          </span>
          <a className="help-link" href="#how-it-works">How it works</a>
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-line" /> YOUR DOCUMENTS, IN CONVERSATION</div>
          <h1>Ask your PDFs.<br /><span>Get to the point.</span></h1>
          <p>Upload a document and ask questions in plain language. PageWise finds the relevant passages and shows you exactly where they came from.</p>
        </div>
        <div className="hero-badge" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="hero-document">
            <div className="hero-document-shine" />
            <span className="hero-doc-label">PDF</span>
            <span className="hero-doc-line line-long" />
            <span className="hero-doc-line" />
            <span className="hero-doc-highlight" />
            <span className="hero-doc-line line-short" />
            <span className="hero-doc-line line-long" />
            <span className="hero-doc-line" />
          </div>
          <span className="sparkle sparkle-a">✦</span>
          <span className="sparkle sparkle-b">✧</span>
          <span className="hero-orbit-dot" />
        </div>
      </section>

      <section className="workspace" aria-label="PDF question answering workspace">
        <aside className="upload-panel">
          <div className="panel-heading">
            <div>
              <span className="step-label">STEP 01</span>
              <h2>Add your PDFs</h2>
            </div>
            <span className="step-icon" aria-hidden="true">↥</span>
          </div>
          <p className="panel-description">Select up to {MAX_FILES} text-based PDFs. Your original files stay in this browser.</p>

          <div
            className={`dropzone ${dragging ? "dropzone-active" : ""} ${files.length ? "dropzone-compact" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false); }}
            onDrop={handleDrop}
          >
            <input
              ref={fileInput}
              className="visually-hidden"
              type="file"
              accept="application/pdf,.pdf"
              multiple
              onChange={handleFileChange}
              aria-label="Choose PDF files"
            />
            <div className="upload-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4.5 15v3.25c0 .69.56 1.25 1.25 1.25h12.5c.69 0 1.25-.56 1.25-1.25V15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <strong>{files.length ? "Add another PDF" : "Drop PDFs here"}</strong>
            <span>or</span>
            <button type="button" className="browse-button" onClick={() => fileInput.current?.click()}>
              Browse files
            </button>
            <small>PDF only · up to 25 MB total</small>
          </div>

          {files.length > 0 && (
            <div className="file-list" aria-label="Selected PDFs">
              {files.map((file, index) => (
                <div className="file-row" key={`${file.name}-${file.size}`}>
                  <span className="file-icon" aria-hidden="true">PDF</span>
                  <span className="file-meta">
                    <strong title={file.name}>{file.name}</strong>
                    <small>{formatSize(file.size)}</small>
                  </span>
                  <button
                    type="button"
                    className="icon-button remove-file"
                    onClick={() => removeFile(index)}
                    disabled={busy}
                    aria-label={`Remove ${file.name}`}
                    title="Remove PDF"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            className="primary-button index-button"
            type="button"
            onClick={prepareDocuments}
            disabled={files.length === 0 || busy || serviceStatus === null}
          >
            {phase === "extracting" ? <><span className="spinner" /> Reading pages…</> : null}
            {phase === "embedding" ? <><span className="spinner" /> Creating search index…</> : null}
            {phase !== "extracting" && phase !== "embedding" ? <>{hasIndex ? "Refresh document index" : "Prepare documents"}<span aria-hidden="true">→</span></> : null}
          </button>
          {busy && phase !== "asking" && (
            <div className="progress-track" aria-label={`Processing progress ${progress}%`}>
              <span style={{ width: `${progress}%` }} />
            </div>
          )}
          {hasIndex && (
            <div className="indexed-note">
              <span className="checkmark">✓</span>
              <span>{indexedDocument.chunks.length} sections ready · {indexedDocument.mode === "huggingface" ? "semantic search enabled" : "keyword search ready"}</span>
            </div>
          )}

          <div className="privacy-note">
            <span className="privacy-icon" aria-hidden="true">◈</span>
            <p>
              {serviceStatus?.aiEnabled
                ? "PDFs are read in your browser. Text passages and questions go to this app’s API and Hugging Face for AI processing; nothing is saved by this demo."
                : "Local mode keeps PDF text and search in this browser. Add an HF_TOKEN in Vercel to enable semantic search and AI-generated answers."}
            </p>
          </div>
          {serviceStatus?.aiEnabled && (
            <details className="model-details">
              <summary>Models in use</summary>
              <p><b>Embeddings</b><br />{serviceStatus.embeddingModel}</p>
              <p><b>Answer model</b><br />{serviceStatus.chatModel}</p>
            </details>
          )}
        </aside>

        <section className="chat-panel" aria-label="Ask questions about the PDFs">
          <div className="chat-header">
            <div>
              <span className="step-label">STEP 02</span>
              <h2>Ask a question</h2>
            </div>
            <span className={`ready-badge ${hasIndex ? "ready" : "not-ready"}`}>
              <span className="status-dot" />{hasIndex ? "Document ready" : "Waiting for a PDF"}
            </span>
          </div>

          <div className="chat-body" aria-live="polite">
            {!hasIndex && messages.length === 0 && (
              <div className="empty-chat">
                <div className="empty-illustration" aria-hidden="true">
                  <div className="empty-page"><span /><span /><span /><i /></div>
                  <span className="empty-question">?</span>
                </div>
                <h3>Your next answer is in there.</h3>
                <p>Add a PDF and prepare it. Then ask about a detail, date, conclusion, or anything you want to find.</p>
                <div className="suggestion-preview">
                  <span>TRY ASKING</span>
                  <p>“What are the main points?”</p>
                </div>
                <a className="sample-download" href="/sample-research-summary.pdf" download>
                  Download a sample PDF to try <span aria-hidden="true">↗</span>
                </a>
              </div>
            )}

            {hasIndex && messages.length === 0 && (
              <div className="question-starters">
                <div className="starter-mark" aria-hidden="true">✳</div>
                <h3>What would you like to know?</h3>
                <p>Your document is indexed. Choose a question or write your own.</p>
                <div className="starter-list">
                  {EXAMPLE_QUESTIONS.map((example) => (
                    <button type="button" key={example} onClick={() => setQuestion(example)}>
                      <span>{example}</span><span aria-hidden="true">↗</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <article className={`message-row ${message.role === "user" ? "message-user" : "message-assistant"}`} key={message.id}>
                {message.role === "assistant" && <div className="assistant-avatar" aria-hidden="true">✦</div>}
                <div className="message-content">
                  <div className="message-label">{message.role === "user" ? "YOU" : "PAGEWISE"}</div>
                  <div className={`message-bubble ${message.role === "user" ? "user-bubble" : "assistant-bubble"}`}>
                    <p>{message.content}</p>
                  </div>
                  {message.sources && message.sources.length > 0 && (
                    <div className="sources-list">
                      <span className="sources-title">SOURCES</span>
                      {message.sources.map((source, index) => (
                        <details className="source-card" key={`${message.id}-${source.chunk.id}`}>
                          <summary>
                            <span className="source-number">{index + 1}</span>
                            <span className="source-name">
                              <strong>{source.chunk.fileName}</strong>
                              <small>Page {source.chunk.page}</small>
                            </span>
                            {indexedDocument?.mode === "huggingface" && (
                              <span className="source-score">{source.score.toFixed(2)}</span>
                            )}
                            <span className="source-chevron" aria-hidden="true">⌄</span>
                          </summary>
                          <p>{source.chunk.text}</p>
                        </details>
                      ))}
                    </div>
                  )}
                </div>
              </article>
            ))}

            {phase === "asking" && (
              <div className="thinking-row">
                <div className="assistant-avatar" aria-hidden="true">✦</div>
                <div className="thinking-bubble"><span /><span /><span /></div>
                <small>{indexedDocument?.mode === "huggingface" ? "Searching passages and composing an answer…" : "Finding matching passages…"}</small>
              </div>
            )}
            <div ref={chatEnd} />
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <span aria-hidden="true">!</span><p>{error}</p>
              <button type="button" onClick={() => setError("")} aria-label="Dismiss error">×</button>
            </div>
          )}

          <form className="question-form" onSubmit={submitQuestion}>
            <label className="visually-hidden" htmlFor="question">Ask about your PDF</label>
            <input
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={hasIndex ? "Ask something about your document…" : "Prepare a PDF to start asking questions…"}
              disabled={!hasIndex || busy}
              maxLength={800}
              autoComplete="off"
            />
            <button className="send-button" type="submit" disabled={!hasIndex || busy || !question.trim()} aria-label="Send question">
              {phase === "asking" ? <span className="spinner spinner-light" /> : <span aria-hidden="true">↑</span>}
            </button>
          </form>
          <div className="chat-footnote">
            <span>Answers can make mistakes. Check the cited page for important details.</span>
            {hasIndex && messages.length > 0 && (
              <button type="button" onClick={() => setMessages([])} disabled={busy}>Clear chat</button>
            )}
          </div>
        </section>
      </section>

      <section className="how-it-works" id="how-it-works">
        <div className="how-heading">
          <span className="step-label">THE SIMPLE VERSION</span>
          <h2>From PDF to answer, in three steps.</h2>
        </div>
        <div className="how-grid">
          <div className="how-card"><span>01</span><h3>Read</h3><p>Text is extracted from each page right in your browser. The PDF itself is not uploaded.</p></div>
          <div className="how-card"><span>02</span><h3>Find</h3><p>Text is divided into small overlapping sections. Search finds the best passages for your question.</p></div>
          <div className="how-card"><span>03</span><h3>Answer</h3><p>In Hugging Face mode, the AI answers from those passages and shows the source pages.</p></div>
        </div>
      </section>

      <footer className="footer">
        <a className="brand footer-brand" href="/">
          <span className="brand-mark" aria-hidden="true">P</span>
          <span className="brand-name">Page<span>Wise</span></span>
        </a>
        <span>Made for curious readers. · PDF Assistant</span>
      </footer>
    </main>
  );
}
