# IITM BS Course Assistant - RAG Pipeline
# pip install langchain langchain-core langchain-community langchain-text-splitters faiss-cpu pypdf requests numpy

import os
import requests
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
PDF_PATH   = r"C:/Users/Md Saad Khan/Downloads/BS ES - Student Handbook.pdf"
HF_TOKEN   = "hf_your_new_token_here"   # <-- paste your NEW token here
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
TOP_K_DOCS = 3

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ── 2. CUSTOM EMBEDDINGS CLASS (calls HF API directly via requests) ───────────
class HFEmbeddings(Embeddings):
    def __init__(self, token: str):
        self.url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {token}"}

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Send in batches of 32 to avoid timeouts
        all_embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            response = requests.post(self.url, headers=self.headers, json={"inputs": batch, "options": {"wait_for_model": True}})
            response.raise_for_status()
            result = response.json()
            # Mean pool if token-level embeddings returned
            for item in result:
                if isinstance(item[0], list):
                    vec = np.mean(item, axis=0).tolist()
                else:
                    vec = item
                all_embeddings.append(vec)
        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

# ── 3. LOAD & CHUNK DOCUMENT ──────────────────────────────────────────────────
print("[1/4] Loading PDF...")
pages  = PyPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
).split_documents(pages)
print(f"      {len(pages)} pages -> {len(chunks)} chunks")

# ── 4. EMBED & INDEX INTO FAISS ───────────────────────────────────────────────
print("[2/4] Embedding chunks via HuggingFace API...")
embeddings   = HFEmbeddings(HF_TOKEN)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
print("      Vector store saved!")

# ── 5. BUILD RAG CHAIN ────────────────────────────────────────────────────────
print("[3/4] Building RAG chain...")

PROMPT_TEMPLATE = """You are a helpful IITM BS course assistant.
Use ONLY the context below to answer concisely.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""

prompt    = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
llm       = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                task="conversational",
                temperature=0.2,
                max_new_tokens=256,
                huggingfacehub_api_token=HF_TOKEN
            )
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_DOCS})

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# ── 6. INTERACTIVE Q&A LOOP ───────────────────────────────────────────────────
print("[4/4] Ready! Ask anything (or type 'quit').\n")
while True:
    query = input("You: ").strip()
    if query.lower() in {"quit", "exit", "q"}:
        break
    if not query:
        continue
    answer = rag_chain.invoke(query)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    print(f"\nAssistant: {answer}\n")
