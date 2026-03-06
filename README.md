# IITM BS Course Assistant — RAG Pipeline

A Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, 
and Mistral-7B to answer questions from IITM BS course PDFs.

## Tech Stack
- LangChain · FAISS Vector DB · HuggingFace Mistral-7B
- Prompt Engineering · Semantic Search · Python

## How it works
1. Loads a PDF and chunks it into 500-token segments
2. Embeds chunks into a FAISS vector store
3. On query, retrieves top-3 relevant chunks
4. Feeds context + question to Mistral-7B with a custom prompt

## Run
pip install langchain langchain-core langchain-community langchain-huggingface langchain-text-splitters faiss-cpu pypdf requests numpy
python ragpipeline.py
