
# PDF Intelligence Lab

A document-grounded RAG (Retrieval-Augmented Generation)
application built with Python, LangChain, ChromaDB,
local embeddings, Groq, and Gradio.

## Features

- Upload and index PDF documents
- Persistent ChromaDB vector storage
- Local HuggingFace embeddings
- Similarity and MMR retrieval
- Document-aware retrieval for summary questions
- Cross-encoder reranking
- Evidence filtering with a two-chunk limit
- Skip LLM calls when no evidence is selected
- Page references and retrieved evidence inspection
- Retrieval and generation performance metrics
- Automated unit tests with pytest

## Architecture

PDF Upload
    |
    v
PDF Ingestion and Chunking
    |
    v
Local Embeddings
    |
    v
ChromaDB
    |
    v
Similarity / MMR Retrieval
    |
    v
Document-Aware Context Enhancement
    |
    v
Cross-Encoder Reranking
    |
    v
Evidence Quality Filter
    |
    v
Groq LLM
    |
    v
Gradio Chat Interface

## Project Structure

- `pdf_chat_app.py` - Gradio interface
- `pdf_ingestion.py` - PDF processing and indexing
- `retrieval_service.py` - Retrieval and evidence selection
- `rag_service.py` - Groq answer generation
- `prompts.py` - Grounding instructions
- `evaluate_rag.py` - Retrieval evaluation
- `evaluation_questions.json` - Evaluation dataset
- `tests/` - Automated tests

## Setup

Recommended: Python 3.11 or 3.12.

1. Create a virtual environment:

   python -m venv .venv

2. Activate it on Windows:

   .venv\Scripts\Activate.ps1

3. Install dependencies:

   python -m pip install -r requirements.txt

4. Configure GROQ_API_KEY in a `.env` file.

   The current application expects `.env` two
   directory levels above `rag-lab`, at the
   FirstClass directory.

5. Start the application:

   python pdf_chat_app.py

## Testing

Run:

    python -m pytest tests/test_rag_services.py -v

Six unit tests cover:
- Document-level question detection
- Page-number preservation
- Evidence rejection
- Document-level fallback
- Skipping Groq when evidence is missing
- Mocked answer generation

## Known Limitations

- The cross-encoder score threshold is experimental
  and is not a calibrated confidence probability.
- Introductory-page fallback can admit irrelevant evidence.
- Page references are not independently verified citations.
- Retrieval evaluation uses a small, document-specific dataset.
- Persistent local storage is not a multi-user security boundary.
- The application is intended as a learning prototype,
  not a production document-processing service.

## Models

- Embeddings: all-MiniLM-L6-v2
- Reranker: cross-encoder/ms-marco-MiniLM-L6-v2
- LLM: openai/gpt-oss-20b via Groq
