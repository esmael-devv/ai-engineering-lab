
import os
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from retrieval_service import select_evidence
from rag_service import generate_answer
from rag_resources import load_collection, get_reranker


from pdf_ingestion import ingest_pdf


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "pdf_chroma_db"

# Shared .env location from your previous projects
load_dotenv(
    Path(__file__).resolve().parents[2] / ".env"
)

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not found in .env")


# =========================================================
# 2. INITIALIZE MODELS
# =========================================================

print("Loading embedding model...")

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


# =========================================================
# 5. DOCUMENT-LEVEL QUESTION DETECTION
# =========================================================

DOCUMENT_LEVEL_PHRASES = [
    "main objective",
    "main purpose",
    "main goal",
    "primary objective",
    "what is this document about",
    "what is this paper about",
    "what is the document about",
    "what is the paper about",
    "summarize this document",
    "summarize the paper",
    "summary of this document",
    "summary of the paper"
]


# =========================================================
#  PDF UPLOAD AND INDEXING
# =========================================================

def upload_pdf(pdf_path):

    if not pdf_path:
        return (
            None,
            "Please select a PDF.",
            [],
            "",
            ""
        )

    start = time.perf_counter()

    try:

        # Reuse the ingestion pipeline
        db = ingest_pdf(pdf_path)

        collection_name = db._collection.name
        chunk_count = db._collection.count()

        elapsed = time.perf_counter() - start

        status = (
            f"✅ **Indexed:** {Path(pdf_path).name}\n\n"
            f"**Chunks:** {chunk_count} | "
            f"**Indexing:** {elapsed:.2f}s\n\n"
            f"**Collection:** `{collection_name}`"
        )

        return (
            collection_name,
            status,
            [],
            "",
            ""
        )

    except Exception as exc:

        return (
            None,
            f"❌ PDF indexing failed: {exc}",
            [],
            "",
            ""
        )


# =========================================================
# RAG QUESTION ANSWERING
# =========================================================


def ask_pdf(question, history, collection_name, strategy):

    history = list(history or [])
    question = (question or "").strip()

    # 1. Validate user input
    if not question:
        return history, "", "", ""

    if not collection_name:
        return (
            history,
            question,
            "Please upload a PDF first.",
            ""
        )

    try:
        # 2. Load the active Chroma collection
        db = load_collection(collection_name)

        # 3. Retrieve, rerank, and filter evidence
        docs, ranking_details, retrieval_metrics = select_evidence(
            db=db,
            question=question,
            strategy=strategy,
            reranker=get_reranker()
        )

        retrieval_time = retrieval_metrics["retrieval_time"]
        rerank_time = retrieval_metrics["rerank_time"]
        document_aware = retrieval_metrics["document_aware"]

        print("\n--- RETRIEVAL SUMMARY ---")
        print(f"Strategy: {strategy}")
        print(
            f"Candidate chunks: "
            f"{retrieval_metrics['candidate_count']}"
        )
        print(f"Selected chunks: {len(docs)}")
        print(f"Total retrieval: {retrieval_time:.3f}s")
        print(f"Reranking: {rerank_time:.3f}s")

        # 4. Build the LLM context
        context = "\n\n".join(
            (
                f"[Page {doc.metadata.get('page', 0) + 1}]\n"
                f"{doc.page_content}"
            )
            for doc in docs
        )

        # 5. Prepare source evidence for Gradio
        evidence_parts = []

        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", 0) + 1

            evidence_parts.append(
                f"### Source {i} — Page {page}\n\n"
                f"{doc.page_content[:800]}"
            )

        evidence = (
            "\n\n---\n\n".join(evidence_parts)
            if evidence_parts
            else "No evidence passed the relevance filter."
        )

        # 6. Generate answer using the RAG service
        answer, llm_time = generate_answer(
            question=question,
            documents=docs
        )


        # 7. Identify retrieved source pages
        pages = sorted({
            doc.metadata.get("page", 0) + 1
            for doc in docs
        })

        source_list = (
            ", ".join(f"Page {p}" for p in pages)
            if pages
            else "None"
        )

        # These identify selected evidence pages.
        # They are not independently verified citations.
        answer_display = (
            f"{answer}\n\n"
            f"**Retrieved pages:** {source_list}"
        )

        # 8. Update conversation history
        history.extend([
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer_display
            }
        ])

        # 9. Display performance metrics
        # retrieval_time already includes reranking.
        metrics = (
            f"**Strategy:** {strategy} | "
            f"**Document-aware:** "
            f"{'Yes' if document_aware else 'No'} | "
            f"**Candidates:** "
            f"{retrieval_metrics['candidate_count']} | "
            f"**Selected chunks:** {len(docs)} | "
            f"**Total retrieval:** {retrieval_time:.3f}s | "
            f"**Reranking:** {rerank_time:.3f}s | "
            f"**Groq:** {llm_time:.3f}s"
        )

        return (
            history,
            "",
            evidence,
            metrics
        )

    except Exception as exc:
        error_message = f"❌ Error: {exc}"

        print(error_message)

        history.extend([
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": error_message
            }
        ])

        return (
            history,
            "",
            error_message,
            ""
        )


# =========================================================
# 10. GRADIO USER INTERFACE
# =========================================================

with gr.Blocks(
    title="PDF Intelligence Lab"
) as demo:

    gr.Markdown(
        """
# 📄 PDF Intelligence Lab

### Retrieval-Augmented Generation with Groq

Upload a research paper or PDF and ask questions
grounded in its contents.

**Features**
- Local embeddings
- Chroma vector search
- Similarity vs MMR retrieval
- Document-aware retrieval
- Page citations
- Performance metrics
"""
    )

    # Per-session selected collection
    collection_state = gr.State(None)

    with gr.Row():

        pdf_input = gr.File(
            label="Upload your PDF",
            file_types=[".pdf"],
            type="filepath"
        )

        with gr.Column():

            strategy = gr.Radio(
                choices=[
                    "Similarity",
                    "MMR"
                ],
                value="Similarity",
                label="Retrieval Strategy"
            )

            status = gr.Markdown(
                "Upload a PDF to begin."
            )

    chatbot = gr.Chatbot(
        label="Document Assistant",
        height=450
    )

    with gr.Row():

        question = gr.Textbox(
            label="Your Question",
            placeholder="What is the main objective of this document?",
            scale=5
        )

        ask_button = gr.Button(
            "Ask",
            variant="primary",
            scale=1
        )

    gr.Markdown("### ⚡ Performance Metrics")

    metrics = gr.Markdown()

    with gr.Accordion(
        "🔎 Retrieved Evidence",
        open=False
    ):

        evidence = gr.Markdown()

    # PDF upload event
    pdf_input.upload(
        fn=upload_pdf,
        inputs=[pdf_input],
        outputs=[
            collection_state,
            status,
            chatbot,
            evidence,
            metrics
        ]
    )

    # Ask button event
    ask_button.click(
        fn=ask_pdf,
        inputs=[
            question,
            chatbot,
            collection_state,
            strategy
        ],
        outputs=[
            chatbot,
            question,
            evidence,
            metrics
        ]
    )

    # Enter key event
    question.submit(
        fn=ask_pdf,
        inputs=[
            question,
            chatbot,
            collection_state,
            strategy
        ],
        outputs=[
            chatbot,
            question,
            evidence,
            metrics
        ]
    )


# =========================================================
# 11. START APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch(
        share=False
    )
