
import json
import time
from pathlib import Path

from pdf_chat_app import (
    load_collection,
    retrieve_documents,
    enhance_with_document_context,
    rerank_documents
)

BASE_DIR = Path(__file__).resolve().parent

COLLECTION_NAME = "pdf_b27573cba17ebf18"

with open(
    BASE_DIR / "evaluation_questions.json",
    encoding="utf-8"
) as file:
    questions = json.load(file)

db = load_collection(COLLECTION_NAME)

print("\n=== RAG RETRIEVAL EVALUATION ===")

for test in questions:
    question = test["question"]
    expected_pages = set(test["expected_pages"])

    print(f"\nQuestion: {question}")

    for strategy in ["Similarity", "MMR"]:
        start = time.perf_counter()

        docs, _ = retrieve_documents(
            db, question, strategy
        )

        docs, _ = enhance_with_document_context(
            db, question, docs
        )

        docs = rerank_documents(
            question, docs, top_k=3
        )

        elapsed = time.perf_counter() - start

        retrieved_pages = {
            doc.metadata.get("page", 0) + 1
            for doc in docs
        }

        # Did retrieval find at least one expected page?
        if expected_pages:
            hit = bool(retrieved_pages & expected_pages)
            result = "PASS" if hit else "FAIL"
        else:
            result = "N/A"

        print(
            f"{strategy}: "
            f"Pages={sorted(retrieved_pages)} | "
            f"Hit@3={result} | "
            f"Time={elapsed:.3f}s"
        )
