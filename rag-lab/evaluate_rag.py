
import json
import time
from pathlib import Path

from rag_resources import load_collection, get_reranker

from retrieval_service import (
    retrieve_documents,
    enhance_with_document_context,
    rerank_documents,
)


BASE_DIR = Path(__file__).resolve().parent

# Existing indexed research paper
COLLECTION_NAME = "pdf_b27573cba17ebf18"


def evaluate_retrieval():
    questions_path = BASE_DIR / "evaluation_questions.json"

    with open(questions_path, encoding="utf-8") as file:
        questions = json.load(file)

    db = load_collection(COLLECTION_NAME)
    reranker = get_reranker()

    print("\n=== RAG RETRIEVAL EVALUATION ===")

    results = []

    for test in questions:
        question = test["question"]
        expected_pages = set(test["expected_pages"])

        print(f"\nQuestion: {question}")

        for strategy in ["Similarity", "MMR"]:
            start = time.perf_counter()

            # Step 1: Retrieve candidates
            docs, _ = retrieve_documents(
                db, question, strategy
            )

            # Step 2: Add opening-page context
            docs, _ = enhance_with_document_context(
                db, question, docs
            )

            # Step 3: Cross-encoder reranking
            docs = rerank_documents(
                question=question,
                documents=docs,
                reranker=reranker,
                top_k=3,
            )

            elapsed = time.perf_counter() - start

            retrieved_pages = {
                doc.metadata.get("page", 0) + 1
                for doc in docs
            }

            # Hit@3 checks whether at least one expected
            # page appears in the top three results.
            if expected_pages:
                hit = bool(retrieved_pages & expected_pages)
                result = "PASS" if hit else "FAIL"
            else:
                hit = None
                result = "N/A"

            record = {
                "question": question,
                "category": test.get("category", "unknown"),
                "strategy": strategy,
                "expected_pages": sorted(expected_pages),
                "retrieved_pages": sorted(retrieved_pages),
                "hit_at_3": hit,
                "elapsed_seconds": round(elapsed, 3),
            }

            results.append(record)

            print(
                f"{strategy}: "
                f"Pages={sorted(retrieved_pages)} | "
                f"Hit@3={result} | "
                f"Time={elapsed:.3f}s"
            )

    print("\n=== SUMMARY ===")

    for strategy in ["Similarity", "MMR"]:
        answerable = [
            item for item in results
            if item["strategy"] == strategy
            and item["hit_at_3"] is not None
        ]

        hits = sum(
            item["hit_at_3"]
            for item in answerable
        )

        total = len(answerable)

        hit_rate = hits / total if total else 0

        print(
            f"{strategy}: "
            f"{hits}/{total} queries hit | "
            f"Hit@3={hit_rate:.1%}"
        )

    return results


if __name__ == "__main__":
    evaluate_retrieval()
