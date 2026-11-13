
import time

from langchain_core.documents import Document


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
    "summary of the paper",
]


def is_document_level_question(question):
    normalized = " ".join(question.lower().split())

    return any(
        phrase in normalized
        for phrase in DOCUMENT_LEVEL_PHRASES
    )


def retrieve_documents(db, question, strategy):
    start = time.perf_counter()

    if strategy == "MMR":
        docs = db.max_marginal_relevance_search(
            question,
            k=4,
            fetch_k=12,
            lambda_mult=0.7
        )
    else:
        docs = db.similarity_search(
            question,
            k=4
        )

    return docs, time.perf_counter() - start


def enhance_with_document_context(db, question, docs):
    docs = list(docs)

    if not is_document_level_question(question):
        return docs, False

    print("\nDocument-level question detected.")
    print("Retrieving opening-page context...")

    opening_data = db.get(
        where={"page": 0},
        include=["documents", "metadatas"]
    )

    opening_chunks = [
        Document(
            page_content=text,
            metadata=metadata or {}
        )
        for text, metadata in zip(
            opening_data["documents"],
            opening_data["metadatas"]
        )
        if text and text.strip()
    ]

    seen = {
        (doc.metadata.get("page"), doc.page_content)
        for doc in docs
    }

    added = 0

    for doc in opening_chunks[:8]:
        key = (doc.metadata.get("page"), doc.page_content)

        if key not in seen:
            docs.append(doc)
            seen.add(key)
            added += 1

    print(f"Added {added} opening-page chunks.")

    return docs, True


def score_documents(question, documents, reranker):
    if not documents:
        return []

    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        {
            "rank": rank,
            "page": doc.metadata.get("page", 0) + 1,
            "score": float(score),
            "document": doc,
        }
        for rank, (doc, score) in enumerate(ranked, 1)
    ]


def rerank_documents(
    question, documents, reranker, top_k=3
):
    ranked = score_documents(
        question, documents, reranker
    )

    return [
        item["document"]
        for item in ranked[:top_k]
    ]


def filter_evidence(
    question, documents, reranker, top_k=2
):
    if not documents:
        return [], []

    ranking_details = score_documents(
        question, documents, reranker
    )

    # Experimental threshold.
    # This is NOT a calibrated probability.
    MIN_SCORE = -5.0

    selected = [
        item["document"]
        for item in ranking_details
        if item["score"] >= MIN_SCORE
    ][:top_k]

    # Preserve the fallback from our successful
    # document-level regression test.
    if (
        not selected
        and is_document_level_question(question)
    ):
        opening_candidates = [
            item for item in ranking_details
            if item["page"] == 1
        ]

        if opening_candidates:
            selected = [
                opening_candidates[0]["document"]
            ]

            print(
                "Document-aware fallback: "
                "selected opening-page evidence."
            )

    print("\n--- EVIDENCE FILTER RESULTS ---")

    for item in ranking_details:
        status = (
            "ACCEPTED"
            if item["score"] >= MIN_SCORE
            else "REJECTED"
        )

        print(
            f"Rank {item['rank']} | "
            f"Page {item['page']} | "
            f"Score {item['score']:.4f} | "
            f"{status}"
        )

    print(f"Selected chunks: {len(selected)}")

    return selected, ranking_details


def select_evidence(db, question, strategy, reranker):
    """
    Complete retrieval pipeline.

    Returns:
        selected documents
        ranking details
        performance metrics
    """

    start = time.perf_counter()

    docs, search_time = retrieve_documents(
        db, question, strategy
    )

    docs, document_aware = enhance_with_document_context(
        db, question, docs
    )

    rerank_start = time.perf_counter()

    selected, ranking_details = filter_evidence(
        question,
        docs,
        reranker,
        top_k=2
    )

    rerank_time = time.perf_counter() - rerank_start
    retrieval_time = time.perf_counter() - start

    metrics = {
        "strategy": strategy,
        "document_aware": document_aware,
        "candidate_count": len(docs),
        "selected_count": len(selected),
        "search_time": search_time,
        "rerank_time": rerank_time,
        "retrieval_time": retrieval_time,
    }

    return selected, ranking_details, metrics
