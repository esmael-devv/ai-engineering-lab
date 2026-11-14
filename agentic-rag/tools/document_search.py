
import json
import sys
from pathlib import Path


# Existing Project 4 directory
REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_LAB = REPO_ROOT / "rag-lab"


def search_documents(collection_name, question):
    """
    Reuse Project 4's existing retrieval pipeline.

    Returns selected documents and retrieval metrics.
    """

    if str(RAG_LAB) not in sys.path:
        sys.path.insert(0, str(RAG_LAB))

    from rag_resources import load_collection, get_reranker
    from retrieval_service import select_evidence

    db = load_collection(collection_name)

    documents, _, metrics = select_evidence(
        db=db,
        question=question,
        strategy="Similarity",
        reranker=get_reranker()
    )

    return documents, metrics


def build_document_search_tool(
    collection_name,
    search_fn=None
):
    """
    Create an executor-compatible document search tool.

    search_fn is injectable for unit testing.
    """

    if not collection_name or not collection_name.strip():
        raise ValueError("Collection name is required")

    search_fn = search_fn or search_documents

    def document_search_tool(instruction, context):
        question = (instruction or "").strip()

        if not question:
            raise ValueError("Search question is required")

        documents, metrics = search_fn(
            collection_name,
            question
        )

        matches = []

        for document in documents:
            page_index = document.metadata.get("page")

            page_number = (
                page_index + 1
                if isinstance(page_index, int)
                else None
            )

            source = document.metadata.get("source")

            matches.append({
                "page": page_number,
                "source": (
                    Path(str(source)).name
                    if source else None
                ),
                "content": document.page_content
            })

        result = {
            "query": question,
            "collection": collection_name,
            "found": bool(matches),
            "matches": matches,
            "candidate_count": metrics.get(
                "candidate_count", 0
            ),
            "selected_count": len(matches)
        }

        return json.dumps(result, ensure_ascii=False)

    return document_search_tool
