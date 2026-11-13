
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# The project uses a hyphenated folder name.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from retrieval_service import (
    is_document_level_question,
    filter_evidence,
)
from rag_service import build_context, generate_answer


def make_document(text, page=0):
    return Document(
        page_content=text,
        metadata={"page": page}
    )


def test_document_level_question_detection():
    assert is_document_level_question(
        "What is the main objective of this document?"
    )

    assert not is_document_level_question(
        "How is PCA used in face recognition?"
    )


def test_context_preserves_page_numbers():
    docs = [
        make_document("PCA reduces dimensions.", page=0),
        make_document("Eigenfaces identify features.", page=8),
    ]

    context = build_context(docs)

    assert "[Page 1]" in context
    assert "[Page 9]" in context
    assert "PCA reduces dimensions." in context


def test_irrelevant_evidence_is_rejected():
    reranker = MagicMock()
    reranker.predict.return_value = [-10.0, -11.0]

    docs = [
        make_document("Shipping policy.", page=2),
        make_document("Software development.", page=3),
    ]

    selected, ranking = filter_evidence(
        question="What is the author's favorite language?",
        documents=docs,
        reranker=reranker,
        top_k=2,
    )

    assert len(selected) == 0
    assert len(ranking) == 2


def test_document_level_fallback():
    reranker = MagicMock()
    reranker.predict.return_value = [-7.0, -9.0]

    docs = [
        make_document(
            "Face recognition using PCA.",
            page=0,
        ),
        make_document(
            "Software development models.",
            page=3,
        ),
    ]

    selected, _ = filter_evidence(
        question="What is the main objective of this document?",
        documents=docs,
        reranker=reranker,
        top_k=2,
    )

    assert len(selected) == 1
    assert selected[0].metadata["page"] == 0


def test_empty_evidence_skips_groq():
    answer, elapsed = generate_answer(
        "Unknown question?",
        []
    )

    assert answer == "I couldn't find that in the document."
    assert elapsed == 0.0



def test_answer_generation_with_mocked_groq():

    # Create a fake response
    fake_response = MagicMock()
    fake_response.content = "PCA reduces dimensions. [Page 1]"

    # Create sample document evidence
    docs = [
        make_document(
            "PCA reduces dimensions.",
            page=0,
        )
    ]

    # Mock the entire LangChain chain
    # rather than its invoke method
    with patch("rag_service.chain") as mock_chain:

        mock_chain.invoke.return_value = fake_response

        answer, elapsed = generate_answer(
            "What does PCA do?",
            docs
        )

        assert "PCA reduces dimensions" in answer
        assert elapsed >= 0

        mock_chain.invoke.assert_called_once()

        # Verify the question and context were passed
        call_args = mock_chain.invoke.call_args.args[0]

        assert call_args["question"] == "What does PCA do?"
        assert "[Page 1]" in call_args["context"]
