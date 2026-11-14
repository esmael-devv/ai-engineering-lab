
import json

import pytest

from answer_generator import (
    generate_grounded_answer,
    extract_citations,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return FakeResponse(
            json.dumps(self.response)
        )


def make_evidence():
    return {
        "question": "Find Kafka documentation",
        "evidence": [
            {
                "id": "E1",
                "type": "web",
                "source": "Apache Kafka",
                "content": "Official Kafka website.",
                "reference": "https://kafka.apache.org/",
            },
            {
                "id": "E2",
                "type": "code",
                "source": "example.py",
                "content": "KAFKA_TOPIC = 'events'",
                "reference": "example.py:L10",
            },
        ],
        "stop_reason": "decision_finished",
    }


def test_valid_grounded_answer():
    model = FakeModel({
        "answer": (
            "Official documentation is available "
            "from Apache Kafka [E1]."
        ),
        "citations": ["E1"],
        "evidence_sufficient": True,
    })

    result = generate_grounded_answer(
        "Find Kafka docs",
        make_evidence(),
        model,
    )

    assert result["citations"] == ["E1"]
    assert result["evidence_sufficient"] is True


def test_multiple_valid_citations():
    model = FakeModel({
        "answer": (
            "Kafka documentation was found [E1]. "
            "A Kafka topic setting appears in code [E2]."
        ),
        "citations": ["E1", "E2"],
        "evidence_sufficient": True,
    })

    result = generate_grounded_answer(
        "Find Kafka docs and code",
        make_evidence(),
        model,
    )

    assert len(result["citations"]) == 2


def test_fabricated_citation_rejected():
    model = FakeModel({
        "answer": "Documentation is available [E99].",
        "citations": ["E99"],
        "evidence_sufficient": True,
    })

    with pytest.raises(ValueError):
        generate_grounded_answer(
            "Find Kafka docs",
            make_evidence(),
            model,
        )



def test_mismatched_citations_are_normalized():
    model = FakeModel({
        "answer": "See Kafka documentation [E1].",
        "citations": ["E2"],
        "evidence_sufficient": True,
    })

    result = generate_grounded_answer(
        "Find Kafka docs",
        make_evidence(),
        model,
    )

    assert result["citations"] == ["E1"]


def test_empty_evidence_skips_model():
    model = FakeModel({
        "answer": "Invented answer",
        "citations": [],
        "evidence_sufficient": True,
    })

    package = {
        "evidence": [],
        "stop_reason": "decision_finished",
    }

    result = generate_grounded_answer(
        "Find something",
        package,
        model,
    )

    assert model.calls == 0
    assert result["evidence_sufficient"] is False
    assert result["citations"] == []


def test_duplicate_inline_citations():
    assert extract_citations(
        "One claim [E1]. Another claim [E1] [E2]."
    ) == ["E1", "E2"]


def test_invalid_json_rejected():
    class InvalidModel:
        def invoke(self, messages):
            return FakeResponse("not JSON")

    with pytest.raises(ValueError):
        generate_grounded_answer(
            "Find Kafka docs",
            make_evidence(),
            InvalidModel(),
        )


def test_empty_question_rejected():
    model = FakeModel({
        "answer": "Example",
        "citations": [],
        "evidence_sufficient": False,
    })

    with pytest.raises(ValueError):
        generate_grounded_answer(
            "",
            make_evidence(),
            model,
        )


def test_redundant_declared_citations_are_normalized():
    model = FakeModel({
        "answer": "Kafka documentation is available [E1].",
        "citations": ["E1", "E2"],
        "evidence_sufficient": True,
    })

    result = generate_grounded_answer(
        "Find Kafka docs",
        make_evidence(),
        model,
    )

    assert result["citations"] == ["E1"]
