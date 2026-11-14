
from evidence_guard import (
    classify_code_evidence,
    apply_evidence_guard,
)


def make_answer(sufficient=True):
    return {
        "answer": "Kafka code was found [E1].",
        "citations": ["E1"],
        "evidence_sufficient": sufficient,
    }


def make_package(source, content):
    return {
        "evidence": [{
            "id": "E1",
            "type": "code",
            "source": source,
            "content": content,
            "reference": f"{source}:L10",
        }]
    }


def test_prompt_string_is_not_implementation():
    item = {
        "source": "agentic-rag/groq_decider.py",
        "content": '"Apache Kafka official documentation"',
    }

    assert classify_code_evidence(item) == (
        "example_or_literal"
    )


def test_test_fixture_is_not_implementation():
    item = {
        "source": "agentic-rag/tests/test_example.py",
        "content": "class KafkaProducer:",
    }

    assert classify_code_evidence(item) == (
        "example_or_test"
    )


def test_class_definition_is_candidate():
    item = {
        "source": "src/KafkaProducer.py",
        "content": "class KafkaProducer:",
    }

    assert classify_code_evidence(item) == (
        "implementation_candidate"
    )


def test_unsupported_code_downgrades_answer():
    package = make_package(
        "agentic-rag/groq_decider.py",
        '"Apache Kafka official documentation"',
    )

    result = apply_evidence_guard(
        "Find Kafka integration code",
        make_answer(),
        package,
    )

    assert result["evidence_sufficient"] is False
    assert result["verification_status"] == "needs_review"
    assert result["verification_warnings"]


def test_guard_never_upgrades_sufficiency():
    package = make_package(
        "src/KafkaProducer.py",
        "class KafkaProducer:",
    )

    result = apply_evidence_guard(
        "Find Kafka implementation",
        make_answer(sufficient=False),
        package,
    )

    assert result["evidence_sufficient"] is False
