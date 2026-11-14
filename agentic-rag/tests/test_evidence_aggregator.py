
import json

from evidence_aggregator import (
    aggregate_evidence,
    normalize_record,
)


def make_record(tool, result, success=True):
    return {
        "tool": tool,
        "success": success,
        "result": result,
    }


def make_execution(records):
    return {
        "request": "Find Kafka docs and code",
        "history": [
            {"record": record, "observation": {}}
            for record in records
        ],
        "tool_calls": len(records),
        "stop_reason": "decision_finished",
    }


def test_aggregate_web_and_code_evidence():
    web = make_record(
        "web_search",
        json.dumps({
            "results": [{
                "title": "Apache Kafka",
                "url": "https://kafka.apache.org/",
                "snippet": "Official Kafka website",
            }]
        })
    )

    code = make_record(
        "code_search",
        json.dumps({
            "results": [{
                "file": "kafka/producer.py",
                "line": 42,
                "snippet": "class KafkaProducer:",
            }]
        })
    )

    result = aggregate_evidence(
        make_execution([web, code])
    )

    assert result["evidence_count"] == 2
    assert result["evidence"][0]["id"] == "E1"
    assert result["evidence"][0]["type"] == "web"
    assert result["evidence"][1]["type"] == "code"
    assert result["evidence"][1]["reference"] == (
        "kafka/producer.py:L42"
    )


def test_failed_tool_provides_no_evidence():
    record = make_record(
        "web_search",
        None,
        success=False,
    )

    result = aggregate_evidence(
        make_execution([record])
    )

    assert result["has_evidence"] is False
    assert result["evidence_count"] == 0


def test_invalid_web_url_excluded():
    record = make_record(
        "web_search",
        json.dumps({
            "results": [{
                "title": "Invalid",
                "url": "javascript:alert(1)",
                "snippet": "Unsafe link",
            }]
        })
    )

    assert normalize_record(record) == []


def test_document_page_preserved():
    record = make_record(
        "document_search",
        json.dumps({
            "matches": [{
                "source": "research_paper.pdf",
                "page": 3,
                "content": "PCA reduces dimensionality.",
            }]
        })
    )

    evidence = normalize_record(record)

    assert evidence[0]["reference"] == (
        "research_paper.pdf:page:3"
    )


def test_memory_evidence_preserved():
    record = make_record(
        "memory_search",
        json.dumps({
            "results": [{
                "key": "favorite_language",
                "value": "Java",
            }]
        })
    )

    evidence = normalize_record(record)

    assert evidence[0]["type"] == "memory"
    assert evidence[0]["content"] == "Java"


def test_calculation_result():
    record = make_record(
        "calculator",
        "127.50",
    )

    evidence = normalize_record(record)

    assert evidence[0]["type"] == "calculation"
    assert evidence[0]["content"] == "127.50"


def test_duplicate_evidence_removed():
    record = make_record(
        "web_search",
        json.dumps({
            "results": [{
                "title": "Kafka",
                "url": "https://kafka.apache.org/",
                "snippet": "Official site",
            }]
        })
    )

    result = aggregate_evidence(
        make_execution([record, record])
    )

    assert result["evidence_count"] == 1


def test_evidence_limit():
    items = [
        {
            "title": f"Result {i}",
            "url": f"https://example.com/{i}",
            "snippet": f"Result text {i}",
        }
        for i in range(10)
    ]

    record = make_record(
        "web_search",
        json.dumps({"results": items})
    )

    result = aggregate_evidence(
        make_execution([record]),
        max_evidence=3,
    )

    assert result["evidence_count"] == 3


def test_malformed_json_ignored():
    record = make_record(
        "web_search",
        "not valid JSON",
    )

    assert normalize_record(record) == []


def test_path_traversal_rejected():
    record = make_record(
        "code_search",
        json.dumps({
            "results": [{
                "file": "../secret.txt",
                "line": 1,
                "snippet": "secret",
            }]
        })
    )

    assert normalize_record(record) == []



def test_later_tool_execution_is_preserved():
    web_results = [
        {
            "title": f"Web result {i}",
            "url": f"https://example.com/{i}",
            "snippet": f"Information {i}",
        }
        for i in range(10)
    ]

    web_record = make_record(
        "web_search",
        json.dumps({"results": web_results}),
    )

    code_record = make_record(
        "code_search",
        json.dumps({
            "results": [{
                "file": "src/kafka/Producer.java",
                "line": 25,
                "snippet": "public class Producer {",
            }]
        }),
    )

    result = aggregate_evidence(
        make_execution([web_record, code_record]),
        max_evidence=5,
    )

    assert result["evidence_count"] == 5

    assert any(
        item["reference"] == "src/kafka/Producer.java:L25"
        for item in result["evidence"]
    )


def test_multiple_code_searches_contribute_evidence():
    first_search = make_record(
        "code_search",
        json.dumps({
            "results": [
                {
                    "file": f"src/file_{i}.java",
                    "line": i + 1,
                    "snippet": f"Example code {i}",
                }
                for i in range(8)
            ]
        }),
    )

    second_search = make_record(
        "code_search",
        json.dumps({
            "results": [{
                "file": "src/KafkaProducer.java",
                "line": 50,
                "snippet": "class KafkaProducer {",
            }]
        }),
    )

    result = aggregate_evidence(
        make_execution([first_search, second_search]),
        max_evidence=4,
    )

    references = {
        item["reference"]
        for item in result["evidence"]
    }

    assert "src/KafkaProducer.java:L50" in references


def test_duplicate_results_across_groups_are_removed():
    record = make_record(
        "web_search",
        json.dumps({
            "results": [{
                "title": "Kafka",
                "url": "https://kafka.apache.org/",
                "snippet": "Kafka documentation",
            }]
        }),
    )

    result = aggregate_evidence(
        make_execution([record, record, record]),
    )

    assert result["evidence_count"] == 1
    assert result["evidence"][0]["id"] == "E1"


def test_empty_execution_returns_no_evidence():
    result = aggregate_evidence(
        make_execution([]),
    )

    assert result["evidence_count"] == 0
    assert result["has_evidence"] is False
    assert result["evidence"] == []
