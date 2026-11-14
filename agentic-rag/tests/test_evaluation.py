
from evaluation.run_eval import (
    evaluate_case,
    run_suite,
)


CASE = {
    "id": "test_01",
    "question": "Find Kafka documentation",
    "expected_tools": ["web_search"],
    "expected_answer_contains": ["kafka.apache.org"],
}


def fake_success(question):
    return {
        "answer": (
            "Official docs: kafka.apache.org [E1]"
        ),
        "sources": [
            {
                "id": "E1",
                "type": "web",
                "reference": "https://kafka.apache.org/",
            }
        ],
        "trace": [
            {
                "tool": "web_search",
                "success": True,
            }
        ],
        "metrics": {
            "tool_calls": 1,
            "failed_calls": 0,
            "total_seconds": 0.1,
            "evidence_sufficient": True,
            "verification_status": "basic_checks_passed",
            "verification_warnings": [],
        },
    }


def test_successful_evaluation():
    result = evaluate_case(
        CASE, run_agent=fake_success
    )
    assert result["passed"] is True


def test_missing_expected_tool_fails():
    def incorrect_tool(question):
        data = fake_success(question)
        data["trace"][0]["tool"] = "code_search"
        return data

    result = evaluate_case(
        CASE, run_agent=incorrect_tool
    )
    assert result["checks"]["tool_routing"] is False
    assert result["passed"] is False


def test_invalid_citation_fails():
    def invalid_citation(question):
        data = fake_success(question)
        data["answer"] = (
            "Official docs: kafka.apache.org [E99]"
        )
        return data

    result = evaluate_case(
        CASE, run_agent=invalid_citation
    )
    assert result["checks"]["citation_validity"] is False


def test_suite_summary():
    report = run_suite(
        [CASE, CASE],
        run_agent=fake_success,
    )

    assert report["summary"]["total"] == 2
    assert report["summary"]["passed"] == 2
    assert report["summary"]["pass_rate"] == 100


def test_tool_budget_failure():
    def over_budget(question):
        result = fake_success(question)
        result["metrics"]["tool_calls"] = 6
        return result

    result = evaluate_case(CASE, run_agent=over_budget)
    assert result["passed"] is False


def test_missing_expected_answer():
    def wrong_answer(question):
        result = fake_success(question)
        result["answer"] = "No relevant documentation found."
        return result

    result = evaluate_case(CASE, run_agent=wrong_answer)
    assert result["checks"]["answer_content"] is False


def test_agent_exception_is_reported():
    def failing_agent(question):
        raise RuntimeError("Simulated tool failure")

    result = evaluate_case(CASE, run_agent=failing_agent)

    assert result["passed"] is False
    assert "Simulated tool failure" in result["error"]
