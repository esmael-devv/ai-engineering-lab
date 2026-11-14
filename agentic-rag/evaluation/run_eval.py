
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard_service import run_dashboard_request


EVAL_DIR = Path(__file__).resolve().parent
CASES_FILE = EVAL_DIR / "eval_cases.json"
REPORT_FILE = EVAL_DIR / "eval_report.json"

CITATION_PATTERN = re.compile(r"\[(E[1-9][0-9]*)\]")


def evaluate_case(case, run_agent=run_dashboard_request):
    question = case["question"]
    expected_tools = set(case.get("expected_tools", []))
    expected_text = case.get("expected_answer_contains", [])

    try:
        result = run_agent(question)

        answer = result["answer"]
        sources = result["sources"]
        trace = result["trace"]
        metrics = result["metrics"]

        successful_tools = {
            entry["tool"]
            for entry in trace
            if entry["success"]
        }

        all_tools = [entry["tool"] for entry in trace]

        source_ids = {
            source["id"] for source in sources
        }

        inline_citations = set(
            CITATION_PATTERN.findall(answer)
        )

        checks = {
            "tool_routing": expected_tools.issubset(
                successful_tools
            ),
            "answer_content": all(
                text.lower() in answer.lower()
                for text in expected_text
            ),
            "citation_validity": inline_citations.issubset(
                source_ids
            ),
            "tool_budget": metrics["tool_calls"] <= 5,
        }

        return {
            "id": case["id"],
            "question": question,
            "passed": all(checks.values()),
            "checks": checks,
            "expected_tools": sorted(expected_tools),
            "successful_tools": sorted(successful_tools),
            "tools_attempted": all_tools,
            "answer": answer,
            "cited_evidence_ids": sorted(inline_citations),
            "evidence_sufficient": metrics.get(
                "evidence_sufficient", False
            ),
            "verification_status": metrics.get(
                "verification_status"
            ),
            "verification_warnings": metrics.get(
                "verification_warnings", []
            ),
            "tool_calls": metrics["tool_calls"],
            "failed_calls": metrics["failed_calls"],
            "elapsed_seconds": metrics["total_seconds"],
            "error": None,
        }

    except Exception as error:
        return {
            "id": case["id"],
            "question": question,
            "passed": False,
            "checks": {},
            "error": f"{type(error).__name__}: {error}",
        }


def run_suite(cases, run_agent=run_dashboard_request):
    results = [
        evaluate_case(case, run_agent=run_agent)
        for case in cases
    ]

    total = len(results)
    passed = sum(item["passed"] for item in results)

    return {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(
                passed / total * 100, 2
            ) if total else 0,
        },
        "results": results,
    }


def main():
    cases = json.loads(
        CASES_FILE.read_text(encoding="utf-8")
    )

    print(f"\nRunning {len(cases)} evaluation cases...\n")

    report = run_suite(cases)

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}: {result['id']}")

        for check, passed in result["checks"].items():
            symbol = "PASS" if passed else "FAIL"
            print(f"  {symbol}: {check}")

        if result["error"]:
            print(f"  ERROR: {result['error']}")

    summary = report["summary"]

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']}%")
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
