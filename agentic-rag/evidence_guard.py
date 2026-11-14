
import re


IMPLEMENTATION_TERMS = re.compile(
    r"\b(implement|implementation|"
    r"producer|consumer|integration|"
    r"related code|source code)\b",
    re.IGNORECASE,
)

CODE_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\("),
    re.compile(r"^\s*class\s+\w+"),
    re.compile(
        r"^\s*(?:public|private|protected)?\s*"
        r"(?:abstract\s+|final\s+)?class\s+\w+"
    ),
    re.compile(r"^\s*(?:from|import)\s+\S+"),
    re.compile(r"^\s*[\w.]+\s*\("),
)


def classify_code_evidence(item):
    """
    Heuristic classification of a code-search match.

    This is not proof of executable implementation.
    """

    source = str(item.get("source", "")).lower()
    content = str(item.get("content", "")).strip()

    if (
        "/tests/" in f"/{source}"
        or source.startswith("tests/")
        or source.endswith("_test.py")
        or source.endswith(".md")
    ):
        return "example_or_test"

    if (
        content.startswith(('"', "'", "#", "//"))
        or content.startswith("<CodeBlock")
        or content.startswith('"instruction"')
        or content.startswith('"arguments"')
    ):
        return "example_or_literal"

    if any(
        pattern.search(content)
        for pattern in CODE_PATTERNS
    ):
        return "implementation_candidate"

    return "unknown"


def apply_evidence_guard(
    question,
    grounded_answer,
    evidence_package,
):
    """
    Apply conservative, deterministic safeguards.

    Never upgrades evidence_sufficient.
    Only downgrades an answer when a known
    evidence-quality problem is detected.
    """

    result = dict(grounded_answer)
    warnings = []

    evidence = evidence_package.get("evidence", [])

    if not evidence:
        result["evidence_sufficient"] = False

        warnings.append(
            "No supporting evidence was retrieved."
        )

    code_items = [
        item
        for item in evidence
        if item.get("type") == "code"
    ]

    if IMPLEMENTATION_TERMS.search(question):
        if code_items:
            candidates = [
                item
                for item in code_items
                if classify_code_evidence(item)
                == "implementation_candidate"
            ]

            if not candidates:
                result["evidence_sufficient"] = False

                warnings.append(
                    "Code search found references, but no "
                    "clear implementation candidates."
                )

    result["verification_warnings"] = warnings

    result["verification_status"] = (
        "needs_review"
        if warnings
        else "basic_checks_passed"
    )

    return result
