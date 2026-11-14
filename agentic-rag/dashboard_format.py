
import re
from urllib.parse import urlparse


CITATION_RE = re.compile(r"\[(E[1-9][0-9]*)\]")


def safe_web_url(value):
    if not isinstance(value, str):
        return False

    parsed = urlparse(value)

    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
    )


def format_answer(answer, sources):
    """Link valid web citations without changing their IDs."""

    source_lookup = {
        item["id"]: item
        for item in sources
    }

    def replace_citation(match):
        citation_id = match.group(1)
        source = source_lookup.get(citation_id)

        if not source:
            return match.group(0)

        reference = source.get("reference")

        if (
            source.get("type") == "web"
            and safe_web_url(reference)
        ):
            return f"[[{citation_id}]]({reference})"

        return f"[{citation_id}]"

    return CITATION_RE.sub(replace_citation, answer)


def format_execution_summary(metrics):
    """Create compact performance and reliability summary."""

    return (
        f"**Tool calls:** {metrics['tool_calls']}  |  "
        f"**Successful:** {metrics['successful_calls']}  |  "
        f"**Failed:** {metrics['failed_calls']}  |  "
        f"**Total time:** {metrics['total_seconds']:.2f}s  |  "
        f"**Evidence:** {metrics['evidence_count']}"
    )


def format_evidence_status(metrics):
    """Display evidence status without overstating verification."""

    warnings = metrics.get("verification_warnings", [])

    if (
        not metrics.get("evidence_sufficient", False)
        or warnings
    ):
        message = (
            "### ⚠️ Evidence needs review\n"
            "Some requested information could not be "
            "established from the retrieved evidence."
        )
    else:
        message = (
            "### ✅ Basic evidence checks passed\n"
            "The answer passed the available checks. "
            "Individual claims have not been "
            "independently fact-verified."
        )

    if warnings:
        message += "\n\n" + "\n".join(
            f"- {warning}" for warning in warnings
        )

    return message
