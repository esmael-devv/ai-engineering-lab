
import json
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse


MAX_EVIDENCE = 12
MAX_TEXT_LENGTH = 1200


def clean_text(value):
    if not isinstance(value, str):
        return ""
    return value.strip()[:MAX_TEXT_LENGTH]


def valid_web_url(url):
    if not isinstance(url, str):
        return False

    parsed = urlparse(url)

    return (
        parsed.scheme in {"https", "http"}
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
    )


def normalize_record(record):
    """
    Convert one successful tool execution into
    zero or more evidence items.

    Tool results are treated as untrusted data.
    """

    if not record.get("success"):
        return []

    tool = record.get("tool")
    raw = record.get("result")

    if tool == "calculator":
        if not isinstance(raw, str):
            return []

        try:
            number = Decimal(raw.strip())
        except (InvalidOperation, ValueError):
            return []

        if not number.is_finite():
            return []

        return [{
            "type": "calculation",
            "source": "calculator",
            "content": str(number),
            "reference": None,
        }]

    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []

    if not isinstance(data, dict):
        return []

    if tool == "web_search":
        items = data.get("results", [])
        if not isinstance(items, list):
            return []

        evidence = []

        for item in items[:MAX_EVIDENCE]:
            if not isinstance(item, dict):
                continue

            url = item.get("url")

            if not valid_web_url(url):
                continue

            evidence.append({
                "type": "web",
                "source": clean_text(item.get("title")),
                "content": clean_text(item.get("snippet")),
                "reference": url,
            })

        return evidence

    if tool == "code_search":
        items = data.get("results", [])
        if not isinstance(items, list):
            return []

        evidence = []

        for item in items[:MAX_EVIDENCE]:
            if not isinstance(item, dict):
                continue

            path = item.get("file")
            line = item.get("line")

            if not isinstance(path, str) or not path:
                continue

            if (
                path.startswith(("/", "\\"))
                or ".." in path.replace("\\", "/").split("/")
                or ":" in path
                or type(line) is not int
                or line < 1
            ):
                continue

            evidence.append({
                "type": "code",
                "source": path,
                "content": clean_text(item.get("snippet")),
                "reference": f"{path}:L{line}",
            })

        return evidence

    if tool == "document_search":
        items = data.get("matches", [])
        if not isinstance(items, list):
            return []

        evidence = []

        for item in items[:MAX_EVIDENCE]:
            if not isinstance(item, dict):
                continue

            source = item.get("source")
            page = item.get("page")

            if (
                not isinstance(source, str)
                or not source.strip()
                or type(page) is not int
                or page < 1
            ):
                continue

            evidence.append({
                "type": "document",
                "source": source,
                "content": clean_text(item.get("content")),
                "reference": f"{source}:page:{page}",
            })

        return evidence

    if tool == "memory_search":
        items = data.get("results", [])
        if not isinstance(items, list):
            return []

        evidence = []

        for item in items[:MAX_EVIDENCE]:
            if not isinstance(item, dict):
                continue

            key = item.get("key")
            value = item.get("value")

            if not isinstance(key, str) or not key.strip():
                continue

            if not isinstance(value, str) or not value.strip():
                continue

            evidence.append({
                "type": "memory",
                "source": key,
                "content": clean_text(value),
                "reference": f"memory:{key}",
            })

        return evidence

    return []



def aggregate_evidence(
    execution_output,
    max_evidence=MAX_EVIDENCE,
):
    """
    Aggregate evidence fairly across tool executions.

    Round-robin selection prevents earlier searches
    from exhausting the evidence budget.

    Preserves:
    - source references
    - execution order
    - duplicate removal
    - maximum evidence limit
    """

    if not isinstance(max_evidence, int) or isinstance(
        max_evidence, bool
    ):
        raise ValueError("Invalid evidence limit")

    if not 1 <= max_evidence <= MAX_EVIDENCE:
        raise ValueError("Invalid evidence limit")

    evidence_groups = []

    # Build an independent evidence group for
    # each successful tool execution.
    for attempt in execution_output.get("history", []):
        record = attempt.get("record", {})

        items = normalize_record(record)

        if items:
            evidence_groups.append(items)

    evidence = []
    seen = set()

    # Track our position within each tool result.
    positions = [0] * len(evidence_groups)

    while len(evidence) < max_evidence:
        progress = False

        for group_index, group in enumerate(evidence_groups):
            position = positions[group_index]

            # Skip already-consumed groups.
            if position >= len(group):
                continue

            item = group[position]
            positions[group_index] += 1
            progress = True

            identity = (
                item["type"],
                item["reference"],
                item["content"],
            )

            if identity in seen:
                continue

            seen.add(identity)

            evidence.append({
                "id": f"E{len(evidence) + 1}",
                **item,
            })

            if len(evidence) >= max_evidence:
                break

        if not progress:
            break

    return {
        "question": execution_output.get("request", ""),
        "evidence": evidence,
        "evidence_count": len(evidence),
        "has_evidence": bool(evidence),
        "stop_reason": execution_output.get("stop_reason"),
        "tool_calls": execution_output.get("tool_calls", 0),
    }
