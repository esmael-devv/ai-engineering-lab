
import json
import re
from decimal import Decimal, InvalidOperation


REFERENCE_PATTERN = re.compile(
    r"^\$\{step_([1-9][0-9]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\}$"
)


def resolve_value(value, context):
    """
    Resolve a numeric reference from a previous tool result.

    Example:
        ${step_1.revenue_2024}

    Ordinary values are returned unchanged.
    """

    if not isinstance(value, str):
        return value

    match = REFERENCE_PATTERN.fullmatch(value)

    if not match:
        # Reject malformed reference-like strings.
        if "${" in value:
            raise ValueError("Invalid tool reference")

        return value

    step_number = int(match.group(1))
    field_name = match.group(2)

    previous_results = context.previous_results

    if step_number > len(previous_results):
        raise ValueError(
            f"Step {step_number} has not completed"
        )

    record = previous_results[step_number - 1]

    # Never resolve from unsuccessful steps.
    if not record.get("success"):
        raise ValueError(
            f"Step {step_number} did not succeed"
        )

    try:
        data = json.loads(record["result"])
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        raise ValueError(
            "Referenced tool result is not valid JSON"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "Referenced result must be a JSON object"
        )

    if field_name not in data:
        raise ValueError(
            f"Field {field_name!r} not found in "
            f"step {step_number}"
        )

    raw_value = data[field_name]

    if isinstance(raw_value, bool) or not isinstance(
        raw_value, (int, float, str)
    ):
        raise ValueError(
            "Referenced value must be numeric"
        )

    try:
        number = Decimal(str(raw_value))
    except InvalidOperation as error:
        raise ValueError(
            "Referenced value is not numeric"
        ) from error

    if not number.is_finite():
        raise ValueError(
            "Referenced value must be finite"
        )

    return str(number)


FACT_REFERENCE_PATTERN = re.compile(
    r"^\$\{step_([1-9][0-9]*)"
    r"\.facts\[(20[0-9]{2})\]\.amount_usd\}$"
)


def resolve_fact_reference(value, context):
    """
    Resolve a revenue fact by reporting year.

    Returns:
        {
            "value": str,
            "provenance": dict
        }
    """

    match = FACT_REFERENCE_PATTERN.fullmatch(value)

    if not match:
        raise ValueError("Invalid fact reference")

    step_number = int(match.group(1))
    year = match.group(2)

    if step_number > len(context.previous_results):
        raise ValueError(
            f"Referenced step {step_number} has not completed"
        )

    record = context.previous_results[step_number - 1]

    if (
        not record.get("success")
        or record.get("tool") != "fact_extractor"
    ):
        raise ValueError(
            "Fact reference requires a successful "
            "fact_extractor step"
        )

    try:
        output = json.loads(record["result"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Invalid fact extraction result"
        ) from error

    if not isinstance(output, dict):
        raise ValueError("Fact output must be an object")

    if output.get("fact_type") != "annual_revenue":
        raise ValueError("Unsupported fact type")

    facts = output.get("facts")

    if not isinstance(facts, list):
        raise ValueError("Facts must be a list")

    matches = [
        fact for fact in facts
        if isinstance(fact, dict)
        and fact.get("year") == year
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one revenue fact for {year}"
        )

    fact = matches[0]

    if fact.get("currency") != "USD":
        raise ValueError("Expected USD revenue")

    source = fact.get("source")
    page = fact.get("page")
    evidence = fact.get("evidence")

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Missing fact source")

    if type(page) is not int or page < 1:
        raise ValueError("Invalid source page")

    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("Missing supporting evidence")

    raw_amount = fact.get("amount_usd")

    if (
        isinstance(raw_amount, bool)
        or not isinstance(raw_amount, (str, int, float))
    ):
        raise ValueError("Invalid revenue amount")

    try:
        number = Decimal(str(raw_amount))
    except InvalidOperation as error:
        raise ValueError("Revenue amount is not numeric") from error

    if not number.is_finite() or number < 0:
        raise ValueError("Revenue amount must be finite and nonnegative")

    return {
        "value": str(number),
        "provenance": {
            "year": year,
            "source": source,
            "page": page,
            "evidence": evidence,
            "origin_step": step_number,
        },
    }
