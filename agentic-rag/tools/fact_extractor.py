
import json
import re
from decimal import Decimal


REVENUE_PATTERN = re.compile(
    r"\bRevenue\s+in\s+"
    r"(?P<year>20\d{2})\s+was\s+"
    r"\$\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s+"
    r"(?P<unit>thousand|million|billion)\b",
    re.IGNORECASE,
)

MULTIPLIERS = {
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
}


def extract_revenue_facts(search_result):
    """
    Extract explicitly stated USD revenue facts.

    Every fact retains:
    - reporting year
    - normalized USD amount
    - source document
    - page number
    - exact supporting excerpt

    Unsupported text produces no facts.
    """

    matches = search_result.get("matches", [])

    if not isinstance(matches, list):
        raise ValueError("matches must be a list")

    facts_by_year = {}

    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("Invalid document match")

        content = match.get("content")
        source = match.get("source")
        page = match.get("page")

        if not isinstance(content, str):
            raise ValueError("Document content must be text")

        for found in REVENUE_PATTERN.finditer(content):
            if not isinstance(source, str) or not source:
                raise ValueError("Revenue fact has no source")

            if type(page) is not int or page < 1:
                raise ValueError("Revenue fact has no valid page")

            year = found.group("year")
            unit = found.group("unit").lower()

            raw_amount = found.group("amount")
            amount = Decimal(raw_amount.replace(",", ""))

            amount_usd = amount * MULTIPLIERS[unit]

            fact = {
                "year": year,
                "amount_usd": str(amount_usd),
                "currency": "USD",
                "source": source,
                "page": page,
                "evidence": found.group(0),
            }

            if year in facts_by_year:
                existing = facts_by_year[year]

                if existing["amount_usd"] != fact["amount_usd"]:
                    raise ValueError(
                        f"Conflicting revenue values for {year}"
                    )

                continue

            facts_by_year[year] = fact

    return {
        "fact_type": "annual_revenue",
        "facts": list(facts_by_year.values()),
        "count": len(facts_by_year),
    }


def fact_extractor_tool(instruction, context):
    """
    Extract facts from a previous document_search result.

    For this version, document_search must be the
    immediately preceding successful tool.
    """

    if not context.previous_results:
        raise ValueError("No previous tool results")

    previous = context.previous_results[-1]

    if (
        previous.get("tool") != "document_search"
        or not previous.get("success")
    ):
        raise ValueError(
            "Fact extraction requires a successful "
            "document_search result"
        )

    try:
        search_result = json.loads(previous["result"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Invalid document search result"
        ) from error

    if not isinstance(search_result, dict):
        raise ValueError("Search result must be an object")

    extracted = extract_revenue_facts(search_result)

    return json.dumps(extracted)
