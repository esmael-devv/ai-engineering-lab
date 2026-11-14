
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from executor import ExecutionContext
from reference_resolver import resolve_value


def make_context(result):
    context = ExecutionContext(
        original_request="Calculate revenue growth"
    )

    context.previous_results.append({
        "step": 1,
        "tool": "web_search",
        "result": result,
        "success": True,
    })

    return context


def test_resolve_numeric_reference():
    context = make_context(
        '{"revenue_2024":100,"revenue_2025":120}'
    )

    assert resolve_value(
        "${step_1.revenue_2024}", context
    ) == "100"

    assert resolve_value(
        "${step_1.revenue_2025}", context
    ) == "120"


def test_literal_number_unchanged():
    context = make_context("{}")

    assert resolve_value(850, context) == 850


def test_missing_field_rejected():
    context = make_context('{"revenue":120}')

    with pytest.raises(ValueError):
        resolve_value(
            "${step_1.unknown}", context
        )


def test_future_step_rejected():
    context = make_context('{"revenue":120}')

    with pytest.raises(ValueError):
        resolve_value(
            "${step_2.revenue}", context
        )


def test_nonnumeric_reference_rejected():
    context = make_context('{"revenue":"unknown"}')

    with pytest.raises(ValueError):
        resolve_value(
            "${step_1.revenue}", context
        )


def test_malformed_reference_rejected():
    context = make_context('{"revenue":120}')

    with pytest.raises(ValueError):
        resolve_value(
            "${step_1.revenue", context
        )


def test_boolean_reference_rejected():
    context = make_context('{"revenue":true}')

    with pytest.raises(ValueError):
        resolve_value(
            "${step_1.revenue}", context
        )
