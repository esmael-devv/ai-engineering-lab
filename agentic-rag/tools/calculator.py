
import json
from decimal import Decimal, InvalidOperation


SUPPORTED_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "percentage_of",
    "percent_change",
}


def parse_number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean values are not numbers")

    if not isinstance(value, (int, float, str)):
        raise ValueError("Expected a numeric value")

    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid number") from error

    if not number.is_finite():
        raise ValueError("Number must be finite")

    return number


def calculate(operation, a, b):
    """
    Perform a deterministic calculation.

    percentage_of:
        a percent of b

    percent_change:
        percentage change from a to b
    """

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    a = parse_number(a)
    b = parse_number(b)

    if operation == "add":
        result = a + b

    elif operation == "subtract":
        result = a - b

    elif operation == "multiply":
        result = a * b

    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b

    elif operation == "percentage_of":
        result = (a / 100) * b

    elif operation == "percent_change":
        if a == 0:
            raise ValueError(
                "Original value cannot be zero"
            )
        result = ((b - a) / a) * 100

    return result


def calculator_tool(instruction, context):
    """
    Executor-compatible tool.

    Expects a JSON instruction:
    {
        "operation": "add",
        "a": 10,
        "b": 20
    }

    Context is reserved for future tool-result
    references; it is not used in this version.
    """

    try:
        data = json.loads(instruction)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            "Calculator requires a JSON instruction"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "Calculator instruction must be an object"
        )

    required_keys = {"operation", "a", "b"}

    if set(data) != required_keys:
        raise ValueError(
            "Calculator requires exactly: "
            "operation, a, b"
        )

    result = calculate(
        operation=data["operation"],
        a=data["a"],
        b=data["b"],
    )

    return str(result)
