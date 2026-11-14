
import json
import re

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from answer_prompt import ANSWER_SYSTEM_PROMPT


CITATION_PATTERN = re.compile(r"\[(E[1-9][0-9]*)\]")


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str]
    evidence_sufficient: bool


def extract_citations(answer):
    return list(dict.fromkeys(
        CITATION_PATTERN.findall(answer)
    ))


def validate_grounded_answer(
    answer,
    evidence_package,
):
    """
    Validate citation identifiers and consistency.

    This does NOT verify claim-level factual support.
    """

    allowed_ids = {
        item["id"]
        for item in evidence_package.get("evidence", [])
    }

    inline_ids = extract_citations(answer.answer)
    declared_ids = answer.citations

    if len(declared_ids) != len(set(declared_ids)):
        raise ValueError("Duplicate citation identifiers")

    if set(inline_ids) != set(declared_ids):
        raise ValueError(
            "Inline citations and citation list differ"
        )

    unknown = set(declared_ids) - allowed_ids

    if unknown:
        raise ValueError(
            f"Unsupported citations: {sorted(unknown)}"
        )

    if not allowed_ids:
        if answer.evidence_sufficient:
            raise ValueError(
                "Cannot claim sufficient evidence "
                "when none was retrieved"
            )

        if declared_ids:
            raise ValueError(
                "Cannot cite missing evidence"
            )

    return answer


def generate_grounded_answer(
    question,
    evidence_package,
    model,
):
    """
    Generate and validate a grounded response.

    The model is injectable for unit testing.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question is required")

    if not isinstance(evidence_package, dict):
        raise TypeError(
            "Evidence package must be a dictionary"
        )

    evidence = evidence_package.get("evidence", [])

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list")

    # Do not spend Groq tokens if nothing was retrieved.
    if not evidence:
        return {
            "answer": (
                "I could not establish an answer from "
                "the available evidence."
            ),
            "citations": [],
            "evidence_sufficient": False,
        }

    payload = {
        "question": question,
        "evidence": evidence,
        "stop_reason": evidence_package.get(
            "stop_reason"
        ),
    }

    response = model.invoke([
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(
            content=json.dumps(
                payload,
                ensure_ascii=False,
            )
        ),
    ])

    if not isinstance(response.content, str):
        raise ValueError(
            "Answer model must return JSON text"
        )

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Answer model returned invalid JSON"
        ) from error

    
    answer = GroundedAnswer.model_validate(parsed)

    # The answer text is the canonical source of
    # which evidence identifiers were actually cited.
    inline_citations = extract_citations(answer.answer)

    # Reject unknown citations before normalization.
    allowed_ids = {
        item["id"]
        for item in evidence_package.get("evidence", [])
    }

    unknown = set(inline_citations) - allowed_ids

    if unknown:
        raise ValueError(
            f"Unsupported inline citations: {sorted(unknown)}"
        )

    # Normalize the declared list to match valid
    # citations appearing in the answer.
    answer.citations = inline_citations

    validate_grounded_answer(
        answer,
        evidence_package,
    )


    return answer.model_dump()
