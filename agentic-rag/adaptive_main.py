
import json

from adaptive_agent import run_adaptive_agent
from groq_decider import create_groq_decider
from planner import create_planner_model
from main import build_registry
from evidence_aggregator import aggregate_evidence
from answer_generator import generate_grounded_answer
from evidence_guard import apply_evidence_guard


def run_with_model(
    request,
    model,
    registry,
    max_tool_calls=5,
):
    """
    Connect a decision model to the adaptive executor.

    Works with both real Groq and mocked models.
    """

    decide_next = create_groq_decider(
        model=model,
        allowed_tools=registry.names(),
    )

    return run_adaptive_agent(
        request=request,
        registry=registry,
        decide_next=decide_next,
        max_tool_calls=max_tool_calls,
    )


def main():
    registry = build_registry()
    
    model = create_planner_model().bind(
        response_format={"type": "json_object"}
    )


    print("\nUniversal Knowledge Agent — Adaptive Mode")
    print("Type /exit to quit.\n")

    while True:
        request = input("Request: ").strip()

        if request.lower() == "/exit":
            break

        if not request:
            continue

        try:
            output = run_with_model(
                request=request,
                model=model,
                registry=registry,
            )


            evidence_package = aggregate_evidence(output)

            print("\n=== AGGREGATED EVIDENCE ===")
            print(
                json.dumps(
                    evidence_package,
                    indent=2,
                    ensure_ascii=False,
                )
            )


            answer_model = model.bind(
                response_format={"type": "json_object"}
            )

            grounded_answer = generate_grounded_answer(
                question=request,
                evidence_package=evidence_package,
                model=answer_model,
            )

            grounded_answer = apply_evidence_guard(
                question=request,
                grounded_answer=grounded_answer,
                evidence_package=evidence_package,
            )

            print("\n=== GROUNDED ANSWER ===")
            print(grounded_answer["answer"])

            print("\n=== CITATIONS ===")
            for citation_id in grounded_answer["citations"]:
                item = next(
                    evidence
                    for evidence in evidence_package["evidence"]
                    if evidence["id"] == citation_id
                )

                print(f"{citation_id}: {item['reference']}")

            print(
                "\nEvidence sufficient:",
                grounded_answer["evidence_sufficient"]
            )


            print(
                "Verification status:",
                grounded_answer["verification_status"],
            )

            for warning in grounded_answer["verification_warnings"]:
                print("Evidence warning:", warning)


            print("\n=== ADAPTIVE EXECUTION TRACE ===")

            for index, entry in enumerate(
                output["history"], start=1
            ):
                record = entry["record"]
                observation = entry["observation"]

                print(f"\nAttempt {index}")
                print("Tool:", record["tool"])
                print("Instruction:", record["instruction"])
                print("Success:", record["success"])
                print("Observation:", observation["status"])

                if not record["success"]:
                    print("Error:", record.get("error"))

            print("\n=== EXECUTION SUMMARY ===")
            print("Tool calls:", output["tool_calls"])
            print("Stop reason:", output["stop_reason"])

            if output["results"]:
                print("\n=== LAST TOOL RESULT ===")
                print(output["results"][-1]["result"])

            print(
                "\nNote: Answer synthesis is not "
                "implemented yet."
            )

        except Exception as error:
            print(
                f"\nAgent error: "
                f"{type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    main()
