
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY is missing")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
    timeout=60.0,
    max_retries=1
)

MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
]

SYSTEM_PROMPT = (
    "You are a knowledgeable technical assistant. "
    "Provide accurate, clear and concise answers."
)


def evaluate_model(model: str, prompt: str) -> dict:
    start_time = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_completion_tokens=1024
        )

        latency = time.perf_counter() - start_time

        usage = response.usage

        return {
            "model": model,
            "answer": (
                response.choices[0].message.content or ""
            ),
            "latency": latency,
            "input_tokens": (
                usage.prompt_tokens if usage else 0
            ),
            "output_tokens": (
                usage.completion_tokens if usage else 0
            ),
            "total_tokens": (
                usage.total_tokens if usage else 0
            ),
            "error": None
        }

    except OpenAIError as error:
        return {
            "model": model,
            "answer": "",
            "latency": time.perf_counter() - start_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": str(error)
        }


def compare_models(prompt: str):
    print("\n" + "=" * 65)
    print("LLM EVALUATION ARENA")
    print("=" * 65)

    results = []

    for model in MODELS:
        print(f"\nEvaluating {model}...")
        result = evaluate_model(model, prompt)
        results.append(result)

    for result in results:
        print("\n" + "-" * 65)
        print(f"MODEL: {result['model']}")
        print("-" * 65)

        if result["error"]:
            print(f"ERROR: {result['error']}")
            continue

        print(f"\nRESPONSE:\n{result['answer']}")
        print(f"\nLatency: {result['latency']:.2f} seconds")
        print(f"Input tokens: {result['input_tokens']}")
        print(f"Output tokens: {result['output_tokens']}")
        print(f"Total tokens: {result['total_tokens']}")

    successful = [
        r for r in results if r["error"] is None
    ]

    if len(successful) == 2:
        fastest = min(
            successful,
            key=lambda r: r["latency"]
        )

        print("\n" + "=" * 65)
        print("PERFORMANCE SUMMARY")
        print("=" * 65)
        print(f"Fastest response: {fastest['model']}")
        print(f"Time: {fastest['latency']:.2f} seconds")


def main():
    print("Welcome to LLM Evaluation Arena")
    print("Type 'exit' to quit")

    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not prompt:
            continue

        compare_models(prompt)


if __name__ == "__main__":
    main()
