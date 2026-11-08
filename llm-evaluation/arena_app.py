
import json
import os
import random
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
    timeout=60.0,
    max_retries=1
)

# Free-tier models available through Groq
MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# Add an optional fourth chat model through .env
fourth = os.getenv("ARENA_FOURTH_MODEL")
if fourth and fourth not in MODELS:
    MODELS.append(fourth)

VOTE_FILE = Path(__file__).parent / "votes.jsonl"
vote_lock = threading.Lock()


def ask_model(model, prompt):
    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful technical assistant. "
                        "Answer accurately and clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=4096,
            extra_body={
                "reasoning_format": "hidden",
                "reasoning_effort": (
                    "none" if model == "qwen/qwen3.8-27b"
                    else "low"
                )
            }
        )

        usage = response.usage

        return {
            "model": model,
            "answer": (
                response.choices[0].message.content
                or "(No text returned)"
            ),
            "latency": round(
                time.perf_counter() - start, 2
            ),
            "input_tokens": (
                usage.prompt_tokens if usage else 0
            ),
            "output_tokens": (
                usage.completion_tokens if usage else 0
            ),
            "error": None,
        }

    except Exception as error:
        return {
            "model": model,
            "error": str(error),
            "latency": round(
                time.perf_counter() - start, 2
            ),
        }


def make_card(letter, result, revealed=False):
    if result["error"]:
        return (
            f"### Contender {letter}\n\n"
            f"Request failed: {result['error']}"
        )

    heading = (
        result["model"] if revealed
        else f"Contender {letter} — Identity hidden"
    )

    card = f"### {heading}\n\n{result['answer']}"

    if revealed:
        card += (
            f"\n\n---\n"
            f"**Latency:** {result['latency']} seconds\n\n"
            f"**Input tokens:** {result['input_tokens']}\n\n"
            f"**Output tokens:** {result['output_tokens']}"
        )

    return card


def battle(prompt, selected_models):
    empty_cards = [""] * 4

    if not prompt or not prompt.strip():
        return (
            *empty_cards,
            gr.update(choices=[], value=None),
            "Enter a prompt first.",
            {},
        )

    if not 2 <= len(selected_models) <= 4:
        return (
            *empty_cards,
            gr.update(choices=[], value=None),
            "Select between 2 and 4 models.",
            {},
        )

    with ThreadPoolExecutor(
        max_workers=len(selected_models)
    ) as executor:
        results = list(
            executor.map(
                lambda m: ask_model(m, prompt),
                selected_models,
            )
        )

    successful = [
        result for result in results
        if result["error"] is None
    ]

    random.shuffle(successful)

    letters = "ABCD"
    cards = [""] * 4
    state = {}

    for index, result in enumerate(successful):
        letter = letters[index]
        state[letter] = result
        cards[index] = make_card(letter, result)

    failed = [
        result for result in results
        if result["error"] is not None
    ]

    choices = list(state.keys())

    status = (
        f"**{len(successful)} models responded.** "
        "Choose the best answer to reveal their identities."
    )

    if failed:
        status += (
            f"\n\n{len(failed)} model(s) failed. "
            "Check your terminal for details."
        )
        for result in failed:
            print(
                f"Model {result['model']} failed: "
                f"{result['error']}"
            )

    if len(choices) < 2:
        status = "Not enough models responded for a battle."

    return (
        *cards,
        gr.update(choices=choices, value=None),
        status,
        state,
    )



def submit_vote(choice, state):
    if not state or choice not in state:
        return (
            *[""] * 4,
            "Choose a contender from an active battle."
        )

    selected = state[choice]
    model = selected["model"]

    # Consider only successful responses.
    results = [
        result for result in state.values()
        if result["error"] is None
        and result.get("answer")
    ]

    fastest = min(
        results,
        key=lambda r: r["latency"]
    )

    lowest_tokens = min(
        results,
        key=lambda r: r["output_tokens"]
    )

    # Difference from the fastest model.
    if fastest["latency"] > 0:
        latency_diff = (
            (selected["latency"] - fastest["latency"])
            / fastest["latency"]
        ) * 100
    else:
        latency_diff = 0

    # Difference from lowest output token usage.
    if lowest_tokens["output_tokens"] > 0:
        token_diff = (
            (
                selected["output_tokens"]
                - lowest_tokens["output_tokens"]
            )
            / lowest_tokens["output_tokens"]
        ) * 100
    else:
        token_diff = None

    latency_description = (
        "Fastest response in this battle"
        if abs(latency_diff) < 0.01
        else f"{latency_diff:.1f}% slower than the fastest"
    )

    if token_diff is None:
        token_description = (
            "Token comparison unavailable"
        )
    elif abs(token_diff) < 0.01:
        token_description = (
            "Lowest output token usage in this battle"
        )
    else:
        token_description = (
            f"{token_diff:.1f}% more output tokens "
            "than the lowest-token response"
        )

    # Save the vote and comparison metrics.
    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "preferred_model": model,
        "latency": selected["latency"],
        "output_tokens": selected["output_tokens"],
        "latency_difference_percent": round(
            latency_diff, 2
        ),
        "token_difference_percent": (
            round(token_diff, 2)
            if token_diff is not None else None
        ),
        "compared_models": [
            result["model"] for result in results
        ],
    }

    with vote_lock:
        with VOTE_FILE.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(record) + "\n")

    cards = [""] * 4

    for index, (letter, result) in enumerate(
        state.items()
    ):
        cards[index] = make_card(
            letter,
            result,
            revealed=True
        )

    verdict = f"""
## ❤️ Your Preferred Response

You preferred **{model}**.

### Performance

| Metric | Your selected model | Comparison |
|---|---|---|
| Latency | {selected['latency']:.2f}s | {latency_description} |
| Input tokens | {selected['input_tokens']} | — |
| Output tokens | {selected['output_tokens']} | {token_description} |

**Fastest model:** {fastest['model']}
({fastest['latency']:.2f}s)

**Lowest output-token usage:** {lowest_tokens['model']}
({lowest_tokens['output_tokens']} tokens)

---

*These comparisons are based on this single battle.
They do not establish which model is best overall.*
"""

    return (*cards, verdict)


with gr.Blocks(
    title="AI Battle Arena",
) as demo:

    gr.Markdown(
        "# ⚔️ AI Battle Arena\n"
        "### One prompt. Multiple models. Your verdict.\n"
        "Compare responses anonymously and vote "
        "for the most useful answer."
    )

    model_selector = gr.CheckboxGroup(
        choices=MODELS,
        value=MODELS[:3],
        label="Choose your contenders (2–4)",
    )

    prompt = gr.Textbox(
        label="Challenge prompt",
        placeholder=(
            "Explain Kafka partition ordering "
            "and how it affects consumers..."
        ),
        lines=3,
    )

    fight_button = gr.Button(
        "⚔️ Start Battle",
        variant="primary",
    )

    with gr.Row():
        with gr.Column():
            card_a = gr.Markdown()
            card_c = gr.Markdown()
        with gr.Column():
            card_b = gr.Markdown()
            card_d = gr.Markdown()

    gr.Markdown("## 🗳️ Cast your vote")

    choice = gr.Radio(
        choices=[],
        label="Which contender gave the best answer?",
    )

    vote_button = gr.Button("Vote & Reveal")
    status = gr.Markdown()
    battle_state = gr.State({})

    cards = [card_a, card_b, card_c, card_d]

    fight_button.click(
        fn=battle,
        inputs=[prompt, model_selector],
        outputs=[
            *cards,
            choice,
            status,
            battle_state,
        ],
    )

    vote_button.click(
        fn=submit_vote,
        inputs=[choice, battle_state],
        outputs=[*cards, status],
    )


if __name__ == "__main__":
    demo.launch(share=False)
