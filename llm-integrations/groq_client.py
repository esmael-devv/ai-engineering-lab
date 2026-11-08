
import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"


# ============================================================
# VERSION 1: SINGLE QUESTION / SINGLE RESPONSE
# ============================================================
# Preserved for reference.
# Uncomment this section to run the original implementation.
#
# def ask_llm(prompt: str) -> str:
#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful AI assistant."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )
#
#     return response.choices[0].message.content or ""
#
#
# if __name__ == "__main__":
#     question = input("Enter your question: ")
#     print(ask_llm(question))


# ============================================================
# VERSION 2: MULTI-TURN CONVERSATION
# ============================================================
# Supports follow-up questions using conversation history.
# Commands:
#   exit  - End the conversation
#   clear - Reset conversation history
# ============================================================

SYSTEM_PROMPT = "You are a helpful AI assistant."


def main():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("Groq AI Chat")
    print("Commands: exit, clear")
    print("-" * 35)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if question.lower() == "clear":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            print("Conversation history cleared.")
            continue

        messages.append({
            "role": "user",
            "content": question
        })

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7
            )

            answer = response.choices[0].message.content or ""

            print(f"\nGroq: {answer}")

            messages.append({
                "role": "assistant",
                "content": answer
            })

        except OpenAIError as error:
            messages.pop()
            print(f"\nAPI error: {error}")


if __name__ == "__main__":
    main()
