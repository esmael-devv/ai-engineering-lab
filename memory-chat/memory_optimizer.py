
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda


def optimize_history(inputs: dict, max_tokens: int = 1000):
    history = inputs.get("history", [])

    if not history or max_tokens <= 0:
        return {**inputs, "history": []}

    exchanges = []
    current = []

    # Group messages into complete human/AI exchanges
    for message in history:
        if message.type == "human":
            if current:
                current = []

            current = [message]

        elif message.type == "ai" and current:
            exchanges.append(current + [message])
            current = []

    retained = []
    used_tokens = 0

    # Process exchanges from newest to oldest
    for exchange in reversed(exchanges):
        exchange_tokens = count_tokens_approximately(exchange)

        # Stop instead of skipping newer exchanges
        if used_tokens + exchange_tokens > max_tokens:
            break

        retained = exchange + retained
        used_tokens += exchange_tokens

    return {
        **inputs,
        "history": retained
    }


def create_memory_optimizer(max_tokens: int = 1000):
    return RunnableLambda(
        lambda inputs: optimize_history(inputs, max_tokens)
    )
