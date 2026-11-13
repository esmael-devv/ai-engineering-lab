
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from prompts import get_rag_prompt


# ==========================================
# 1. ENVIRONMENT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR.parents[1] / ".env"
)

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not found")


# ==========================================
# 2. INITIALIZE GROQ
# ==========================================

print("Initializing Groq RAG service...")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = get_rag_prompt()

chain = prompt | llm


# ==========================================
# 3. FORMAT SELECTED EVIDENCE
# ==========================================

def build_context(documents):

    return "\n\n".join(
        (
            f"[Page {doc.metadata.get('page', 0) + 1}]\n"
            f"{doc.page_content}"
        )
        for doc in documents
    )


# ==========================================
# 4. GENERATE GROUNDED ANSWER
# ==========================================

def generate_answer(question, documents):
    """
    Generate an answer using selected evidence.

    Returns:
        answer: str
        llm_time: float
    """

    # Never call Groq without evidence
    if not documents:
        print("No evidence selected. Skipping Groq.")

        return (
            "I couldn't find that in the document.",
            0.0
        )

    context = build_context(documents)

    start = time.perf_counter()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    llm_time = time.perf_counter() - start

    return str(response.content), llm_time
