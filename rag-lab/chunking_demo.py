
from langchain_text_splitters import RecursiveCharacterTextSplitter

document = """
Retrieval-Augmented Generation (RAG) is a technique that
helps language models answer questions using external data.

A RAG system first retrieves relevant information from
documents, databases, or other knowledge sources.

Embeddings convert text into numerical vectors that
represent semantic meaning. Similar text generally
produces vectors that are close together.

Vector databases store these embeddings and allow
efficient similarity searches.

The retrieved text is then passed to a language model
along with the user's question. The model uses this
context to generate a grounded answer.

Chunking is an important part of this process because
large documents need to be divided into smaller sections.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    length_function=len
)

chunks = splitter.split_text(document.strip())

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print(f"Characters: {len(chunk)}")
