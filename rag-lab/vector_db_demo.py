
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

documents = [
    "Customers may return products within 30 days with a receipt.",
    "Standard shipping takes 5 to 7 business days.",
    "Premium customers receive free shipping on all orders.",
    "Refunds are processed within 3 business days after approval.",
    "Customer support is available Monday through Friday."
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.create_documents(documents)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="rag_learning",
    embedding_function=embedder,
    persist_directory=str(DB_PATH)
)

# Use stable IDs so rerunning updates rather than duplicates.
db.add_documents(
    documents=chunks,
    ids=[f"sample-{i}" for i in range(len(chunks))]
)

print(f"Stored {db._collection.count()} chunks")

question = "How long does it take to get my money back?"

results = db.similarity_search_with_score(question, k=3)

print(f"\nQuestion: {question}")

for i, (doc, distance) in enumerate(results, start=1):
    print(f"\nResult {i}")
    print(f"Distance: {distance:.4f}")
    print(f"Text: {doc.page_content}")
