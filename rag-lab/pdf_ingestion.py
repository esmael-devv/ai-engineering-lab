
import hashlib
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "pdf_chroma_db"


def ingest_pdf(pdf_path):
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError("The path is not a file.")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    # 1. Load the PDF page by page
    pages = PyPDFLoader(str(pdf_path)).load()

    print(f"Total PDF pages: {len(pages)}")

    # 2. Split pages into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["document_section"] = (
            "opening" if chunk.metadata.get("page") == 0
            else "body"
        )

    # Remove empty chunks
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError(
            "No extractable text found. "
            "Scanned PDFs may require OCR."
        )

    print(f"Total chunks: {len(chunks)}")

    # 3. Generate free local embeddings
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Identify the document using its contents
    document_id = hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()[:16]

    collection_name = f"pdf_{document_id}"

    # 5. Store embeddings persistently
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=str(DB_DIR)
    )

    # Stable IDs prevent duplicate chunks on reruns
    chunk_ids = [
        f"{document_id}-{i}"
        for i in range(len(chunks))
    ]

    db.add_documents(
        documents=chunks,
        ids=chunk_ids
    )

    print(f"\nIndexed collection: {collection_name}")
    print(f"Stored chunks: {db._collection.count()}")

    # 6. Verify page metadata
    print("\nFirst 3 chunks:")

    for chunk in chunks[:3]:
        page_number = chunk.metadata.get("page", 0) + 1

        print(f"\n--- Page {page_number} ---")
        print(chunk.page_content[:300])

    return db


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_ingestion.py <pdf_path>")
        sys.exit(1)

    ingest_pdf(sys.argv[1])
