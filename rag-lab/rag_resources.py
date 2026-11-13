
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "pdf_chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder():
    print("Loading embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_reranker():
    print("Loading reranker...")

    return CrossEncoder(
        RERANKER_MODEL
    )


def load_collection(collection_name):
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(DB_DIR),
        embedding_function=get_embedder()
    )
