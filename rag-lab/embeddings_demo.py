
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How can I return a purchased product?",
    "What is the refund policy?",
    "Python is a programming language.",
    "The weather is beautiful today."
]

vectors = model.encode(sentences, normalize_embeddings=True)

print("Embedding dimensions:", vectors.shape)

query = model.encode(
    "How do I get my money back?",
    normalize_embeddings=True
)

scores = vectors @ query

for idx in np.argsort(scores)[::-1]:
    print(f"{scores[idx]:.3f} | {sentences[idx]}")
