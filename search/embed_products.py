"""
Embed all products using OpenAI text-embedding-3-small and store in a FAISS index.

Produces:
  - search/faiss_index.bin
  - search/product_id_map.json
  - search/index_metadata.json
"""

import hashlib
import json
import os
import time

import faiss
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, wait_exponential

load_dotenv()

client = openai.OpenAI()

BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
PRODUCT_ID_MAP_PATH = os.path.join(BASE_DIR, "product_id_map.json")
INDEX_METADATA_PATH = os.path.join(BASE_DIR, "index_metadata.json")


def compute_csv_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for integrity tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_product_texts(df: pd.DataFrame) -> list[str]:
    """Create combined text strings for embedding."""
    texts = []
    for _, row in df.iterrows():
        text = f"{row['title']}. {row['description']}. Category: {row['category']}. Brand: {row['brand']}."
        texts.append(text)
    return texts


@retry(
    wait=wait_exponential(min=1, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
)
def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with automatic retry on rate limits."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def main():
    print(f"Loading products from {PRODUCTS_CSV}...")
    df = pd.read_csv(PRODUCTS_CSV)
    print(f"  Loaded {len(df)} products.")

    csv_hash = compute_csv_hash(PRODUCTS_CSV)
    texts = build_product_texts(df)
    product_ids = df["product_id"].tolist()

    all_embeddings = []
    start_time = time.time()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_embeddings = embed_batch(batch)
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} products...")

    elapsed = time.time() - start_time
    print(f"  Embedding complete in {elapsed:.1f}s")

    # Build FAISS index
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product (cosine after normalization)
    faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity
    index.add(embeddings_np)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"  Saved FAISS index to {FAISS_INDEX_PATH} ({index.ntotal} vectors)")

    # Save product ID mapping
    with open(PRODUCT_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(product_ids, f)
    print(f"  Saved product ID map to {PRODUCT_ID_MAP_PATH}")

    # Save metadata for integrity tracking
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_products": len(product_ids),
        "model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "source_csv_hash": csv_hash,
    }
    with open(INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved index metadata to {INDEX_METADATA_PATH}")

    print(f"Done! Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
