"""
Build a BM25 keyword index over product data.

Produces:
  - search/bm25_index.pkl

Exposes:
  - build_bm25_index() — builds and saves the index
  - load_bm25() — loads the saved index
  - get_bm25_scores(query: str) -> list[float] — scores all documents for a query
"""

import os
import pickle
import re

import pandas as pd
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_product_corpus(df: pd.DataFrame) -> list[list[str]]:
    """Build tokenized corpus from product data."""
    corpus = []
    for _, row in df.iterrows():
        combined = f"{row['title']} {row['description']} {row['category']} {row['brand']}"
        corpus.append(tokenize(combined))
    return corpus


def build_bm25_index() -> BM25Okapi:
    """Build BM25 index from products.csv and save to disk."""
    print(f"Loading products from {PRODUCTS_CSV}...")
    df = pd.read_csv(PRODUCTS_CSV)
    print(f"  Loaded {len(df)} products.")

    print("Building BM25 index...")
    corpus = build_product_corpus(df)
    bm25 = BM25Okapi(corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved BM25 index to {BM25_INDEX_PATH}")

    return bm25


def load_bm25() -> BM25Okapi:
    """Load a previously saved BM25 index."""
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def get_bm25_scores(bm25: BM25Okapi, query: str) -> list[float]:
    """Return BM25 scores for all documents given a query string."""
    tokenized_query = tokenize(query)
    return bm25.get_scores(tokenized_query).tolist()


if __name__ == "__main__":
    build_bm25_index()
    print("Done! BM25 index built successfully.")
