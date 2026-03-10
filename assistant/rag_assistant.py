"""
RAG-based shopping assistant using LangChain + OpenAI.

Exposes:
  - ask_assistant(question) -> dict  (async)
  - build_vectorstore() — one-time build from products.csv

When run as __main__, builds the vectorstore from scratch.
"""

import os
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")

MAX_QUESTION_LENGTH = 1000

SYSTEM_PROMPT = """\
You are a helpful shopping assistant for an e-commerce platform.
Your job is to help users find products and answer product-related questions.

RULES:
- Answer based on the product context provided below.
- Use your judgment to infer which products are relevant to the user's intent.
  For example, if they ask "home workout products", recommend dumbbells, yoga mats, and resistance bands.
- If no products are relevant, say: "I don't have that specific product, but we have: [suggest alternatives]"
- Never make up product names or specifications not in the context.
- Always mention the price in Indian Rupees (₹).

CONTEXT:
{context}"""

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def _build_documents(df: pd.DataFrame) -> list[Document]:
    """Convert product DataFrame rows into LangChain Documents."""
    docs = []
    for _, row in df.iterrows():
        page_content = (
            f"Product: {row['title']}\n"
            f"Brand: {row['brand']}\n"
            f"Category: {row['category']}\n"
            f"Price: ₹{row['price']}\n"
            f"Description: {row['description']}\n"
            f"Attributes: {row['attributes']}"
        )
        metadata = {
            "product_id": row["product_id"],
            "category": row["category"],
            "price": float(row["price"]),
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def build_vectorstore():
    """Build FAISS vectorstore from products.csv and save locally."""
    print(f"Loading products from {PRODUCTS_CSV}...")
    df = pd.read_csv(PRODUCTS_CSV)
    print(f"  Loaded {len(df)} products.")

    print("Building LangChain documents...")
    docs = _build_documents(df)

    print("Creating FAISS vectorstore (this calls OpenAI embeddings)...")
    start = time.time()
    vectorstore = FAISS.from_documents(docs, _embeddings)
    elapsed = time.time() - start
    print(f"  Vectorstore built in {elapsed:.1f}s")

    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"  Saved vectorstore to {VECTORSTORE_DIR}")
    return vectorstore


def _load_vectorstore() -> FAISS:
    """Load vectorstore from disk if it exists."""
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Vectorstore not found at {VECTORSTORE_DIR}. "
            "Run 'python assistant/rag_assistant.py' first to build it."
        )
    return FAISS.load_local(
        VECTORSTORE_DIR,
        _embeddings,
        allow_dangerous_deserialization=True,
    )


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# Build the LCEL chain
def _build_chain(vectorstore: FAISS):
    """Build the RAG chain using LCEL."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | _llm
        | StrOutputParser()
    )
    return chain, retriever


# Lazy-loaded module state
_vectorstore = None
_chain = None
_retriever = None


def _ensure_loaded():
    """Lazy-load vectorstore and chain on first use."""
    global _vectorstore, _chain, _retriever
    if _vectorstore is None:
        _vectorstore = _load_vectorstore()
        _chain, _retriever = _build_chain(_vectorstore)


async def ask_assistant(question: str) -> dict:
    """
    Ask the RAG shopping assistant a question.

    Args:
        question: User's question (1-1000 chars).

    Returns:
        {"answer": str, "source_products": list[str]}
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters.")

    question = question.strip()
    _ensure_loaded()

    # Get answer and source documents in parallel
    answer = await _chain.ainvoke(question)
    source_docs = await _retriever.ainvoke(question)

    # Extract unique product IDs from retrieved documents
    source_product_ids = list(dict.fromkeys(
        doc.metadata["product_id"]
        for doc in source_docs
        if "product_id" in doc.metadata
    ))

    return {
        "answer": answer,
        "source_products": source_product_ids,
    }


if __name__ == "__main__":
    build_vectorstore()
    print("Done! Vectorstore built successfully.")
