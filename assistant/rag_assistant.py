"""
RAG-based shopping assistant using LangChain + OpenAI.

Exposes:
  - ask_assistant(question) -> dict  (async)
  - build_vectorstore() — one-time build from products.csv

When run as __main__, builds the vectorstore from scratch.
"""

import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from monitoring.metrics import observe_llm_tokens

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")

MAX_QUESTION_LENGTH = 1000

# Category keywords for filtering
CATEGORY_KEYWORDS = {
    "Electronics": ["phone", "laptop", "headphone", "speaker", "monitor", "keyboard", "mouse", "tablet", "camera", "tv", "watch", "earbuds"],
    "Clothing": ["shirt", "dress", "jeans", "pants", "jacket", "sweater", "hoodie", "t-shirt", "blazer", "cloth"],
    "Footwear": ["shoes", "sneaker", "boot", "sandal", "flip-flop", "slipper", "footwear"],
    "Books": ["book", "guide", "manual", "programming", "python", "javascript"],
    "Sports": ["badminton", "cricket", "basketball", "yoga", "weights", "dumbbell", "sport", "fitness", "exercise"],
    "Home": ["mixer", "grinder", "blender", "purifier", "fan", "coolant", "heater", "kettle", "appliance", "home"],
    "Beauty": ["makeup", "foundation", "lipstick", "cream", "lotion", "shampoo", "skincare", "perfume", "beauty"],
    "Toys": ["toy", "game", "puzzle", "action", "doll", "lego", "board game"],
}

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


def _extract_relevant_categories(question: str) -> list[str]:
    """
    Extract likely product categories from the user's question.
    Returns list of matching category names.
    """
    question_lower = question.lower()
    matched_categories = []
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            matched_categories.append(category)
    
    return matched_categories if matched_categories else list(CATEGORY_KEYWORDS.keys())


def _extract_budget(question: str) -> tuple[float | None, float | None]:
    """
    Extract budget/price range from the question.
    Returns (min_price, max_price) or (None, None) if not found.
    
    Looks for patterns like:
    - "under 5000" -> max_price=5000
    - "below 1000" -> max_price=1000
    - "budget 500-1000" -> min=500, max=1000
    - "200 to 500" -> min=200, max=500
    """
    question_lower = question.lower()
    min_price = None
    max_price = None
    
    # Pattern: "under/below/within X"
    under_match = re.search(r'(?:under|below|within|less than|max|upto)\s+(?:₹)?\s*(\d+(?:,\d{3})*)', question_lower)
    if under_match:
        max_price = float(under_match.group(1).replace(',', ''))
    
    # Pattern: "X to Y" or "X-Y"
    range_match = re.search(r'(\d+(?:,\d{3})*)\s*(?:to|-|\bas\s)\s*(\d+(?:,\d{3})*)', question_lower)
    if range_match:
        min_price = float(range_match.group(1).replace(',', ''))
        max_price = float(range_match.group(2).replace(',', ''))
    
    # Pattern: "budget/price X"
    budget_match = re.search(r'(?:budget|price)\s+(?:of\s+)?(?:₹)?\s*(\d+(?:,\d{3})*)', question_lower)
    if budget_match and not range_match:
        max_price = float(budget_match.group(1).replace(',', ''))
    
    return (min_price, max_price)


def _filter_and_rank_documents(
    docs: list[Document],
    question: str,
    max_results: int = 5,
) -> list[Document]:
    """
    Filter documents by category and price, rank by relevance.
    Returns top 3-5 high-confidence products.
    """
    # Extract category and budget filters
    relevant_categories = _extract_relevant_categories(question)
    min_price, max_price = _extract_budget(question)
    
    # Filter by category
    filtered_docs = [
        doc for doc in docs
        if doc.metadata.get("category") in relevant_categories
    ]
    
    # Filter by price range if budget mentioned
    if min_price is not None or max_price is not None:
        filtered_docs = [
            doc for doc in filtered_docs
            if (
                (min_price is None or doc.metadata.get("price", float('inf')) >= min_price)
                and (max_price is None or doc.metadata.get("price", float('inf')) <= max_price)
            )
        ]
    
    # If no results after filtering, relax constraints
    if not filtered_docs:
        # Relax category constraint, keep price
        filtered_docs = docs
        if min_price is not None or max_price is not None:
            filtered_docs = [
                doc for doc in filtered_docs
                if (
                    (min_price is None or doc.metadata.get("price", float('inf')) >= min_price)
                    and (max_price is None or doc.metadata.get("price", float('inf')) <= max_price)
                )
            ]
    
    # Return top results (3-5, or fewer if not available)
    return filtered_docs[:max_results]


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# Build the LCEL chain
def _build_chain(vectorstore: FAISS):
    """Build the RAG chain using LCEL (without retriever - we handle retrieval separately)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        prompt
        | _llm
        | StrOutputParser()
    )
    return chain


# Lazy-loaded module state
_vectorstore = None
_chain = None


def _ensure_loaded():
    """Lazy-load vectorstore and chain on first use."""
    global _vectorstore, _chain
    if _vectorstore is None:
        _vectorstore = _load_vectorstore()
        _chain = _build_chain(_vectorstore)


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

    # Phase 1: Retrieve candidates (get more than we need for filtering)
    raw_docs = await _vectorstore.asimilarity_search(question, k=10)
    
    # Phase 2: Filter by category and price (with smart fallback)
    filtered_docs = _filter_and_rank_documents(raw_docs, question, max_results=5)
    
    # Phase 3: Format context for the prompt
    context = _format_docs(filtered_docs)
    
    # Phase 4: Invoke the chain with the filtered context and question
    config = {}
    # Token tracking (best-effort, depends on LangChain version)
    get_openai_callback = None
    try:
        from langchain_community.callbacks import get_openai_callback as _get_openai_callback  # type: ignore

        get_openai_callback = _get_openai_callback
    except Exception:
        try:
            from langchain.callbacks import get_openai_callback as _get_openai_callback  # type: ignore

            get_openai_callback = _get_openai_callback
        except Exception:
            get_openai_callback = None

    if get_openai_callback:
        with get_openai_callback() as cb:
            answer = await _chain.ainvoke(
                {"context": context, "question": question},
                config=config,
            )

        try:
            observe_llm_tokens(
                model=getattr(_llm, "model_name", getattr(_llm, "model", "unknown")),
                prompt_tokens=int(getattr(cb, "prompt_tokens", 0)),
                completion_tokens=int(getattr(cb, "completion_tokens", 0)),
                total_tokens=int(getattr(cb, "total_tokens", 0)),
            )
        except Exception:
            pass
    else:
        answer = await _chain.ainvoke(
            {"context": context, "question": question},
            config=config,
        )
    
    # Extract unique product IDs from filtered documents
    source_product_ids = list(dict.fromkeys(
        doc.metadata["product_id"]
        for doc in filtered_docs
        if "product_id" in doc.metadata
    ))

    return {
        "answer": answer,
        "source_products": source_product_ids,
    }


if __name__ == "__main__":
    build_vectorstore()
    print("Done! Vectorstore built successfully.")
