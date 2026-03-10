# src/retriever.py
# -------------------------------------------------------
# This script searches your ChromaDB vector database
# and returns the most relevant chunks for a given query.
#
# Think of it like a smart search engine over your news articles.
# Instead of matching exact keywords, it matches MEANING.
# -------------------------------------------------------

import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

# Load the same embedding model we used in ingest.py
# We MUST use the same model — otherwise the numbers won't match
print("🧠 Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Connect to ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name="market_news",
    metadata={"hnsw:space": "cosine"}
)


# -------------------------------------------------------
# MAIN FUNCTION: Search the database
# -------------------------------------------------------

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Searches the vector database for chunks most relevant to the query.

    How it works:
    1. Converts your query into an embedding (list of numbers)
    2. Compares it against all stored embeddings using cosine similarity
    3. Returns the top_k most similar chunks

    Args:
        query:  the user's question e.g. "What's happening with Tesla?"
        top_k:  how many chunks to return (5 is a good default)

    Returns:
        A list of dicts, each with 'text', 'source', 'url', 'title', 'score'
    """

    # Convert the query into an embedding
    query_embedding = model.encode(query).tolist()

    # Search ChromaDB for the most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns nested lists — flatten them
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Package results into clean dicts
    chunks = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        # Convert distance to a similarity score (0 = identical, 2 = opposite)
        # We flip it so higher = more relevant (easier to understand)
        similarity_score = round(1 - distance, 3)

        chunks.append({
            "text": text,
            "title": metadata.get("title", "Unknown"),
            "source": metadata.get("source", "Unknown"),
            "url": metadata.get("url", ""),
            "published_at": metadata.get("published_at", ""),
            "score": similarity_score,
        })

    return chunks


def print_results(query: str, chunks: list[dict]):
    """
    Pretty-prints the retrieved chunks in the terminal.
    Useful for debugging and understanding what the retriever found.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Query: {query}")
    print(f"{'='*60}")

    if not chunks:
        print("❌ No results found.")
        return

    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 Result {i} — Score: {chunk['score']}")
        print(f"   Source : {chunk['source']}")
        print(f"   Title  : {chunk['title'][:80]}...")
        print(f"   URL    : {chunk['url']}")
        print(f"   Text   : {chunk['text'][:200]}...")
        print(f"   {'-'*56}")


# -------------------------------------------------------
# TEST: Run this file directly to test retrieval
# -------------------------------------------------------

if __name__ == "__main__":
    # Try a few test queries to make sure retrieval is working
    test_queries = [
        "What is happening with Tesla stock?",
        "What did the Federal Reserve decide about interest rates?",
        "How are Microsoft earnings performing?",
    ]

    for query in test_queries:
        chunks = retrieve(query, top_k=3)
        print_results(query, chunks)