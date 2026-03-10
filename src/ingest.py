# src/ingest.py
# -------------------------------------------------------
# This script does 3 things:
# 1. Fetches real financial news articles from NewsAPI
# 2. Splits them into small chunks of text
# 3. Stores those chunks in ChromaDB (your local vector database)
# -------------------------------------------------------

import os
import requests
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load your API keys from the .env file
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

# -------------------------------------------------------
# STEP 1: Fetch news articles from NewsAPI
# -------------------------------------------------------

def fetch_news(topic: str, max_articles: int = 20) -> list[dict]:
    """
    Fetches recent news articles about a given topic.
    
    Args:
        topic: e.g. "Apple stock", "Tesla earnings", "S&P 500"
        max_articles: how many articles to fetch (free tier allows 100/day)
    
    Returns:
        A list of articles, each with 'title', 'content', 'url', 'source'
    """
    print(f"\n📰 Fetching news about: '{topic}'...")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",       # Most recent first
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ NewsAPI error: {response.status_code} - {response.text}")
        return []

    articles = response.json().get("articles", [])
    print(f"✅ Found {len(articles)} articles")

    # Clean up the articles — keep only what we need
    cleaned = []
    for article in articles:
        # Skip articles with no content
        if not article.get("content") or not article.get("title"):
            continue

        cleaned.append({
            "title": article["title"],
            "content": article["content"],
            "url": article["url"],
            "source": article["source"]["name"],
            "published_at": article["publishedAt"],
        })

    return cleaned


# -------------------------------------------------------
# STEP 2: Split articles into chunks
# -------------------------------------------------------

def chunk_article(article: dict, chunk_size: int = 500) -> list[dict]:
    """
    Splits a long article into smaller chunks of text.
    
    Why do we chunk? Because embedding models work best on
    short pieces of text, and we want precise retrieval —
    not entire articles dumped into the LLM context.
    
    Args:
        article: a single article dict from fetch_news()
        chunk_size: max characters per chunk (500 is a good start)
    
    Returns:
        A list of chunks, each with the text + metadata
    """
    # Combine title + content for better context
    full_text = f"{article['title']}. {article['content']}"

    # Split into chunks by character count
    chunks = []
    words = full_text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for the space

        if current_length >= chunk_size:
            chunks.append({
                "text": " ".join(current_chunk),
                "source": article["source"],
                "url": article["url"],
                "title": article["title"],
                "published_at": article["published_at"],
            })
            current_chunk = []
            current_length = 0

    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "source": article["source"],
            "url": article["url"],
            "title": article["title"],
            "published_at": article["published_at"],
        })

    return chunks


# -------------------------------------------------------
# STEP 3: Store chunks in ChromaDB
# -------------------------------------------------------

def store_chunks(chunks: list[dict], collection_name: str = "market_news"):
    """
    Embeds each chunk and stores it in ChromaDB.
    
    What is embedding? It converts text into a list of numbers
    that captures the meaning of the text. Similar texts get
    similar numbers — this is what makes semantic search work.
    
    Args:
        chunks: list of chunk dicts from chunk_article()
        collection_name: the ChromaDB "table" to store in
    """
    print(f"\n🧠 Loading embedding model...")
    # This model runs locally on your machine — no API call needed
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    print(f"💾 Connecting to ChromaDB at {CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Get or create the collection (like a table in a database)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
    )

    print(f"📦 Storing {len(chunks)} chunks in ChromaDB...")

    # Process in batches to avoid memory issues
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]

        texts = [chunk["text"] for chunk in batch]
        
        # Create embeddings for each chunk
        embeddings = model.encode(texts).tolist()

        # Create unique IDs for each chunk
        ids = [f"chunk_{i + j}" for j, _ in enumerate(batch)]

        # Store metadata so we can show sources later
        metadatas = [{
            "source": chunk["source"],
            "url": chunk["url"],
            "title": chunk["title"],
            "published_at": chunk["published_at"],
        } for chunk in batch]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"✅ Successfully stored {len(chunks)} chunks!")
    print(f"📊 Total chunks in database: {collection.count()}")


# -------------------------------------------------------
# MAIN: Run the full ingestion pipeline
# -------------------------------------------------------

def ingest(topics: list[str], max_articles_per_topic: int = 10):
    """
    Full pipeline: fetch → chunk → store for a list of topics.
    
    Args:
        topics: list of topics to fetch news for
        max_articles_per_topic: how many articles per topic
    """
    all_chunks = []

    for topic in topics:
        # Fetch articles
        articles = fetch_news(topic, max_articles=max_articles_per_topic)

        # Chunk each article
        for article in articles:
            chunks = chunk_article(article)
            all_chunks.extend(chunks)

    print(f"\n📝 Total chunks to store: {len(all_chunks)}")

    # Store all chunks in ChromaDB
    if all_chunks:
        store_chunks(all_chunks)
    else:
        print("⚠️  No chunks to store. Check your NewsAPI key.")


if __name__ == "__main__":
    # These are the topics we'll fetch news about
    # You can change these to anything you're interested in
    topics = [
        "Apple stock",
        "Tesla stock",
        "S&P 500",
        "Federal Reserve interest rates",
        "Microsoft earnings",
    ]

    ingest(topics, max_articles_per_topic=10)