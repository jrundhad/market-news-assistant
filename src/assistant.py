# src/assistant.py
# -------------------------------------------------------
# This is the brain of the chatbot.
# It takes the chunks found by retriever.py and sends
# them to Claude along with the user's question.
#
# Flow:
# user question → retrieve chunks → build prompt → ask Claude → return answer
# -------------------------------------------------------

import os
import anthropic
from dotenv import load_dotenv
from retriever import retrieve

# Load your API keys from .env
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Create the Anthropic client (this is how we talk to Claude)
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# -------------------------------------------------------
# STEP 1: Build the prompt
# -------------------------------------------------------

def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Combines the retrieved news chunks + the user's question
    into a single prompt we send to Claude.

    This is called 'prompt engineering' — how you frame the
    question massively affects the quality of the answer.

    Args:
        question: the user's question
        chunks:   the relevant chunks from retriever.py

    Returns:
        A formatted string prompt ready to send to Claude
    """

    # Format each chunk with its source so Claude can cite it
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']} — {chunk['title'][:60]}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a financial news assistant. Your job is to answer questions about stocks and financial markets based ONLY on the news articles provided below.

Important rules:
- Only use information from the provided context
- If the context doesn't contain enough information to answer, say "I don't have enough recent news to answer that confidently"
- Always mention which sources you used
- Keep answers clear and easy to understand
- Do not give investment advice or recommendations

---
CONTEXT (Recent News Articles):
{context}

---
QUESTION: {question}

ANSWER:"""

    return prompt


# -------------------------------------------------------
# STEP 2: Ask Claude
# -------------------------------------------------------

def ask_claude(prompt: str) -> str:
    """
    Sends the prompt to Claude and returns its response.

    Args:
        prompt: the full prompt from build_prompt()

    Returns:
        Claude's answer as a string
    """
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text


# -------------------------------------------------------
# STEP 3: The main function — ties it all together
# -------------------------------------------------------

def answer(question: str, top_k: int = 5) -> dict:
    """
    Full pipeline: question → retrieve → prompt → Claude → answer

    Args:
        question: the user's question in plain English
        top_k:    how many news chunks to retrieve (default 5)

    Returns:
        A dict with the answer, sources used, and relevance scores
    """

    print(f"\n🔍 Searching news database...")
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant news articles to answer that question.",
            "sources": [],
            "chunks_used": 0,
        }

    # Check if the best match is relevant enough
    # If the top score is very low, the database probably
    # doesn't have articles on this topic
    best_score = chunks[0]["score"]
    if best_score < 0.5:
        return {
            "answer": "I don't have enough relevant news to answer that confidently. Try asking about Apple, Tesla, Microsoft, the S&P 500, or Federal Reserve rates.",
            "sources": [],
            "chunks_used": 0,
        }

    print(f"📰 Found {len(chunks)} relevant chunks (best score: {best_score})")
    print(f"💭 Asking Claude...")

    # Build the prompt and ask Claude
    prompt = build_prompt(question, chunks)
    response = ask_claude(prompt)

    # Collect unique sources used
    sources = []
    seen = set()
    for chunk in chunks:
        if chunk["url"] not in seen:
            sources.append({
                "title": chunk["title"],
                "source": chunk["source"],
                "url": chunk["url"],
            })
            seen.add(chunk["url"])

    return {
        "answer": response,
        "sources": sources,
        "chunks_used": len(chunks),
    }


def print_answer(question: str, result: dict):
    """
    Pretty-prints the answer and sources in the terminal.
    """
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    print(f"{'='*60}")
    print(f"\n💬 Answer:\n{result['answer']}")

    if result["sources"]:
        print(f"\n📚 Sources used:")
        for i, source in enumerate(result["sources"], 1):
            print(f"   {i}. {source['source']} — {source['title'][:60]}")
            print(f"      {source['url']}")

    print(f"\n{'='*60}\n")


# -------------------------------------------------------
# TEST: Run this file directly to test the assistant
# -------------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        "What is happening with Tesla stock?",
        "How is the S&P 500 performing?",
        "What did Microsoft report in their latest earnings?",
    ]

    for question in test_questions:
        result = answer(question)
        print_answer(question, result)