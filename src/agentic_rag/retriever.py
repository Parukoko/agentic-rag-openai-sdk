"""Custom RAG tool: keyword-overlap search over the local knowledge base.

The scoring logic (`search`) is a plain, dependency-free function so it can
be unit tested without touching disk or the Agents SDK. `search_knowledge_base`
is the thin wrapper exposed to the Data Retriever agent as a tool.
"""

import os
import string
from pathlib import Path

from agents import function_tool

DEFAULT_KB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base.txt"
KB_PATH = Path(os.environ.get("KNOWLEDGE_BASE_PATH", DEFAULT_KB_PATH))

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "on", "in", "to", "for", "and", "or", "but", "not", "no", "so",
    "at", "by", "from", "into", "over", "under", "up", "down", "out",
    "what", "what's", "which", "who", "whom", "how", "why", "when", "where",
    "does", "do", "did", "doing", "have", "has", "had", "having",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "i", "you", "he", "she", "it", "we", "they", "them", "their", "theirs",
    "this", "that", "these", "those", "there", "here",
    "policy", "about", "with", "our", "your", "its",
    "also", "more", "most", "other", "than", "then", "than",
    "need", "needs", "want", "wants", "get", "gets",
}


def load_chunks(path: Path = KB_PATH) -> list[str]:
    """Split a knowledge-base text file into paragraph-level chunks."""
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, and drop stopwords/short tokens."""
    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def document_frequencies(chunks: list[str]) -> dict[str, int]:
    """Count how many chunks each word appears in, to down-weight ubiquitous terms."""
    df: dict[str, int] = {}
    for chunk in chunks:
        for word in tokenize(chunk):
            df[word] = df.get(word, 0) + 1
    return df


def search(query: str, chunks: list[str], max_results: int = 3) -> str:
    """Rank chunks by keyword overlap with the query and return the top matches.

    Query words appearing in more than half the chunks (e.g. a company name
    repeated in every paragraph) are ignored as too common to be a useful
    relevance signal. Falls back to the raw query words if that filtering
    removes everything.
    """
    query_words = tokenize(query)
    if not query_words:
        return "No searchable keywords found in the query."

    df = document_frequencies(chunks)
    informative_words = {w for w in query_words if df.get(w, 0) <= len(chunks) / 2}
    search_words = informative_words or query_words

    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        overlap = search_words & tokenize(chunk)
        if overlap:
            scored.append((len(overlap), chunk))

    if not scored:
        return "No relevant information found in the knowledge base."

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return "\n\n---\n\n".join(chunk for _, chunk in scored[:max_results])


@function_tool
def search_knowledge_base(query: str, max_results: int = 3) -> str:
    """Search the knowledge base for the paragraphs most relevant to a query.

    Args:
        query: The user's question or topic to search for.
        max_results: Maximum number of relevant chunks to return.
    """
    return search(query, load_chunks(), max_results)
