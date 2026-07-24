"""Custom RAG tool: keyword-overlap search over a local knowledge_base.txt file."""

import os
import string

from agents import function_tool

KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base.txt")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "on", "in", "to", "for", "and",
    "what", "what's", "how", "does", "do", "policy", "about", "with", "our",
}


def _load_chunks() -> list[str]:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _tokenize(text: str) -> set[str]:
    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _document_frequencies(chunks: list[str]) -> dict[str, int]:
    """Count how many chunks each word appears in, to down-weight ubiquitous terms."""
    df: dict[str, int] = {}
    for chunk in chunks:
        for word in _tokenize(chunk):
            df[word] = df.get(word, 0) + 1
    return df


@function_tool
def search_knowledge_base(query: str, max_results: int = 3) -> str:
    """Search knowledge_base.txt for the paragraphs most relevant to a query.

    Uses keyword-overlap scoring between the query and each paragraph (chunk)
    in the knowledge base, ignoring words so common across chunks (e.g. a
    repeated company name) that they carry no discriminating signal, and
    returns the top-scoring raw text chunks.

    Args:
        query: The user's question or topic to search for.
        max_results: Maximum number of relevant chunks to return.
    """
    chunks = _load_chunks()
    query_words = _tokenize(query)
    if not query_words:
        return "No searchable keywords found in the query."

    # Drop query words that appear in more than half the chunks: they're too
    # common (e.g. a company name repeated in every paragraph) to indicate
    # relevance. Fall back to the full query if that filters everything out.
    df = _document_frequencies(chunks)
    informative_words = {w for w in query_words if df.get(w, 0) <= len(chunks) / 2}
    search_words = informative_words or query_words

    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        chunk_words = _tokenize(chunk)
        overlap = search_words & chunk_words
        if overlap:
            scored.append((len(overlap), chunk))

    if not scored:
        return "No relevant information found in the knowledge base."

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:max_results]]
    return "\n\n---\n\n".join(top_chunks)
