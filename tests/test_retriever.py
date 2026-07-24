from agentic_rag.retriever import search, tokenize

SAMPLE_CHUNKS = [
    "Acme Corp International Travel Policy: international travel requires "
    "approval and a valid passport.",
    "Acme Corp Domestic Travel Policy: domestic travel within the country is "
    "simpler and needs less notice.",
    "Acme Corp PTO Policy: employees accrue fifteen days of paid time off "
    "per year.",
]


def test_search_ranks_most_relevant_chunk_first():
    result = search("What is the international travel policy?", SAMPLE_CHUNKS)
    assert result.startswith("Acme Corp International Travel Policy")


def test_search_ignores_terms_common_to_most_chunks():
    # "Acme" appears in every chunk, so it should not, by itself, count as a match.
    result = search("What is Acme's drone delivery policy?", SAMPLE_CHUNKS)
    assert result == "No relevant information found in the knowledge base."


def test_search_returns_message_when_query_has_no_keywords():
    result = search("???", SAMPLE_CHUNKS)
    assert result == "No searchable keywords found in the query."


def test_tokenize_strips_punctuation_and_stopwords():
    assert tokenize("What is the PTO policy?") == {"pto"}
