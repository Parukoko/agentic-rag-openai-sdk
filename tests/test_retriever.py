from agentic_rag.retriever import search, tokenize

SAMPLE_CHUNKS = [
    "Cat Domestication: Cats were domesticated from African wildcats roughly "
    "10,000 years ago in the Near East.",
    "Cat Behavior: Cats communicate through meowing, purring, and body "
    "language such as tail position.",
    "Cat Nutrition: Cats are obligate carnivores and require taurine from "
    "animal-based food.",
]


def test_search_ranks_most_relevant_chunk_first():
    result = search("How were cats domesticated?", SAMPLE_CHUNKS)
    assert result.startswith("Cat Domestication")


def test_search_ignores_terms_common_to_most_chunks():
    # "Cats" appears in every chunk, so it should not, by itself, count as a match.
    result = search("What is the cats' policy on drone deliveries?", SAMPLE_CHUNKS)
    assert result == "No relevant information found in the knowledge base."


def test_search_returns_message_when_query_has_no_keywords():
    result = search("???", SAMPLE_CHUNKS)
    assert result == "No searchable keywords found in the query."


def test_tokenize_strips_punctuation_and_stopwords():
    assert tokenize("What is the diet policy?") == {"diet"}
