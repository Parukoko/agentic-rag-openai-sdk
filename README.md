# Agentic RAG — Data Retriever + Report Generator (OpenAI Agents SDK)

A minimal two-agent system built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), demonstrating an **agent-as-tool** orchestration pattern with a custom Retrieval-Augmented Generation (RAG) tool.

## Architecture

```
User query
   |
   v
Report Generator (Agent)  -- expert writer/synthesizer
   |
   | calls as a tool
   v
Data Retriever (Agent)    -- expert at retrieval only, never answers directly
   |
   | calls as a tool
   v
search_knowledge_base()   -- custom Python function: keyword-overlap search
   |
   v
data/knowledge_base.txt
```

- **Data Retriever** is instructed to only call `search_knowledge_base` and return the raw snippets it finds — no summarizing, no answering.
- **Data Retriever** is exposed to the **Report Generator** as a callable tool via `agent.as_tool(...)`, so the Report Generator automatically invokes it, receives the raw snippets, and synthesizes them into one non-redundant, well-formatted answer. It's instructed to stick strictly to retrieved content and never fall back to outside knowledge.
- `search_knowledge_base` is a thin wrapper around `search()` (`retriever.py`), a plain, dependency-free function that ranks paragraph-level chunks of the knowledge base by keyword overlap with the query, ignoring terms so common across chunks (e.g. a company name repeated in every paragraph) that they carry no relevance signal.
- The LLM (`gpt-5-mini`) is served behind an Azure API Management gateway exposing an OpenAI-compatible **Responses API** at `POST {LLM_API_BASE_URL}/responses`, authenticated with an `api-key` header. `config.py` points the Agents SDK's default OpenAI client at this gateway.

## Project layout

```
.
├── pyproject.toml              # package metadata + dependencies
├── requirements.txt            # convenience alias for `pip install -e .[dev]`
├── .env.example                # template for gateway credentials
├── data/
│   └── knowledge_base.txt      # sample knowledge base (fictional company policies)
├── src/agentic_rag/
│   ├── config.py                # env vars + LLM client wiring
│   ├── retriever.py              # RAG tool: search() + search_knowledge_base
│   ├── agents.py                  # Data Retriever + Report Generator agent definitions
│   └── main.py                    # CLI entrypoint / sample run
├── tests/
│   └── test_retriever.py       # unit tests for the retrieval/scoring logic
└── screenshots/                # sample run screenshots for submission
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # editable install of this package + pytest

cp .env.example .env
# then edit .env with your real API key
```

The included `.env.example` is pre-filled with the gateway base URL and model given for this test:

- base URL: `https://apimsdbxcandidate01.azure-api.net/llm` (so requests go to `.../llm/responses`)
- model: `gpt-5-mini`

You only need to supply `LLM_API_KEY`.

> To use a plain OpenAI key instead of this gateway, drop the `default_headers` override and `base_url` in `config.py` and just do `AsyncOpenAI(api_key=...)`, then set `model="gpt-4o-mini"` (or any chat model) in `agents.py`.

## Run

```bash
agentic-rag
# or: python -m agentic_rag.main
```

This runs a few sample queries (including one with no answer in the knowledge base, to show the system doesn't hallucinate) and prints each query with its synthesized answer.

## Test

```bash
pytest
```

Covers the keyword-overlap ranking logic in `retriever.py` directly, with no network/API calls involved.

## Example query

```
Q: What is the policy on international travel?
A: <synthesized, well-formatted answer based only on the retrieved snippets>
```

Screenshots of runs for multiple queries are included in this repo under `screenshots/`.
