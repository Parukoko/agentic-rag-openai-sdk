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
knowledge_base.txt
```

- **Data Retriever** is instructed to only call `search_knowledge_base` and return the raw snippets it finds — no summarizing, no answering.
- **Data Retriever** is exposed to the **Report Generator** as a callable tool via `agent.as_tool(...)`, so the Report Generator automatically invokes it, receives the raw snippets, and synthesizes them into one non-redundant, well-formatted answer.
- `search_knowledge_base` is a plain Python function (`retriever_tool.py`) that reads `knowledge_base.txt`, splits it into paragraph-level chunks, and ranks chunks by keyword overlap with the query — a simple, dependency-free RAG mechanism.
- The LLM (`gpt-5-mini`) is served behind an Azure API Management gateway exposing an OpenAI-compatible **Responses API** at `POST {LLM_API_BASE_URL}/responses`, authenticated with an `api-key` header. `agents_app.py` points the Agents SDK's default OpenAI client at this gateway via `set_default_openai_client(...)`.

## Files

| File | Purpose |
|---|---|
| `knowledge_base.txt` | Sample knowledge base (fictional company policies + product info). |
| `retriever_tool.py` | Custom RAG tool: keyword-overlap search over `knowledge_base.txt`. |
| `agents_app.py` | Agent definitions, Azure OpenAI wiring, orchestration, sample run. |
| `.env.example` | Template for the Azure OpenAI credentials. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env with your real API key
```

The included `.env.example` is pre-filled with the gateway base URL and model given for this test:

- base URL: `https://apimsdbxcandidate01.azure-api.net/llm` (so requests go to `.../llm/responses`)
- model: `gpt-5-mini`

You only need to supply `LLM_API_KEY`.

> To use a plain OpenAI key instead of this gateway, drop the `default_headers` override and `base_url` in `agents_app.py` and just do `AsyncOpenAI(api_key=...)`, then set `model="gpt-4o-mini"` (or any chat model) on both agents.

## Run

```bash
python agents_app.py
```

This runs a few sample queries (including one with no answer in the knowledge base, to show the system doesn't hallucinate) and prints each query with its synthesized answer.

## Example query

```
Q: What is the policy on international travel?
A: <synthesized, well-formatted answer based only on the retrieved snippets>
```

Screenshots of runs for multiple queries are included in this repo (see `/screenshots`).
