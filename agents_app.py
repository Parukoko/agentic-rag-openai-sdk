"""Two-agent RAG system built with the OpenAI Agents SDK, using an
agent-as-tool pattern:

    Report Generator (Agent)
        -> uses Data Retriever (Agent, wrapped as a tool)
            -> uses search_knowledge_base (custom Python function tool)

Data Retriever: searches knowledge_base.txt and returns raw, relevant
snippets. It is instructed NOT to answer the question itself.

Report Generator: receives those snippets (by calling the Data Retriever
as a tool) and synthesizes a single, cohesive, well-formatted answer.

LLM: "gpt-5-mini" served through an Azure API Management gateway that
exposes an OpenAI-compatible Responses API (configured via environment
variables, see .env.example).
"""

import asyncio
import os

from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

from retriever_tool import search_knowledge_base

load_dotenv()

# The gateway exposes POST {LLM_API_BASE_URL}/responses, e.g.
#   https://apimsdbxcandidate01.azure-api.net/llm/responses
# and authenticates via an "api-key" header rather than "Authorization: Bearer".
LLM_API_BASE_URL = os.environ.get(
    "LLM_API_BASE_URL", "https://apimsdbxcandidate01.azure-api.net/llm"
)
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")

_client = AsyncOpenAI(
    base_url=LLM_API_BASE_URL,
    api_key=LLM_API_KEY,
    default_headers={"api-key": LLM_API_KEY},
)
set_default_openai_client(_client, use_for_tracing=False)
set_tracing_disabled(True)


data_retriever_agent = Agent(
    name="Data Retriever",
    instructions=(
        "You are an expert information-retrieval specialist. Your only job "
        "is to call the search_knowledge_base tool with the user's request "
        "and return the relevant raw text snippets it finds. "
        "Do NOT answer the question yourself, do NOT summarize or "
        "paraphrase, and do NOT add any commentary of your own — return the "
        "retrieved snippets essentially as-is so another agent can use them."
    ),
    tools=[search_knowledge_base],
    model=LLM_MODEL,
)

retrieve_information_tool = data_retriever_agent.as_tool(
    tool_name="retrieve_information",
    tool_description=(
        "Retrieve raw, relevant text snippets from the company knowledge "
        "base for a given question or topic."
    ),
)

report_generator_agent = Agent(
    name="Report Generator",
    instructions=(
        "You are an expert writer and synthesizer. For every user question, "
        "first call the retrieve_information tool to gather relevant raw "
        "snippets from the knowledge base. Then combine those snippets into "
        "a single, cohesive, non-redundant, well-formatted answer using "
        "clear prose and, where helpful, bullet points. "
        "Only use facts present in the retrieved snippets — never invent "
        "information, and never draw on outside or general knowledge. "
        "If the snippets do not answer the question, say plainly that the "
        "knowledge base does not contain that information, and stop there — "
        "do not offer to search public/external sources, browse the web, or "
        "speculate about what the answer might be."
    ),
    tools=[retrieve_information_tool],
    model=LLM_MODEL,
)


async def answer_query(query: str) -> str:
    """Run the full pipeline for a single query and return the final answer."""
    result = await Runner.run(report_generator_agent, query)
    return result.final_output


SAMPLE_QUERIES = [
    "What is the policy on international travel?",
    "How much PTO do employees get and does it carry over?",
    "What are the SLA and support response times for enterprise customers?",
    "What's Acme's policy on drone deliveries?",  # expected: not in KB
]


async def main() -> None:
    for query in SAMPLE_QUERIES:
        print(f"Q: {query}")
        answer = await answer_query(query)
        print(f"A: {answer}\n{'-' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
