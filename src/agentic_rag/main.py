"""CLI entrypoint: runs the Report Generator (which calls the Data
Retriever as a tool) against a handful of sample queries."""

import asyncio

from agents import Runner

from .agents import report_generator_agent

SAMPLE_QUERIES = [
    "What is the policy on international travel?",
    "How much PTO do employees get and does it carry over?",
    "What are the SLA and support response times for enterprise customers?",
    "What's Acme's policy on drone deliveries?",  # expected: not in KB
]


async def answer_query(query: str) -> str:
    """Run the full two-agent pipeline for a single query."""
    result = await Runner.run(report_generator_agent, query)
    return result.final_output


async def _run_samples() -> None:
    for query in SAMPLE_QUERIES:
        print(f"Q: {query}")
        answer = await answer_query(query)
        print(f"A: {answer}\n{'-' * 80}")


def run() -> None:
    asyncio.run(_run_samples())


if __name__ == "__main__":
    run()
