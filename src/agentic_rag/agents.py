"""Agent definitions, wired together with the agent-as-tool pattern.

Data Retriever: calls the search_knowledge_base tool and returns raw,
relevant snippets. It is instructed NOT to answer the question itself, and
uses tool_use_behavior="stop_on_first_tool" so the tool's raw output is
returned directly as the agent's final output — skipping the extra LLM
round-trip it would otherwise take to restate that output.

Report Generator: exposes the Data Retriever as a callable tool
(`retrieve_information`) and, when asked a question, calls it to gather
snippets, then synthesizes them into a single, cohesive, well-formatted
answer grounded only in what was retrieved.
"""

from textwrap import dedent

from agents import Agent

from .config import LLM_MODEL
from .retriever import search_knowledge_base

DATA_RETRIEVER_INSTRUCTIONS = dedent("""\
    # Role
    You are an expert information-retrieval specialist.

    # Steps
    1. Read the user's request.
    2. Call the search_knowledge_base tool, passing the user's request as the query.
    3. Return the raw text snippets the tool returns.

    # Rules
    - Do NOT answer the question yourself.
    - Do NOT summarize, paraphrase, or add any commentary of your own.
    - Return the retrieved snippets essentially as-is so another agent can use them.
    """)

data_retriever_agent = Agent(
    name="Data Retriever",
    instructions=DATA_RETRIEVER_INSTRUCTIONS,
    tools=[search_knowledge_base],
    model=LLM_MODEL,
    # The tool's output already *is* the desired final output (raw
    # snippets), so skip the extra LLM call that would otherwise just
    # restate it — cuts one full round-trip out of the pipeline per query.
    tool_use_behavior="stop_on_first_tool",
)

retrieve_information_tool = data_retriever_agent.as_tool(
    tool_name="retrieve_information",
    tool_description=(
        "Retrieve raw, relevant text snippets from the knowledge base for "
        "a given question or topic."
    ),
)

REPORT_GENERATOR_INSTRUCTIONS = dedent("""\
    # Role
    You are an expert writer and synthesizer.

    # Steps
    1. Call the retrieve_information tool to gather relevant raw snippets from the knowledge base.
    2. Read the returned snippets and identify the facts relevant to the user's question.
    3. Combine those facts into a single, cohesive, non-redundant answer using clear prose and, where helpful, bullet points.

    # Rules
    - Only use facts present in the retrieved snippets — never invent information, and never draw on outside or general knowledge.
    - If the snippets do not answer the question, say plainly that the knowledge base does not contain that information, and stop there.
    - Do not offer to search public/external sources, browse the web, or speculate about what the answer might be.
    """)

report_generator_agent = Agent(
    name="Report Generator",
    instructions=REPORT_GENERATOR_INSTRUCTIONS,
    tools=[retrieve_information_tool],
    model=LLM_MODEL,
)
