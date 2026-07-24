"""Agent definitions, wired together with the agent-as-tool pattern.

Data Retriever: calls the search_knowledge_base tool and returns raw,
relevant snippets. It is instructed NOT to answer the question itself.

Report Generator: exposes the Data Retriever as a callable tool
(`retrieve_information`) and, when asked a question, calls it to gather
snippets, then synthesizes them into a single, cohesive, well-formatted
answer grounded only in what was retrieved.
"""

from agents import Agent

from .config import LLM_MODEL
from .retriever import search_knowledge_base

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
