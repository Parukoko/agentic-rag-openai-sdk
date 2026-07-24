"""Environment configuration and LLM client wiring for the Agents SDK.

The model ("gpt-5-mini") is served behind an Azure API Management gateway
that exposes an OpenAI-compatible Responses API at
POST {LLM_API_BASE_URL}/responses, authenticating via an "api-key" header
rather than the SDK's default "Authorization: Bearer". Importing this
module points the Agents SDK's default OpenAI client at that gateway.
"""

import os

from agents import set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

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
