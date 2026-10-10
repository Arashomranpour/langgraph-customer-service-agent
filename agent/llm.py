"""Groq chat model factory. The only place the LLM client is built."""

import os
from collections.abc import Callable
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

DEFAULT_MODEL = "openai/gpt-oss-120b"
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
PROMPT_GUARD_MAX_CHARS = 2000  # the classifier has a 512-token window


def _require_key() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    _require_key()
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=0,
        reasoning_effort="low",
        max_retries=int(os.getenv("GROQ_MAX_RETRIES", "2")),
    )


@lru_cache(maxsize=1)
def get_prompt_guard() -> Callable[[str], float]:
    """Llama Prompt Guard 2 on Groq: returns the probability (0-1) that text is a prompt injection or jailbreak."""
    _require_key()
    client = Groq(max_retries=int(os.getenv("GROQ_MAX_RETRIES", "2")))

    def injection_score(text: str) -> float:
        response = client.chat.completions.create(
            model=PROMPT_GUARD_MODEL, messages=[{"role": "user", "content": text[:PROMPT_GUARD_MAX_CHARS]}]
        )
        return float(response.choices[0].message.content)

    return injection_score
