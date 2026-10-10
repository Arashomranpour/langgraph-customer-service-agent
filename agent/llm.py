"""Groq client factories. The only place Groq clients are built.

The web app is bring-your-own-key: it calls `make_llm(api_key)` / `make_prompt_guard(api_key)` with the visitor's
key, which lives only in that browser session. `get_llm()` / `get_prompt_guard()` read GROQ_API_KEY from the
environment and are for the CLI and evals only.
"""

import os
from collections.abc import Callable
from functools import lru_cache

from dotenv import load_dotenv
from groq import AuthenticationError, Groq
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

DEFAULT_MODEL = "openai/gpt-oss-120b"
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
PROMPT_GUARD_MAX_CHARS = 2000  # the classifier has a 512-token window
REQUIRED_MODELS = (DEFAULT_MODEL, PROMPT_GUARD_MODEL)


def _max_retries() -> int:
    return int(os.getenv("GROQ_MAX_RETRIES", "2"))


def make_llm(api_key: str) -> BaseChatModel:
    return ChatGroq(
        api_key=api_key,
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=0,
        reasoning_effort="low",
        max_retries=_max_retries(),
    )


def make_prompt_guard(api_key: str) -> Callable[[str], float]:
    """Llama Prompt Guard 2 on Groq: returns the probability (0-1) that text is a prompt injection or jailbreak."""
    client = Groq(api_key=api_key, max_retries=_max_retries())

    def injection_score(text: str) -> float:
        response = client.chat.completions.create(
            model=PROMPT_GUARD_MODEL, messages=[{"role": "user", "content": text[:PROMPT_GUARD_MAX_CHARS]}]
        )
        return float(response.choices[0].message.content)

    return injection_score


def validate_api_key(api_key: str) -> str | None:
    """Cheap check (lists models, no tokens spent). Returns an error message, or None if the key works."""
    if not api_key.strip().startswith("gsk_"):
        return "That doesn't look like a Groq API key (they start with `gsk_`)."
    try:
        available = {m.id for m in Groq(api_key=api_key.strip(), max_retries=1).models.list().data}
    except AuthenticationError:
        return "Groq rejected this key. Check it at console.groq.com/keys."
    except Exception:
        return "Couldn't reach Groq to check the key. Please try again."
    missing = [m for m in REQUIRED_MODELS if m not in available]
    return f"This key can't use: {', '.join(missing)}." if missing else None


def _env_key() -> str:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    return make_llm(_env_key())


@lru_cache(maxsize=1)
def get_prompt_guard() -> Callable[[str], float]:
    return make_prompt_guard(_env_key())
