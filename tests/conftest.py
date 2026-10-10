import sys
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.state import IntentResult  # noqa: E402


class FakeLLM(BaseChatModel):
    """Scripted chat model: queued AIMessages for chat/tool calls, queued IntentResults for classification."""

    replies: list = []
    intents: list = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: self.intents.pop(0))


def no_injection(_text: str) -> float:
    return 0.0


def intent(
    name: str, confidence: float = 0.95, sentiment: str = "neutral", order_id: str | None = None
) -> IntentResult:
    return IntentResult(intent=name, confidence=confidence, sentiment=sentiment, order_id=order_id)


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture(autouse=True)
def reset_stores():
    from agent import tools

    tools.REFUNDS.clear()
    tools.TICKETS.clear()
    yield


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(replies=[], intents=[])
