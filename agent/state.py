"""Graph state and the classifier's structured output schema."""

from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

Intent = Literal["faq", "order_status", "refund_return", "complaint", "human_request", "chitchat"]


class IntentResult(BaseModel):
    """Structured output of the intent classifier."""

    intent: Intent = Field(description="The customer's main intent in their latest message.")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident you are in the intent, 0 to 1.")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="The customer's sentiment in their latest message."
    )
    order_id: str | None = Field(default=None, description="Order ID mentioned in the conversation, e.g. ORD-1001.")


class SupportState(MessagesState):
    """Conversation state persisted per thread by the checkpointer."""

    intent: Intent | None
    confidence: float
    sentiment: str | None
    order_id: str | None
    negative_turns: int
    blocked_reason: str | None
    injection_score: float | None
    ticket_id: str | None
