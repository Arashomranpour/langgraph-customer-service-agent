"""Graph nodes and routers. Nodes needing the LLM are built by factories so tests can inject a fake model."""

import json
import logging
import re
from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from agent.state import IntentResult, SupportState
from agent.tools import TICKETS, create_refund_request, create_ticket, normalize_order_id

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 2000
INJECTION_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
NEGATIVE_TURNS_BEFORE_ESCALATION = 2
CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")

STORE_CONTEXT = (
    "You are 'Nova', the customer support assistant for ShopNova, an online store. "
    "Be concise, warm and accurate. Use short paragraphs or bullet points (Markdown is fine). "
    "Only state facts that come from your tools or the conversation; never invent order details or policies."
)

CLASSIFIER_PROMPT = """Classify the customer's LATEST message for a customer support router.
Intents:
- faq: general questions about policies, shipping, payments, accounts, warranty, hours
- order_status: where is my order, order details, tracking
- refund_return: wants to return an item or get a refund for an order
- complaint: angry or dissatisfied about service/product and wants it addressed
- human_request: explicitly asks for a human, agent, manager or person
- chitchat: greetings, thanks, small talk, or anything off-topic
Use earlier messages only for context (e.g. a bare order ID after the assistant asked for one continues that intent).
Extract the order ID if one appears anywhere in the conversation."""

FAQ_PROMPT = (
    STORE_CONTEXT
    + " Answer store policy questions using the search_faq tool. If the FAQ has no answer, say so and offer to connect the customer to a human."
)
ORDER_ID_HINT = (
    " Order IDs may be written loosely (1003, #1003, ord-1003); pass them to the tools as given, "
    "the tools normalize them. Never ask the customer to reformat an ID."
)
ORDER_PROMPT = (
    STORE_CONTEXT
    + " Help with order status and tracking using your tools. If no order ID is known, ask for it."
    + ORDER_ID_HINT
)
REFUND_PROMPT = (
    STORE_CONTEXT
    + ORDER_ID_HINT
    + (
        " Handle returns and refunds. You need an order ID; ask for it if missing. "
        "Always call check_refund_eligibility before discussing a refund. "
        "If eligible, summarise the order and refund amount and tell the customer a confirmation button will appear "
        "below to approve or decline it. "
        "If not eligible, explain why kindly and suggest alternatives. You cannot submit refunds yourself; never claim a refund was submitted."
    )
)
CHITCHAT_PROMPT = (
    STORE_CONTEXT + " Reply briefly and steer toward how you can help (orders, returns, refunds, store questions)."
)


def conversation_text(messages: Sequence[AnyMessage], limit: int = 8) -> list[AnyMessage]:
    """Human/assistant text turns only (no tool traffic), for nodes that don't call tools."""
    turns = [
        m
        for m in messages
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content and not m.tool_calls)
    ]
    return [m if isinstance(m, HumanMessage) else AIMessage(content=m.content) for m in turns[-limit:]]


def last_human_index(messages: Sequence[AnyMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return -1


# ---------- guardrails (deterministic) ----------


def guardrails(state: SupportState) -> dict:
    msg = state["messages"][-1]
    text = msg.content if isinstance(msg.content, str) else ""
    if not text.strip():
        return {"blocked_reason": "empty"}
    if len(text) > MAX_MESSAGE_CHARS:
        return {"blocked_reason": "too_long"}
    if CARD_PATTERN.search(text):
        # Redact the card number from persisted history (same id replaces the message).
        redacted = HumanMessage(content=CARD_PATTERN.sub("[REDACTED CARD NUMBER]", text), id=msg.id)
        return {"blocked_reason": "card_number", "messages": [redacted]}
    return {"blocked_reason": None}


REFUSALS = {
    "empty": "It looks like your message was empty. How can I help you today?",
    "too_long": f"That message is a bit long for me. Could you summarise it in under {MAX_MESSAGE_CHARS} characters?",
    "card_number": (
        "For your security I've removed the card number from your message. Please never share card details "
        "in chat; I never need them to help with orders, returns or refunds. What can I help you with?"
    ),
    "prompt_injection": (
        "I can't act on instructions that try to change how I work. I'm happy to help with orders, "
        "tracking, returns, refunds or store questions."
    ),
}


def refuse_reply(state: SupportState) -> dict:
    return {"messages": [AIMessage(content=REFUSALS[state["blocked_reason"]])]}


def route_after_guardrails(state: SupportState) -> str:
    return "refuse_reply" if state.get("blocked_reason") else "injection_check"


def make_injection_check(injection_score: Callable[[str], float]) -> Callable[[SupportState], dict]:
    """Model-based prompt-injection screen. Fails open (logs and continues) if the classifier is unavailable."""

    def injection_check(state: SupportState) -> dict:
        try:
            score = injection_score(state["messages"][-1].content)
        except Exception:
            logger.warning("Prompt guard unavailable; continuing without injection screening", exc_info=True)
            return {"injection_score": None}
        if score >= INJECTION_THRESHOLD:
            return {"injection_score": score, "blocked_reason": "prompt_injection"}
        return {"injection_score": score}

    return injection_check


def route_after_injection_check(state: SupportState) -> str:
    return "refuse_reply" if state.get("blocked_reason") else "classify_intent"


# ---------- classification ----------


def make_classifier(llm: BaseChatModel) -> Callable[[SupportState], dict]:
    structured = llm.with_structured_output(IntentResult)

    def classify_intent(state: SupportState) -> dict:
        result: IntentResult = structured.invoke(
            [SystemMessage(content=CLASSIFIER_PROMPT), *conversation_text(state["messages"])]
        )
        negative = state.get("negative_turns", 0)
        negative = (
            negative + 1 if result.sentiment == "negative" else (0 if result.sentiment == "positive" else negative)
        )
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "sentiment": result.sentiment,
            "order_id": normalize_order_id(result.order_id) if result.order_id else state.get("order_id"),
            "negative_turns": negative,
        }

    return classify_intent


INTENT_ROUTES = {
    "faq": "faq_agent",
    "order_status": "order_agent",
    "refund_return": "refund_agent",
    "chitchat": "chitchat",
}


def route_after_classify(state: SupportState) -> str:
    if (
        state["intent"] in ("complaint", "human_request")
        or state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
        or state.get("negative_turns", 0) >= NEGATIVE_TURNS_BEFORE_ESCALATION
    ):
        return "create_ticket"
    return INTENT_ROUTES[state["intent"]]


# ---------- tool-calling sub-agents (Graph 2) ----------


def build_tool_agent(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, name: str):
    """ReAct loop: agent -> tools -> agent ... until the model answers without tool calls."""
    model = llm.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        context_note = SystemMessage(content=system_prompt)
        return {"messages": [model.invoke([context_note, *state["messages"]])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(name=name)


def make_chitchat(llm: BaseChatModel) -> Callable[[SupportState], dict]:
    def chitchat(state: SupportState) -> dict:
        reply = llm.invoke([SystemMessage(content=CHITCHAT_PROMPT), *conversation_text(state["messages"])])
        return {"messages": [AIMessage(content=reply.content)]}

    return chitchat


# ---------- refund confirmation (human-in-the-loop: the customer) ----------


def latest_eligibility(state: SupportState) -> dict | None:
    """The check_refund_eligibility result from the current turn, if any."""
    messages = state["messages"]
    for m in reversed(messages[last_human_index(messages) + 1 :]):
        if isinstance(m, ToolMessage) and m.name == "check_refund_eligibility":
            try:
                return json.loads(m.content)
            except (TypeError, json.JSONDecodeError):
                return None
    return None


def route_after_refund(state: SupportState) -> str:
    result = latest_eligibility(state)
    return "confirm_refund" if result and result.get("eligible") else END


def confirm_refund(state: SupportState) -> dict:
    result = latest_eligibility(state)
    decision = interrupt(
        {
            "type": "refund_confirmation",
            "order_id": result["order_id"],
            "amount": result["amount"],
            "question": f"Submit a refund of ${result['amount']:.2f} for order {result['order_id']}?",
        }
    )
    if decision == "approve":
        refund = create_refund_request(result["order_id"], result["amount"])
        text = (
            f"Done! Refund {refund['refund_id']} for ${refund['amount']:.2f} on order {refund['order_id']} "
            "has been submitted. You'll see it on your original payment method within 5-7 business days."
        )
    else:
        text = "No problem, I haven't submitted the refund. Is there anything else I can help with?"
    return {"messages": [AIMessage(content=text)]}


# ---------- escalation (Graph 3: human-in-the-loop with a support agent) ----------


def open_ticket(state: SupportState) -> dict:
    last = state["messages"][last_human_index(state["messages"])].content
    reason = state.get("intent") or "unknown"
    priority = "high" if reason == "complaint" or state.get("sentiment") == "negative" else "normal"
    summary = f"[{reason}] {last[:200]}" + (f" (order {state['order_id']})" if state.get("order_id") else "")
    ticket = create_ticket(summary, priority)
    text = (
        f"I've opened support ticket {ticket['ticket_id']} ({priority} priority) and a member of our team "
        "will join this chat shortly."
    )
    return {"ticket_id": ticket["ticket_id"], "messages": [AIMessage(content=text)]}


def human_handoff(state: SupportState) -> dict:
    ticket = TICKETS[state["ticket_id"]]
    reply = interrupt(
        {
            "type": "human_handoff",
            "ticket_id": ticket["ticket_id"],
            "priority": ticket["priority"],
            "summary": ticket["summary"],
        }
    )
    return {
        "messages": [AIMessage(content=str(reply), name="support_agent")],
        "negative_turns": 0,
    }
