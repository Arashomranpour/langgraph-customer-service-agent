"""Builds the main supervisor graph (Graph 1) with sub-agents (Graph 2) and escalation (Graph 3)."""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent import nodes
from agent.state import SupportState
from agent.tools import FAQ_TOOLS, ORDER_TOOLS, REFUND_TOOLS


def build_graph(
    llm: BaseChatModel | None = None,
    injection_score: Callable[[str], float] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    api_key: str | None = None,
) -> CompiledStateGraph:
    """Pass `llm` and `injection_score` fakes in tests. Otherwise real Groq models are built from `api_key`
    (the web app's per-session key) or, if that is None, from GROQ_API_KEY in the environment (CLI, evals)."""
    from agent import llm as groq

    if llm is None:
        llm = groq.make_llm(api_key) if api_key else groq.get_llm()
    if injection_score is None:
        injection_score = groq.make_prompt_guard(api_key) if api_key else groq.get_prompt_guard()

    builder = StateGraph(SupportState)
    builder.add_node("guardrails", nodes.guardrails)
    builder.add_node("injection_check", nodes.make_injection_check(injection_score))
    builder.add_node("refuse_reply", nodes.refuse_reply)
    builder.add_node("classify_intent", nodes.make_classifier(llm))
    builder.add_node("faq_agent", nodes.build_tool_agent(llm, FAQ_TOOLS, nodes.FAQ_PROMPT, "faq_agent"))
    builder.add_node("order_agent", nodes.build_tool_agent(llm, ORDER_TOOLS, nodes.ORDER_PROMPT, "order_agent"))
    builder.add_node("refund_agent", nodes.build_tool_agent(llm, REFUND_TOOLS, nodes.REFUND_PROMPT, "refund_agent"))
    builder.add_node("chitchat", nodes.make_chitchat(llm))
    builder.add_node("confirm_refund", nodes.confirm_refund)
    builder.add_node("create_ticket", nodes.open_ticket)
    builder.add_node("human_handoff", nodes.human_handoff)

    builder.add_edge(START, "guardrails")
    builder.add_conditional_edges("guardrails", nodes.route_after_guardrails, ["refuse_reply", "injection_check"])
    builder.add_conditional_edges(
        "injection_check", nodes.route_after_injection_check, ["refuse_reply", "classify_intent"]
    )
    builder.add_conditional_edges(
        "classify_intent",
        nodes.route_after_classify,
        ["faq_agent", "order_agent", "refund_agent", "chitchat", "create_ticket"],
    )
    builder.add_conditional_edges("refund_agent", nodes.route_after_refund, ["confirm_refund", END])
    builder.add_edge("create_ticket", "human_handoff")
    for node in ("refuse_reply", "faq_agent", "order_agent", "chitchat", "confirm_refund", "human_handoff"):
        builder.add_edge(node, END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
