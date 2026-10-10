"""Streamlit chat UI. No agent logic here: it only streams the compiled graph and renders its state."""

import json
import logging
import uuid
from collections.abc import Iterator

import streamlit as st
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command

from agent.graph import build_graph
from agent.tools import _orders
from ui.graph_view import SUBAGENTS, Trace, node_key, steps_markdown, to_svg

logger = logging.getLogger(__name__)

STREAMED_NODES = {"agent", "chitchat"}
SAMPLE_PROMPTS = [
    "What's your return policy?",
    "Where is my order ORD-1002?",
    "I want a refund for ORD-1001",
    "Can I get a refund for ORD-1004?",
    "This is unacceptable, I want a manager!",
]

st.set_page_config(page_title="ShopNova Support", page_icon="🛍️", layout="wide")


@st.cache_resource
def get_graph():
    load_dotenv()
    return build_graph()


try:
    graph = get_graph()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "trace" not in st.session_state:
    st.session_state.trace = Trace()
config = {"configurable": {"thread_id": st.session_state.thread_id}}
trace: Trace = st.session_state.trace


def queue(key: str, value: str) -> None:
    st.session_state[key] = value


snapshot = graph.get_state(config)
values = snapshot.values or {}
pending = snapshot.interrupts[0].value if snapshot.interrupts else None

# ---------- sidebar ----------
with st.sidebar:
    st.title("🛍️ ShopNova Support")
    st.caption("LangGraph + Groq customer service agent")
    if st.button("➕ New conversation", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.trace = Trace()
        st.rerun()

    st.subheader("Try asking")
    for prompt in SAMPLE_PROMPTS:
        # on_click runs exactly once per click, even if the script reruns mid-stream
        st.button(prompt, width="stretch", disabled=pending is not None, on_click=queue, args=("queued_prompt", prompt))

    if pending and pending["type"] == "human_handoff":
        st.subheader("🧑‍💼 Support agent console")
        st.warning(f"Ticket **{pending['ticket_id']}** · {pending['priority']} priority\n\n{pending['summary']}")
        with st.form("agent_reply", clear_on_submit=True):
            reply = st.text_area("Reply to the customer as a human agent")
            if st.form_submit_button("Send as support agent", type="primary") and reply.strip():
                st.session_state.queued_resume = reply.strip()

    with st.expander("🔍 Agent state", expanded=False):
        st.json(
            {
                "thread_id": st.session_state.thread_id,
                "intent": values.get("intent"),
                "confidence": values.get("confidence"),
                "sentiment": values.get("sentiment"),
                "order_id": values.get("order_id"),
                "negative_turns": values.get("negative_turns"),
                "ticket_id": values.get("ticket_id"),
                "next": list(snapshot.next),
            }
        )
    with st.expander("📦 Demo orders"):
        st.table(
            [
                {"order": oid, "status": o["status"], "total": f"${o['total']:.2f}", "refunded": o["refunded"]}
                for oid, o in _orders().items()
            ]
        )

chat_col, graph_col = st.columns([1, 1], gap="large")

# ---------- live graph ----------
with graph_col:
    st.subheader("🕸️ Live agent graph")
    st.caption("🟨 running · 🟩 visited this turn · 🟥 waiting for a person · badges = step order · hover for tools")
    graph_view = st.empty()


def draw_graph() -> None:
    with graph_view.container():
        st.markdown(to_svg(trace), unsafe_allow_html=True)  # st.html sanitizes SVG away
        st.markdown(steps_markdown(trace))


draw_graph()


def run_graph(payload) -> Iterator[str]:
    """Yield answer tokens for st.write_stream while redrawing the graph as nodes start and finish."""
    for namespace, mode, data in graph.stream(payload, config, stream_mode=["updates", "messages"], subgraphs=True):
        if mode == "messages":
            chunk, meta = data
            key = node_key(namespace, meta["langgraph_node"])
            if trace.active != key:
                trace.active = key
                draw_graph()
            if (
                meta["langgraph_node"] in STREAMED_NODES
                and isinstance(chunk, AIMessageChunk)
                and isinstance(chunk.content, str)
            ):
                yield chunk.content
            continue
        for node, update in data.items():
            if node == "__interrupt__" or (not namespace and node in SUBAGENTS):
                continue  # sub-agent internals are recorded individually
            trace.record(node_key(namespace, node), update)
        draw_graph()


def send(payload, new_turn: bool) -> None:
    trace.start_run(new_turn)
    with chat_col, st.chat_message("assistant", avatar="🛍️"):
        try:
            st.write_stream(run_graph(payload))
        except RateLimitError:
            trace.finish(None)
            st.warning("The demo has reached its free model quota for now. Please try again in a few minutes.")
            return
        except Exception:  # never show raw provider errors (they include account details) to customers
            logger.exception("Graph run failed")
            trace.finish(None)
            st.error("Sorry, something went wrong on our side. Please try again.")
            return
    after = graph.get_state(config)
    trace.finish(after.next[0] if after.interrupts else None)
    st.rerun()


# ---------- conversation ----------
with chat_col:
    st.header("How can we help today?")
    messages = values.get("messages", [])
    if not messages:
        with st.chat_message("assistant", avatar="🛍️"):
            st.markdown(
                "Hi! I'm **Nova**. I can check orders and tracking, handle returns and refunds, or answer store questions."
            )

    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            calls = ", ".join(f"`{c['name']}({json.dumps(c['args'])})`" for c in msg.tool_calls)
            st.caption(f"🔧 Nova used {calls}")
        elif isinstance(msg, AIMessage) and msg.content:
            if msg.name == "support_agent":
                with st.chat_message("assistant", avatar="🧑‍💼"):
                    st.markdown(f"**Support agent:** {msg.content}")
            else:
                with st.chat_message("assistant", avatar="🛍️"):
                    st.markdown(msg.content)

    if pending and pending["type"] == "refund_confirmation":
        with st.container(border=True):
            st.markdown(f"**Confirm refund** · {pending['question']}")
            approve, decline = st.columns(2)
            approve.button(
                "✅ Approve refund", type="primary", width="stretch", on_click=queue, args=("queued_resume", "approve")
            )
            decline.button("❌ Decline", width="stretch", on_click=queue, args=("queued_resume", "decline"))
    elif pending and pending["type"] == "human_handoff":
        st.info("⏳ Waiting for a support agent to reply (use the console in the sidebar).")

placeholder = {
    "refund_confirmation": "Please approve or decline the refund above",
    "human_handoff": "Waiting for a support agent…",
}.get(pending["type"] if pending else "", "Type your message…")
typed = st.chat_input(placeholder, disabled=pending is not None)

if resume := st.session_state.pop("queued_resume", None):
    send(Command(resume=resume), new_turn=False)
elif prompt := typed or st.session_state.pop("queued_prompt", None):
    with chat_col, st.chat_message("user"):
        st.markdown(prompt)
    send({"messages": [HumanMessage(content=prompt)]}, new_turn=True)
