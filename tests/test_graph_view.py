from langchain_core.messages import AIMessage

from agent.graph import build_graph
from tests.conftest import no_injection
from ui.graph_view import EDGES, END, LABELS, POS, START, Trace, node_key, steps_markdown, to_svg


def test_node_key_namespaces_subagent_nodes():
    assert node_key((), "guardrails") == "guardrails"
    assert node_key(("faq_agent:1234-abcd",), "tools") == "faq_agent/tools"


def test_view_covers_every_real_graph_node(fake_llm):
    real = set(build_graph(fake_llm, no_injection).get_graph(xray=1).nodes)
    drawn = {k.replace("/", ":") for k in LABELS} | {START, END}
    # sub-agent entry/exit points are internal plumbing, not drawn
    assert {n for n in real if not n.endswith((":__start__", ":__end__"))} <= drawn


def test_trace_highlights_walked_path_and_tool_loop():
    trace = Trace()
    trace.start_run(new_turn=True)
    for key in ("guardrails", "classify_intent", "order_agent/agent", "order_agent/tools", "order_agent/agent"):
        trace.record(key, {"messages": [AIMessage(content="x")]} if key.endswith("agent") else {})
    trace.finish(None)
    svg = to_svg(trace)
    assert '<g class="edge walked" data-edge="order_agent/tools->order_agent/agent">' in svg
    assert '<g class="edge walked" data-edge="order_agent/agent->__end__">' in svg
    assert '<g class="edge" data-edge="faq_agent/agent->__end__">' in svg  # untouched branch stays grey
    assert '<g class="node visited" data-node="order_agent/tools">' in svg
    assert "🏁 Done" in steps_markdown(trace)


def test_every_node_has_a_layout_position():
    assert set(POS) == set(LABELS)


def test_interrupt_marks_waiting_and_resume_continues_path():
    trace = Trace()
    trace.start_run(new_turn=True)
    trace.record("guardrails", {})
    trace.record("classify_intent", {"intent": "human_request", "confidence": 0.9, "sentiment": "negative"})
    trace.record("create_ticket", {"ticket_id": "TCK-1"})
    trace.finish("human_handoff")
    assert '<g class="node waiting" data-node="human_handoff">' in to_svg(trace)
    assert "Paused at" in steps_markdown(trace)

    trace.start_run(new_turn=False)
    trace.record("human_handoff", {})
    trace.finish(None)
    assert ("create_ticket", "human_handoff") in set(zip(trace.path(), trace.path()[1:], strict=False))
    assert trace.path()[-1] == END


def test_every_edge_endpoint_is_drawn():
    for src, dst, _ in EDGES:
        assert src in LABELS or src == START
        assert dst in LABELS or dst == END
