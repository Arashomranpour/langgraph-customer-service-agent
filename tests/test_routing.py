from langchain_core.messages import HumanMessage

from agent.nodes import guardrails, route_after_classify


def _state(**kwargs):
    return {"messages": [HumanMessage(content="hi", id="m1")], "negative_turns": 0, "confidence": 0.9, **kwargs}


def test_guardrails_blocks_empty_and_long():
    assert guardrails({"messages": [HumanMessage(content="  ", id="m1")]})["blocked_reason"] == "empty"
    assert guardrails({"messages": [HumanMessage(content="x" * 2001, id="m1")]})["blocked_reason"] == "too_long"


def test_guardrails_redacts_card_numbers_in_place():
    out = guardrails({"messages": [HumanMessage(content="card 4111 1111 1111 1111 pls", id="m1")]})
    assert out["blocked_reason"] == "card_number"
    assert out["messages"][0].id == "m1"
    assert "4111" not in out["messages"][0].content


def test_guardrails_allows_order_ids():
    assert guardrails({"messages": [HumanMessage(content="where is ORD-1001", id="m1")]})["blocked_reason"] is None


def test_route_by_intent():
    assert route_after_classify(_state(intent="faq")) == "faq_agent"
    assert route_after_classify(_state(intent="order_status")) == "order_agent"
    assert route_after_classify(_state(intent="refund_return")) == "refund_agent"
    assert route_after_classify(_state(intent="chitchat")) == "chitchat"


def test_route_escalates():
    assert route_after_classify(_state(intent="complaint")) == "create_ticket"
    assert route_after_classify(_state(intent="human_request")) == "create_ticket"
    assert route_after_classify(_state(intent="faq", confidence=0.3)) == "create_ticket"
    assert route_after_classify(_state(intent="faq", negative_turns=2)) == "create_ticket"
