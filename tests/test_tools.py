from agent.tools import (
    check_refund_eligibility,
    create_refund_request,
    get_order_status,
    get_tracking,
    normalize_order_id,
    search_faq,
)


def test_normalize_order_id_accepts_loose_formats():
    assert normalize_order_id("ord-1001") == "ORD-1001"
    assert normalize_order_id("1001") == "ORD-1001"
    assert normalize_order_id("order #1001") == "ORD-1001"


def test_search_faq_ranks_relevant_answer_first():
    result = search_faq.invoke({"query": "what is the return policy"})
    assert result["results"][0]["question"] == "What is your return policy?"


def test_search_faq_no_match_returns_empty():
    assert search_faq.invoke({"query": "zzz qqq"})["results"] == []


def test_get_order_status_found_and_missing():
    assert get_order_status.invoke({"order_id": "ORD-1002"})["status"] == "shipped"
    assert get_order_status.invoke({"order_id": "ORD-9999"}) == {"found": False, "order_id": "ORD-9999"}


def test_get_tracking():
    assert get_tracking.invoke({"order_id": "1002"})["carrier"] == "FedEx"


def test_refund_eligibility_rules():
    assert check_refund_eligibility.invoke({"order_id": "ORD-1001"})["eligible"] is True
    assert "window" in check_refund_eligibility.invoke({"order_id": "ORD-1004"})["reason"]
    assert "already been refunded" in check_refund_eligibility.invoke({"order_id": "ORD-1005"})["reason"]
    assert "processing" in check_refund_eligibility.invoke({"order_id": "ORD-1003"})["reason"]
    assert "not found" in check_refund_eligibility.invoke({"order_id": "ORD-9999"})["reason"]


def test_refund_cannot_be_submitted_twice():
    create_refund_request("ORD-1002", 145.0)
    assert check_refund_eligibility.invoke({"order_id": "ORD-1002"})["eligible"] is False
