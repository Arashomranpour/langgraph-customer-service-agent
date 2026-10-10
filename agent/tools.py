"""Customer service tools backed by mock data in data/. Tools return dicts, not prose."""

import itertools
import json
import re
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RETURN_WINDOW_DAYS = 30

_ticket_counter = itertools.count(1001)
_refund_counter = itertools.count(5001)
_lock = threading.Lock()
TICKETS: dict[str, dict] = {}
REFUNDS: dict[str, dict] = {}


@lru_cache(maxsize=1)
def _orders() -> dict[str, dict]:
    return json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _faq() -> list[dict]:
    return json.loads((DATA_DIR / "faq.json").read_text(encoding="utf-8"))


def normalize_order_id(order_id: str) -> str:
    digits = re.sub(r"\D", "", order_id or "")
    return f"ORD-{digits}" if digits else (order_id or "").strip().upper()


def _get_order(order_id: str) -> dict | None:
    return _orders().get(normalize_order_id(order_id))


@tool
def search_faq(query: str) -> dict:
    """Search the store FAQ (returns, refunds, shipping, payments, accounts, warranty, support hours)."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for entry in _faq():
        text_words = set(re.findall(r"[a-z0-9]+", entry["question"].lower())) | set(entry["tags"])
        score = len(words & text_words)
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return {"query": query, "results": [{"question": e["question"], "answer": e["answer"]} for _, e in scored[:3]]}


@tool
def get_order_status(order_id: str) -> dict:
    """Look up an order's status, items and total by order ID (e.g. ORD-1001)."""
    order = _get_order(order_id)
    if order is None:
        return {"found": False, "order_id": normalize_order_id(order_id)}
    return {
        "found": True,
        "order_id": normalize_order_id(order_id),
        "status": order["status"],
        "items": order["items"],
        "total": order["total"],
        "estimated_delivery": order.get("estimated_delivery"),
        "refunded": order["refunded"],
    }


@tool
def get_tracking(order_id: str) -> dict:
    """Get the carrier and tracking number for a shipped or delivered order."""
    order = _get_order(order_id)
    if order is None:
        return {"found": False, "order_id": normalize_order_id(order_id)}
    return {
        "found": True,
        "order_id": normalize_order_id(order_id),
        "status": order["status"],
        "carrier": order["carrier"],
        "tracking_number": order["tracking_number"],
    }


@tool
def check_refund_eligibility(order_id: str) -> dict:
    """Check whether an order can be refunded under the 30-day return policy."""
    oid = normalize_order_id(order_id)
    order = _get_order(oid)
    if order is None:
        return {"order_id": oid, "eligible": False, "reason": "Order not found."}
    if order["refunded"] or oid in {r["order_id"] for r in REFUNDS.values()}:
        return {"order_id": oid, "eligible": False, "reason": "This order has already been refunded."}
    if order["status"] != "delivered":
        return {
            "order_id": oid,
            "eligible": False,
            "reason": f"Order is '{order['status']}'. Only delivered orders can be refunded; "
            "orders still processing can be cancelled instead.",
        }
    days = order["days_since_delivery"]
    if days > RETURN_WINDOW_DAYS:
        return {
            "order_id": oid,
            "eligible": False,
            "reason": f"Delivered {days} days ago, outside the {RETURN_WINDOW_DAYS}-day return window.",
        }
    return {"order_id": oid, "eligible": True, "amount": order["total"], "reason": "Within the return window."}


def create_refund_request(order_id: str, amount: float) -> dict:
    """Submit a refund. Called by the graph only after the customer confirms, never by the LLM."""
    with _lock:
        refund_id = f"RFD-{next(_refund_counter)}"
        REFUNDS[refund_id] = {"refund_id": refund_id, "order_id": normalize_order_id(order_id), "amount": amount}
    return REFUNDS[refund_id]


def create_ticket(summary: str, priority: str = "normal") -> dict:
    """Open a support ticket for a human agent."""
    with _lock:
        ticket_id = f"TCK-{next(_ticket_counter)}"
        TICKETS[ticket_id] = {"ticket_id": ticket_id, "summary": summary, "priority": priority, "status": "open"}
    return TICKETS[ticket_id]


FAQ_TOOLS = [search_faq]
ORDER_TOOLS = [get_order_status, get_tracking]
REFUND_TOOLS = [get_order_status, check_refund_eligibility, search_faq]
