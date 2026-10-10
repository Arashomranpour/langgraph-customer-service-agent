"""Live view of the agent graph: a hand-laid SVG diagram with the current turn's path highlighted.

SVG (not Graphviz) so the layout has a fixed, legible scale in a half-width column and can animate.
"""

import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, ToolMessage

START, END = "__start__", "__end__"

SUBAGENTS = {
    "faq_agent": ("📚 FAQ agent", "search_faq"),
    "order_agent": ("📦 Order agent", "get_order_status, get_tracking"),
    "refund_agent": ("💸 Refund agent", "check_refund_eligibility, get_order_status, search_faq"),
}

# Short labels keep the diagram's natural size small so it stays legible when scaled to the column.
LABELS = {
    "guardrails": "🛡️ Guardrails",
    "injection_check": "🕵️ Injection guard",
    "refuse_reply": "🚫 Refuse",
    "classify_intent": "🧭 Classify",
    "chitchat": "💬 Chitchat",
    "confirm_refund": "✅ Confirm refund",
    "create_ticket": "🎫 Ticket",
    "human_handoff": "🧑‍💼 Human agent",
    **{f"{name}/agent": "🤖 LLM" for name in SUBAGENTS},
    **{f"{name}/tools": "🔧 Tools" for name in SUBAGENTS},
}
TOOLTIPS = {f"{name}/tools": tools for name, (_, tools) in SUBAGENTS.items()}

EDGES: list[tuple[str, str, str]] = [
    (START, "guardrails", ""),
    ("guardrails", "refuse_reply", "blocked"),
    ("guardrails", "injection_check", "ok"),
    ("injection_check", "refuse_reply", "injection"),
    ("injection_check", "classify_intent", "safe"),
    ("classify_intent", "faq_agent/agent", "faq"),
    ("classify_intent", "order_agent/agent", "order"),
    ("classify_intent", "refund_agent/agent", "refund"),
    ("classify_intent", "chitchat", "chitchat"),
    ("classify_intent", "create_ticket", "escalate"),
    *[
        edge
        for name in SUBAGENTS
        for edge in ((f"{name}/agent", f"{name}/tools", ""), (f"{name}/tools", f"{name}/agent", ""))
    ],
    ("refund_agent/agent", "confirm_refund", "eligible"),
    ("create_ticket", "human_handoff", ""),
    ("refuse_reply", END, ""),
    ("chitchat", END, ""),
    ("faq_agent/agent", END, ""),
    ("order_agent/agent", END, ""),
    ("refund_agent/agent", END, ""),
    ("confirm_refund", END, ""),
    ("human_handoff", END, ""),
]

# Hand-laid layout in SVG user units (viewBox 780 x 570). Values are (x center, y center, width).
# Compact on purpose: the view is usually ~450-650px wide, so 1 unit ≈ 0.75-1px and 15px text stays legible.
VIEW_W, VIEW_H, NODE_H = 600, 566, 32
CLUSTER_X = {"faq_agent": 112, "order_agent": 222, "refund_agent": 332}
POS: dict[str, tuple[int, int, int]] = {
    "guardrails": (290, 72, 140),
    "injection_check": (290, 128, 176),
    "refuse_reply": (80, 128, 96),
    "classify_intent": (290, 192, 128),
    "chitchat": (432, 318, 100),
    "create_ticket": (534, 318, 88),
    "human_handoff": (534, 384, 120),
    "confirm_refund": (320, 470, 140),
    **{f"{name}/agent": (x, 318, 92) for name, x in CLUSTER_X.items()},
    **{f"{name}/tools": (x, 380, 92) for name, x in CLUSTER_X.items()},
}
START_XY, END_XY = (290, 26), (290, 546)
CLUSTER_TOP, CLUSTER_BOTTOM, CLUSTER_W = 266, 410, 104

# Which side of each node an edge leaves/enters (default bottom -> top).
SIDES = {
    ("guardrails", "refuse_reply"): ("left", "top"),
    ("injection_check", "refuse_reply"): ("left", "right"),
    ("chitchat", END): ("bottom", "right"),
    ("confirm_refund", END): ("bottom", "right"),
    ("human_handoff", END): ("bottom", "right"),
}
# Refuse -> end runs down the left margin, outside the FAQ cluster.
FIXED_PATHS = {("refuse_reply", END): ("M32 128 C-10 200 -10 520 277 546", (14, 330))}
LABEL_AT = {(src, dst): 0.72 for src, dst, _ in EDGES if src == "classify_intent"}
DIRECTIONS = {"top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}


def node_key(namespace: tuple[str, ...], node: str) -> str:
    """'agent' inside the faq_agent sub-graph becomes 'faq_agent/agent'."""
    return f"{namespace[0].split(':')[0]}/{node}" if namespace else node


def describe(key: str, update: dict | None) -> str:
    update = update or {}
    messages = update.get("messages", [])
    if key == "guardrails":
        return f"blocked: {update['blocked_reason']}" if update.get("blocked_reason") else "passed"
    if key == "injection_check":
        score = update.get("injection_score")
        return "guard unavailable (fail-open)" if score is None else f"injection risk {score:.1%}"
    if key == "classify_intent":
        return f"{update.get('intent')} · {update.get('confidence', 0):.0%} sure · {update.get('sentiment')}"
    if key.endswith("/agent") and messages and isinstance(messages[-1], AIMessage):
        calls = messages[-1].tool_calls
        return "calls " + ", ".join(c["name"] for c in calls) if calls else "wrote the answer"
    if key.endswith("/tools"):
        return "ran " + ", ".join(m.name for m in messages if isinstance(m, ToolMessage))
    if key == "create_ticket":
        return f"opened {update.get('ticket_id')}"
    return ""


@dataclass
class Trace:
    """Ordered steps of the current turn, kept across an interrupt so the resumed run continues the path."""

    steps: list[dict] = field(default_factory=list)
    active: str | None = None
    waiting: str | None = None
    done: bool = False
    _last: float = field(default_factory=time.perf_counter)

    def start_run(self, new_turn: bool) -> None:
        if new_turn:
            self.steps.clear()
        self.active, self.waiting, self.done = None, None, False
        self._last = time.perf_counter()

    def record(self, key: str, update: dict | None) -> None:
        now = time.perf_counter()
        self.steps.append({"node": key, "detail": describe(key, update), "ms": (now - self._last) * 1000})
        self._last = now
        self.active = None

    def finish(self, waiting_on: str | None) -> None:
        self.active = None
        self.waiting = waiting_on
        self.done = waiting_on is None and bool(self.steps)

    def path(self) -> list[str]:
        return [START, *(s["node"] for s in self.steps)] + ([END] if self.done else [])


CSS = """
<style>
.lg{--idle:rgba(148,163,184,.55);--ok:#22c55e;--run:#f59e0b;--wait:#ef4444;color:inherit;font-family:inherit}
.lg svg{width:100%;height:auto;display:block;max-width:760px;margin:auto;overflow:visible}
.lg .node rect,.lg .term{fill:rgba(148,163,184,.10);stroke:var(--idle);stroke-width:1.2}
.lg .node text{font-size:15px;fill:currentColor;dominant-baseline:central;text-anchor:middle}
.lg .node.visited rect,.lg .term.visited{fill:rgba(34,197,94,.18);stroke:var(--ok);stroke-width:2}
.lg .node.active rect{fill:rgba(245,158,11,.22);stroke:var(--run);stroke-width:2.5;animation:lg-pulse 1s ease-in-out infinite}
.lg .node.waiting rect{fill:rgba(239,68,68,.20);stroke:var(--wait);stroke-width:2.5;stroke-dasharray:6 4;
  animation:lg-flow 1.2s linear infinite}
.lg .cluster rect{fill:none;stroke:var(--idle);stroke-dasharray:4 4;rx:12}
.lg .cluster text{font-size:12.5px;fill:currentColor;opacity:.7;text-anchor:middle}
.lg .edge path{fill:none;stroke:var(--idle);stroke-width:1.3}
.lg .edge.walked path{stroke:var(--ok);stroke-width:2.6;stroke-dasharray:7 5;animation:lg-flow .8s linear infinite}
.lg .edge text{font-size:12px;fill:currentColor;opacity:.6;text-anchor:middle;paint-order:stroke;
  stroke:var(--lg-bg,transparent);stroke-width:3}
.lg .edge.walked text{fill:var(--ok);opacity:1}
.lg .badge rect{fill:var(--ok)} .lg .badge text{font-size:11px;font-weight:700;fill:#052e16;text-anchor:middle;
  dominant-baseline:central}
.lg .term-label{font-size:11px;fill:currentColor;opacity:.75;text-anchor:middle;dominant-baseline:central}
@keyframes lg-flow{to{stroke-dashoffset:-24}}
@keyframes lg-pulse{50%{stroke-width:4;fill:rgba(245,158,11,.38)}}
@media (prefers-reduced-motion:reduce){.lg *{animation:none!important}}
</style>
"""


def _anchor(key: str, side: str) -> tuple[float, float]:
    if key in (START, END):
        x, y = START_XY if key == START else END_XY
        dx, dy = DIRECTIONS[side]
        return x + dx * 13, y + dy * 13
    x, y, w = POS[key]
    dx, dy = DIRECTIONS[side]
    return x + dx * w / 2, y + dy * NODE_H / 2


def _edge_geometry(src: str, dst: str) -> tuple[str, tuple[float, float]]:
    """SVG path and label point for an edge."""
    if (src, dst) in FIXED_PATHS:
        return FIXED_PATHS[(src, dst)]
    if src.endswith("/agent") and dst.endswith("/tools") or src.endswith("/tools") and dst.endswith("/agent"):
        x = POS[src][0] + (-14 if src.endswith("/agent") else 14)
        y1, y2 = (335, 363) if src.endswith("/agent") else (363, 335)
        return f"M{x} {y1} L{x} {y2}", (x, (y1 + y2) / 2)
    if src.endswith("/agent") and (dst == END or dst == "confirm_refund"):
        x = POS[src][0] + (-18 if dst == END and src.startswith("refund") else 0)
        if dst == "confirm_refund":
            x += 18
        x1, y1 = x, CLUSTER_BOTTOM
        x2, y2 = _anchor(dst, "top")
        k = max(30, (y2 - y1) * 0.5)
        return f"M{x1} {y1} C{x1} {y1 + k} {x2} {y2 - k} {x2} {y2}", (x1 + (x2 - x1) * 0.5, y1 + (y2 - y1) * 0.5)
    side_out, side_in = SIDES.get((src, dst), ("bottom", "top"))
    (x1, y1), (x2, y2) = _anchor(src, side_out), _anchor(dst, side_in)
    k = max(28, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 * 0.4)
    (ax, ay), (bx, by) = DIRECTIONS[side_out], DIRECTIONS[side_in]
    c1, c2 = (x1 + ax * k, y1 + ay * k), (x2 + bx * k, y2 + by * k)
    t = LABEL_AT.get((src, dst), 0.5)
    lx = (1 - t) ** 3 * x1 + 3 * (1 - t) ** 2 * t * c1[0] + 3 * (1 - t) * t**2 * c2[0] + t**3 * x2
    ly = (1 - t) ** 3 * y1 + 3 * (1 - t) ** 2 * t * c1[1] + 3 * (1 - t) * t**2 * c2[1] + t**3 * y2
    return f"M{x1:.0f} {y1:.0f} C{c1[0]:.0f} {c1[1]:.0f} {c2[0]:.0f} {c2[1]:.0f} {x2:.0f} {y2:.0f}", (lx, ly)


def to_svg(trace: Trace) -> str:
    """Self-contained HTML (inline CSS + SVG) of the graph with the trace's path highlighted."""
    path = trace.path()
    visited = set(path)
    walked = set(zip(path, path[1:], strict=False))
    order: dict[str, list[str]] = {}
    for i, step in enumerate(trace.steps, 1):
        order.setdefault(step["node"], []).append(str(i))

    parts = [
        f'<div class="lg">{CSS}<svg viewBox="0 0 {VIEW_W} {VIEW_H}" role="img" aria-label="Agent graph">',
        '<defs><marker id="lg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="context-stroke"/></marker></defs>',
    ]
    for name, (title, tools) in SUBAGENTS.items():
        x = POS[f"{name}/agent"][0]
        parts.append(
            f'<g class="cluster"><title>{tools}</title><rect x="{x - CLUSTER_W / 2}" y="{CLUSTER_TOP}" '
            f'width="{CLUSTER_W}" height="{CLUSTER_BOTTOM - CLUSTER_TOP}"/>'
            f'<text x="{x}" y="{CLUSTER_TOP + 18}">{title}</text></g>'
        )
    for src, dst, label in EDGES:
        d, (lx, ly) = _edge_geometry(src, dst)
        cls = "edge walked" if (src, dst) in walked else "edge"
        text = f'<text x="{lx:.0f}" y="{ly - 4:.0f}">{label}</text>' if label else ""
        parts.append(f'<g class="{cls}" data-edge="{src}->{dst}"><path d="{d}" marker-end="url(#lg-arrow)"/>{text}</g>')
    for key, (x, y, w) in POS.items():
        state = (
            "waiting"
            if key == trace.waiting
            else "active"
            if key == trace.active
            else "visited"
            if key in visited
            else "idle"
        )
        tooltip = TOOLTIPS.get(key, LABELS[key])
        parts.append(
            f'<g class="node {state}" data-node="{key}"><title>{tooltip}</title>'
            f'<rect x="{x - w / 2}" y="{y - NODE_H / 2}" width="{w}" height="{NODE_H}" rx="9"/>'
            f'<text x="{x}" y="{y}">{LABELS[key]}</text></g>'
        )
        if key in order:
            badge = ",".join(order[key])
            bw = 8 + 7 * len(badge)
            parts.append(
                f'<g class="badge"><rect x="{x + w / 2 - bw + 6}" y="{y - NODE_H / 2 - 9}" width="{bw}" height="16" '
                f'rx="8"/><text x="{x + w / 2 - bw / 2 + 6}" y="{y - NODE_H / 2 - 1}">{badge}</text></g>'
            )
    for key, (x, y), label in ((START, START_XY, "start"), (END, END_XY, "end")):
        cls = "term visited" if key in visited else "term"
        ring = f'<circle class="{cls}" cx="{x}" cy="{y}" r="9" style="fill:none"/>' if key == END else ""
        parts.append(f'<circle class="{cls}" cx="{x}" cy="{y}" r="13"/>{ring}')
        parts.append(f'<text class="term-label" x="{x}" y="{y}">{label}</text>')
    parts.append("</svg></div>")
    return "".join(parts)


def step_title(key: str) -> str:
    if "/" in key:
        agent, part = key.split("/")
        return f"{SUBAGENTS[agent][0]} → {'🤖 LLM' if part == 'agent' else '🔧 tools'}"
    return LABELS[key].splitlines()[0]


def steps_markdown(trace: Trace) -> str:
    if not trace.steps and not trace.active:
        return "_Send a message to watch the agent move through the graph._"
    rows = [
        f"{i}. **{step_title(s['node'])}**" + (f" · {s['detail']}" if s["detail"] else "") + f" · `{s['ms']:.0f} ms`"
        for i, s in enumerate(trace.steps, 1)
    ]
    if trace.active:
        rows.append(f"{len(rows) + 1}. **{step_title(trace.active)}** · ⏳ running…")
    status = ""
    if trace.waiting:
        status = f"⏸ **Paused at {step_title(trace.waiting)}**, waiting for a person"
    elif trace.done:
        status = f"🏁 Done in `{sum(s['ms'] for s in trace.steps) / 1000:.2f} s`"
    return "\n".join(rows) + (f"\n\n{status}" if status else "")
