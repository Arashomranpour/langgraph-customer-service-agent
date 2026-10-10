# Project structure

```
app.py              Streamlit UI: chat + live graph view, refund buttons, support agent console
agent/
  state.py          SupportState (messages + intent, confidence, sentiment, order_id, negative_turns,
                    blocked_reason, injection_score, ticket_id) and IntentResult schema
  llm.py            get_llm() and get_prompt_guard(): the only places Groq clients are built
  tools.py          Mock-data tools (search_faq, get_order_status, get_tracking, check_refund_eligibility)
                    + graph-only actions (create_refund_request, create_ticket)
  nodes.py          guardrails, injection_check, classifier, routers, build_tool_agent (ReAct sub-graph),
                    confirm_refund / human_handoff interrupts
  graph.py          build_graph(llm=None, injection_score=None, checkpointer=None)
ui/graph_view.py    Trace + Graphviz DOT for the live graph; LABELS/EDGES must mirror agent/graph.py
evals/              dataset.jsonl (50 labelled cases), run_evals.py, RESULTS.md (generated)
data/               faq.json, orders.json (mock)
tests/              pytest with FakeLLM + no_injection (conftest.py); no network
.github/workflows/  ci.yml (lint + tests on push), evals.yml (manual, needs GROQ_API_KEY secret)
```

## Graph shape
guardrails → (refuse_reply | injection_check) → (refuse_reply | classify_intent) →
(faq_agent | order_agent | refund_agent | chitchat | create_ticket).
refund_agent → confirm_refund (interrupt, customer approves) when eligible. create_ticket → human_handoff (interrupt).
Sub-agents are compiled `StateGraph(MessagesState)` added directly as nodes, so their tokens stream with `subgraphs=True`.

When adding or renaming a graph node, update `ui/graph_view.py` too; `tests/test_graph_view.py` fails if the view
misses a real node.
