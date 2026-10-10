---
paths:
  - "agent/**/*.py"
---

# Agent (LangGraph) rules

- The graph must be importable and testable without Streamlit; no `streamlit` imports in `agent/`.
- Nodes are small and single-purpose; type hints on all functions.
- Tools return structured data (dicts), not prose; the LLM phrases the reply.
- Always provide a path to a human (escalation node) for complaints, low confidence, or explicit requests.
- Conversation memory uses the checkpointer keyed by `thread_id` passed in the graph config.
- Build the Groq client only in `agent/llm.py`.
