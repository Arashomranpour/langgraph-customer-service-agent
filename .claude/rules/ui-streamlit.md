---
paths:
  - "app.py"
  - "ui/**/*.py"
---

# Streamlit UI rules

- No agent logic here; call the compiled graph from `agent.graph` only.
- Create one `thread_id` per user session, store it in `st.session_state`, pass it in the graph config.
- Stream responses to the UI.
- Buttons that start a graph run use `on_click` callbacks to queue work in session state (runs once per click).
- The live graph comes from `ui/graph_view.py`; keep presentation logic there, not in `app.py`.
- Load `.env` once at startup; show a clear error if `GROQ_API_KEY` is missing.
