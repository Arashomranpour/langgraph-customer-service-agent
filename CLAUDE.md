# Customer Service Agent

Online customer service agent. LangGraph runs the agent logic; Streamlit provides the chat UI; Groq is the LLM provider.

README.md is the public, CV-facing description; keep its numbers in sync with `evals/RESULTS.md`.

Detailed rules live in `.claude/rules/`:
- `stack-and-commands.md`, `project-structure.md`, `secrets.md` load every session.
- `agent-langgraph.md` (`agent/**`), `ui-streamlit.md` (`app.py`), `testing.md` (`tests/**`) load only when working on matching files.
