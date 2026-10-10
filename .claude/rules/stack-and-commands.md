# Stack and commands

- Python 3.12 in `.venv` (created with `uv venv --python 3.12 .venv`); langgraph 1.x, langchain-core 1.x
- `langgraph` (StateGraph, checkpointer, interrupts), `langchain-groq` (LLM), `streamlit` (UI), `python-dotenv`, `pytest`
- Model set via `GROQ_MODEL` (default `openai/gpt-oss-120b`). `llama-3.3-70b-versatile` is retired on Groq; list available models with `GET https://api.groq.com/openai/v1/models`.

## Commands
```
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py      # port 8502 via .streamlit/config.toml (Docker holds 8501)
.venv\Scripts\python.exe -m pytest -q
uvx ruff check . ; uvx ruff format .                  # CI runs both
.venv\Scripts\python.exe -m evals.run_evals            # live Groq, ~200 requests; --ids a,b for a subset
```

Groq free tier: 8k tokens/min and 1k requests/day on gpt-oss-120b. Don't run the full eval suite casually.
Prompt injection uses `meta-llama/llama-prompt-guard-2-86m` (returns a 0-1 score as text; threshold 0.5).
