# Configuration and secrets

Follow the user-level `groq-byok` skill.

- **Web app is bring-your-own-key.** `app.py` asks each visitor for a Groq key, validates it with
  `validate_api_key` (lists models, no tokens spent) and builds that session's graph with `build_graph(api_key=...)`.
  The key exists only in that session's clients: no `st.cache_resource`, files, logs or env fallback in the app.
  `tests/test_app.py` checks that an env key can't bypass the gate.
- `GROQ_API_KEY` in `.env` (gitignored) is for the CLI, tests and evals only (`get_llm()`, `get_prompt_guard()`).
  Ask the user for the key at the start of a session; never reuse one from memory.
- Never commit, log, or print API keys. Before committing: `git grep --cached -n "gsk_"` must be empty.
- Never show raw Groq errors in the UI; they contain the organization ID.
- Use only mock or fake customer data.
