---
paths:
  - "tests/**/*.py"
---

# Testing rules

- Tests mock the LLM so they run offline and without an API key.
- Test nodes and tools individually, then the compiled graph with a fake model.
- Run with `pytest`.
- `build_graph(fake_llm, no_injection)`: always pass a fake injection scorer too, or the test calls Groq.
- Live-model behaviour is measured by `evals/`, not unit tests. Add a dataset case when fixing a model-behaviour bug.
