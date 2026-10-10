"""Bring-your-own-key gate. Offline: no real key is ever entered."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from agent.llm import validate_api_key

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_asks_for_key_before_chat(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_should_not_be_used")  # an env key must not bypass the gate
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert len(at.chat_input) == 0
    assert any("Connect your Groq API key" in h.value for h in at.subheader)
    key_input = at.text_input[0].proto
    assert type(key_input).Type.Name(key_input.type) == "PASSWORD"


def test_malformed_key_is_rejected_without_network():
    at = AppTest.from_file(APP).run()
    at.text_input[0].input("not-a-key")
    at.button[0].click().run()
    assert any("doesn't look like a Groq API key" in e.value for e in at.error)
    assert "graph" not in at.session_state
    assert len(at.chat_input) == 0


def test_validate_rejects_non_groq_keys():
    assert "doesn't look like" in validate_api_key("sk-openai-123")
