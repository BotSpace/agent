"""Tests for model-specific prompt addenda (PART 4)."""
from pathlib import Path

from agent.prompts.system import (
    _MODEL_ADDENDA_DIR,
    load_model_addendum,
)


def test_qwen_addendum_file_exists_and_non_empty():
    p = _MODEL_ADDENDA_DIR / "qwen.txt"
    assert p.exists()
    text = p.read_text(encoding="utf-8").strip()
    assert text
    # Must steer toward autonomy / not asking permission.
    low = text.lower()
    assert "autonom" in low or "avtonom" in low
    assert "ask_user" in low


def test_load_model_addendum_substring_match():
    # Any qwen-family model id should pick up qwen.txt.
    for model in ("qwen", "Qwen3-Coder", "bedrock/qwen3-coder-480b-instruct"):
        assert load_model_addendum(model).strip(), f"no addendum for {model}"


def test_load_model_addendum_no_match_returns_empty():
    assert load_model_addendum("gpt-5.2") == ""
    assert load_model_addendum("claude-3-5-sonnet") == ""
    assert load_model_addendum("") == ""


def test_load_model_addendum_longest_match_wins(tmp_path, monkeypatch):
    # Two overlapping names: longer stem must win.
    d = tmp_path / "addenda"
    d.mkdir()
    (d / "qwen.txt").write_text("short", encoding="utf-8")
    (d / "qwen3-coder.txt").write_text("long", encoding="utf-8")
    monkeypatch.setattr("agent.prompts.system._MODEL_ADDENDA_DIR", d)
    assert load_model_addendum("bedrock/qwen3-coder-480b") == "long"
    assert load_model_addendum("bedrock/qwen2-7b") == "short"


def test_static_block_includes_addendum_when_qwen(monkeypatch):
    # Simulate a qwen executor and rebuild the addendum lookup.
    from agent.prompts import system as sysmod

    monkeypatch.setattr(
        "agent.config.get_settings",
        lambda: type("S", (), {"model_executor": "bedrock/qwen3-coder"})(),
    )
    addendum = sysmod._executor_model_addendum()
    assert "ask_user" in addendum.lower()


def test_addenda_dir_is_a_directory():
    assert isinstance(_MODEL_ADDENDA_DIR, Path)
    assert _MODEL_ADDENDA_DIR.is_dir()
