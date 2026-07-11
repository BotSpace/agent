"""Tests for the universal autonomy rule + the model-addenda mechanism.

Autonomy guidance used to be a Qwen-only addendum (model_addenda/qwen.txt),
which is why Qwen executed decisively while Sonnet/others greeted and asked.
It is now a universal block (autonomy.txt) applied to the dispatcher (all
models) AND every sub-agent. The model_addenda mechanism is kept for genuinely
model-specific tweaks (the dir may be empty).
"""
# ruff: noqa: I001
from pathlib import Path

import agent.tools  # noqa: F401  — ensure tools package is initialized first
from agent.prompts.system import (
    _AUTONOMY_FILE,
    _MODEL_ADDENDA_DIR,
    STATIC_SYSTEM_BLOCK,
    load_autonomy_rule,
    load_model_addendum,
)
from agent.prompts.subagent_specs import REGISTRY


def test_autonomy_file_exists_and_non_empty():
    text = _AUTONOMY_FILE.read_text(encoding="utf-8").strip()
    assert text
    low = text.lower()
    assert "autonom" in low or "avtonom" in low
    assert "ask_user" in low


def test_autonomy_is_universal_dispatcher_and_subagents():
    # Dispatcher static block carries autonomy for EVERY model (was Qwen-only).
    assert load_autonomy_rule()
    assert "ask_user" in STATIC_SYSTEM_BLOCK.lower()
    # The analyzer sub-agent prompt carries it too (appended via NARRATE_RULE).
    assert "avtonom" in REGISTRY["analyzer"].full_prompt.lower()


# ── model-addenda mechanism (dir may be empty, mechanism still works) ───────────


def test_load_model_addendum_no_match_returns_empty():
    assert load_model_addendum("gpt-5.2") == ""
    assert load_model_addendum("claude-3-5-sonnet") == ""
    assert load_model_addendum("") == ""


def test_load_model_addendum_substring_and_longest_match(tmp_path, monkeypatch):
    # The substring + longest-stem-wins matching still works for any future
    # model-specific addendum files.
    d = tmp_path / "addenda"
    d.mkdir()
    (d / "qwen.txt").write_text("short", encoding="utf-8")
    (d / "qwen3-coder.txt").write_text("long", encoding="utf-8")
    monkeypatch.setattr("agent.prompts.system._MODEL_ADDENDA_DIR", d)
    assert load_model_addendum("bedrock/qwen3-coder-480b") == "long"
    assert load_model_addendum("bedrock/qwen2-7b") == "short"
    assert load_model_addendum("gpt-5.2") == ""


def test_addenda_dir_is_a_directory():
    assert isinstance(_MODEL_ADDENDA_DIR, Path)
    assert _MODEL_ADDENDA_DIR.is_dir()
