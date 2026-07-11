"""Tests for the ask_user tool and the flow.clear approval gate (PART 1 & 2)."""
# ruff: noqa: I001  — agent.tools MUST import before agent.prompts.subagent_specs
# to fully initialize the tools package first; otherwise subagent_specs imported
# in isolation hits a partial circular import (tools.task → subagent_specs).
import asyncio

import pytest
from pydantic import ValidationError

from agent.tools import build_default_registry, build_main_registry
from agent.prompts.subagent_specs import REGISTRY, tool_allowed
from agent.tools.ask_user import AskUserInput, ask_user
from agent.tools.base import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(run_id="r", project_id=None, user_id=None)


def _run(coro):
    return asyncio.run(coro)


def test_ask_user_text_interrupt_payload():
    res = _run(ask_user.run({"question": "Bot nomi nima?"}, _ctx()))
    assert res.interrupt is not None
    assert res.interrupt["kind"] == "clarify"
    assert res.interrupt["question"] == "Bot nomi nima?"
    assert res.interrupt["input_type"] == "text"
    assert res.interrupt["options"] is None
    assert res.interrupt["placeholder"] is None


def test_ask_user_select_carries_options():
    res = _run(
        ask_user.run(
            {
                "question": "Qaysi rang?",
                "input_type": "select",
                "options": ["qizil", "ko'k"],
                "placeholder": "tanlang",
            },
            _ctx(),
        )
    )
    # Cross-stream contract: input_type + options reach the dashboard payload.
    assert res.interrupt["input_type"] == "select"
    assert res.interrupt["options"] == ["qizil", "ko'k"]
    assert res.interrupt["placeholder"] == "tanlang"


def test_ask_user_multiselect_carries_options():
    res = _run(
        ask_user.run(
            {
                "question": "Qaysi tillar?",
                "input_type": "multiselect",
                "options": ["uz", "ru", "en"],
            },
            _ctx(),
        )
    )
    assert res.interrupt["input_type"] == "multiselect"
    assert res.interrupt["options"] == ["uz", "ru", "en"]


def test_ask_user_select_requires_options():
    with pytest.raises(ValidationError):
        AskUserInput.model_validate({"question": "q", "input_type": "select"})
    with pytest.raises(ValidationError):
        AskUserInput.model_validate({"question": "q", "input_type": "multiselect"})


def test_ask_user_registered_main_and_subagents():
    assert build_main_registry().get("ask_user") is not None
    assert build_default_registry().get("ask_user") is not None
    for spec in REGISTRY.values():
        assert tool_allowed("ask_user", spec.allowed_tools), (
            f"ask_user must be allowed for {spec.name}"
        )


def test_flow_clear_is_removed():
    # flow.clear was removed — the AI kept wiping the working flow mid-work.
    assert build_default_registry().get("flow.clear") is None


def test_flow_save_saves_without_approval():
    # flow.save now persists immediately — NO approval prompt (neither
    # destructive-gated nor always_approve). The AI kept double-saving when it
    # had to re-issue the call after an approval interrupt.
    fs = build_default_registry().get("flow.save")
    assert fs is not None
    assert fs.destructive is False
    assert fs.always_approve is False
