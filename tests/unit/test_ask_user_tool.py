"""Tests for the ask_user tool and the flow.clear approval gate (PART 1 & 2)."""
import asyncio

import pytest
from pydantic import ValidationError

from agent.prompts.subagent_specs import REGISTRY, tool_allowed
from agent.tools import build_default_registry, build_main_registry
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


def test_flow_clear_is_destructive_and_always_approve():
    fc = build_default_registry().get("flow.clear")
    assert fc is not None
    assert fc.destructive is True
    assert fc.always_approve is True


def test_flow_save_destructive_but_not_always_approve():
    # flow.save / deploy stay ungated inside sub-agents (approve_destructive=False).
    fs = build_default_registry().get("flow.save")
    assert fs.destructive is True
    assert fs.always_approve is False
