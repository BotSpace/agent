"""Regression tests for prod incidents, updated for the single-agent design:

A. Build discipline — the main agent (which now builds bots directly) must NOT
   wipe an EXISTING working_flow on a save/deploy/add/fix task, must template
   only on an EMPTY flow, must use a CMS collection for dynamic data, and must
   default AI to cloud Ollama while asking the user for the LLM choice/keys.
   These guarantees now live in the main STATIC_SYSTEM_BLOCK (the building guide
   was folded in) rather than a flow-builder sub-agent prompt.

D. Nested-emit isolation — task.run's _wrap_emit must NOT overwrite a
   `sub_agent_id` a child already stamped, and must not count a foreign child's
   tool_call toward this sub-agent's tool_call_count.
"""

from __future__ import annotations

import pytest

import agent.graph as graph_mod
import agent.integrations.dynamic_config as dyn_mod
from agent.llm import ModelRoles
from agent.llm.base import Message, TextPart
from agent.prompts.subagent_specs import REGISTRY
from agent.prompts.system import STATIC_SYSTEM_BLOCK
from agent.tools.base import ToolContext
from agent.tools.task import TaskRunInput
from agent.tools.task import task_run as _task_run_tool

_handler = _task_run_tool.handler


# ── A. build discipline (main-prompt guarantees) ────────────────────────────


def test_prompt_guards_against_wiping_existing_flow_on_save_deploy():
    p = STATIC_SYSTEM_BLOCK
    # The "don't wipe an existing flow" guard must exist.
    assert "Mavjud flow'ni O'CHIRMA" in p
    # save / deploy / ishga tushir must be named as never-wipe cases.
    for word in ("saqla", "deploy", "ishga tushir"):
        assert word in p, f"wipe-guard missing the {word!r} case"
    # The removed flow.clear tool must not be referenced anywhere.
    assert "flow.clear" not in p


def test_prompt_template_first_is_conditional_on_empty_flow():
    p = STATIC_SYSTEM_BLOCK
    assert "working_flow BO'SH" in p
    assert "template QIDIRMA" in p


def test_prompt_teaches_collections_first():
    """Dynamic/list data must go through a CMS collection the agent creates
    itself, never hardcoded into the flow."""
    p = STATIC_SYSTEM_BLOCK
    assert "collections-first" in p
    assert "Dinamik ma'lumot" in p
    # Collections are created by the main agent itself now (no cms-manager).
    assert "cms-manager" not in p


def test_prompt_ai_provider_ollama_is_cloud():
    p = STATIC_SYSTEM_BLOCK
    assert "flow.add_ai_ollama_provider" in p
    # base_url moved to the credential — the agent must not set baseURL anymore.
    assert "https://ollama.com/v1" not in p
    assert "localhost:11434" not in p
    assert "http://localhost" not in p
    for tool in ("flow.add_ai_openai_provider", "flow.add_ai_gemini_provider",
                 "flow.add_ai_anthropic_provider"):
        assert tool in p
    assert "acc:ai_model" in p
    assert "acc:ai_tool" in p


def test_prompt_asks_user_for_llm_choice():
    p = STATIC_SYSTEM_BLOCK
    assert "ask_user" in p
    assert "Ollama" in p and ("bepul" in p or "tavsiya" in p)
    assert "apiKey" in p


def test_analyzer_spec_has_no_dead_budget_fields():
    spec = REGISTRY["analyzer"]
    assert not hasattr(spec, "max_tool_calls")
    assert not hasattr(spec, "max_wall_seconds")


def test_autonomy_rule_is_universal_across_main_and_analyzer():
    assert "AVTONOM" in STATIC_SYSTEM_BLOCK
    assert "AVTONOM" in REGISTRY["analyzer"].full_prompt


# ── D. nested-emit isolation ────────────────────────────────────────────────


def _ctx(checkpointer):
    emitted: list = []

    async def _emit(name, payload):
        emitted.append((name, payload))

    ctx = ToolContext(
        run_id="run-XYZ",
        project_id="proj-1",
        user_id="user-1",
        state={"thread_id": "conv-STABLE"},
        emit=_emit,
        services={"checkpointer": checkpointer},
    )
    ctx._emitted = emitted  # type: ignore[attr-defined]
    return ctx


class _FakeCheckpointer:
    async def aget(self, config):
        return None  # always fresh


class _NestedEmitGraph:
    """On ainvoke emits an OWN tool_call (parent stamps + counts it) and a
    foreign child event that already carries a sub_agent_id (parent passes it
    through untouched and does NOT count it)."""

    def __init__(self, emit) -> None:
        self._emit = emit

    async def ainvoke(self, base, config=None):
        await self._emit("tool_call", {"tool": "flow.add_node"})
        await self._emit(
            "tool_call",
            {"tool": "cms.create_collection", "sub_agent_id": "sa-child-deadbeef",
             "sub_agent_name": "child"},
        )
        return {
            "messages": [Message(role="assistant", content=[TextPart(text="ok")])]
        }


@pytest.fixture()
def patched_nested(monkeypatch):
    holder: dict = {}

    def _build_graph(cfg, checkpointer=None):
        g = _NestedEmitGraph(cfg.emit)
        holder["graph"] = g
        holder["cfg"] = cfg
        return g

    monkeypatch.setattr(graph_mod, "build_graph", _build_graph)

    async def _fake_build_client(*a, **k):
        return object()

    async def _fake_roles(*a, **k):
        return ModelRoles(planner="m", executor="m", critic="m", embedding="m")

    monkeypatch.setattr(dyn_mod, "build_effective_client", _fake_build_client)
    monkeypatch.setattr(dyn_mod, "effective_model_roles", _fake_roles)
    return holder


@pytest.mark.asyncio
async def test_nested_child_event_passes_through_untouched(patched_nested):
    ctx = _ctx(_FakeCheckpointer())
    res = await _handler(
        TaskRunInput(subagent_type="analyzer", prompt="inspect + nested"),
        ctx,
    )
    assert not res.is_error

    tool_calls = [(n, p) for n, p in ctx._emitted if n == "tool_call"]
    assert len(tool_calls) == 2

    own = tool_calls[0][1]
    nested = tool_calls[1][1]

    # (1) the analyzer's OWN event was stamped with ITS id.
    assert own["sub_agent_id"].startswith("sa-analyzer-")
    assert own["sub_agent_name"] == "analyzer"

    # (2) the foreign child's event kept the CHILD's id — not overwritten.
    assert nested["sub_agent_id"] == "sa-child-deadbeef"
    assert nested["sub_agent_name"] == "child"


@pytest.mark.asyncio
async def test_nested_child_tool_call_not_counted_by_parent(patched_nested):
    ctx = _ctx(_FakeCheckpointer())
    res = await _handler(
        TaskRunInput(subagent_type="analyzer", prompt="inspect + nested"),
        ctx,
    )
    assert not res.is_error
    assert res.data["tool_calls"] == 1
    finished = [p for n, p in ctx._emitted if n == "sub_agent_finished"]
    assert finished and finished[-1]["tool_calls"] == 1


@pytest.mark.asyncio
async def test_subagent_system_block_is_spec_prompt_not_dispatcher(patched_nested):
    """The analyzer must run with ITS OWN spec prompt as the system message, not
    the main agent's persona."""
    ctx = _ctx(_FakeCheckpointer())
    await _handler(
        TaskRunInput(subagent_type="analyzer", prompt="analyze project 1185"),
        ctx,
    )
    cfg = patched_nested["cfg"]
    assert cfg.system_block == REGISTRY["analyzer"].full_prompt
