"""Regression tests for two prod incidents:

A. flow.clear discipline — the flow-builder prompt must NOT instruct the
   sub-agent to clear/re-template an EXISTING working_flow when the task is
   save/deploy/add/fix. It must only allow flow.clear for an explicit
   "rebuild from scratch / wipe" intent.

D. Nested sub-agent event isolation — task.run's _wrap_emit must NOT overwrite
   a `sub_agent_id` that a nested child already stamped (otherwise the outer
   parent's card replays the inner child's steps — data mixing), and must not
   count a nested child's tool_call toward the parent's tool_call_count.
"""

from __future__ import annotations

import pytest

import agent.graph as graph_mod
import agent.integrations.dynamic_config as dyn_mod
import agent.llm as llm_mod
from agent.llm import ModelRoles
from agent.llm.base import Message, TextPart
from agent.prompts.subagent_specs import REGISTRY
from agent.tools.base import ToolContext
from agent.tools.task import TaskRunInput
from agent.tools.task import task_run as _task_run_tool

_handler = _task_run_tool.handler


# ── A. flow.clear discipline (prompt-level guard) ───────────────────────────


def test_flow_builder_prompt_guards_flow_clear_against_save_deploy():
    prompt = REGISTRY["flow-builder"].system_prompt
    # The strict guard section must exist.
    assert "QAT'IY himoya" in prompt
    # save / deploy / ishga tushir must be explicitly named as NEVER-clear.
    for word in ("saqla", "deploy", "ishga tushir"):
        assert word in prompt, f"clear-guard missing the {word!r} case"
    # The old unconditional "clear the existing working_flow first" line is gone.
    assert "Mavjud working_flow bo'lsa, avval `flow.clear`" not in prompt
    # The old eager "first step is flow.clear" advice is gone.
    assert "birinchi qadam **`flow.clear(confirm=true)`**" not in prompt


def test_flow_builder_prompt_template_first_is_conditional_on_empty_flow():
    prompt = REGISTRY["flow-builder"].system_prompt
    # Template-first must be tied to an EMPTY working_flow, and incremental
    # edits must explicitly forbid template search + clear.
    assert "working_flow BO'SH" in prompt
    assert "template QIDIRMA" in prompt
    assert "`flow.clear` QILMA" in prompt


def test_flow_builder_prompt_teaches_collections_first():
    """Dynamic/list data must go through a CMS collection (seeded by cms-manager
    via nested task.run), never hardcoded into the flow."""
    prompt = REGISTRY["flow-builder"].system_prompt
    assert "collections-first" in prompt
    assert "Dinamik ma'lumot" in prompt
    # Demo seeding is delegated to cms-manager (flow-builder has no cms.* tools).
    assert "cms-manager" in prompt
    assert "cms.bulk_insert" in prompt
    assert 'subagent_type="cms-manager"' in prompt


def test_flow_builder_prompt_ai_provider_ollama_is_cloud():
    """Pattern 8 rewrite: Ollama default baseURL must be the CLOUD endpoint
    (https://ollama.com/v1) requiring an apiKey — NOT local 11434."""
    prompt = REGISTRY["flow-builder"].system_prompt
    assert "flow.add_ai_ollama_provider" in prompt
    assert "https://ollama.com/v1" in prompt
    # The wrong local endpoint must not be presented as a baseURL/default.
    assert "localhost:11434" not in prompt
    assert "http://localhost" not in prompt
    # All four provider typed tools are documented.
    for tool in ("flow.add_ai_openai_provider", "flow.add_ai_gemini_provider",
                 "flow.add_ai_anthropic_provider"):
        assert tool in prompt
    # connect_accessory ports are taught.
    assert "acc:ai_model" in prompt
    assert "acc:ai_tool" in prompt


def test_flow_builder_prompt_asks_user_for_llm_choice():
    """LLM provider/model and API keys must be asked via ask_user, with Ollama
    recommended as the free default — never assumed."""
    prompt = REGISTRY["flow-builder"].system_prompt
    assert "ask_user" in prompt
    # Ollama bepul/default tavsiya sifatida tilga olinadi.
    assert "Ollama" in prompt and ("bepul" in prompt or "tavsiya" in prompt)
    # API key haqiqiy qiymat sifatida ask_user orqali olinadi (placeholder emas).
    assert "ask_user" in prompt and "apiKey" in prompt


def test_flow_builder_spec_has_no_dead_budget_fields():
    spec = REGISTRY["flow-builder"]
    # Fix B: the dead metadata fields were removed (never enforced anywhere).
    assert not hasattr(spec, "max_tool_calls")
    assert not hasattr(spec, "max_wall_seconds")


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
    """On ainvoke, emits:
      1. an OWN tool_call (no sub_agent_id) — parent should stamp it + count it.
      2. a NESTED child event (already carries a foreign sub_agent_id) —
         parent must pass it through untouched and NOT count it.
    Then returns a final assistant message.
    """

    def __init__(self, emit) -> None:
        self._emit = emit

    async def ainvoke(self, base, config=None):
        # (1) this sub-agent's own tool_call — unstamped.
        await self._emit("tool_call", {"tool": "flow.add_node"})
        # (2) a tool_call bubbling up from a nested child (CMS Manager) that
        #     already stamped its own identity.
        await self._emit(
            "tool_call",
            {"tool": "cms.create_collection", "sub_agent_id": "sa-cms-manager-deadbeef",
             "sub_agent_name": "cms-manager"},
        )
        return {
            "messages": [Message(role="assistant", content=[TextPart(text="ok")])]
        }


@pytest.fixture()
def patched_nested(monkeypatch):
    holder: dict = {}

    def _build_graph(cfg, checkpointer=None):
        # cfg.emit is the wrapped emit (task.run's _wrap_emit). The fake graph
        # drives events THROUGH it exactly like the real sub-graph would.
        g = _NestedEmitGraph(cfg.emit)
        holder["graph"] = g
        holder["cfg"] = cfg
        return g

    monkeypatch.setattr(graph_mod, "build_graph", _build_graph)

    # task.run now resolves its LLM via the dynamic-config helpers (PR #27),
    # not the static llm.build_client. Mock those so the test never reaches a
    # real provider client (which would need API keys / a backend).
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
        TaskRunInput(subagent_type="flow-builder", prompt="build + nested cms"),
        ctx,
    )
    assert not res.is_error

    tool_calls = [(n, p) for n, p in ctx._emitted if n == "tool_call"]
    assert len(tool_calls) == 2

    own = tool_calls[0][1]
    nested = tool_calls[1][1]

    # (1) the sub-agent's OWN event was stamped with ITS id (flow-builder).
    assert own["sub_agent_id"].startswith("sa-flow-builder-")
    assert own["sub_agent_name"] == "flow-builder"

    # (2) the nested child's event kept the CHILD's id — NOT overwritten by
    #     the outer flow-builder wrapper (this is the data-mixing bug fix).
    assert nested["sub_agent_id"] == "sa-cms-manager-deadbeef"
    assert nested["sub_agent_name"] == "cms-manager"


@pytest.mark.asyncio
async def test_nested_child_tool_call_not_counted_by_parent(patched_nested):
    ctx = _ctx(_FakeCheckpointer())
    res = await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="build + nested cms"),
        ctx,
    )
    assert not res.is_error
    # Only the OWN tool_call counts; the nested child's does not inflate it.
    assert res.data["tool_calls"] == 1
    finished = [p for n, p in ctx._emitted if n == "sub_agent_finished"]
    assert finished and finished[-1]["tool_calls"] == 1


@pytest.mark.asyncio
async def test_subagent_system_block_is_spec_prompt_not_dispatcher(patched_nested):
    """A sub-agent must run with ITS OWN spec prompt as the system message, not
    the dispatcher persona — otherwise Sonnet-class models greet instead of
    doing the delegated task (deploy-manager replied 'Salom!' bug)."""
    ctx = _ctx(_FakeCheckpointer())
    await _handler(
        TaskRunInput(subagent_type="deploy-manager", prompt="deploy project 1185"),
        ctx,
    )
    cfg = patched_nested["cfg"]
    assert cfg.system_block == REGISTRY["deploy-manager"].full_prompt
    # And it must NOT be the dispatcher persona.
    assert "dispatcher" not in (cfg.system_block or "").lower()[:200]


def test_autonomy_rule_is_universal_across_models_and_subagents():
    """Autonomy guidance must reach the dispatcher (all models) AND every
    sub-agent — previously it was a Qwen-only addendum, so other models greeted
    / asked instead of executing."""
    from agent.prompts.system import STATIC_SYSTEM_BLOCK

    assert "AVTONOM" in STATIC_SYSTEM_BLOCK
    for name in ("flow-builder", "deploy-manager", "cms-manager", "crm-manager"):
        assert "AVTONOM" in REGISTRY[name].full_prompt, f"{name} missing autonomy"
