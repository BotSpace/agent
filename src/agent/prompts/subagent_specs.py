"""Sub-agent registry — Claude Code'dagi Agent tool kabi.

Botmother arxitekturasi:
- **Main agent** — dispatcher. Faqat plan.*, project.*, task.run, search.docs ga ega.
  Domen ishini sub-agent'ga uzatadi. Main history'da sub-agent qadamlari saqlanmaydi —
  faqat qisqa xulosa.
- **Sub-agent** — domen ekspert. Alohida tool to'plami, alohida thread, alohida
  system prompt. Oxirida main'ga 1-3 jumlali natija qaytaradi.

Allowed-tools sintaksis: aniq nom (`plan.set`) yoki prefiks (`flow.*`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tools.flow.node_data_models import render_summary_table
from .loader import load_prompt

# Universal autonomy guidance, shared with the dispatcher system block. Appended
# to every sub-agent's prompt so ALL models (not just Qwen) act decisively
# instead of greeting / asking permission. '' on read error.
_AUTONOMY_RULE = load_prompt("shared/autonomy.txt")


NARRATE_RULE = load_prompt("shared/narrate_rule.txt")

# Make autonomy part of every sub-agent prompt (all models), not just Qwen.
if _AUTONOMY_RULE:
    NARRATE_RULE = NARRATE_RULE + "\n" + _AUTONOMY_RULE + "\n"


@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    description: str
    """Foydalanuvchiga ko'rinadigan qisqa tavsif."""
    system_prompt: str
    allowed_tools: list[str]
    """Tool name'lar yoki prefikslar: `plan.set` yoki `flow.*`."""
    model_hint: str = "executor"
    """`planner` / `executor` / `critic` — ModelRoles dan tanlanadi."""

    @property
    def full_prompt(self) -> str:
        return self.system_prompt + NARRATE_RULE


def tool_allowed(tool_name: str, patterns: list[str]) -> bool:
    """Sub-agent allowed_tools ro'yxati bo'yicha tool nomini tekshiradi.

    Patterns:
      - aniq nom: `plan.set` → faqat shu
      - prefiks: `flow.*` → flow.add_*, flow.connect_*, va h.k.
    """
    for pat in patterns:
        if pat.endswith(".*"):
            if tool_name.startswith(pat[:-1]):
                return True
        elif pat == tool_name:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Sub-agents
# ─────────────────────────────────────────────────────────────────────────────

# Flow-builder prompt = asosiy yo'riqnoma + to'liq bot misollari (alohida fayl,
# foydalanuvchi dinamik kengaytira oladi). Ikkalasi birlashtirilib beriladi.
_FLOW_BUILDER_PROMPT = (
    load_prompt("subagents/flow_builder.txt")
    + "\n\n"
    + load_prompt("subagents/flow_builder_examples.txt")
)


FLOW_BUILDER = SubAgentSpec(
    name="flow-builder",
    description="Bot visual flow muharriri — node, edge, validation, layout, save.",
    system_prompt=_FLOW_BUILDER_PROMPT.replace(
        "{node_params_table}", render_summary_table()
    ),
    allowed_tools=[
        "plan.*",
        "ask_user",
        "flow.*",
        "task.run",
    ],
    model_hint="executor",
)


CRM_MANAGER = SubAgentSpec(
    name="crm-manager",
    description="Foydalanuvchilar, segmentlar, broadcast — CMS query va chat tarixi.",
    system_prompt=load_prompt("subagents/crm_manager.txt"),
    allowed_tools=[
        "plan.*",
        "ask_user",
        "cms.list_collections",
        "cms.get_collection",
        "cms.find",
        "cms.get_document",
        "cms.aggregate",
        "chat.*",
        "broadcast.*",
        "task.run",
    ],
    model_hint="executor",
)


CMS_MANAGER = SubAgentSpec(
    name="cms-manager",
    description="MongoDB collections schema va document CRUD — cms.*",
    system_prompt=load_prompt("subagents/cms_manager.txt"),
    allowed_tools=[
        "plan.*",
        "ask_user",
        "cms.*",
        "task.run",
    ],
    model_hint="executor",
)


DEPLOY_MANAGER = SubAgentSpec(
    name="deploy-manager",
    description="Bot deploy lifecycle — env, release, tarif, to'lov, loglar.",
    system_prompt=load_prompt("subagents/deploy_manager.txt"),
    allowed_tools=[
        "plan.*",
        "ask_user",
        "env.*",
        "release.*",
        "plans.*",
        "subscription.*",
        "payment.*",
        "file.*",
        "task.run",
    ],
    model_hint="executor",
)


REGISTRY: dict[str, SubAgentSpec] = {
    spec.name: spec
    for spec in (FLOW_BUILDER, CRM_MANAGER, CMS_MANAGER, DEPLOY_MANAGER)
}


def get_subagent(name: str) -> SubAgentSpec | None:
    return REGISTRY.get(name)


def list_subagents() -> list[SubAgentSpec]:
    return list(REGISTRY.values())
