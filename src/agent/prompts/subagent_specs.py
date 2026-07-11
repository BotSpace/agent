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
# Sub-agent — bitta umumiy ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
#
# Avval 4 ta domen sub-agent bor edi (flow-builder / crm-manager / cms-manager /
# deploy-manager) va main agent BUTUN ishni ularga uzatardi. Muammo: sub-agentlar
# main chat tarixini ko'rmaydi → chalkashib, tugagan ishni qaytadan qurishga
# urinishardi va token portlashiga sabab bo'lishardi. Endi butun domen ishi
# (bot qurish, CRM, CMS, deploy) MAIN agentning O'ZIDA bajariladi (u tarixni
# to'liq ko'radi). Faqat bitta `analyzer` sub-agent qoladi — TAHLIL/TEKSHIRUV
# uchun (bot qurish uchun EMAS). Analyzer to'liq toolset oladi, FAQAT `task.run`
# yo'q — ichma-ich sub-agent bo'lmaydi.

# analyzer'ga task.run'dan BOSHQA hamma imkoniyat beriladi. Prefikslar bilan
# barcha namespace'lar (task.* ataylab yo'q).
_ANALYZER_ALLOWED = [
    "plan.*",
    "ask_user",
    "project.*",
    "flow.*",
    "env.*",
    "release.*",
    "plans.*",
    "subscription.*",
    "payment.*",
    "file.*",
    "cms.*",
    "chat.*",
    "broadcast.*",
    "credentials.*",
    "modules.*",
    "search.*",
]


ANALYZER = SubAgentSpec(
    name="analyzer",
    description="Tahlil/tekshiruv yordamchisi — flow, CRM, CMS, loyiha holatini "
    "o'qib-tekshirib main agentga qisqa xulosa qaytaradi. Bot qurish uchun emas.",
    system_prompt=load_prompt("subagents/analyzer.txt").replace(
        "{node_params_table}", render_summary_table()
    ),
    allowed_tools=_ANALYZER_ALLOWED,
    model_hint="executor",
)


REGISTRY: dict[str, SubAgentSpec] = {ANALYZER.name: ANALYZER}


def list_subagents() -> list[SubAgentSpec]:
    return list(REGISTRY.values())
