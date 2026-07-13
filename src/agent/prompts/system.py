"""Main agent system prompt — dispatcher persona.

Main agent o'zi domen ishi qilmaydi. U:
1. Foydalanuvchi savolini tushunadi
2. Kerak bo'lsa plan tuzadi (`plan.set`)
3. Kerakli sub-agent'ga `task.run` orqali aniq vazifa beradi
4. Sub-agent natijasini foydalanuvchiga sodda tilda yetkazadi

Sub-agent tool call'lari main history'ga kirmaydi — faqat qisqa xulosa.
Dashboard ularni accordion'da ko'rsatadi.

Cache strategiyasi (rolling incremental):
  STATIC_SYSTEM_BLOCK  = CORE_IDENTITY + compact node catalog  (system, cached prefix, breakpoint 1)
  conversation history = user/assistant/tool messages          (rolling, breakpoint 2-3)
  dynamic context tail = joriy session kontekst                (LAST user message, uncached, changes per request)

Request shape:
  [ static-system (cached) ] → tools (cached) → [ conversation history (rolling cache) ] → [ dynamic-context tail (uncached) ]

The dynamic block is a trailing "user" message so it never invalidates the
conversation history cache.  For Anthropic: appended as an extra text block
on a user-role Message.  For OpenAI/OpenRouter: appended as a {"role": "user"}
dict — both paths work identically since neither rejects extra user messages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..flow.manifest_compiler.node_catalog import render_compact_catalog
from ..llm.base import Message
from .loader import load_prompt

logger = logging.getLogger(__name__)

_MODEL_ADDENDA_DIR = Path(__file__).parent / "model_addenda"

# Universal, model-INDEPENDENT autonomy guidance ("execute, don't ask for
# permission"). Previously this lived as a Qwen-only addendum, which is why Qwen
# behaved decisively while Sonnet/others greeted and asked. It applies to EVERY
# model now (dispatcher + sub-agents share the same file).
_AUTONOMY_FILE = Path(__file__).parent / "texts" / "shared" / "autonomy.txt"


def load_autonomy_rule() -> str:
    """Return the universal autonomy block; '' on any read error (never raises)."""
    try:
        return _AUTONOMY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def load_model_addendum(model: str) -> str:
    """Return the extra system prompt for `model` from model_addenda/*.txt.

    Scans the model_addenda dir for `.txt` files whose name (without extension,
    case-insensitive) is a SUBSTRING of the model id. Longest matching name wins
    (so `qwen3-coder.txt` beats `qwen.txt` for a qwen3-coder model). Returns ""
    when nothing matches or on any read error — never raises.

    Example: model="bedrock/qwen3-coder-480b" matches model_addenda/qwen.txt.
    """
    if not model:
        return ""
    model_lc = model.lower()
    try:
        candidates = sorted(
            _MODEL_ADDENDA_DIR.glob("*.txt"),
            key=lambda p: len(p.stem),
            reverse=True,
        )
    except Exception:
        return ""
    for path in candidates:
        if path.stem.lower() in model_lc:
            try:
                return path.read_text(encoding="utf-8").strip()
            except Exception:
                logger.warning("Failed to read model addendum %s", path, exc_info=True)
                return ""
    return ""


def _executor_model_addendum() -> str:
    """Addendum for the configured executor model (the model that runs tools)."""
    try:
        from ..config import get_settings

        return load_model_addendum(get_settings().model_executor)
    except Exception:
        return ""

CORE_IDENTITY: str = load_prompt("dispatcher/core_identity.txt")


def _build_guide() -> str:
    """Concise bot-building guide folded into the main agent's cached prefix.

    LAZY schema loading: the full per-node field schemas are NOT sent up front
    (that ballooned the prompt to ~82K/req). The prompt carries only the compact
    catalog (names + one-line summaries) + this concise guide; the agent fetches
    a specific node's full schema on demand via flow.list_node_types(type=...).
    Never raises — returns '' on any load error.
    """
    try:
        return load_prompt("dispatcher/build_guide.txt").strip()
    except Exception:
        logger.warning("Failed to build bot-building guide for system prompt", exc_info=True)
        return ""


# ── Static cached prefix ──────────────────────────────────────────────────────
# Built once at import time; byte-identical across all requests.
# render_compact_catalog() is deterministic (no timestamps, no random content).
# A model-specific addendum (model_addenda/<id>.txt for the configured executor
# model) is appended so it is part of the cached prefix and reaches Bedrock.
def _lazy_tools_catalog() -> str:
    """Compact name+summary catalog of LAZY tools (loaded on demand via
    tools.search). Never raises — '' on any error."""
    try:
        from ..tools import build_default_registry
        from ..tools.lazy import lazy_catalog

        return lazy_catalog(build_default_registry().all())
    except Exception:
        logger.warning("Failed to build lazy tool catalog", exc_info=True)
        return ""


_AUTONOMY_RULE: str = load_autonomy_rule()
_MODEL_ADDENDUM: str = _executor_model_addendum()
_BUILD_GUIDE: str = _build_guide()
_LAZY_TOOL_CATALOG: str = _lazy_tools_catalog()
STATIC_SYSTEM_BLOCK: str = (
    CORE_IDENTITY
    + "\n\n"
    + render_compact_catalog()
    # Detailed building mechanics + examples (main agent builds bots directly).
    + (("\n\n" + _BUILD_GUIDE) if _BUILD_GUIDE else "")
    # Lazy tools shown as name+summary; full schema via tools.search on demand.
    + (("\n\n" + _LAZY_TOOL_CATALOG) if _LAZY_TOOL_CATALOG else "")
    # Autonomy applies to ALL models (was Qwen-only — the reason other models
    # behaved worse). Model-specific addenda (if any) come after it.
    + (("\n\n" + _AUTONOMY_RULE) if _AUTONOMY_RULE else "")
    + (("\n\n" + _MODEL_ADDENDUM) if _MODEL_ADDENDUM else "")
)


def build_dynamic_block(state: dict[str, Any] | None = None) -> str:
    """Return the per-request dynamic context block (NOT cached).

    This is intentionally small — only session-level values that change
    between requests (project_id, flow stats, dirty flag).
    """
    state = state or {}
    project_id = state.get("project_id") or "—"
    user_id = state.get("user_id") or "—"
    lines: list[str] = ["## Kontekst"]
    goal = state.get("goal")
    if isinstance(goal, str) and goal.strip():
        g = goal.strip()
        if len(g) > 500:
            g = g[:500] + "…"
        lines.append(
            f"- 🎯 MAQSAD (shu vazifaning asosiy talabi — undan CHIQMA, har "
            f"qadamda shunga xizmat qil): {g}"
        )
    lines.append(f"- project_id: {project_id}")
    lines.append(f"- user_id: {user_id}")
    if state.get("baseline_flow"):
        wf = state["baseline_flow"]
        n_nodes = len((wf or {}).get("nodes") or [])
        n_edges = len((wf or {}).get("edges") or [])
        lines.append(
            f"- Joriy saqlangan flow: {n_nodes} node, {n_edges} edge (allaqachon "
            f"kontekstingda). flow.get'ni qayta-qayta yuklama; mavjudini o'chirma."
        )
    if state.get("flow_dirty"):
        lines.append(
            "- DIQQAT: saqlanmagan flow o'zgarishlari mavjud. flow.save qilishni "
            "unutma."
        )
    collections = state.get("collections") or []
    if collections:
        rendered = []
        for c in collections:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            if not name:
                continue
            fields = c.get("fields") or []
            if fields:
                rendered.append(f"{name}({', '.join(str(f) for f in fields)})")
            else:
                rendered.append(str(name))
        if rendered:
            lines.append("- Available collections: " + "; ".join(rendered))
    module_nodes = state.get("module_nodes") or []
    if module_nodes:
        # Each ENABLED module is shown as its OWN category (name + node count)
        # only — its node types/schemas are NOT dumped here. The agent loads a
        # module's nodes on demand with flow.list_node_types(module="<slug>").
        by_module: dict[str, dict[str, Any]] = {}
        for m in module_nodes:
            if not isinstance(m, dict) or not m.get("type"):
                continue
            slug = str(m.get("module") or m["type"].split(".", 1)[0])
            entry = by_module.setdefault(slug, {"name": m.get("module_name") or slug, "n": 0})
            entry["n"] += 1
        if by_module:
            mod_lines = "; ".join(
                f'{e["name"]} ({e["n"]} node, module="{slug}")'
                for slug, e in by_module.items()
            )
            lines.append(
                "- Enabled MODULES (each is a separate category; nodes NOT loaded here — "
                "fetch a module's nodes on demand with "
                "flow.list_node_types(module=\"<slug>\")): " + mod_lines
                + ". Add with flow.add_node(type='moduleSlug.NodeName'); module "
                "credential fields take a credential_id of the manifest type_key. "
                "Need a capability NO enabled module covers? modules.search → ask the "
                "user → modules.enable → reload session."
            )
    else:
        # No modules enabled for this project: nudge the discovery path so the
        # agent reaches for modules.search instead of inventing a node type.
        lines.append(
            "- Modules: none enabled for this project. Only built-in nodes are "
            "available. If the user needs a capability built-ins lack (another LLM "
            "provider, a third-party/own service), use modules.search to find a "
            "catalog module, confirm with the user, then modules.enable (SIDE-EFFECTFUL) "
            "and reload the session. See docs_guide(topic='modules')."
        )
    return "\n".join(lines) + "\n"


def build_system_prompt(state: dict[str, Any] | None = None) -> str:
    """Legacy single-string form — kept for tests / callers that need plain text.

    For LLM requests use build_system_message() which returns a two-block Message
    with the cache breakpoint on the static prefix.
    """
    return STATIC_SYSTEM_BLOCK + "\n\n" + build_dynamic_block(state)


def build_system_message(state: dict[str, Any] | None = None) -> Message:
    """Return the STATIC-ONLY system Message ready for the LLM clients.

    Contains ONLY STATIC_SYSTEM_BLOCK (CORE_IDENTITY + compact catalog).
    The LLM clients place a cache_control breakpoint on this block.

    The dynamic flow-state context is NO LONGER embedded here.
    It is injected as a trailing user message by build_dynamic_context_message()
    so it sits AFTER the conversation history and never invalidates the cached prefix.
    """
    return Message.system_text(STATIC_SYSTEM_BLOCK)


def build_dynamic_context_message(state: dict[str, Any] | None = None) -> Message:
    """Return the per-request dynamic context as a trailing USER message.

    This message is appended AFTER the conversation history in every LLM call:
      [static-system] → tools → [history (cached)] → [this message (uncached)]

    Using a user role is provider-correct for both Anthropic (which only
    allows "user" / "assistant" in the messages array) and OpenAI/OpenRouter.
    The dynamic content is small (project_id, flow stats, dirty flag) and
    changes every turn, so it must NOT receive a cache_control breakpoint.
    """
    return Message.user_text(f"[context]\n{build_dynamic_block(state)}")
