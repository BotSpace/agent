"""Node data validatorlari — manifestdan RUNTIME'da kompilyatsiya qilinadi.

Avval har node uchun qo'lda Pydantic model yozilardi (~1750 qator) va u
engine'dan drift qilardi. Endi modellar frontend manifestidan
(`flow/manifest_compiler`) kompilyatsiya qilinadi — manifest engine bilan mos
(yagona haqiqat manbasi). Xavfsizlik (to'lov maxAmount clamp, toolName/operations
default, searchFields parsing) ENGINE'da, server-side — manifest/agent ishonch
chegarasi emas.

Ommaviy interfeys (o'zgarmadi): `NODE_DATA_MODELS` (dict-like), `get_data_model`,
`render_node_schema`, `render_params_doc`, `render_summary_table`.

Cross-field qoidalar (manifest ifoda eta olmaydigan) uchun `_ADDENDA` escape-hatch:
node_type → (model) -> model. Hozircha bo'sh (kerakli guard'lar engine'da).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel

from ...flow.manifest_compiler.compiler import LocalizedText, compile_node_model
from ...flow.manifest_compiler.loader import load_node_manifests

__all__ = [
    "NODE_DATA_MODELS",
    "LocalizedText",
    "get_data_model",
    "render_node_schema",
    "render_params_doc",
    "render_summary_table",
]

def _check_membership_addendum(model: type[BaseModel]) -> type[BaseModel]:
    """CheckMembership: channels/channelsCollection/chatId — kamida bittasi shart.

    Engine, hech qaysi manba bo'lmasa, HAMMANI a'zo deb o'tkazib yuboradi (gate
    ochiq qoladi). Manifest "kamida bittasi" cross-field qoidasini ifoda eta
    olmaydi — shu addendum bilan ushlaymiz.
    """
    from pydantic import model_validator

    class CheckMembershipNodeData(model):  # type: ignore[valid-type,misc]
        @model_validator(mode="after")
        def _at_least_one_channel(self):
            dumped = self.model_dump(by_alias=True, exclude_none=True)
            if not (dumped.get("channels") or dumped.get("channelsCollection") or dumped.get("chatId")):
                raise ValueError(
                    "kamida bitta kanal manbasi kerak: channels / channelsCollection / chatId"
                )
            return self

    return CheckMembershipNodeData


# Cross-field post-processorlar: kompilyatsiya qilingan modelga validator ulaydi.
# node_type → (model_cls) -> model_cls. Manifest ifoda eta olmaydigan qoidalar
# shu yerga (xavfsizlik guard'lari ENGINE'da — bu yerda emas).
_ADDENDA: dict[str, Callable[[type[BaseModel]], type[BaseModel]]] = {
    "CheckMembershipNode": _check_membership_addendum,
}


class _CompiledRegistry(Mapping):
    """Manifestdan lazy kompilyatsiya qilinadigan dict-like model registr.

    `NODE_DATA_MODELS[type]` → kompilyatsiya qilingan Pydantic model. Birinchi
    murojaatda kompilyatsiya qilinadi va keshlanadi.
    """

    def __init__(self) -> None:
        self._cache: dict[str, type[BaseModel]] = {}

    def __getitem__(self, node_type: str) -> type[BaseModel]:
        if node_type in self._cache:
            return self._cache[node_type]
        manifests = load_node_manifests()
        manifest = manifests.get(node_type)
        if manifest is None:
            raise KeyError(node_type)
        model = compile_node_model(manifest)
        addendum = _ADDENDA.get(node_type)
        if addendum is not None:
            model = addendum(model)
        self._cache[node_type] = model
        return model

    def __contains__(self, node_type: object) -> bool:
        return isinstance(node_type, str) and node_type in load_node_manifests()

    def __iter__(self):
        return iter(load_node_manifests())

    def __len__(self) -> int:
        return len(load_node_manifests())


NODE_DATA_MODELS: Mapping[str, type[BaseModel]] = _CompiledRegistry()


def get_data_model(node_type: str) -> type[BaseModel] | None:
    return NODE_DATA_MODELS.get(node_type)


# ── Render helperlar (flow.list_node_types / tool javoblari / system prompt) ──────


def render_params_doc(node_type: str) -> str:
    """Bitta node maydonlarining markdown jadvali."""
    model = NODE_DATA_MODELS.get(node_type)
    if model is None:
        return f"_(no validator for `{node_type}`)_"
    rows: list[str] = []
    for name, info in model.model_fields.items():
        alias = info.alias or name
        type_str = _format_type(info.annotation)
        req = "yes" if info.is_required() else "no"
        desc = (info.description or "").replace("\n", " ")
        rows.append(f"| `{alias}` | {type_str} | {req} | {desc} |")
    if not rows:
        return f"`{node_type}` — no data fields."
    head = "| field | type | required | description |\n|---|---|---|---|"
    return f"{head}\n" + "\n".join(rows)


def render_node_schema(node_type: str) -> dict[str, Any]:
    """flow.list_node_types uchun JSON-schema-like dict."""
    model = NODE_DATA_MODELS.get(node_type)
    if model is None:
        return {"type": node_type, "fields": [], "validator": False}
    fields = []
    for name, info in model.model_fields.items():
        fields.append({
            "name": info.alias or name,
            "type": _format_type(info.annotation),
            "required": info.is_required(),
            "default": None if info.is_required() else _safe_default(info.default),
            "description": info.description or "",
        })
    return {"type": node_type, "fields": fields, "validator": True}


def render_summary_table() -> str:
    """BARCHA node'larning ixcham jadvali (flow-builder system prompt uchun)."""
    from ...flow.manifest_compiler.node_catalog import NODE_TYPES

    rows_by_cat: dict[str, list[str]] = {}
    for type_name, spec in NODE_TYPES.items():
        model = NODE_DATA_MODELS.get(type_name)
        if model is None:
            req = ", ".join(spec.required_data) or "—"
            opt = ", ".join(spec.optional_data) or "—"
        else:
            req_fields, opt_fields = [], []
            for fname, info in model.model_fields.items():
                disp = f"{info.alias or fname}"
                annot = _format_type(info.annotation)
                if "|" in annot and "[" not in annot:  # Literal enums
                    disp = f"{disp}({annot})"
                (req_fields if info.is_required() else opt_fields).append(disp)
            req = ", ".join(req_fields) or "—"
            opt = ", ".join(opt_fields) or "—"
        rows_by_cat.setdefault(spec.category, []).append(
            f"| `{type_name}` | **{req}** | {opt} |"
        )
    parts: list[str] = []
    for cat in ("trigger", "send", "edit", "flow_control", "data", "integration", "interaction", "subflow", "ai"):
        if cat not in rows_by_cat:
            continue
        parts.append(f"\n### {cat}")
        parts.append("| node type | required (with enums) | optional |")
        parts.append("|---|---|---|")
        parts.extend(rows_by_cat[cat])
    return "\n".join(parts)


def _safe_default(value: Any) -> Any:
    try:
        if str(value).startswith("PydanticUndefined"):
            return None
        return value
    except Exception:
        return None


def _format_type(annot: Any) -> str:
    """Pydantic annotatsiyani docs uchun ixcham matnga aylantiradi."""
    try:
        s = str(annot)
    except Exception:
        return "any"
    for prefix in ("typing.", "<class '", "'>", "agent.flow.manifest_compiler.compiler."):
        s = s.replace(prefix, "")
    if "Literal[" in s:
        import re
        s = re.sub(
            r"Literal\[([^\]]+)\]",
            lambda m: m.group(1).replace("'", "").replace(", ", "|"),
            s,
        )
    return s
