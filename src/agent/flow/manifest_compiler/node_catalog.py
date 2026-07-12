"""Node katalogi — MANIFESTDAN derive qilinadi (qo'lyozma catalog.py o'rniga).

Avval `flow/catalog.py` har node metadatasini (tavsif, handle'lar, required/optional)
QO'LDA saqlardi. Endi hammasi backend manifestidan keladi:
- `description` ← manifest `aiDoc` (AI uchun yo'riqnoma)
- `required_data`/`optional_data` ← kompilyatsiya qilingan model (manifest field'lari)
- `source_handles`/`target_handles` ← manifest `sourceHandles`/`targetHandles`
- `is_trigger` ← category=="trigger"; `is_accessory` ← manifest `accessory`
- `accessory_target_handles` ← manifest `accessoryPorts`

Ommaviy API catalog.py bilan bir xil: `NODE_TYPES`, `all_node_types`,
`get_node_spec`, `is_trigger`, `summary_for_llm`, `render_compact_catalog`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from .compiler import compile_node_model
from .loader import load_node_category_docs, load_node_manifests

_CATEGORY_ORDER = (
    "trigger", "send", "edit", "flow_control", "data",
    "integration", "interaction", "subflow", "payment", "ai",
)

# Categories whose node list is ALWAYS spelled out in the cached system prompt
# (the core building blocks — few, common). Every OTHER category is listed by
# NAME + count only; the agent pulls that category's nodes on demand via
# flow.list_node_types(category="..."). This keeps the always-sent catalog tiny.
_BASE_CATEGORIES = frozenset(
    {"trigger", "send", "edit", "flow_control", "interaction"}
)


@dataclass(frozen=True)
class NodeTypeSpec:
    type: str
    category: str
    description: str
    summary: str = ""  # bir qatorli qisqa tavsif (compact katalog uchun; params yo'q)
    required_data: list[str] = field(default_factory=list)
    optional_data: list[str] = field(default_factory=list)
    source_handles: list[str] = field(default_factory=lambda: ["source-handler"])
    target_handles: list[str] = field(default_factory=lambda: ["target-handler"])
    is_trigger: bool = False
    is_accessory: bool = False
    accessory_target_handles: list[str] = field(default_factory=list)


def _spec_from_manifest(m: dict) -> NodeTypeSpec:
    node_type = m["type"]
    # `agentCategory` — agent semantik kategoriyasi (trigger/send/edit/flow_control/...).
    # Manifest `category` esa frontend grouping (triggers/flowControl/...) — boshqa.
    category = m.get("agentCategory") or m.get("category") or "integration"
    # required/optional — kompilyatsiya qilingan modeldan (manifest field'lari).
    model = compile_node_model(m)
    required, optional = [], []
    for name, info in model.model_fields.items():
        alias = info.alias or name
        (required if info.is_required() else optional).append(alias)
    acc_ports = ["acc:" + p.get("portType", "") for p in m.get("accessoryPorts") or []]
    # `[]` (bo'sh — masalan SubFlowExit, chiqishsiz leaf) falsy → None tekshir.
    src = m.get("sourceHandles")
    tgt = m.get("targetHandles")
    return NodeTypeSpec(
        type=node_type,
        category=category,
        description=m.get("aiDoc") or m.get("descriptionFallback") or "",
        summary=m.get("summary") or "",
        required_data=required,
        optional_data=optional,
        source_handles=src if src is not None else ["source-handler"],
        target_handles=tgt if tgt is not None else ["target-handler"],
        is_trigger=category == "trigger",
        is_accessory=bool(m.get("accessory")),
        accessory_target_handles=acc_ports,
    )


@lru_cache(maxsize=1)
def _node_types() -> dict[str, NodeTypeSpec]:
    return {t: _spec_from_manifest(m) for t, m in load_node_manifests().items()}


class _NodeTypesView:
    """`NODE_TYPES` dict-like (lazy, manifestdan)."""

    def __getitem__(self, k: str) -> NodeTypeSpec:
        return _node_types()[k]

    def get(self, k: str, default=None):
        return _node_types().get(k, default)

    def values(self):
        return _node_types().values()

    def keys(self):
        return _node_types().keys()

    def items(self):
        return _node_types().items()

    def __iter__(self):
        return iter(_node_types())

    def __contains__(self, k):
        return k in _node_types()

    def __len__(self):
        return len(_node_types())


NODE_TYPES = _NodeTypesView()


def all_node_types() -> list[NodeTypeSpec]:
    return list(_node_types().values())


# ── Manifestdan derive qilinadigan node-turi to'plamlari (avval qotirilgan edi) ──

# MessageNode — manifestda yo'q legacy alias; button-edge uchun qo'shib qo'yamiz.
_LEGACY_BUTTON_EDGE = frozenset({"MessageNode"})


@lru_cache(maxsize=1)
def button_edge_types() -> frozenset[str]:
    """Engine button-edge'ni qo'llab-quvvatlaydigan klaviaturali node'lar
    (manifest `buttonEdge:true`) + legacy `MessageNode`."""
    man = load_node_manifests()
    return frozenset(t for t, m in man.items() if m.get("buttonEdge")) | _LEGACY_BUTTON_EDGE


@lru_cache(maxsize=1)
def state_saving_types() -> frozenset[str]:
    """`data.type` ni MAJBURIY talab qiladigan state-saqlovchi node'lar
    (manifest `savesState:true`)."""
    return frozenset(t for t, m in load_node_manifests().items() if m.get("savesState"))


@lru_cache(maxsize=1)
def acc_port_accepts() -> dict[str, frozenset[str]]:
    """`acc:<portType>` → qabul qilinadigan node turlari (manifest `accessoryPorts`)."""
    out: dict[str, set[str]] = {}
    for m in load_node_manifests().values():
        for port in m.get("accessoryPorts") or []:
            pt = port.get("portType")
            if pt:
                out.setdefault("acc:" + pt, set()).update(port.get("accepts") or [])
    return {k: frozenset(v) for k, v in out.items()}


def get_node_spec(type_name: str) -> NodeTypeSpec | None:
    return _node_types().get(type_name)


def is_trigger(type_name: str) -> bool:
    spec = _node_types().get(type_name)
    return spec.is_trigger if spec else False


def summary_for_llm() -> str:
    """System prompt uchun qisqa katalog dump."""
    lines: list[str] = []
    categories: dict[str, list[NodeTypeSpec]] = {}
    for spec in _node_types().values():
        categories.setdefault(spec.category, []).append(spec)
    for cat in _CATEGORY_ORDER:
        items = categories.get(cat, [])
        if not items:
            continue
        lines.append(f"\n[{cat.upper()}]")
        for spec in items:
            req = ", ".join(spec.required_data) if spec.required_data else "—"
            lines.append(f"  {spec.type}: {spec.description}  required: {req}")
    return "\n".join(lines)


def render_compact_catalog() -> str:
    """Cache'lanadigan system-prompt uchun ixcham node katalogi.

    Faqat node NOMI + bir qatorli `summary` (nima qilishi) — params EMAS.
    Node maydonlarini (params/schema) va to'liq `aiDoc`ni bilish uchun agent
    `flow.list_node_types(type=...)` chaqiradi. Summary bo'lmasa aiDoc'ga tushadi.
    """
    buckets: dict[str, list[NodeTypeSpec]] = {cat: [] for cat in _CATEGORY_ORDER}
    for spec in _node_types().values():
        if spec.category in buckets:
            buckets[spec.category].append(spec)
    cat_docs = load_node_category_docs()

    lines: list[str] = [
        "## Node tiplari kataloqi\n"
        "Asosiy (base) kategoriyalar to'liq keltirilgan (nom + nima qilishi). "
        "Boshqa kategoriyalar faqat NOMI bilan — ichidagi node'larni kerak bo'lganda "
        "`flow.list_node_types(category=\"<nom>\")` bilan yukla. Bitta node maydonlari "
        "uchun `flow.list_node_types(type=...)`.\n\n"
    ]
    for cat in _CATEGORY_ORDER:
        items = buckets[cat]
        if not items:
            continue
        if cat not in _BASE_CATEGORIES:
            # Specialized category — name + count + a short doc (from the backend
            # manifest's node_categories) only; nodes loaded on demand so their
            # summaries don't sit in every request's prompt.
            doc = (cat_docs.get(cat) or {}).get("doc", "")
            lines.append(
                f"### {cat.upper()} ({len(items)} node) — "
                f"`flow.list_node_types(category=\"{cat}\")` bilan yukla\n"
            )
            if doc:
                lines.append(f"  {doc}\n")
            continue
        lines.append(f"### {cat.upper()}\n")
        for spec in items:
            trigger_flag = " [trigger]" if spec.is_trigger else ""
            accessory_flag = " [accessory]" if spec.is_accessory else ""
            handles = spec.source_handles
            handles_str = (
                "" if (len(handles) == 1 and handles[0] == "source-handler")
                else f"  handles: {', '.join(handles)}"
            )
            lines.append(
                f"- **{spec.type}**{trigger_flag}{accessory_flag} "
                f"[{spec.category}]{handles_str}\n"
                f"  {spec.summary or spec.description}\n"
            )
    return "".join(lines)
