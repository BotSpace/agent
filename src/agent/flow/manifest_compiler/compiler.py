"""Manifest → Pydantic model kompilyatori.

`compile_node_model(manifest)` frontend node manifestidan runtime'da strict
Pydantic model yasaydi. Bu qo'lda yozilgan `node_data_models.py` o'rnini bosadi —
manifest engine bilan mos (yagona haqiqat manbasi), shuning uchun kompilyatsiya
qilingan model engine kutgan data-key'larni aniq aks ettiradi.

Falsafa: STRUKTURANI tekshir (noma'lum kalit, yetishmagan required, enum, bool,
number), erkin/murakkab maydonlarda (textarea/json/condition-list) LENIENT bo'l —
engine bardoshli o'qiydi, agent vazifasi yo'qotilgan/xato kalitni ushlash.
"""

from __future__ import annotations

import keyword
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from .field_type_map import FIELD_TYPE_MAP, FieldKind
from .loader import NodeManifest

# Localizable text — oddiy string YOKI i18n marker obyekt.
LocalizedText = str | dict[str, Any]

# Bir data-key uchun emit: (data_key, annotation, description, required, default)
_Emit = tuple[str, Any, str | None, bool, Any]


class _StrictData(BaseModel):
    """Noma'lum kalitlarni rad etadi; alias yoki nom bilan to'ldirish mumkin.

    `coerce_numbers_to_str`: LLM string maydonlarga (masalan PaymentNode.amount)
    tez-tez SON yozadi (`1000`), string emas (`"1000"`). Engine ikkalasini ham
    o'qiydi, shuning uchun bu yerda ham sonni satrga coerce qilamiz — aks holda
    agent "Input should be a valid string" xatosiga tushib takror urinadi.
    """

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, coerce_numbers_to_str=True
    )


# Bu widget turlari DOIM yaroqli qiymatga ega (select→biror option, switch→bool,
# read-only→avto-to'ldiriladi) — shuning uchun har doim ixtiyoriy (required emas).
_ALWAYS_OPTIONAL_TYPES = {"select", "switch", "checkbox", "boolean", "read-only"}


def _is_optional(f: dict) -> bool:
    """Field ixtiyoriymi: optional bayrog'i, shartli ko'rinish, default-yashirin,
    yoki doim-yaroqli widget turi (select/switch/checkbox/read-only)."""
    return (
        bool(f.get("optional"))
        or bool(f.get("visibleWhen"))
        or f.get("defaultVisible") is False
        or f.get("type") in _ALWAYS_OPTIONAL_TYPES
    )


def _scalar_type(kind: str, options: list[str] | None):
    if kind == FieldKind.ENUM and options:
        return Literal[tuple(options)]  # type: ignore[valid-type]
    if kind == FieldKind.NUMBER:
        return int | float
    if kind == FieldKind.BOOL:
        return bool
    if kind == FieldKind.STR_LIST:
        return list[str]
    if kind == FieldKind.LOCALIZED:
        return LocalizedText
    if kind == FieldKind.OBJECT:
        return dict[str, Any] | list[Any] | str  # textarea(JSON-string)/dict/list
    if kind == FieldKind.NESTED_LIST:
        return list[dict[str, Any]]
    return str  # SCALAR_STR, CREDENTIAL_REF, NODE_REF, FILE_REF


def _describe(f: dict) -> str | None:
    """LLM-facing field doc. Manifestdagi `description` (AI uchun, boy) afzal;
    aks holda UI matni (label/placeholder/helpText) fallback."""
    return f.get("description") or f.get("label") or f.get("placeholder") or f.get("helpText")


def _emit_field(f: dict, defaults: dict) -> list[_Emit]:
    """Bitta manifest field → uning yozadigan data-key(lar)i.

    Ko'pchilik field `key` orqali bitta data-key yozadi; murakkablari
    (code-editor, condition-list, ...) bir nechta data-key yozadi va kalit
    NOMLARI manifest atributlarida (outputsKey, conditionsKey, ...) turadi.
    """
    ftype = f.get("type")
    spec = FIELD_TYPE_MAP.get(ftype)
    if spec is None or spec.kind in (FieldKind.UI_ONLY, FieldKind.COMPOSITE):
        return []  # composite default'lar orqali alohida qo'shiladi

    optional = _is_optional(f)
    desc = _describe(f)
    out: list[_Emit] = []

    if ftype == "code-editor":
        key = f.get("key")
        if key:
            out.append((key, str, desc, not optional, None))
        ok = f.get("outputsKey")
        if ok:
            out.append((ok, list[dict[str, Any]], "Named exit outputs.", False, defaults.get(ok)))
        return out

    if ftype == "condition-list":
        ck = f.get("conditionsKey") or "conditions"
        out.append((ck, list[dict[str, Any]], "Condition rows.", not optional, defaults.get(ck)))
        opk = f.get("operatorKey")
        if opk:
            out.append((opk, str, "Combine operator (and/or).", False, defaults.get(opk)))
        return out

    if ftype == "dynamic-params":
        for attr in ("schemaKey", "paramsKey", "descriptionKey"):
            dk = f.get(attr)
            if dk:
                out.append((dk, dict[str, Any] | list[Any] | str, None, False, defaults.get(dk)))
        return out

    if ftype == "state-type-select":
        for attr in ("keyField", "typeField"):
            dk = f.get(attr)
            if dk:
                out.append((dk, str, None, False, defaults.get(dk)))
        return out

    if ftype == "collection-select":
        key = f.get("key")
        if key:
            out.append((key, str, desc, not optional, defaults.get(key)))
        context_key = f.get("contextKeyField")
        if context_key:
            out.append((context_key, str, None, False, defaults.get(context_key)))
        if f.get("withFieldMapping"):
            out.append(("fieldMappings", dict[str, Any], "Collection field mappings.", False, defaults.get("fieldMappings") or {}))
        return out

    if ftype == "mediaMessage":
        # The media widget (frontend) writes `<kind>SourceType` + one of
        # `<kind>FileId` / `<kind>Url` / upload — the conditional source fields
        # aren't in the manifest, so the compiler must add them or the agent
        # can't set a video/photo/etc. source (strict model rejects the key).
        kind = str(f.get("mediaKind") or "media")
        return [
            (
                f"{kind}SourceType",
                Literal["upload", "file_id", "url"],  # type: ignore[valid-type]
                f"How the {kind} is provided: 'file_id' (Telegram file_id OR a "
                f"platform file UUID like {{{{yt_file}}}} from a module), 'url' "
                f"(direct link), or 'upload'.",
                False,
                defaults.get(f"{kind}SourceType") or "upload",
            ),
            (
                f"{kind}FileId",
                str,
                f"{kind} file_id/UUID — set when {kind}SourceType='file_id' "
                f"(e.g. {{{{yt_file}}}}).",
                False,
                defaults.get(f"{kind}FileId"),
            ),
            (
                f"{kind}Url",
                str,
                f"Direct {kind} URL — set when {kind}SourceType='url'.",
                False,
                defaults.get(f"{kind}Url"),
            ),
        ]

    # Oddiy: bitta `key`.
    key = f.get("key")
    if not key:
        return []
    options = [o["value"] for o in f.get("options") or [] if isinstance(o, dict) and "value" in o]
    out.append((key, _scalar_type(spec.kind, options or None), desc, not optional, defaults.get(key)))
    # collection-select / dynamic_select ikkilamchi kalitlari (ixtiyoriy).
    for attr in ("contextKeyField", "credentialKey"):
        dk = f.get(attr)
        if dk:
            out.append((dk, str, None, False, defaults.get(dk)))
    return out


def _composite_keys(manifest: NodeManifest, defaults: dict) -> list[_Emit]:
    """COMPOSITE field'lar (action-recipients/keyboard) yozadigan data-key'lar."""
    out: list[_Emit] = []
    for f in _field_entries(manifest):
        spec = FIELD_TYPE_MAP.get(f.get("type"))
        if spec is None or spec.kind != FieldKind.COMPOSITE:
            continue
        for dk in spec.data_key_fields:
            out.append((dk, Any, None, False, defaults.get(dk)))
    return out


def _field_entries(manifest: NodeManifest) -> list[dict]:
    entries: list[dict] = []
    for section in ("content", "configFields"):
        for f in manifest.get(section) or []:
            if isinstance(f, dict):
                entries.append(f)
    return entries


def _safe_name(key: str) -> tuple[str, str | None]:
    """(field_name, alias). Python keyword/identifier bo'lmasa nom sanitatsiya + alias."""
    if keyword.iskeyword(key) or key.endswith("_"):
        return key + "_", key
    if not key.isidentifier():
        return "f_" + "".join(c if c.isalnum() else "_" for c in key), key
    return key, None


# Manifest drift normalisatsiyasi — qarang: docs/manifest_drift_report.md.
# Ba'zi backend manifestlari hali eskirgan data-key'larni chiqaradi; engine'ning
# kanonik kontrakti (va AI-credential UUID-string kontrakti) yangi nomlarni
# kutadi. Kompilyatsiya paytida qayta nomlaymiz — shunda manifest drift bo'lsa
# ham kompilyatsiya qilingan model DOIM to'g'ri, engine bilan mos kalitni beradi.
# NODE-SCOPED (global emas) — chunki bir xil kalit boshqa node'da yaroqli:
#   * Anthropic provider ATAYIN `apiKey` ishlatadi (uning credential type'i yo'q,
#     validator.py buni tasdiqlaydi) — shuning uchun u ro'yxatda YO'Q.
_NODE_KEY_RENAMES: dict[str, dict[str, str]] = {
    "VariableNode": {"name": "variableName"},
    "AIOpenAIProviderNode": {"apiKey": "credential_id"},
    "AIGeminiProviderNode": {"apiKey": "credential_id"},
    "AIOllamaProviderNode": {"apiKey": "credential_id"},
}


def _rename_key(node_type: str, data_key: str) -> str:
    """Eskirgan manifest data-key'ini engine-kanonik nomga o'giradi (node bo'yicha)."""
    return _NODE_KEY_RENAMES.get(node_type, {}).get(data_key, data_key)


def compile_node_model(manifest: NodeManifest) -> type[BaseModel]:
    """Bitta node manifestdan strict Pydantic model yasaydi."""
    node_type = manifest.get("type", "Unknown")
    defaults = manifest.get("defaults") or {}

    emits: list[_Emit] = []
    for f in _field_entries(manifest):
        emits.extend(_emit_field(f, defaults))
    emits.extend(_composite_keys(manifest, defaults))

    fields: dict[str, tuple[Any, Any]] = {}
    seen: set[str] = set()
    for data_key, annotation, desc, required, default in emits:
        data_key = _rename_key(node_type, data_key)
        if not data_key or data_key in seen:
            continue
        seen.add(data_key)
        _add_field(fields, data_key, annotation, desc, required, default)

    # `defaults` ichidagi, lekin biror content-field qoplamagan kalitlar:
    # engine-kontrakt default'lari (tool/toolName/maxAmount/embeddingProvider/...)
    # ko'pincha shu yerda yashaydi (content'da emas). Ularni ixtiyoriy field
    # sifatida tipi default qiymatdan chiqarib emit qilamiz — agent shu kalitlarni
    # to'g'ri default bilan yozadi, engine esa baribir o'zi default beradi.
    for data_key, value in defaults.items():
        data_key = _rename_key(node_type, data_key)
        if data_key in seen:
            continue
        seen.add(data_key)
        _add_field(fields, data_key, _infer_type(value), None, False, value)

    return create_model(  # type: ignore[call-overload]
        f"{node_type}Data", __base__=_StrictData, **fields
    )


def _infer_type(value: Any):
    """Default qiymatdan Python tip annotatsiyasi."""
    if isinstance(value, bool):
        return bool
    if isinstance(value, (int, float)):
        return int | float
    if isinstance(value, str):
        return str
    if isinstance(value, list):
        return list[Any]
    if isinstance(value, dict):
        return dict[str, Any]
    return Any


def _add_field(fields: dict, data_key: str, annotation: Any, desc, required, default) -> None:
    name, alias = _safe_name(data_key)
    kw: dict[str, Any] = {}
    if alias:
        kw["alias"] = alias
    if desc:
        kw["description"] = desc
    if required:
        ann = annotation
    else:
        ann = annotation | None
        kw["default"] = default
    fields[name] = (ann, Field(**kw))
