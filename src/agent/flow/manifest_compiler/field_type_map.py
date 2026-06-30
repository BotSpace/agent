"""Field-type mapping kontrakti — manifest field type → Pydantic/JSON-schema.

Bu jadval kompilyator (`compiler.py`, Bosqich 2) uchun YAGONA haqiqat manbasi:
frontend `NodeFieldManifest.type` (UI widget turi) qaysi data-key'larga, qaysi
Python tipiga aylanishini aytadi.

Ikki manba qoplanadi:
  1. Frontend `NodeFieldManifest["type"]` union — `bot-constructor-panel/.../manifest/types.ts`
     (built-in node manifestlari shu type'lardan foydalanadi).
  2. Backend `ALLOWED_FIELD_TYPES` — `backend/.../services/module_service.py`
     (modul manifestlari shu kichik to'plamdan foydalanadi).

Yangi field type qo'shilsa, shu jadvalga ham qo'shilishi SHART — aks holda
`tests/unit/test_manifest_field_type_map.py` sinadi (kontrakt himoyasi).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Field "kind" — kompilyator qanday Python tipini yasashini aniqlaydi ──────────
# Bu UI widget turidan (text/select/...) ko'ra dag'alroq sinf: bir nechta manifest
# widget turi bitta kind'ga tushadi (masalan text/textarea/code → scalar_str).


class FieldKind:
    SCALAR_STR = "scalar_str"        # str
    NUMBER = "number"               # int | float (min/max → ge/le)
    BOOL = "bool"                   # bool
    ENUM = "enum"                   # Literal[*options.value]
    STR_LIST = "str_list"           # list[str]
    LOCALIZED = "localized"         # LocalizedText (str | dict)  — i18n marker
    OBJECT = "object"               # dict[str, Any] (erkin nested config)
    NESTED_LIST = "nested_list"     # list[<sub-model>] — itemFields/columnFields/conditions
    CREDENTIAL_REF = "credential_ref"  # str (credential_id) + credentialType metadata
    NODE_REF = "node_ref"           # str (flow node id)
    FILE_REF = "file_ref"           # str (file url/id)
    COMPOSITE = "composite"         # panel sub-section bir nechta data-key yozadi (recipients, rows)
    UI_ONLY = "ui_only"             # data-key yo'q — schema'ga kirmaydi (description/widget)


@dataclass(frozen=True)
class FieldTypeSpec:
    """Bitta manifest field type'ning kompilyatsiya qoidasi."""

    kind: str
    # Ushbu field type yozadigan data-key manba(lar)i. "key" = field.key;
    # boshqalar field manifestidagi maxsus kalitlar (outputsKey, conditionsKey, ...).
    data_key_fields: tuple[str, ...] = ("key",)
    note: str = ""
    # COMPOSITE/UI_ONLY uchun: schema'da to'g'ridan-to'g'ri field yaratmaydi
    # (kompilyator maxsus ishlov beradi yoki e'tiborsiz qoldiradi).
    schema_field: bool = True


def _spec(kind: str, **kw) -> FieldTypeSpec:
    return FieldTypeSpec(kind=kind, **kw)


# ── Asosiy jadval ───────────────────────────────────────────────────────────────
# Reja "Field-type → schema mapping" jadvaliga mos.
FIELD_TYPE_MAP: dict[str, FieldTypeSpec] = {
    # — Oddiy skalar —
    "text": _spec(FieldKind.SCALAR_STR, note="label/placeholder → description"),
    "textarea": _spec(FieldKind.SCALAR_STR),
    "code": _spec(FieldKind.SCALAR_STR, note="modul code field (backend ALLOWED)"),
    "code-editor": _spec(
        FieldKind.SCALAR_STR,
        data_key_fields=("key", "outputsKey"),
        note="key=kod (str); outputsKey → outputs: list[str]; language metadata",
    ),
    "color": _spec(FieldKind.SCALAR_STR, note="hex rang string"),
    "datetime": _spec(FieldKind.SCALAR_STR, note="ISO datetime string"),
    "read-only": _spec(FieldKind.SCALAR_STR, note="str | None — round-trip qiymat (webhook url va h.k.)"),
    "number": _spec(FieldKind.NUMBER, note="min/max → ge/le; step metadata"),
    "checkbox": _spec(FieldKind.BOOL),
    "switch": _spec(FieldKind.BOOL),
    "boolean": _spec(FieldKind.BOOL, note="modul boolean (backend ALLOWED)"),
    "select": _spec(FieldKind.ENUM, note="options[].value → Literal; bittasi default"),
    "stringArray": _spec(FieldKind.STR_LIST, note="dinamik string ro'yxat"),
    # — i18n matn —
    "richTextMessage": _spec(FieldKind.LOCALIZED, note="LocalizedText (str|dict) i18n marker"),
    "mediaMessage": _spec(FieldKind.LOCALIZED, note="caption LocalizedText; mediaKind metadata"),
    # — Erkin obyekt / config —
    "json": _spec(FieldKind.OBJECT, note="modul json field (backend ALLOWED)"),
    "collection-select": _spec(
        FieldKind.OBJECT,
        data_key_fields=("key", "contextKeyField"),
        note="collection + (ixtiyoriy) field-mapping/filter/sort nested config",
    ),
    "collection-binding": _spec(
        FieldKind.OBJECT, note="collection + field map + filter + pagination nested config"
    ),
    "dynamic-params": _spec(
        FieldKind.OBJECT,
        data_key_fields=("schemaKey", "paramsKey", "descriptionKey"),
        note="schema + params + description data-key'lari",
    ),
    "tool-params": _spec(FieldKind.OBJECT, note="AI tool parametr config"),
    "dynamic_inputs": _spec(FieldKind.OBJECT, note="dinamik kiritma (schema auto-load)"),
    "dynamic_select": _spec(
        FieldKind.SCALAR_STR,
        data_key_fields=("key", "credentialKey"),
        note="tanlangan qiymat str; resource/dependsOn metadata (backend load-options)",
    ),
    "cron-editor": _spec(FieldKind.SCALAR_STR, note="cron ifoda string"),
    "trigger-config": _spec(
        FieldKind.OBJECT,
        note="message/callback/reply trigger nested config (kind metadata)",
    ),
    # — State (normalizatsiya nuqtasi — Bosqich 4a flat kanonik) —
    "stateConfig": _spec(
        FieldKind.OBJECT,
        note="state config; Bosqich 4a: flat (state_key/state_type/save_state) kanonik",
    ),
    "state-type-select": _spec(
        FieldKind.OBJECT,
        data_key_fields=("keyField", "typeField"),
        note="state key/type select; Bosqich 4a: flat kanonik",
    ),
    # — Nested ro'yxat (sub-model generatsiya) —
    "condition-list": _spec(
        FieldKind.NESTED_LIST,
        data_key_fields=("operatorKey", "conditionsKey"),
        note="operator: Literal + conditions: list[<sub>] (conditionFieldKey/conditionOpKey)",
    ),
    "array-editor": _spec(
        FieldKind.NESTED_LIST, note="list[<itemFields → sub-model>] (maxItems → max_length)"
    ),
    "nested-rows": _spec(
        FieldKind.NESTED_LIST, note="list[Row] — columnFields → sub-model (keyboard tugmalari)"
    ),
    # — Maxsus referenslar —
    "credential": _spec(
        FieldKind.CREDENTIAL_REF,
        note="key=credential_id (str); credentialType metadata bilan filtrlanadi",
    ),
    "node-select": _spec(FieldKind.NODE_REF, note="flow node id (str)"),
    "file": _spec(FieldKind.FILE_REF, note="fayl url/id (str) — modul ALLOWED"),
    # — Panel sub-section (bir nechta data-key) —
    "action-recipients": _spec(
        FieldKind.COMPOSITE,
        data_key_fields=("recipients",),
        schema_field=False,
        note="recipients: {sendToSender, additionalIds} — kompilyator default'dan oladi",
    ),
    "action-keyboard": _spec(
        FieldKind.COMPOSITE,
        data_key_fields=("keyboard_kind", "rows"),
        schema_field=False,
        note="keyboard_kind + rows — kompilyator default'dan oladi",
    ),
    # — UI-only (schema'ga kirmaydi) —
    "description": _spec(FieldKind.UI_ONLY, data_key_fields=(), schema_field=False,
                         note="faqat ko'rsatma matni"),
    "widget": _spec(FieldKind.UI_ONLY, data_key_fields=(), schema_field=False,
                    note="maxsus UI widget — data yo'q"),
}


# ── Manba ro'yxatlari (test bilan sinxronlik darvozasi) ─────────────────────────

# Backend modul manifestlari uchun ruxsat etilgan field type'lar.
# Manba: backend/core/apps/api/services/module_service.py :: ALLOWED_FIELD_TYPES
# Bu yerda NUSXASI — drift bo'lsa test ushlaydi.
BACKEND_ALLOWED_FIELD_TYPES: frozenset[str] = frozenset({
    "text", "number", "textarea", "select", "checkbox", "switch", "description",
    "widget", "json", "color", "file", "datetime", "boolean", "code", "credential",
})

# Frontend `NodeFieldManifest["type"]` union NUSXASI.
# Manba: bot-constructor-panel/src/features/bot-constructor/manifest/types.ts
# types.ts o'zgarsa shu ro'yxat YANGILANISHI va FIELD_TYPE_MAP qoplashi SHART.
FRONTEND_NODE_FIELD_TYPES: frozenset[str] = frozenset({
    "description", "widget", "text", "number", "textarea", "select", "checkbox",
    "switch", "stringArray", "stateConfig", "richTextMessage", "mediaMessage",
    "array-editor", "code-editor", "collection-select", "state-type-select",
    "condition-list", "cron-editor", "trigger-config", "dynamic-params",
    "tool-params", "action-recipients", "action-keyboard", "nested-rows",
    "collection-binding", "credential", "dynamic_inputs", "dynamic_select",
    "node-select", "read-only",
})


def known_field_types() -> frozenset[str]:
    """Kompilyator qoplashi kerak bo'lgan barcha field type'lar (ikki manba birlashmasi)."""
    return BACKEND_ALLOWED_FIELD_TYPES | FRONTEND_NODE_FIELD_TYPES
