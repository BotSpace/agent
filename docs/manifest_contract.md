# Manifest → schema kontrakti

Node'ning data-maydonlari (field) **frontend manifestda** (`NodeFieldManifest`)
ta'riflanadi va agent uni **runtime'da Pydantic schema'ga kompilyatsiya** qiladi
(qo'lda yozilgan `node_data_models.py` o'rniga). Bu hujjat field-type
semantikasini va kompilyatsiya qoidalarini belgilaydi.

Reja: `~/.claude/plans/agent-arxitikturasini-yangilash-kerak-eager-thimble.md`

## Manbalar

| Rol | Joy |
|---|---|
| Field type union (frontend) | `bot-constructor-panel/.../manifest/types.ts` → `NodeFieldManifest["type"]` |
| Modul field type whitelist (backend) | `backend/.../services/module_service.py` → `ALLOWED_FIELD_TYPES` |
| Mapping kontrakti (agent) | `agent/src/agent/flow/manifest_compiler/field_type_map.py` |
| Kontrakt darvozasi (test) | `agent/tests/unit/test_manifest_field_type_map.py` |

Yangi field type qo'shilsa, `field_type_map.py` ga ham qo'shilishi SHART —
aks holda kontrakt testi sinadi.

## Field → data-key

Har manifest field `key` orqali `node.data[key]` ga yoziladi. Ba'zi murakkab
field turlari bir nechta data-key yozadi (manifestdagi maxsus kalitlar orqali):

- `code-editor` → `key` (kod), `outputsKey` (`outputs: list[str]`)
- `condition-list` → `operatorKey`, `conditionsKey`
- `dynamic-params` → `schemaKey`, `paramsKey`, `descriptionKey`
- `action-recipients` → `recipients`; `action-keyboard` → `keyboard_kind`, `rows`
  (panel sub-section'lar — data `defaults` dan keladi, alohida `key` yo'q)

`description` va `widget` — UI-only, schema'ga KIRMAYDI.

## FieldKind → Python tip

`field_type_map.py::FieldKind` har manifest type'ni dag'al sinfga soladi:

| FieldKind | Python tip | Manifest type'lar |
|---|---|---|
| `scalar_str` | `str` | text, textarea, code, code-editor, color, datetime, read-only, cron-editor, dynamic_select |
| `number` | `int \| float` (min/max → ge/le) | number |
| `bool` | `bool` | checkbox, switch, boolean |
| `enum` | `Literal[*options.value]` | select |
| `str_list` | `list[str]` | stringArray |
| `localized` | `LocalizedText` (`str \| dict`) | richTextMessage, mediaMessage |
| `object` | `dict[str, Any]` / nested config | json, collection-select, collection-binding, dynamic-params, tool-params, dynamic_inputs, trigger-config, stateConfig, state-type-select |
| `nested_list` | `list[<sub-model>]` | array-editor (itemFields), nested-rows (columnFields), condition-list (conditions) |
| `credential_ref` | `str` (credential_id) | credential |
| `node_ref` | `str` | node-select |
| `file_ref` | `str` | file |
| `composite` | bir nechta data-key (schema_field=False) | action-recipients, action-keyboard |
| `ui_only` | — (data yo'q) | description, widget |

## Metadata (schema'ga ta'sir qilmaydi, lekin saqlanadi)

`label`/`placeholder`/`helpText` → Pydantic `Field(description=...)`.
`language` (code-editor), `mediaKind` (mediaMessage), `credentialType`
(credential), `resource`/`dependsOn` (dynamic_select) — kompilyator izoh/metadata
sifatida saqlaydi, lekin tip cheklovi emas.

## Normalizatsiya nuqtasi (Bosqich 4a)

State auto-save hozir IKKI shaklda: frontend flat (`save_state`+`state_key`+
`state_type`) ↔ agent nested (`state:{key,type}`). Engine `extractStateConfig()`
IKKALASINI ham o'qiydi. **Bosqich 4a: flat kanonik** — kompilyator flat chiqaradi,
saqlangan eski nested flow'lar engine'da ishlayveradi (migratsiya no-op).

## Escape hatch — validator addenda

Manifest ifoda eta olmaydigan cross-field qoidalar (masalan `CheckMembershipData`
ning `@model_validator`'i) `manifest_compiler/addenda.py::VALIDATOR_ADDENDA` da
saqlanadi va kompilyatsiya qilingan modelga ulanadi.
