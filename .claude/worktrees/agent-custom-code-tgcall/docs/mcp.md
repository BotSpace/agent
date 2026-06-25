# Botmother Agent — MCP server

agent ichidagi tool registry'ni **Model Context Protocol** (MCP) orqali tashqi
klientlarga (Claude Code, Cursor, boshqa MCP host'lar) ochadi. Shu orqali siz
o'z botingiz flow'ini, CMS kolleksiyalarini, env/release'larini to'g'ridan-to'g'ri
AI klientidan boshqarasiz.

- **Transport:** Streamable HTTP, stateless + JSON response (load-balancer
  do'st). Path: `/mcp`.
- **Auth:** `Authorization: Bearer <token>` — `bsp_` personal access token yoki
  oddiy JWT access token.

## 1. Token olish

Dashboard'da (`admin.botspace.uz`) profil sozlamalaridan **Personal Access
Token** yarating — `bsp_` bilan boshlanadi. Bu token MCP server tomonidan
backend `POST /auth/token/exchange/` orqali qisqa muddatli JWT'ga almashtiriladi
(va muddati tugashidan oldin avtomatik yangilanadi).

> Tokenni hech kimga bermang — u sizning loyihalaringizga to'liq kirish beradi.

## 2. Ulanish

### Claude Code

```bash
# Prod
claude mcp add --transport http botspace https://botspace.uz/mcp \
  --header "Authorization: Bearer bsp_xxxxxxxxxxxxxxxxxxxx"

# Lokal dev (agent 8100-portda ishlayotgan bo'lsa)
claude mcp add --transport http botspace-local http://localhost:8100/mcp \
  --header "Authorization: Bearer bsp_xxxxxxxxxxxxxxxxxxxx"
```

### Standalone server (ixtiyoriy)

API'dan alohida, faqat MCP server kerak bo'lsa:

```bash
python -m agent.mcp          # AGENT_MCP_PORT (default 8200), path /mcp
```

`AGENT_MCP_ENABLED=false` qilib `create_app()` ichidagi `/mcp` mount'ni
o'chirish mumkin.

## 3. Ish oqimi

MCP stateless bo'lgani uchun aktiv loyiha server tomonida sessiya holatida
saqlanadi. Flow tool'laridan oldin **albatta** loyihani tanlang:

1. `session_use_project` — loyihani tanlaydi, saqlangan flow va CMS
   kolleksiyalarini sessiyaga yuklaydi.
2. `flow_*` — flow'ni o'zgartirasiz (node qo'shish, ulash, validatsiya, ...).
3. `flow_save` — backend'ga saqlaysiz (**DESTRUCTIVE** — saqlangan flow'ni
   qayta yozadi).
4. `session_status` — istalgan paytda aktiv loyiha + saqlanmagan o'zgarishlarni
   ko'rsatadi.

Aktiv loyiha tanlanmagan holda `flow_*` chaqirilsa, server tushunarli xato
qaytaradi: *"avval session_use_project bilan loyiha tanlang"*.

## 4. Tool ro'yxati (namespace bo'yicha)

MCP nomlari registry nomidagi nuqtani pastki chiziqqa almashtiradi
(`flow.add_node` → `flow_add_node`). Agent-ichki namespace'lar — `plan.*`,
`task.*`, `ask_user` — MCP'da **ko'rsatilmaydi**.

| Namespace | Maqsad | Misol tool'lar |
|-----------|--------|----------------|
| `session_*` | MCP sessiya boshqaruvi (MCP-only) | `session_use_project`, `session_status` |
| `flow_*` | Flow CRUD, validatsiya, layout, saqlash | `flow_get`, `flow_add_node`, `flow_connect`, `flow_validate`, `flow_save` |
| `project_*` | Loyiha CRUD | `project_list`, `project_get`, `project_create` |
| `env_*` | Environment o'zgaruvchilari | `env_list`, `env_set`, `env_delete` |
| `release_*` | Deploy / debug bot | `release_list`, `release_create`, `release_logs` |
| `file_*` | Loyiha fayllari | `file_list`, `file_delete` |
| `cms_*` | CMS kolleksiya + hujjat ops | `cms_list_collections`, `cms_insert`, `cms_query`, `cms_aggregate` |
| `broadcast_*` | Broadcast xabarlar | `broadcast_list`, `broadcast_send` |
| `chat_*` | CRM chat foydalanuvchilari + tarix | `chat_users`, `chat_history` |
| `search_*` | Botmother dokumentatsiya/template qidiruvi | `search_docs`, `search_templates` |

To'liq ro'yxat va input sxemalar uchun klient `tools/list` chaqiradi yoki
agent API'sining `GET /tools` endpoint'iga qarang.

### Destructive tool'lar

`flow_save`, `*_delete`, `flow_clear` kabi qaytarib bo'lmaydigan amallar MCP
`destructiveHint` annotatsiyasi bilan belgilangan — klient ularni
foydalanuvchidan tasdiq so'rab bajaradi. Read-only tool'lar (`*_get`, `*_list`,
`flow_diff`, `flow_validate`, `*_search`) `readOnlyHint` bilan belgilangan.

## 5. Arxitektura eslatmalari

- **Stateless transport, server-side workspace:** flow tahriri `ctx.state`
  asosida ishlagani uchun, har `(user_id, project_id)` uchun in-memory workspace
  (`working_flow`/`baseline_flow`/`flow_dirty`/`auth_claims`/`collections`)
  saqlanadi, 2 soat TTL bilan.
- **Token rotation:** `auth_claims` dict'i reference bo'yicha saqlanadi —
  BackendClient mid-call access/refresh tokenni in-place yangilaganda, keyingi
  tool call'lar avtomatik yangi token bilan ishlaydi.
- **Auth:** Starlette middleware `Authorization` header'ini tekshiradi, AuthedUser'ni
  `contextvar` orqali tool handler'ga yetkazadi; auth yiqilsa MCP so'rovi
  backend'ga bormaydi — to'g'ridan-to'g'ri `401`.
