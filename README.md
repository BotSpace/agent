# agent — Botmother AI Agent

Claude Code darajasidagi autonomous AI agent. Foydalanuvchi bilan tabiiy tilda
gaplashib, **visual bot flow** yaratadi, **botlarni deploy qiladi**, **CRM**
(MongoDB collectionlar) bilan ishlaydi, **broadcast** yuboradi va **boshqa botlarni**
boshqaradi.

## Stack

- **Python 3.12** + **FastAPI**
- **LangGraph** 0.3+ — orchestrator graph (plan/act/observe/reflect loop)
- **LangChain Core** — faqat tool/message abstractions
- **Direct SDKs**: Anthropic, OpenAI, AWS Bedrock — provider-agnostik
- **Postgres + pgvector** — runs, messages, checkpoints, semantic memory
- **Redis** — SSE pubsub, rate limit, cache
- **MongoDB (motor)** — CRM tool'lari uchun project-scoped DB
- **Pydantic v2** — schemas, JSON schemas
- **PyJWT** — dashboard bilan shared session

## Loyihaning fayl tuzilishi

```
src/agent/
├── config.py             # pydantic-settings
├── logging.py            # structlog
├── main.py               # ASGI entry (uvicorn)
├── cli.py                # `agent serve|tools|chat`
│
├── llm/                  # provider abstraction
│   ├── base.py           # Message, ToolSpec, GenerateOptions, LLMClient protocol
│   ├── anthropic.py
│   ├── openai.py
│   ├── bedrock.py
│   ├── router.py         # build_client, ModelRoles
│   └── budgets.py        # per-run guardrails
│
├── flow/                 # flow domain (engine-golang bilan to'g'ridan-to'g'ri mos)
│   ├── schema.py
│   ├── catalog.py        # 47+ node type
│   ├── validator.py
│   ├── layout.py
│   └── diff.py
│
├── tools/                # namespaced tool registry
│   ├── base.py
│   ├── registry.py
│   ├── plan/             # plan.set, plan.update, plan.complete
│   ├── flow/             # flow.get/upsert_node/connect/validate/save/...
│   ├── project/          # (Phase 4)
│   ├── env/              # (Phase 4)
│   ├── release/          # (Phase 4)
│   ├── crm/              # (Phase 5)
│   ├── broadcast/        # (Phase 5)
│   ├── chat/             # (Phase 5)
│   ├── file/             # (Phase 4)
│   ├── search/           # (Phase 7)
│   ├── docs/             # (Phase 5)
│   └── task/             # (Phase 6) sub-agent delegation
│
├── graph/                # LangGraph orchestrator
│   ├── state.py          # AgentState TypedDict
│   └── builder.py        # build_graph()
│
├── prompts/
│   └── system.py         # core identity + dynamic context
│
├── runtime/              # (Phase 3-4) store, worker, events
│   └── events.py
│
├── api/                  # FastAPI
│   └── app.py            # /health, /tools, (future) /runs, /events
│
├── integrations/         # (Phase 4-5) external clients
└── memory/               # (Phase 7) checkpoint + pgvector RAG
```

## Asosiy printsiplar

1. **Plan-then-act** — har murakkab task uchun `plan.set` (TodoWrite ekvivalenti).
2. **Tool-first** — agent JSON ko'rsatma o'rniga `flow.upsert_node`, `flow.connect`
   kabi tool'larni chaqiradi.
3. **Working flow → Save** — agent `working_flow`'da ishlaydi, `flow.save` orqali
   backend'ga yozadi (approval interrupt bilan).
4. **Provider-agnostic** — bitta `LLMClient` protokol, har provider o'z client'i.
5. **Approval gating** — destructive operatsiyalar (save, deploy, delete,
   broadcast) — har doim foydalanuvchi tasdig'idan keyin.
6. **Budget guardrails** — token, tool call, wall-clock per-run limit.
7. **SSE events** — `tool_call`, `tool_result`, `assistant_message`,
   `approval_required`, `clarification_required`, `plan_set/update`, h.k.

## Lokal ishga tushirish

```bash
# 1) Dependencies
cd agent
pip install -e ".[dev]"

# 2) Env
cp .env.example .env
# .env ichidan ANTHROPIC_API_KEY (yoki OPENAI_API_KEY) ni belgilang

# 3) Postgres + Redis (Docker)
docker compose up -d postgres redis

# 4) Dev server
agent serve --reload
# yoki: uvicorn agent.main:app --reload --port 8100
```

`http://localhost:8100/health` → `{"status":"ok"}`
`http://localhost:8100/tools` → registered tools list (JSON schema bilan)

### CLI smoke test

```bash
agent tools                              # tool ro'yxati
agent chat "Build a /start flow that greets the user"
```

## Implementation phases

| Phase | Status |
|---|---|
| 0 — Scaffold | ✅ |
| 1 — LLM router (Claude/OpenAI/Bedrock) | ✅ |
| 2 — Tool framework + flow + plan tools | ✅ |
| 3 — Graph orchestrator + SSE + approval | ✅ |
| 4 — Backend integration (project/env/release/file) | ✅ |
| 5 — CRM + Broadcast + Chat history | ✅ |
| 6 — Sub-agent (task.run) | ✅ |
| 7 — Memory (Postgres + pgvector RAG) | ✅ |
| 8 — Dashboard chat UI | ✅ (dashboard repo) |
| 9 — Observability + tests + k8s + docs | ✅ |

## REST API

To'liq API hujjat: [docs/api.md](docs/api.md). Asosiy endpoints:

```
POST   /projects/{pid}/runs              # new run
GET    /projects/{pid}/runs/{rid}        # status
GET    /projects/{pid}/runs/{rid}/events # SSE stream
POST   /projects/{pid}/runs/{rid}/clarify
POST   /projects/{pid}/runs/{rid}/approve
GET    /tools                            # introspection
GET    /health
```

## Tool katalogi

35+ tool — `plan.*`, `flow.*`, `project.*`, `env.*`, `release.*`, `file.*`,
`crm.*`, `broadcast.*`, `chat.*`, `search.semantic`, `task.run`.
Har destructive operatsiya foydalanuvchi tasdig'ini so'raydi.

## Sub-agentlar

`task.run(subagent_type=..., prompt=...)` orqali quyidagilar:
- **flow-builder** — visual flow yaratish
- **crm-analyst** — MongoDB tahlili va segmentatsiya
- **bot-deployer** — project/env/release lifecycle
- **broadcaster** — targetlangan xabar yuborish

Har sub-agent o'z `allowed_tools` subset bilan ishlaydi, isolated context.

## Dashboard integratsiyasi

- `/uz/app/agent`, `/en/app/agent`, `/ru/app/agent` — sahifa
- `/api/agent/{projectId}/runs/{requestId}/events` — Next.js SSE proxy route
- Komponentlar: `ChatPanel`, `PlanPanel`, `ToolCallBubble`,
  `ApprovalDialog`, `ClarifyDialog`
- Server actions: `startRunAction`, `clarifyRunAction`, `approveRunAction`
- Yangi env: `NEXT_PUBLIC_AGENT_URL`, `AGENT_INTERNAL_API_TOKEN`

## Engine-golang bilan moslik

`flow/catalog.py` engine-golang'dagi 47+ node turini ro'yxatga oladi. Agent har bir
node'ni shu katalogdan tanlaydi, `data` maydonlarini to'ldiradi, `flow.validate`
runtime engine kutgan schema bilan moslashtirib tekshiradi.

## Backend bilan aloqalar

- **Django backend** (`/api/project/`, `/api/relase/`, `/api/enviroment/`,
  `/api/file/`, `/api/collection/`, `/api/message/`, `/api/chat/...`) — agent
  REST orqali chaqiradi (Phase 4-5).
- **MongoDB** — `motor` orqali project-scoped DB (`project_<id>`) — CRM tool'lari.
- **engine-golang** — agent to'g'ridan-to'g'ri murojaat qilmaydi; flow JSON faqat
  backend'ga yoziladi, deploy keyin engine'da bo'ladi.

## Dashboard integratsiyasi (Phase 8)

`/app/agent` sahifasi — Claude Code kabi chat UI:
- SSE stream (tool calls, plan, approvals)
- Sticky composer (multi-line input)
- Plan panel (TodoWrite real-time)
- Approval modal (destructive operatsiyalar uchun)
- Diff view (flow changes)

## Sub-agent (Phase 6)

`task.run(subagent_type, prompt)` — Claude Code'dagi Agent tool kabi:
- Sub-agent'lar registered specs (flow-builder, crm-analyst, bot-deployer)
- Parallel ishlatish (asyncio gather)
- Isolated context (parent state'ni o'qiy oladi, lekin yozish faqat o'z tool'i)
