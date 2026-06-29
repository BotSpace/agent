# agent — REST API

Base URL: `http://localhost:8100` (dev), `https://agent.botmother.uz` (prod).

## Auth

Two modes:
1. **User JWT** — `Authorization: Bearer <jwt>`. JWT shared with dashboard
   (`SESSION_SECRET`). Must contain `user.id` or `sub`. Optional `access` claim
   forwards a user access token to backend tool calls.
2. **Internal token** — `X-Internal-Token: <AGENT_INTERNAL_API_TOKEN>`. For
   service-to-service calls (e.g. Django backend → agent).

## Endpoints

### `GET /health`
Liveness/readiness. → `{ "status": "ok", "service": "agent", "provider": "..." }`

### `GET /tools`
Tool introspection — JSON schema bilan.

### `POST /projects/{project_id}/runs`
New run boshlash.

Request body:
```json
{ "prompt": "...", "thread_id": "optional", "baseline_flow": { "nodes": [], "edges": [] } }
```
Response: `{ "request_id": "<hex>", "status": "queued" }`

### `GET /projects/{project_id}/runs/{request_id}`
Run holatini olish.

Response:
```json
{
  "request_id": "...",
  "status": "queued|running|waiting_for_approval|waiting_for_clarification|completed|failed",
  "final_response": "string | null",
  "error": "string | null",
  "pending_approval": { ... } | null,
  "pending_clarification": { ... } | null,
  "current_plan": [ { "id": "...", "title": "...", "status": "..." } ]
}
```

### `GET /projects/{project_id}/runs/{request_id}/events`
**SSE stream**. Replay (since_seq) + live events.

Event turlari:
- `run_started`
- `assistant_message` `{ text, usage }`
- `thinking_delta` `{ delta }`
- `tool_call` `{ id, name, arguments }`
- `tool_result` `{ tool_call_id, content, is_error, data }`
- `approval_required` `{ tool_call_id, tool_name, arguments, reason }`
- `approval_resolved` `{ tool_call_id, approved }`
- `clarification_required` `{ tool_call_id, question, options? }`
- `clarification_resolved` `{ tool_call_id, answer }`
- `run_completed` `{ final_response }`
- `run_failed` `{ error }`
- `close`

Query param `since_seq=<int>` — replay events from given sequence.

### `POST /projects/{project_id}/runs/{request_id}/clarify`
Foydalanuvchi javobini yuborish (clarification interrupt javobi).

Body: `{ "answer": "..." }` → `{ "status": "resuming" }`

### `POST /projects/{project_id}/runs/{request_id}/approve`
Destructive operatsiyani tasdiqlash/rad etish.

Body: `{ "approved": true, "reason": "optional" }` → `{ "status": "resumed", "approved": "true" }`

## Tool tree (default registry)

```
plan.set        plan.update        plan.complete
flow.get        flow.list_node_types
flow.upsert_node   flow.remove_node   flow.patch
flow.connect    flow.disconnect    flow.validate
flow.layout_auto   flow.diff       flow.save       (destructive)
project.list    project.get        project.create
project.update  project.delete    (destructive)
env.list        env.set            env.unset       (destructive)
release.list    release.deploy    (destructive)    release.status
file.list       file.delete       (destructive)
crm.list_collections   crm.find   crm.count
crm.aggregate   crm.insert         crm.update
crm.delete     (destructive)       crm.segment
broadcast.preview   broadcast.send (destructive)   broadcast.status
chat.list_users     chat.user_history
search.semantic
task.run
```

## Sample call

```bash
# 1) Create run
curl -X POST $AGENT/projects/42/runs \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  -d '{"prompt":"Add a /start handler that greets the user by first_name"}'
# → { "request_id": "abc...", "status": "queued" }

# 2) Stream events
curl -N $AGENT/projects/42/runs/abc.../events \
  -H "Authorization: Bearer $JWT"

# 3) If approval_required appears:
curl -X POST $AGENT/projects/42/runs/abc.../approve \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  -d '{"approved": true}'
```
