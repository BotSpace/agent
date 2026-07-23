import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# Ensure tests don't try to hit network
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-fake")
os.environ.setdefault("OPENAI_API_KEY", "test-key-fake")
os.environ.setdefault("AGENT_DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")

# Production has one manifest source: the Go backend endpoint. Unit tests use a
# tiny HTTP fixture serving that backend-owned data instead of reintroducing an
# agent-local manifest fallback.
_manifest_dir = (
    Path(__file__).resolve().parents[2] / "backend-golang" / "internal" / "modules" / "plugin" / "data"
)
_manifest_payload = json.dumps(
    {
        "manifests": json.loads((_manifest_dir / "builtin_node_manifests.json").read_text()),
        "categories": json.loads((_manifest_dir / "node_categories.json").read_text()),
    }
).encode()


class _ManifestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/api/v1/shared/manifest/builtin-nodes":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_manifest_payload)))
        self.end_headers()
        self.wfile.write(_manifest_payload)

    def log_message(self, _format: str, *args: object) -> None:
        return


_manifest_server = ThreadingHTTPServer(("127.0.0.1", 0), _ManifestHandler)
threading.Thread(target=_manifest_server.serve_forever, daemon=True).start()
os.environ["BACKEND_BASE_URL"] = f"http://127.0.0.1:{_manifest_server.server_address[1]}"


def pytest_sessionfinish() -> None:
    _manifest_server.shutdown()


class _MemoryDraftStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict] = {}
        self.active: tuple[str, str] | None = None

    async def upsert(self, *, user_id, project_id, **values) -> None:
        self.rows[(user_id, project_id)] = dict(values)

    async def load(self, *, user_id, project_id):
        return self.rows.get((user_id, project_id))

    async def load_latest(self, *, user_id):
        if self.active and self.active[0] == user_id:
            row = self.rows.get(self.active)
            if row:
                return {"project_id": self.active[1], **row}
        return None

    async def touch_active(self, *, user_id, project_id) -> None:
        self.active = (user_id, project_id)

    async def clear(self, *, user_id, project_id) -> None:
        self.rows.pop((user_id, project_id), None)


class _MemoryVersionStore:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {}

    async def append(self, *, project_id, flow, node_count, edge_count, **_kwargs):
        rows = self.rows.setdefault(project_id, [])
        rows.append(
            {
                "version": len(rows) + 1,
                "flow": flow,
                "node_count": node_count,
                "edge_count": edge_count,
            }
        )
        return len(rows)

    async def list(self, *, project_id):
        return list(reversed(self.rows.get(project_id, [])))

    async def get(self, *, project_id, version):
        rows = self.rows.get(project_id, [])
        return rows[version - 1] if 0 < version <= len(rows) else None


class _MemoryFlowTestStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict] = {}

    async def save(self, *, project_id, name, steps, enabled=True, **_kwargs):
        self.rows[(project_id, name)] = {
            "name": name,
            "steps": steps,
            "enabled": enabled,
        }
        return True

    async def list(self, *, project_id, only_enabled=False):
        rows = [row for (pid, _), row in self.rows.items() if pid == project_id]
        return [row for row in rows if row["enabled"]] if only_enabled else rows

    async def get(self, *, project_id, name):
        return self.rows.get((project_id, name))

    async def delete(self, *, project_id, name):
        return self.rows.pop((project_id, name), None) is not None


@pytest.fixture(autouse=True)
def _durable_store_fakes():
    from agent.mcp.draft_store import set_draft_store
    from agent.mcp.flow_tests import set_flow_test_store
    from agent.mcp.flow_versions import set_flow_version_store

    set_draft_store(_MemoryDraftStore())
    set_flow_version_store(_MemoryVersionStore())
    set_flow_test_store(_MemoryFlowTestStore())
    yield
    set_draft_store(None)
    set_flow_version_store(None)
    set_flow_test_store(None)
