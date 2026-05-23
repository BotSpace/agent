"""CLI — local dev va smoke testing uchun.

Asosiy buyruqlar:
    agent serve            — uvicorn dev server
    agent tools            — registry'dagi tool'larni ko'rsatish
    agent chat "prompt"    — bir martalik prompt (local LLM call, store yo'q)
"""

from __future__ import annotations

import asyncio
import json

import typer
import uvicorn

from .config import get_settings
from .graph import GraphConfig, build_graph, initial_state
from .llm import get_client
from .tools import build_default_registry


app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def serve(
    host: str = typer.Option(None, help="Override AGENT_HOST"),
    port: int = typer.Option(None, help="Override AGENT_PORT"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    """Run FastAPI dev server."""
    s = get_settings()
    uvicorn.run(
        "agent.main:app",
        host=host or s.host,
        port=port or s.port,
        reload=reload,
        log_level=s.log_level.lower(),
    )


@app.command()
def tools() -> None:
    """Print available tools as JSON."""
    reg = build_default_registry()
    payload = [
        {"name": t.name, "destructive": t.destructive, "description": t.description}
        for t in reg.all()
    ]
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User prompt"),
    project_id: str | None = typer.Option(None),
) -> None:
    """One-shot agent run (no persistence, console output only)."""

    async def _run() -> None:
        s = get_settings()
        cfg = GraphConfig(
            llm=get_client(),
            registry=build_default_registry(),
            model=s.model_executor,
        )
        graph = build_graph(cfg)
        state = initial_state(
            run_id="cli",
            project_id=project_id,
            user_id="cli",
            thread_id="cli",
            user_prompt=prompt,
        )
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "cli"}})
        messages = result.get("messages") or []
        for m in messages:
            if getattr(m, "role", None) == "assistant":
                typer.echo(f"\n--- assistant ---\n{m.text}")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
