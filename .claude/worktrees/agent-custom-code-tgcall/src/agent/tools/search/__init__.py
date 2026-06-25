"""search.* tools — semantic recall over the agent's long-term memory."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...memory import get_retriever
from ..base import Tool, ToolContext, ToolResult, tool


class SearchSemanticInput(BaseModel):
    query: str = Field(description="What to recall — e.g. 'previous CRM segments', 'last error'")
    kind: str | None = Field(default=None, description="Optional filter: 'flow' | 'message' | 'doc' | 'project_summary'")
    top_k: int = Field(default=6, ge=1, le=20)


@tool(
    name="search.semantic",
    description=(
        "Recall relevant items from long-term project memory using semantic search. "
        "Useful for: prior flow examples, past CRM segments, project decisions, doc snippets."
    ),
    input_model=SearchSemanticInput,
)
async def search_semantic(input: SearchSemanticInput, ctx: ToolContext) -> ToolResult:
    pid = ctx.state.get("project_id")
    if pid is None:
        return ToolResult.error("No project_id in context.")
    retriever = ctx.services.get("retriever") or get_retriever()
    hits = await retriever.recall(
        project_id=str(pid),
        query=input.query,
        kind=input.kind,
        top_k=input.top_k,
    )
    if not hits:
        return ToolResult.ok("No relevant memories found.", data={"hits": []})
    lines = [f"[{h['similarity']:.2f}] ({h['kind']}) {h['content'][:200]}" for h in hits]
    return ToolResult.ok("\n".join(lines), data={"hits": hits})


def build_search_tools() -> list[Tool]:
    return [search_semantic]
