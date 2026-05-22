"""Tool registry — namespaced collection, JSON-schema export, execution."""

from __future__ import annotations

from collections.abc import Iterable

from ..llm.base import ToolSpec
from .base import Tool, ToolContext, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool `{tool.name}` already registered")
        self._tools[tool.name] = tool

    def extend(self, tools: Iterable[Tool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def by_namespace(self, namespace: str) -> list[Tool]:
        prefix = f"{namespace}."
        return [t for t in self._tools.values() if t.name.startswith(prefix)]

    def specs(self) -> list[ToolSpec]:
        return [t.to_spec() for t in self._tools.values()]

    def filter_specs(self, allowed: set[str] | None) -> list[ToolSpec]:
        if allowed is None:
            return self.specs()
        return [t.to_spec() for t in self._tools.values() if t.name in allowed]

    async def execute(
        self,
        name: str,
        args: dict,
        ctx: ToolContext,
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.error(f"Unknown tool `{name}`")
        return await tool.run(args, ctx)
