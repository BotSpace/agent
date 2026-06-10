"""Tool framework — decorator, ToolContext, ToolResult, JSON-schema auto.

Har tool Pydantic input modeli orqali aniqlanadi:
    class FlowAddNodeInput(BaseModel):
        id: str
        type: str
        data: dict[str, Any] = {}

    @tool(
        name="flow.add_node",
        description="...",
        input_model=FlowAddNodeInput,
        destructive=False,
    )
    async def add_node(input: FlowAddNodeInput, ctx: ToolContext) -> ToolResult:
        ...
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.base import ToolSpec


InputModel = TypeVar("InputModel", bound=BaseModel)


@dataclass
class ToolContext:
    """Tool execution context — har chaqirilganda fresh."""

    run_id: str
    project_id: str | None
    user_id: str | None
    state: dict[str, Any] = field(default_factory=dict)
    emit: Callable[[str, dict[str, Any]], None] | None = None
    services: dict[str, Any] = field(default_factory=dict)
    """services: backend HTTP client, mongo client, etc. — DI bag."""


@dataclass
class ToolResult:
    """Tool natijasi — LLM ga (content) va graph state'ga (updates) tushuvchi."""

    content: str
    is_error: bool = False
    state_updates: dict[str, Any] = field(default_factory=dict)
    """Agent state'iga (working_flow, current_plan, ...) qo'shiladigan patch."""
    interrupt: dict[str, Any] | None = None
    """Approval/clarification bo'lsa — graph interrupt payload."""
    data: dict[str, Any] | None = None
    """Strukturlangan natija (UI uchun foydali)."""

    @classmethod
    def ok(cls, content: str, **kwargs: Any) -> "ToolResult":
        return cls(content=content, is_error=False, **kwargs)

    @classmethod
    def error(cls, content: str, **kwargs: Any) -> "ToolResult":
        return cls(content=content, is_error=True, **kwargs)


ToolFn = Callable[[Any, ToolContext], Awaitable[ToolResult]]


@dataclass
class Tool(Generic[InputModel]):
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolFn
    destructive: bool = False
    """Destructive bo'lsa, graph approval interrupt yaratadi (agar
    cfg.approve_destructive=True bo'lsa)."""
    always_approve: bool = False
    """True bo'lsa — cfg.approve_destructive holatidan QAT'I NAZAR har doim
    approval interrupt yaratadi (foydalanuvchi tasdiqlamasa bajarilmaydi).
    Qaytarib bo'lmaydigan, butun ishni yo'q qiladigan amallar uchun (flow.clear)."""
    untrusted_output: bool = False
    """True bo'lsa — tool natijasi ishonchsiz (oxirgi foydalanuvchi kiritgan)
    kontentni o'z ichiga oladi: CMS hujjatlari, chat tarixi, foydalanuvchi
    ismlari. Graph bunday natijani delimiter bilan o'rab, modelga uning ichidagi
    "ko'rsatmalar"ni bajarmaslikni aytadi (prompt-injection himoyasi)."""
    tags: list[str] = field(default_factory=list)

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=_pydantic_to_json_schema(self.input_model),
        )

    async def run(self, raw_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_args = _coerce_stringified_json(raw_args, self.input_model)
        try:
            args = self.input_model.model_validate(raw_args)
        except ValidationError as e:
            return ToolResult.error(f"Invalid arguments for `{self.name}`:\n{e}")
        return await self.handler(args, ctx)


def tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    destructive: bool = False,
    always_approve: bool = False,
    untrusted_output: bool = False,
    tags: list[str] | None = None,
) -> Callable[[ToolFn], Tool]:
    """Decorator — Tool obyekti qaytadi."""

    def decorator(fn: ToolFn) -> Tool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Tool `{name}` handler must be async")
        return Tool(
            name=name,
            description=description,
            input_model=input_model,
            handler=fn,
            destructive=destructive,
            always_approve=always_approve,
            untrusted_output=untrusted_output,
            tags=tags or [],
        )

    return decorator


def _pydantic_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic → JSON schema, lekin LLM uchun pollangan."""
    schema = model.model_json_schema(ref_template="#/definitions/{model}")
    # Anthropic/OpenAI ikkalasi `definitions` o'rniga `$defs` ham qabul qiladi
    schema.pop("title", None)
    return schema


def _coerce_stringified_json(
    raw_args: Any, model_cls: type[BaseModel]
) -> Any:
    """LLM ba'zan nested dict/list field'larni JSON string sifatida yuboradi.

    Qwen, Gemma va boshqa kuchsizroq modellar tipik xulqi:
        {"data": "{\\"command\\": \\"/start\\"}"}
    Pydantic dict kutadi, string keladi → rad etadi.

    Bu funksiya: input modeldagi dict/list/BaseModel-tipli field'larni topib,
    ularning qiymati `{...}` yoki `[...]` ko'rinishidagi string bo'lsa,
    JSON parse qiladi. Boshqa string'lar tegmaydi.
    """
    import json

    if not isinstance(raw_args, dict):
        return raw_args

    fields = getattr(model_cls, "model_fields", {})
    if not fields:
        return raw_args

    out = dict(raw_args)
    for key, info in fields.items():
        value = out.get(key)
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped:
            continue
        # Faqat JSON object/list ko'rinishidagi string'larni unwrap qilamiz
        if not (stripped.startswith("{") or stripped.startswith("[")):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        out[key] = parsed
    return out
