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
    data_in_content: bool = False
    """True bo'lsa — `data` allaqachon `content`ga (o'qishli ko'rinishda)
    renderlangan; MCP qatlami YAML(data)ni QAYTA qo'shmaydi (token takrori
    oldini oladi). Data'ni content'ga o'zi chiqaradigan tool'lar uchun
    (list_node_types, flow.get, cms.find/get_document, ...)."""

    @classmethod
    def ok(cls, content: str, **kwargs: Any) -> ToolResult:
        return cls(content=content, is_error=False, **kwargs)

    @classmethod
    def error(cls, content: str, **kwargs: Any) -> ToolResult:
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
    Qaytarib bo'lmaydigan, butun ishni yo'q qiladigan amallar uchun."""
    untrusted_output: bool = False
    """True bo'lsa — tool natijasi ishonchsiz (oxirgi foydalanuvchi kiritgan)
    kontentni o'z ichiga oladi: CMS hujjatlari, chat tarixi, foydalanuvchi
    ismlari. Graph bunday natijani delimiter bilan o'rab, modelga uning ichidagi
    "ko'rsatmalar"ni bajarmaslikni aytadi (prompt-injection himoyasi)."""
    tags: list[str] = field(default_factory=list)
    expose_to_mcp: bool = True
    """False bo'lsa — bu tool MCP klientlariga (dashboard konstruktori) ko'rsatilmaydi.
    Agent-ichki tool'lar uchun (masalan `sleep` — kutish, konstruktorga keraksiz)."""
    expose_to_agent: bool = True
    """False bo'lsa — bu tool AI agent registrida bo'lmaydi (faqat MCP-only tool).
    `expose_to_mcp`ning teskarisi."""
    core: bool = False
    """True bo'lsa — bu tool DOIM to'liq schema bilan yuboriladi (asosiy, ko'p
    ishlatiladigan). False (default) — LAZY: prompt'da faqat summary ko'rinadi,
    to'liq schema `tools.search` bilan yuklanadi (token tejash)."""

    @property
    def summary(self) -> str:
        """Bir qatorli qisqa tavsif — description'ning birinchi jumlasi."""
        d = " ".join(self.description.split())
        for sep in (". ", " — ", "; "):
            if sep in d:
                d = d.split(sep, 1)[0]
                break
        return d[:140]

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
            hint = _format_hint_for_error(e)
            msg = f"Invalid arguments for `{self.name}`:\n{e}"
            if hint:
                msg += "\n" + hint
            return ToolResult.error(msg)
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
    expose_to_mcp: bool = True,
    expose_to_agent: bool = True,
    core: bool = False,
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
            expose_to_mcp=expose_to_mcp,
            expose_to_agent=expose_to_agent,
            core=core,
        )

    return decorator


# Field-specific format hints, keyed by the leaf field name that appears in a
# Pydantic error `loc`. When validation fails on one of these fields we append a
# concrete, copy-pasteable example so the model can self-correct in one turn
# instead of guessing the shape (e.g. searchFields wants dicts, not a string
# array — issue #5).
_FIELD_FORMAT_HINTS: dict[str, str] = {
    "searchFields": (
        "HINT: `searchFields` is an ARRAY OF OBJECTS, each "
        "{\"field\": \"<name>\", \"description\": \"<optional hint>\"} — `field` "
        "is required. Example: "
        "[{\"field\": \"chat_id\", \"description\": \"user id\"}, {\"field\": \"kind\"}]. "
        "A plain string array like [\"chat_id\", \"kind\"] or keys `name`/`type` "
        "are NOT accepted."
    ),
    "jsonSchema": (
        "HINT: `jsonSchema` is the JSON Schema serialized as a STRING (not an "
        "object). Example: "
        "'{\"type\":\"object\",\"properties\":{\"intent\":{\"type\":\"string\"}},"
        "\"required\":[\"intent\"]}'. Only used when structuredOutput=true."
    ),
}


def _format_hint_for_error(err: ValidationError) -> str:
    """Return a concrete format hint for known fields touched by `err`, or ''."""
    seen: set[str] = set()
    hints: list[str] = []
    for e in err.errors():
        for part in e.get("loc", ()):  # type: ignore[union-attr]
            if isinstance(part, str) and part in _FIELD_FORMAT_HINTS and part not in seen:
                seen.add(part)
                hints.append(_FIELD_FORMAT_HINTS[part])
    return "\n".join(hints)


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

    if not isinstance(raw_args, dict):
        return raw_args

    fields = getattr(model_cls, "model_fields", {})
    if not fields:
        return raw_args

    out = dict(raw_args)
    for key, _info in fields.items():
        value = out.get(key)
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped:
            continue
        # Faqat JSON object/list ko'rinishidagi string'larni unwrap qilamiz
        if not (stripped.startswith("{") or stripped.startswith("[")):
            continue
        parsed = _lenient_json_loads(stripped)
        if parsed is None:
            continue
        out[key] = parsed
    return out


def _lenient_json_loads(text: str) -> Any | None:
    """Parse a JSON string, repairing common LLM mistakes. None if unparseable.

    Notably: models (especially via OpenRouter) escape apostrophes as ``\\'``
    inside JSON strings — INVALID JSON (only ``\\"`` / ``\\\\`` / ``\\n`` … are
    allowed). O'zbek matni apostrofga to'la (to'lov, ko'ring), so a stringified
    `data` payload with an Uzbek systemPrompt fails strict json.loads and the
    field stays a string → "Input should be a valid dictionary". We first try
    strict parsing (valid JSON is never altered), then retry after replacing the
    invalid ``\\'`` escape with a bare apostrophe.
    """
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Repair: backslash-apostrophe → apostrophe (invalid JSON escape).
    repaired = text.replace("\\'", "'")
    if repaired != text:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None
