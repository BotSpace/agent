"""Flow blueprint — strong-model rejasidan zaif-model qo'llanish uchun.

`flow.plan` strong model (planner) ga bitta murakkab vazifa beradi va
to'liq FlowBlueprint qaytaradi: nodes va edges ro'yxati. Keyin
`flow.bulk_apply` shu blueprint'ni bir chaqiruvda qo'llaydi.

Bu sxema 200+ turn'lik flow build'ni 3-5 turn'ga tushiradi.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    id: str = Field(description="Stable kebab-case id.")
    type: str = Field(description="Engine node type (CommandTriggerNode, SendTextMessageNode, ...).")
    data: dict[str, Any] = Field(description="Type-specific config.")


class EdgeSpec(BaseModel):
    source: str
    target: str
    source_handle: str | None = Field(default="source-handler")
    target_handle: str | None = Field(default="target-handler")


class FlowBlueprint(BaseModel):
    nodes: list[NodeSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    summary: str | None = Field(default=None, description="One-paragraph human description.")


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_blueprint(text: str) -> FlowBlueprint:
    """LLM matnidan FlowBlueprint chiqarib olish.

    Modelning javobi ```json ... ``` kod-blok'ida bo'lishi mumkin yoki
    butun matn bitta JSON ob'ekt bo'lishi mumkin. Ikkala holatda ham
    raw matni parse qilamiz.
    """
    if not text or not text.strip():
        raise ValueError("Blueprint matni bo'sh.")

    raw = text.strip()
    # JSON kod blok'ini olish
    m = _JSON_FENCE.search(raw)
    if m:
        raw = m.group(1)
    else:
        # Birinchi { dan oxirgi } gacha
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse xato: {e}\n--- raw ---\n{text[:500]}") from e

    return FlowBlueprint.model_validate(payload)
