"""ask_user — ask the END USER for INFORMATION mid-run (not for permission).

Bu tool foydalanuvchidan **ma'lumot** (qiymat, tanlov) so'rash uchun. U
APPROVAL (ruxsat) emas — NON-approval `clarify` interrupt yaratadi.

Mexanizm:
  1. Tool `ToolResult.interrupt={"kind":"clarify", ...}` qaytaradi.
  2. graph (`tool_exec`) buni `pending_clarification` ga ko'chiradi (+ tool_call_id).
  3. `clarify_interrupt` node `interrupt(...)` chaqiradi → graph to'xtaydi.
  4. worker `clarification_required` event chiqaradi (UI widget render qiladi:
     text input / single-select / multi-select — `input_type` + `options` orqali).
  5. Foydalanuvchi javob bergach worker `Command(resume=answer)` bilan davom etadi;
     `clarify_interrupt` javobni tool_result sifatida agentga qaytaradi.

Cross-stream contract — `clarification_required` payload SHAKLI (dashboard shu
bo'yicha widget tanlaydi):
  {
    "kind": "clarify",
    "question": str,
    "input_type": "text" | "select" | "multiselect",
    "options": list[str] | None,
    "placeholder": str | None,
    "tool_call_id": str,   # tool_exec qo'shadi
  }
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .base import ToolContext, ToolResult, tool


class AskUserInput(BaseModel):
    question: str = Field(
        description="Foydalanuvchiga beriladigan aniq savol (1 jumla)."
    )
    input_type: Literal["text", "select", "multiselect"] = Field(
        default="text",
        description=(
            "text — erkin matn kiritish; select — bitta variant tanlash; "
            "multiselect — bir nechta variant tanlash. select/multiselect uchun "
            "`options` MAJBURIY."
        ),
    )
    options: list[str] | None = Field(
        default=None,
        description="select/multiselect uchun tanlanadigan variantlar ro'yxati.",
    )
    placeholder: str | None = Field(
        default=None,
        description="text input uchun ko'rsatma matni (ixtiyoriy).",
    )

    @model_validator(mode="after")
    def _require_options_for_choice(self) -> AskUserInput:
        if self.input_type in ("select", "multiselect") and not self.options:
            raise ValueError(
                f"input_type={self.input_type!r} uchun `options` ro'yxati majburiy."
            )
        return self


@tool(
    name="ask_user",
    description=(
        "Ask the END USER for information you need to continue (a value or a "
        "choice) — NOT for permission. Pauses the run until the user answers, "
        "then the answer is returned to you as the tool result. Use input_type="
        "'text' for free input, 'select' for a single choice, 'multiselect' for "
        "multiple choices (options REQUIRED for select/multiselect). Prefer this "
        "over guessing a value the user has not given."
    ),
    input_model=AskUserInput,
)
async def ask_user(input: AskUserInput, ctx: ToolContext) -> ToolResult:
    interrupt = {
        "kind": "clarify",
        "question": input.question,
        "input_type": input.input_type,
        "options": input.options,
        "placeholder": input.placeholder,
    }
    # content — interrupt sodir bo'lmasa (degrade) ham agentga foydali matn.
    # interrupt ishlaganda clarify_interrupt javobni tool_result bilan almashtiradi.
    return ToolResult.ok(
        f"Awaiting user answer to: {input.question}",
        interrupt=interrupt,
        data={"input_type": input.input_type, "options": input.options},
    )


def build_ask_user_tools() -> list:
    return [ask_user]
