"""LLM provider'lar orasidagi umumiy yordamchilar.

ponytail: tool-name sanitize/restore openai va bedrock'da bir xil edi.
Ba'zi provayderlar tool nomida nuqta qabul qilmaydi → '.' ↔ '--' almashtiramiz.
"""

from __future__ import annotations


def sanitize_tool_name(name: str) -> str:
    return name.replace(".", "--")


def restore_tool_name(name: str) -> str:
    return name.replace("--", ".")
