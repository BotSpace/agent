"""Prompt matnlarini `texts/` papkasidan o'qiydigan umumiy loader.

Barcha prompt matnlari Python kodidan ajratilib, `texts/` ostidagi `.txt`
fayllarda saqlanadi — shunda promptni tahrirlash uchun kodga tegish shart emas,
har bir prompt aniq fayl sifatida topiladi.

Tuzilma:
  texts/
    dispatcher/core_identity.txt      — main agent (dispatcher) personasi
    shared/autonomy.txt               — barcha agentlar uchun autonomy qoidasi
    shared/narrate_rule.txt           — sub-agent narrate/batch qoidalari
    subagents/flow_builder.txt        — flow-builder system prompt
    subagents/flow_builder_examples.txt — to'liq bot misollari
    subagents/crm_manager.txt
    subagents/cms_manager.txt
    subagents/deploy_manager.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TEXTS_DIR = Path(__file__).parent / "texts"


def load_prompt(relpath: str) -> str:
    """texts/<relpath> faylini o'qib, strip qilib qaytaradi.

    Fayl topilmasa yoki o'qishda xato bo'lsa — bo'sh string qaytaradi
    (hech qachon exception tashlamaydi), lekin warning log qiladi.
    """
    path = _TEXTS_DIR / relpath
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        logger.warning("Prompt fayli o'qilmadi: %s", path, exc_info=True)
        return ""
