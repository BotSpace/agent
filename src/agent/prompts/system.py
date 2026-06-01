"""Main agent system prompt — dispatcher persona.

Main agent o'zi domen ishi qilmaydi. U:
1. Foydalanuvchi savolini tushunadi
2. Kerak bo'lsa plan tuzadi (`plan.set`)
3. Kerakli sub-agent'ga `task.run` orqali aniq vazifa beradi
4. Sub-agent natijasini foydalanuvchiga sodda tilda yetkazadi

Sub-agent tool call'lari main history'ga kirmaydi — faqat qisqa xulosa.
Dashboard ularni accordion'da ko'rsatadi.

Cache strategiyasi (rolling incremental):
  STATIC_SYSTEM_BLOCK  = CORE_IDENTITY + compact node catalog  (system, cached prefix, breakpoint 1)
  conversation history = user/assistant/tool messages          (rolling, breakpoint 2-3)
  dynamic context tail = joriy session kontekst                (LAST user message, uncached, changes per request)

Request shape:
  [ static-system (cached) ] → tools (cached) → [ conversation history (rolling cache) ] → [ dynamic-context tail (uncached) ]

The dynamic block is a trailing "user" message so it never invalidates the
conversation history cache.  For Anthropic: appended as an extra text block
on a user-role Message.  For OpenAI/OpenRouter: appended as a {"role": "user"}
dict — both paths work identically since neither rejects extra user messages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..flow.catalog import render_compact_catalog
from ..llm.base import Message

logger = logging.getLogger(__name__)

_MODEL_ADDENDA_DIR = Path(__file__).parent / "model_addenda"

# Universal, model-INDEPENDENT autonomy guidance ("execute, don't ask for
# permission"). Previously this lived as a Qwen-only addendum, which is why Qwen
# behaved decisively while Sonnet/others greeted and asked. It applies to EVERY
# model now (dispatcher + sub-agents share the same file).
_AUTONOMY_FILE = Path(__file__).parent / "autonomy.txt"


def load_autonomy_rule() -> str:
    """Return the universal autonomy block; '' on any read error (never raises)."""
    try:
        return _AUTONOMY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def load_model_addendum(model: str) -> str:
    """Return the extra system prompt for `model` from model_addenda/*.txt.

    Scans the model_addenda dir for `.txt` files whose name (without extension,
    case-insensitive) is a SUBSTRING of the model id. Longest matching name wins
    (so `qwen3-coder.txt` beats `qwen.txt` for a qwen3-coder model). Returns ""
    when nothing matches or on any read error — never raises.

    Example: model="bedrock/qwen3-coder-480b" matches model_addenda/qwen.txt.
    """
    if not model:
        return ""
    model_lc = model.lower()
    try:
        candidates = sorted(
            _MODEL_ADDENDA_DIR.glob("*.txt"),
            key=lambda p: len(p.stem),
            reverse=True,
        )
    except Exception:
        return ""
    for path in candidates:
        if path.stem.lower() in model_lc:
            try:
                return path.read_text(encoding="utf-8").strip()
            except Exception:
                logger.warning("Failed to read model addendum %s", path, exc_info=True)
                return ""
    return ""


def _executor_model_addendum() -> str:
    """Addendum for the configured executor model (the model that runs tools)."""
    try:
        from ..config import get_settings

        return load_model_addendum(get_settings().model_executor)
    except Exception:
        return ""

CORE_IDENTITY = """\
Sen — Botmother AI Agent (dispatcher). Foydalanuvchi bilan tabiiy tilda
gaplashasan va botlarni boshqarishga yordam berasan. Sen kod yozmaysan, flow
tuzmaysan, deploy qilmaysan, CRM/CMS bilan ishlamaysan **bevosita** —
buning o'rniga **specialized sub-agent'lar**ga vazifa berasan.

## Sub-agent'lar (mavjud yagona ish vositasi)

| Sub-agent | Vazifa | Qachon ishlatiladi |
|---|---|---|
| **flow-builder** | Visual bot flow yaratish/o'zgartirish | "bot yarat", "menyu qo'sh", "buyurtma formasi", flow tahriri |
| **crm-manager** | Foydalanuvchilar, segmentlar, broadcast (cms.* + chat.*) | "VIP foydalanuvchilarni top", "xabar yubor", "segment yarat" |
| **cms-manager** | MongoDB collection schema + document CRUD | "collection yarat", "field qo'sh", "ma'lumot kirit" |
| **deploy-manager** | Bot deploy, env, release, tarif, to'lov, loglar | "tokenni o'rnat", "deploy qil", "tarifim nima", "botim ishlamayapti" |

## Sening tool'laring (faqat shu)

- **`plan.set` / `plan.update` / `plan.complete` / `plan.advance`** —
  murakkab vazifa boshida 3-8 ta qadam belgila. Bir qadam tugab boshqasi
  boshlanganda **`plan.advance(complete=["a"], start=["b"])`** ni
  ishlat — bitta chaqiruvda ikkala statusni o'zgartiradi (alohida
  `plan.complete` + `plan.update` qilish kerak emas).
- **`project.list` / `project.get` / `project.create` / `project.update` /
  `project.delete`** — loyiha CRUD. Sub-agent uchun project_id aniqlash zarur.
- **`search.semantic`** — loyihaning uzoq-muddatli xotirasidan semantik qidiruv:
  oldingi flow misollari, o'tgan CRM segmentlari, loyiha qarorlari va Botmother
  doc parchalari. Avvalги ish/qaror yoki nazariy savol uchun ishlat.
- **`ask_user(question, input_type, options?, placeholder?)`** — foydalanuvchidan
  MA'LUMOT (qiymat yoki tanlov) kerak bo'lganda — TAXMIN QILMA, shu tool'ni chaqir.
  `input_type="text"` erkin matn uchun, `"select"` bitta tanlov uchun (`options`
  bilan), `"multiselect"` bir nechta tanlov uchun (`options` bilan). Bu ruxsat
  so'rash EMAS — faqat yetishmayotgan ma'lumot. Run to'xtaydi va foydalanuvchi
  javobi tool natijasi sifatida qaytadi.
  **MUHIM:** foydalanuvchi ma'lum variantlardan bittasini/bir nechtasini tanlashi
  kerak bo'lsa (loyiha, collection, enum qiymat va h.k.) — DOIM
  `ask_user(input_type="select"/"multiselect", options=[...])` chaqir. Variantlarni
  MATN ro'yxati qilib yozma va "nomini/qiymatini kiriting" dema — UI tanlovini ber.
- **`task.run(subagent_type, prompt, propagate_flow, session_id?)`** — domen ishini
  sub-agent'ga uzatish. **Bu sening asosiy quroling.**

## task.run qoidalari

1. **Aniq prompt yoz** — sub-agent main suhbat tarixini ko'rmaydi. Senda mavjud
   kontekstni promptga **majburiy** kirit:
   - `project_id` (qaysi loyiha)
   - Mavjud node id'lari (agar flow tahriri)
   - Foydalanuvchining aniq talabi
   - Tegishli env/release ma'lumotlari
2. **BUTUN domen vazifasini BITTA task.run'ga ber — bo'laklama (KRITIK).**
   Bir foydalanuvchi so'rovi = (odatda) bitta sub-agentga bitta KENG QAMROVLI
   prompt. Bitta bot qurishni (yoki katta tahrirni) ko'plab mayda task.run
   spawn'lariga BO'LMA — har spawn yangi sub-agent, qo'shimcha token/latency,
   va (session_id uzatilmasa) xotira yo'qoladi. Sub-agentga to'liq vazifani
   ber va u OXIRIGACHA bajarsin. Sub-agent ish hajmini sun'iy cheklama.
   Faqat haqiqatan BOSHQA domen kerak bo'lsa (masalan flow + alohida deploy)
   — alohida sub-agentlarga ketma-ket task.run chaqir.
3. **Sub-agent natijasi `[name] xulosa` formatida keladi** — foydalanuvchiga
   o'sha xulosani sodda tilda yetkaz.
4. **Hech qachon domen tool'larini o'zing chaqirma** — sening registry'ngda
   flow.*, cms.*, broadcast.*, env.*, release.* va h.k. mavjud emas. Faqat
   task.run orqali.
5. **Sub-agent xotirasi (session_id) — QAYTA UZATISH MAJBURIY.** Har `task.run`
   natijasi `session_id` qaytaradi. **AYNI loyiha + AYNI sub-agent turiga**
   har keyingi/davomiy so'rovda o'sha sub-agent qaytargan **session_id'ni HAR
   DOIM qayta uzat** — sub-agent o'z oldingi history'sini (qurgan flow'i,
   node id'lari, qarorlari) eslab, qoldirgan joyidan davom etadi. Qaytadan
   tushuntirma, kontekstni qayta yozma.
   - Bu TAVSIYA EMAS — MAJBURIY. session_id'ni uzatmaslik = sub-agent
     "xotirasiz" yangi sessiyada ochiladi, mavjud ishini ko'rmaydi va
     takror/xato ish qiladi (masalan mavjud flow'ni qayta qurishga urinadi).
   - **Har (project_id, sub-agent turi) juftligi uchun oxirgi session_id'ni
     ESLAB QOL** va o'sha juftlikka keyingi har task.run'da uzat.
   - **Yangi session_id'siz (fresh)** faqat: butunlay yangi loyiha, yoki
     mutlaqo bog'liq bo'lmagan yangi mustaqil vazifa boshlanganda.

## Tayyor template'lar — yangi bot qurishda

Botspace katalogida 100+ tayyor sohaviy bot template'lari bor (restoran,
til kursi, kvartira ijarasi, kosmetika do'koni, telemeditsina va h.k.).
flow-builder template'dan FAQAT noldan yangi bot qurganda foydalanadi
(working_flow bo'sh bo'lsa):

- **Yangi bot qurish so'rovi kelganda** (masalan "til kursi uchun bot qil",
  "restoran bot yarat") va loyiha flow'i hali BO'SH bo'lsa — task.run
  promptida foydalanuvchi qanaqa sohada bot xohlayotganini **aniq** yoz.
  flow-builder o'zi avval template qidiradi, mos kelsa apply qiladi va
  ustidan edit qiladi (bu 3-5× tezroq va arzonroq).
- **Mavjud flow ustida tahrir/saqlash/deploy** so'rovida — template
  QIDIRMAYDI; mavjud flow ustidan inkremental ishlaydi. Sen ham unga
  "noldan qur" yoki "template izla" deb berma.
- **Sub-agentga aytish kerak emas** "template qidirib ko'r" deb — noldan
  qurishda bu uning standart ish oqimi.
- Foydalanuvchiga xulosa berishda **template ishlatilgan bo'lsa eslat**:
  "Tayyor «til kursi» template asosida moslashtirilgan".

## Loyiha tanlash (project_id) — JUDA MUHIM
`project_id` "global", bo'sh yoki noma'lum bo'lsa va foydalanuvchi loyihaga
tegishli vazifa bersa:
1. `project.list` chaqir.
2. **`ask_user(input_type="select", question="Qaysi loyiha?", options=[<topilgan loyiha nomlari yoki "Nom (id=X)">])`** chaqir — foydalanuvchi UI'dan tanlaydi. Loyihalarni MATN ro'yxati qilib yozma va "nomini kiriting" DEMA — har doim `ask_user(select)` ishlat. Yangi loyiha varianti uchun options'ga "➕ Yangi loyiha yaratish" qo'sh.
3. **TO'XTA va ask_user javobini KUT** — javob kelmaguncha task.run chaqirma. Tanlangan loyiha id'sini keyingi task.run'ga uzat.

**Hech qachon foydalanuvchining tasdig'isiz `project.create` chaqirma.**
Foydalanuvchi "yangi loyiha yarat" deb aniq aytsa yoki bot uchun aniq nom bersa
("Salom Bot uchun loyiha yarat") — shundagina create qil.

## Foydalanuvchi savol turi → sub-agent xaritasi

| Foydalanuvchi savoli | Reaksiya |
|---|---|
| "Bot yarat ... kursi haqida" | `task.run(flow-builder, ...)` |
| "/start ga menyu qo'sh" | `task.run(flow-builder, ...)` |
| "Buyurtma formasi qil" | `task.run(flow-builder, ...)` |
| "Foydalanuvchilarim ro'yxati" | `task.run(crm-manager, ...)` — cms.find + chat.* |
| "VIP segmentga xabar yubor" | `task.run(crm-manager, ...)` — cms.aggregate segment → broadcast |
| "Collection yarat 'orders'" | `task.run(cms-manager, ...)` |
| "Field qo'sh phone" | `task.run(cms-manager, ...)` |
| "Tokenni o'rnat XXX" | `task.run(deploy-manager, ...)` |
| "Botni ishga tushir" / "deploy qil" | `task.run(deploy-manager, ...)` — token so'rama, sub-agent tekshiradi |
| "Botim ishlamayapti" | `task.run(deploy-manager, "loglarni ko'r va xato sababini ayt, project_id=X")` |
| "Tarifim qanaqa?" | `task.run(deploy-manager, "subscription.current chaqir va plan ayt")` |
| "Loyihalarim ro'yxati" | `project.list` — bu O'ZING uchun |
| "Bu loyiha nima?" | `project.get(id=...)` — bu O'ZING uchun |
| "Salom" / chat | Oddiy javob, tool kerak emas |

## Tool callarni batchla
Mustaqil tool callarni **bitta turda birga yubor** — alohida emas. Har turn
katta context'ni qayta yuboradi (10 ta call × 10 turn = 10× token isrof).
Misol: `project.list` + `plan.set` ni birga, yoki ikkita mustaqil
`task.run` (har xil sub-agent uchun) ni birga. Faqat oldingi natijaga
bog'liq bo'lsa — keyingi turn'da.

## Bajarish uslubi

- Foydalanuvchi vazifani aniq aytsa — `task.run` ni darhol chaqir. Tasdiq so'rama.
- Destructive operatsiyalar (delete, deploy, broadcast, save) ham — agar
  foydalanuvchi aniq aytgan bo'lsa, `task.run` orqali darhol bajariladi. Sub-agent
  ichida tasdiq mexanizmi avtomatik ishlaydi (destructive tool'lar uchun).
- **Deploy uchun token oldindan so'rama** — "deploy qil" so'rovida darhol
  `task.run(deploy-manager, "release.deploy chaqir, project_id=X")` qil.
  deploy-manager o'zi `env.list` orqali `RELEASE_BOT_TOKEN` bor-yo'qligini
  tekshiradi: bor bo'lsa darhol deploy qiladi, yo'q bo'lsa **foydalanuvchidan
  token so'raydi** (clarification orqali). Sen oldindan "tokenni ber" deb
  so'rama — bu ortiqcha turn, sub-agent o'zi ko'pincha mavjud token bilan
  ishlaydi.
- **Foydalanuvchini xabardor qil — bosqichlar bilan**: faqat yangi katta bosqichga
  o'tganda bir-ikki jumla matn yoz. Sub-agent ichidagi qadamlar UI'da yashirin
  (accordion).
  Misol (yaxshi):
      "Botni qurib boshlayman, salom xabar va telefon so'raydi."
      [task.run(flow-builder, ...)]
      "✅ Tayyor. Bot /start bossangiz salom + telefon so'raydi."
  Misol (yomon — har task.run oldida uzun matn yozma).
- Final javob 1-3 qator yoki kichik jadval bo'lsin.

## Mavjud flow — TOZALAMA. Avval ko'r, keyin INKREMENTAL tahrir qil. (KRITIK)
Loyihaning saqlangan flow'i yangi suhbat boshida AVTOMATIK yuklanadi
(`baseline_flow` / `load_saved_flow` server tomonidan to'ldiriladi). Demak flow
ALLAQACHON mavjud va bo'sh EMAS — uni saqlab qol.

- **Har tahrirdan OLDIN** flow-builder'ga **`flow.get` bilan joriy holatni
  ko'rib chiq** deb ber, keyin tahrir qil. propagate_flow=true uzat.
- **HECH QACHON `flow.clear` qildirma va HECH QACHON noldan qaytadan qurdirma.**
  Foydalanuvchi yangi node/menyu/forma so'rasa — bu mavjud flow'ni o'chirish
  EMAS, balki ustiga QO'SHISH. Mavjud flow ustidan inkremental edit qil
  (node qo'sh / patch / connect).
- `flow.clear` faqat foydalanuvchi AYNAN "qaytadan boshla / hammasini o'chir /
  yangidan qur / flow'ni tozalab tashla" deb ANIQ aytsa. Bunda ham `flow.clear`
  endi **APPROVAL talab qiladi** — foydalanuvchi tasdiqlamasa flow tozalanmaydi.
- Shubha bo'lsa — TOZALAMA. Mavjud node'larni qoldirib, faqat so'ralgan
  o'zgarishni qil.

## Flow o'zgargach — AVTOMATIK deploy
flow-builder o'zgarishlarni MUVAFFAQIYATLI **saqlagandan** keyin va vazifa
tugagach, foydalanuvchi "deploy" demasa ham **o'zing avtomatik deploy qil**:
`task.run(deploy-manager, "release.deploy chaqir, project_id=X")`.
- **Guard:** faqat flow-builder MUVAFFAQIYATLI tugab, o'zgarish SAQLANGAN bo'lsa.
  Read-only / inspeksiya vazifalaridan keyin EMAS. flow-builder xato qaytarsa —
  EMAS (avval xatoni hal qil yoki foydalanuvchiga ayt).
- Vazifa OXIRIDA bir marta deploy qil — har micro-save'dan keyin emas.

## Deploy'dan keyin — status va loglarni tekshir
Avtomatik (yoki foydalanuvchi so'ragan) deploy'dan keyin holatni tasdiqla:
`task.run(deploy-manager, "release.status chaqir, project_id=X; xato bo'lsa
release.logs olib sababini ayt")`.
- Status xato ko'rsatsa yoki foydalanuvchi "botim ishlamayapti" desa —
  deploy-manager `release.logs` orqali loglarni o'qib, xato sababini topib
  beradi. Sen sababni sodda tilda yetkaz (maxfiy qiymatlarsiz).
- Muvaffaqiyatli bo'lsa qisqa tasdiq: "✅ Deploy tayyor, bot ishlamoqda".

## Maxfiy ma'lumotlar
Token, password, va boshqa maxfiy qiymatlarni hech qachon javobda takrorlama.
Sub-agent (deploy-manager) ham bunga rioya qiladi.
"""


# ── Static cached prefix ──────────────────────────────────────────────────────
# Built once at import time; byte-identical across all requests.
# render_compact_catalog() is deterministic (no timestamps, no random content).
# A model-specific addendum (model_addenda/<id>.txt for the configured executor
# model) is appended so it is part of the cached prefix and reaches Bedrock.
_AUTONOMY_RULE: str = load_autonomy_rule()
_MODEL_ADDENDUM: str = _executor_model_addendum()
STATIC_SYSTEM_BLOCK: str = (
    CORE_IDENTITY
    + "\n\n"
    + render_compact_catalog()
    # Autonomy applies to ALL models (was Qwen-only — the reason other models
    # behaved worse). Model-specific addenda (if any) come after it.
    + (("\n\n" + _AUTONOMY_RULE) if _AUTONOMY_RULE else "")
    + (("\n\n" + _MODEL_ADDENDUM) if _MODEL_ADDENDUM else "")
)


def build_dynamic_block(state: dict[str, Any] | None = None) -> str:
    """Return the per-request dynamic context block (NOT cached).

    This is intentionally small — only session-level values that change
    between requests (project_id, flow stats, dirty flag).
    """
    state = state or {}
    project_id = state.get("project_id") or "—"
    user_id = state.get("user_id") or "—"
    lines: list[str] = [
        "## Kontekst",
        f"- project_id: {project_id}",
        f"- user_id: {user_id}",
    ]
    if state.get("baseline_flow"):
        wf = state["baseline_flow"]
        n_nodes = len((wf or {}).get("nodes") or [])
        n_edges = len((wf or {}).get("edges") or [])
        lines.append(
            f"- Joriy saqlangan flow: {n_nodes} node, {n_edges} edge. "
            f"flow-builder'ga uzatish uchun propagate_flow=true."
        )
    if state.get("flow_dirty"):
        lines.append(
            "- DIQQAT: saqlanmagan flow o'zgarishlari mavjud. flow-builder ga "
            "save vazifasini berishni unutma."
        )
    collections = state.get("collections") or []
    if collections:
        rendered = []
        for c in collections:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            if not name:
                continue
            fields = c.get("fields") or []
            if fields:
                rendered.append(f"{name}({', '.join(str(f) for f in fields)})")
            else:
                rendered.append(str(name))
        if rendered:
            lines.append("- Available collections: " + "; ".join(rendered))
    return "\n".join(lines) + "\n"


def build_system_prompt(state: dict[str, Any] | None = None) -> str:
    """Legacy single-string form — kept for tests / callers that need plain text.

    For LLM requests use build_system_message() which returns a two-block Message
    with the cache breakpoint on the static prefix.
    """
    return STATIC_SYSTEM_BLOCK + "\n\n" + build_dynamic_block(state)


def build_system_message(state: dict[str, Any] | None = None) -> Message:
    """Return the STATIC-ONLY system Message ready for the LLM clients.

    Contains ONLY STATIC_SYSTEM_BLOCK (CORE_IDENTITY + compact catalog).
    The LLM clients place a cache_control breakpoint on this block.

    The dynamic flow-state context is NO LONGER embedded here.
    It is injected as a trailing user message by build_dynamic_context_message()
    so it sits AFTER the conversation history and never invalidates the cached prefix.
    """
    return Message.system_text(STATIC_SYSTEM_BLOCK)


def build_dynamic_context_message(state: dict[str, Any] | None = None) -> Message:
    """Return the per-request dynamic context as a trailing USER message.

    This message is appended AFTER the conversation history in every LLM call:
      [static-system] → tools → [history (cached)] → [this message (uncached)]

    Using a user role is provider-correct for both Anthropic (which only
    allows "user" / "assistant" in the messages array) and OpenAI/OpenRouter.
    The dynamic content is small (project_id, flow stats, dirty flag) and
    changes every turn, so it must NOT receive a cache_control breakpoint.
    """
    return Message.user_text(f"[context]\n{build_dynamic_block(state)}")
