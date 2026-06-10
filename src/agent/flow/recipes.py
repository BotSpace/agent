"""Bundled starter recipes — complete, pre-validated flows for the most common
bot types.

Why this exists: a WEAK model building a flow from scratch makes structural
mistakes (wrong handles, orphan triggers, missing edges). Starting from a
complete, valid, well-designed flow and only editing surface details (texts,
collection names) is dramatically more reliable.

Unlike `flow.search_template` (backend + pgvector, can be unavailable), these
recipes ship INSIDE the agent — they work offline and are guaranteed to load.
Each `flow` dict is validated against the real flow validator in CI/tests.

Faithful to the canonical patterns in
`prompts/texts/subagents/flow_builder_examples.txt`:
  - Inline buttons connect via Approach A edges (sourceHandle
    `target-handler-<node>-<exact button text>`), NOT separate trigger nodes.
  - Command triggers are the only roots for menu bots.
  - Capture steps use MessageTriggerNode(state={key,type}).
"""

from __future__ import annotations

from typing import Any


def _pos(x: float, y: float) -> dict[str, float]:
    return {"x": x, "y": y}


# ── Recipe 1: Info / menu bot (no collections, beginner) ─────────────────────
_INFO_MENU_FLOW: dict[str, Any] = {
    "nodes": [
        {
            "id": "start",
            "type": "CommandTriggerNode",
            "data": {"command": "/start", "global": True},
            "position": _pos(0, 0),
        },
        {
            "id": "menu",
            "type": "EditOrSendMessageNode",
            "data": {
                "messageText": "Assalomu alaykum, {{first_name}}! 👋\nQuyidagilardan birini tanlang:",
                "parseMode": "HTML",
                "keyboard_kind": "inline",
                "rows": [
                    {"columns": [{"text": "🛠 Xizmatlar", "type": "callback", "value": "services"}]},
                    {"columns": [{"text": "💰 Narxlar", "type": "callback", "value": "prices"}]},
                    {"columns": [{"text": "📞 Aloqa", "type": "callback", "value": "contact"}]},
                ],
            },
            "position": _pos(320, 0),
        },
        {
            "id": "services-page",
            "type": "EditOrSendMessageNode",
            "data": {
                "messageText": "🛠 <b>Xizmatlarimiz:</b>\n\n• Xizmat 1\n• Xizmat 2\n• Xizmat 3",
                "parseMode": "HTML",
                "keyboard_kind": "inline",
                "rows": [{"columns": [{"text": "⬅️ Ortga", "type": "callback", "value": "back"}]}],
            },
            "position": _pos(640, -160),
        },
        {
            "id": "prices-page",
            "type": "EditOrSendMessageNode",
            "data": {
                "messageText": "💰 <b>Narxlar:</b>\n\n• Xizmat 1 — narx\n• Xizmat 2 — narx",
                "parseMode": "HTML",
                "keyboard_kind": "inline",
                "rows": [{"columns": [{"text": "⬅️ Ortga", "type": "callback", "value": "back"}]}],
            },
            "position": _pos(640, 0),
        },
        {
            "id": "contact-page",
            "type": "EditOrSendMessageNode",
            "data": {
                "messageText": "📞 <b>Aloqa:</b>\n\n☎️ +998 90 000 00 00\n📍 Manzil\n🕐 Ish vaqti",
                "parseMode": "HTML",
                "keyboard_kind": "inline",
                "rows": [{"columns": [{"text": "⬅️ Ortga", "type": "callback", "value": "back"}]}],
            },
            "position": _pos(640, 160),
        },
    ],
    "edges": [
        {"id": "e1", "source": "start", "target": "menu"},
        {"id": "e2", "source": "menu", "target": "services-page", "sourceHandle": "target-handler-menu-🛠 Xizmatlar"},
        {"id": "e3", "source": "menu", "target": "prices-page", "sourceHandle": "target-handler-menu-💰 Narxlar"},
        {"id": "e4", "source": "menu", "target": "contact-page", "sourceHandle": "target-handler-menu-📞 Aloqa"},
        {"id": "e5", "source": "services-page", "target": "menu", "sourceHandle": "target-handler-services-page-⬅️ Ortga"},
        {"id": "e6", "source": "prices-page", "target": "menu", "sourceHandle": "target-handler-prices-page-⬅️ Ortga"},
        {"id": "e7", "source": "contact-page", "target": "menu", "sourceHandle": "target-handler-contact-page-⬅️ Ortga"},
    ],
}


# ── Recipe 2: Lead / contact capture form (1 collection, beginner) ───────────
_LEAD_SAVE_JS = (
    "const lead = { name: get_context('lead_name'), phone: get_context('lead_phone'), "
    "telegram_id: chat_id, username: (message && message.from && message.from.username) || '', "
    "status: 'new', created_at: current_timestamp() }; "
    "const ins = db_insert('leads', lead); "
    "if (ins && ins.success) { send_message('✅ Rahmat, ' + lead.name + "
    "'! Arizangiz qabul qilindi. Tez orada bog\\'lanamiz.'); } "
    "else { send_message('❌ Xatolik yuz berdi. Qaytadan urinib ko\\'ring.'); }"
)

_LEAD_CAPTURE_FLOW: dict[str, Any] = {
    "nodes": [
        {
            "id": "start",
            "type": "CommandTriggerNode",
            "data": {"command": "/start", "global": True},
            "position": _pos(0, 0),
        },
        {
            "id": "ask-name",
            "type": "SendTextMessageNode",
            "data": {"messageText": "Salom! 👋 Ariza qoldirish uchun ismingizni yozing:"},
            "position": _pos(320, 0),
        },
        {
            "id": "capture-name",
            "type": "MessageTriggerNode",
            "data": {"type": "text", "global": False, "filter": "any", "state": {"key": "lead_name", "type": "text"}},
            "position": _pos(640, 0),
        },
        {
            "id": "ask-phone",
            "type": "SendTextMessageNode",
            "data": {
                "messageText": "📱 Telefon raqamingizni yuboring:",
                "keyboard_kind": "reply",
                "rows": [{"columns": [{"text": "📱 Telefon yuborish", "type": "request_contact"}]}],
            },
            "position": _pos(960, 0),
        },
        {
            "id": "capture-phone",
            "type": "MessageTriggerNode",
            "data": {"type": "contact", "global": False, "filter": "any", "state": {"key": "lead_phone", "type": "contact_phone_number"}},
            "position": _pos(1280, 0),
        },
        {
            "id": "save-lead",
            "type": "CustomCodeNode",
            "data": {"jsCode": _LEAD_SAVE_JS, "outputs": []},
            "position": _pos(1600, 0),
        },
    ],
    "edges": [
        {"id": "e1", "source": "start", "target": "ask-name"},
        {"id": "e2", "source": "ask-name", "target": "capture-name"},
        {"id": "e3", "source": "capture-name", "target": "ask-phone"},
        {"id": "e4", "source": "ask-phone", "target": "capture-phone"},
        {"id": "e5", "source": "capture-phone", "target": "save-lead"},
    ],
}


# ── Recipe 3: Shop / order bot (collections: products, orders) ───────────────
_SHOP_LIST_JS = (
    "const res = db_find('products', {active: true}); const items = (res && res.data) || []; "
    "if (!items.length) { send_message('Hozircha mahsulot yo\\'q.'); exit('default'); return; } "
    "const rows = items.map(function(p) { return [{ text: p.name + ' — ' + p.price + ' so\\'m', "
    "callback_data: 'prod:' + p._id }]; }); "
    "send_message('🛍 <b>Mahsulotlar:</b>', { parse_mode: 'HTML', inline_keyboard: rows }); exit('default');"
)
_SHOP_DETAIL_JS = (
    "const id = callback_query.data.replace('prod:', ''); const res = db_find_one('products', {_id: id}); "
    "if (!res || !res.data) { send_message('Topilmadi.'); exit('default'); return; } const p = res.data; "
    "set_context('order_product_id', id); set_context('order_product_name', p.name); "
    "send_message('<b>' + p.name + '</b>\\n\\n' + (p.description || '') + '\\n\\n💰 ' + p.price + ' so\\'m', "
    "{ parse_mode: 'HTML', inline_keyboard: [[{ text: '🛒 Buyurtma berish', callback_data: 'buy:' + id }]] }); "
    "exit('default');"
)
_SHOP_SAVE_JS = (
    "const order = { name: get_context('order_name'), phone: get_context('order_phone'), "
    "product_id: get_context('order_product_id'), product_name: get_context('order_product_name'), "
    "telegram_id: chat_id, status: 'new', created_at: current_timestamp() }; "
    "const ins = db_insert('orders', order); "
    "if (ins && ins.success) { send_message('✅ Buyurtmangiz qabul qilindi!\\n\\n📦 ' + order.product_name + "
    "'\\n👤 ' + order.name + '\\n📱 ' + order.phone + '\\n\\nTez orada bog\\'lanamiz.', { parse_mode: 'HTML' }); } "
    "else { send_message('❌ Xatolik. Qaytadan urinib ko\\'ring.'); }"
)

_SHOP_ORDER_FLOW: dict[str, Any] = {
    "nodes": [
        {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start", "global": True}, "position": _pos(0, 0)},
        {"id": "list-products", "type": "CustomCodeNode", "data": {"jsCode": _SHOP_LIST_JS, "outputs": [{"name": "default"}]}, "position": _pos(320, 0)},
        {"id": "prod-cb", "type": "CallbackQueryTriggerNode", "data": {"filter": "starts_with", "context": {"value": "prod:"}, "global": False}, "position": _pos(640, 0)},
        {"id": "product-detail", "type": "CustomCodeNode", "data": {"jsCode": _SHOP_DETAIL_JS, "outputs": [{"name": "default"}]}, "position": _pos(960, 0)},
        {"id": "buy-cb", "type": "CallbackQueryTriggerNode", "data": {"filter": "starts_with", "context": {"value": "buy:"}, "global": False}, "position": _pos(1280, 0)},
        {"id": "ask-name", "type": "SendTextMessageNode", "data": {"messageText": "✍️ Ismingizni yozing:"}, "position": _pos(1600, 0)},
        {"id": "capture-name", "type": "MessageTriggerNode", "data": {"type": "text", "global": False, "filter": "any", "state": {"key": "order_name", "type": "text"}}, "position": _pos(1920, 0)},
        {
            "id": "ask-phone",
            "type": "SendTextMessageNode",
            "data": {"messageText": "📱 Telefon raqamingizni yuboring:", "keyboard_kind": "reply", "rows": [{"columns": [{"text": "📱 Telefon yuborish", "type": "request_contact"}]}]},
            "position": _pos(2240, 0),
        },
        {"id": "capture-phone", "type": "MessageTriggerNode", "data": {"type": "contact", "global": False, "filter": "any", "state": {"key": "order_phone", "type": "contact_phone_number"}}, "position": _pos(2560, 0)},
        {"id": "save-order", "type": "CustomCodeNode", "data": {"jsCode": _SHOP_SAVE_JS, "outputs": []}, "position": _pos(2880, 0)},
    ],
    "edges": [
        {"id": "e1", "source": "start", "target": "list-products"},
        {"id": "e2", "source": "list-products", "target": "prod-cb", "sourceHandle": "output-default"},
        {"id": "e3", "source": "prod-cb", "target": "product-detail"},
        {"id": "e4", "source": "product-detail", "target": "buy-cb", "sourceHandle": "output-default"},
        {"id": "e5", "source": "buy-cb", "target": "ask-name"},
        {"id": "e6", "source": "ask-name", "target": "capture-name"},
        {"id": "e7", "source": "capture-name", "target": "ask-phone"},
        {"id": "e8", "source": "ask-phone", "target": "capture-phone"},
        {"id": "e9", "source": "capture-phone", "target": "save-order"},
    ],
}


# ── Registry ─────────────────────────────────────────────────────────────────
RECIPES: list[dict[str, Any]] = [
    {
        "id": "info-menu",
        "name": "Info / menyu bot",
        "use_cases": "Biznes vizitka, xizmatlar/narxlar/aloqa, statik FAQ — ma'lumot ko'rsatuvchi bot.",
        "difficulty": "beginner",
        "needs_collections": [],
        "adapt": "Menyu tugmalari va sahifa matnlarini o'zingiznikiga moslang. Sahifa qo'shsangiz: EditOrSendMessageNode + menu'ga tugma + Approach A edge.",
        "flow": _INFO_MENU_FLOW,
    },
    {
        "id": "lead-capture",
        "name": "Ariza / lead yig'uvchi forma",
        "use_cases": "Ariza qabul qilish, qayta qo'ng'iroq so'rovi, ro'yxatdan o'tkazish — ism+telefon yig'ib bazaga saqlaydi.",
        "difficulty": "beginner",
        "needs_collections": ["leads (name, phone, telegram_id, username, status, created_at)"],
        "adapt": "`leads` collection'ini yarating (yo'q bo'lsa cms-manager'ga delegate qiling). Savollarni o'zgartirsangiz state key'larini ham moslang.",
        "flow": _LEAD_CAPTURE_FLOW,
    },
    {
        "id": "shop-order",
        "name": "Do'kon / buyurtma bot",
        "use_cases": "Mahsulot katalogi (bazadan dinamik) + buyurtma qabul qilish.",
        "difficulty": "intermediate",
        "needs_collections": [
            "products (name, price, description, active)",
            "orders (name, phone, product_id, product_name, telegram_id, status, created_at)",
        ],
        "adapt": "`products` va `orders` collection'larini yarating va products'ga ma'lumot kiriting. CustomCode'dagi maydon nomlari collection bilan mos bo'lsin.",
        "flow": _SHOP_ORDER_FLOW,
    },
]


def get_recipe(recipe_id: str) -> dict[str, Any] | None:
    return next((r for r in RECIPES if r["id"] == recipe_id), None)
