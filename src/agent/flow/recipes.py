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


# ── Recipe 4: Support — user message → admin (no collections) ────────────────
_SUPPORT_FLOW: dict[str, Any] = {
    "nodes": [
        {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start", "global": True}, "position": _pos(0, 0)},
        {
            "id": "welcome",
            "type": "SendTextMessageNode",
            "data": {"messageText": "Savolingiz yoki murojaatingizni yozing — operatorlarimizga yetkazamiz. 📨"},
            "position": _pos(320, 0),
        },
        {
            "id": "user-msg",
            "type": "MessageTriggerNode",
            "data": {"type": "text", "global": True, "filter": "any", "state": {"key": "support_msg", "type": "text"}},
            "position": _pos(0, 200),
        },
        {
            "id": "notify-admin",
            "type": "SendToAdminNode",
            "data": {"messageText": "📩 Yangi murojaat\n👤 {{first_name}} (id: {{user_id}})\n\n{{state.support_msg}}"},
            "position": _pos(320, 200),
        },
        {
            "id": "ack",
            "type": "SendTextMessageNode",
            "data": {"messageText": "✅ Xabaringiz qabul qilindi. Tez orada javob beramiz."},
            "position": _pos(640, 200),
        },
    ],
    "edges": [
        {"id": "e1", "source": "start", "target": "welcome"},
        {"id": "e2", "source": "user-msg", "target": "notify-admin"},
        {"id": "e3", "source": "notify-admin", "target": "ack"},
    ],
}


# ── Recipe 5: Subscribe-gate — force channel subscription ────────────────────
_SUBSCRIBE_GATE_FLOW: dict[str, Any] = {
    "nodes": [
        {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start", "global": True}, "position": _pos(0, 0)},
        {
            "id": "check",
            "type": "CheckMembershipNode",
            "data": {"channels": [{"value": "@your_channel"}]},
            "position": _pos(320, 0),
        },
        {
            "id": "welcome",
            "type": "EditOrSendMessageNode",
            "data": {"messageText": "✅ Xush kelibsiz, {{first_name}}!\n\nBotdan to'liq foydalanishingiz mumkin.", "parseMode": "HTML"},
            "position": _pos(640, -120),
        },
        {
            "id": "ask-sub",
            "type": "EditOrSendMessageNode",
            "data": {
                "messageText": "📢 Botdan foydalanish uchun avval kanalimizga obuna bo'ling:",
                "parseMode": "HTML",
                "keyboard_kind": "inline",
                "rows": [
                    {"columns": [{"text": "📢 Kanalga obuna bo'lish", "type": "url", "value": "https://t.me/your_channel"}]},
                    {"columns": [{"text": "✅ Tekshirish", "type": "callback", "value": "check-sub"}]},
                ],
            },
            "position": _pos(640, 120),
        },
        {
            "id": "check-cb",
            "type": "CallbackQueryTriggerNode",
            "data": {"filter": "equals", "context": {"value": "check-sub"}, "global": True},
            "position": _pos(320, 240),
        },
    ],
    "edges": [
        {"id": "e1", "source": "start", "target": "check"},
        {"id": "e2", "source": "check", "target": "welcome", "sourceHandle": "is-member"},
        {"id": "e3", "source": "check", "target": "ask-sub", "sourceHandle": "not-member"},
        {"id": "e4", "source": "check-cb", "target": "check"},
    ],
}


# ── Recipe 6: Paid invite — payment → private channel invite link ────────────
_PAID_INVITE_FLOW: dict[str, Any] = {
    "nodes": [
        {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start", "global": True}, "position": _pos(0, 0)},
        {
            "id": "intro",
            "type": "EditOrSendMessageNode",
            "data": {"messageText": "💎 <b>Premium kanal</b>\n\nKirish uchun to'lovni amalga oshiring:", "parseMode": "HTML"},
            "position": _pos(320, 0),
        },
        {
            "id": "pay",
            "type": "PaymentNode",
            "data": {"paymentType": "click", "amount": 50000, "buttonText": "💳 To'lash"},
            "position": _pos(640, 0),
        },
        {
            "id": "invite",
            "type": "CreateInviteLinkNode",
            "data": {"chatId": "-1001234567890", "name": "Premium", "memberLimit": 1},
            "position": _pos(960, -120),
        },
        {
            "id": "send-link",
            "type": "SendTextMessageNode",
            "data": {"messageText": "✅ To'lov qabul qilindi! Premium kanalga shaxsiy havolangiz:\n\n{{invite_link}}\n\n⚠️ Havola faqat siz uchun va 1 marta ishlaydi."},
            "position": _pos(1280, -120),
        },
        {
            "id": "fail-msg",
            "type": "SendTextMessageNode",
            "data": {"messageText": "❌ To'lov amalga oshmadi. Qaytadan urinib ko'rish uchun: /start"},
            "position": _pos(960, 120),
        },
    ],
    "edges": [
        {"id": "e1", "source": "start", "target": "intro"},
        {"id": "e2", "source": "intro", "target": "pay"},
        {"id": "e3", "source": "pay", "target": "invite", "sourceHandle": "paid"},
        {"id": "e4", "source": "invite", "target": "send-link"},
        {"id": "e5", "source": "pay", "target": "fail-msg", "sourceHandle": "failed"},
    ],
}


# ── Recipe 7: Feedback — star rating → save ──────────────────────────────────
_FEEDBACK_SAVE_JS = (
    "const r = (callback_query && callback_query.data || '').replace('rate:', ''); "
    "const ok = db_insert('feedback', { rating: parseInt(r, 10), telegram_id: chat_id, "
    "created_at: current_timestamp() }); "
    "send_message(ok && ok.success ? '🙏 Rahmat! Bahoyingiz (' + r + '⭐) qabul qilindi.' "
    ": '❌ Xatolik yuz berdi.');"
)
_FEEDBACK_FLOW: dict[str, Any] = {
    "nodes": [
        {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start", "global": True}, "position": _pos(0, 0)},
        {
            "id": "ask-rating",
            "type": "EditOrSendMessageNode",
            "data": {
                "messageText": "Xizmatimizni baholang:",
                "keyboard_kind": "inline",
                "rows": [
                    {
                        "columns": [
                            {"text": "⭐1", "type": "callback", "value": "rate:1"},
                            {"text": "⭐2", "type": "callback", "value": "rate:2"},
                            {"text": "⭐3", "type": "callback", "value": "rate:3"},
                            {"text": "⭐4", "type": "callback", "value": "rate:4"},
                            {"text": "⭐5", "type": "callback", "value": "rate:5"},
                        ]
                    }
                ],
            },
            "position": _pos(320, 0),
        },
        {
            "id": "save-rating",
            "type": "CustomCodeNode",
            "data": {"jsCode": _FEEDBACK_SAVE_JS, "outputs": []},
            "position": _pos(640, 0),
        },
    ],
    "edges": [
        {"id": "e1", "source": "start", "target": "ask-rating"},
        {"id": "e2", "source": "ask-rating", "target": "save-rating", "sourceHandle": "target-handler-ask-rating-⭐1"},
        {"id": "e3", "source": "ask-rating", "target": "save-rating", "sourceHandle": "target-handler-ask-rating-⭐2"},
        {"id": "e4", "source": "ask-rating", "target": "save-rating", "sourceHandle": "target-handler-ask-rating-⭐3"},
        {"id": "e5", "source": "ask-rating", "target": "save-rating", "sourceHandle": "target-handler-ask-rating-⭐4"},
        {"id": "e6", "source": "ask-rating", "target": "save-rating", "sourceHandle": "target-handler-ask-rating-⭐5"},
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
    {
        "id": "support",
        "name": "Qo'llab-quvvatlash (support) bot",
        "use_cases": "Foydalanuvchi xabarini operatorga/admin guruhga yetkazadi va tasdiq qaytaradi.",
        "difficulty": "beginner",
        "needs_collections": [],
        "adapt": "Admin chat/guruhni loyiha sozlamalarida belgilang. Xabar matnini va tasdiq matnini moslang.",
        "flow": _SUPPORT_FLOW,
    },
    {
        "id": "subscribe-gate",
        "name": "Kanalga obuna majburlash",
        "use_cases": "Botdan foydalanishdan oldin foydalanuvchini kanalga obuna bo'lishga majburlaydi (a'zolikni tekshiradi).",
        "difficulty": "intermediate",
        "needs_collections": [],
        "adapt": "`@your_channel` va https://t.me/your_channel ni o'z kanalingizga almashtiring. Bot kanalda admin bo'lishi shart. is-member tarmog'iga asosiy kontentni ulang.",
        "flow": _SUBSCRIBE_GATE_FLOW,
    },
    {
        "id": "paid-invite",
        "name": "To'lov → maxfiy kanalga shaxsiy havola",
        "use_cases": "To'lov o'tgach foydalanuvchiga premium/maxfiy kanalga 1 martalik shaxsiy invite havola yuboradi.",
        "difficulty": "intermediate",
        "needs_collections": [],
        "adapt": "amount (summa) va chatId (-100... premium kanal id) ni o'zingiznikiga moslang. Bot premium kanalda admin va 'invite via link' huquqiga ega bo'lishi shart.",
        "flow": _PAID_INVITE_FLOW,
    },
    {
        "id": "feedback",
        "name": "Baholash / feedback bot",
        "use_cases": "Foydalanuvchidan 1–5 yulduz baho oladi va bazaga saqlaydi.",
        "difficulty": "beginner",
        "needs_collections": ["feedback (rating, telegram_id, created_at)"],
        "adapt": "`feedback` collection'ini yarating. Savol matnini moslang; izoh ham kerak bo'lsa rating'dan keyin SendText + MessageTrigger(text) qo'shing.",
        "flow": _FEEDBACK_FLOW,
    },
]


def get_recipe(recipe_id: str) -> dict[str, Any] | None:
    return next((r for r in RECIPES if r["id"] == recipe_id), None)
