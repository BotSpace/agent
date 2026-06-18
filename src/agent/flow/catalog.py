"""Engine-golang node turlari katalogi (47+).

Manba: engine-golang/internal/engine + agentv2 catalog.
Har node turi quyidagilarni qaytaradi: type, kategoriya, qisqa tavsif,
muhim `data` maydonlari (LLM uchun foydali), kerakli handle'lar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NodeCategory = Literal[
    "trigger",
    "send",
    "edit",
    "flow_control",
    "data",
    "integration",
    "interaction",
    "subflow",
    "ai",
]


@dataclass(frozen=True)
class NodeTypeSpec:
    type: str
    category: NodeCategory
    description: str
    required_data: list[str] = field(default_factory=list)
    optional_data: list[str] = field(default_factory=list)
    source_handles: list[str] = field(default_factory=lambda: ["source-handler"])
    target_handles: list[str] = field(default_factory=lambda: ["target-handler"])
    is_trigger: bool = False
    # Accessory sub-node flag: True → node has no flow target handle, only source-default.
    # It attaches to a parent node's acc: port and is never traversed as a flow step.
    is_accessory: bool = False
    # For AIAgentNode: accessory ports it exposes as target handles (e.g. ["acc:ai_model", "acc:ai_tool"]).
    accessory_target_handles: list[str] = field(default_factory=list)


NODE_TYPES: dict[str, NodeTypeSpec] = {
    # ── Triggers ────────────────────────────────────────────────────────
    "CommandTriggerNode": NodeTypeSpec(
        type="CommandTriggerNode",
        category="trigger",
        description=(
            "Telegram /command trigger. global:true har yerda ishlaydi. withArgs:true bo'lsa argument qabul qiladi. "
            "state: {key, type:'text'} — komanda argumentini saqlaydi."
        ),
        required_data=["command"],
        optional_data=["global", "withArgs", "state"],
        is_trigger=True,
    ),
    "MessageTriggerNode": NodeTypeSpec(
        type="MessageTriggerNode",
        category="trigger",
        description=(
            "Xabar turini kutib trigger. type: text/photo/video/audio/voice/video_note/document/contact/location/sticker/animation. "
            "filter: any/equals/contains/starts_with/regex. context.value: filter qiymati. "
            "global:false bo'lsa faqat pauzada ishlaydi (foydalanuvchi javobini kutish). "
            "state: {key, type} — kelgan ma'lumotni AVTOMATIK saqlaydi, alohida StateNode KERAK EMAS. "
            "Misol: state:{key:'user_name', type:'text'} → {{state.user_name}} orqali foydalaniladi."
        ),
        required_data=["type"],
        optional_data=["filter", "context", "global", "state"],
        is_trigger=True,
    ),
    "CallbackQueryTriggerNode": NodeTypeSpec(
        type="CallbackQueryTriggerNode",
        category="trigger",
        description=(
            "Inline button callback trigger. filter: any/equals/contains/starts_with/regex. "
            "context.value: mos keladigan callback_data. global:true bo'lsa har yerda ishlaydi. "
            "state: {key, type} — callback data'ni saqlaydi (type: 'callback'/'data'/'collection_id')."
        ),
        required_data=[],
        optional_data=["filter", "context", "global", "state"],
        is_trigger=True,
    ),
    "CallbackButtonTriggerNode": NodeTypeSpec(
        type="CallbackButtonTriggerNode",
        category="trigger",
        description="Maxsus inline button tomonidan trigger",
        is_trigger=True,
    ),
    "ReplyButtonTriggerNode": NodeTypeSpec(
        type="ReplyButtonTriggerNode",
        category="trigger",
        description="Reply keyboard button bosilganda trigger. selectedButton: {id, text, type:'reply'}",
        required_data=["selectedButton"],
        optional_data=["catchNoButton"],
        is_trigger=True,
    ),
    "CronTriggerNode": NodeTypeSpec(
        type="CronTriggerNode",
        category="trigger",
        description="Cron schedule bo'yicha trigger",
        required_data=["cronExpression"],
        is_trigger=True,
    ),
    "ExternalWebhookTriggerNode": NodeTypeSpec(
        type="ExternalWebhookTriggerNode",
        category="trigger",
        description="Tashqi webhook (stripe/payme/...) tomonidan trigger",
        required_data=["provider"],
        is_trigger=True,
    ),
    "PreCheckoutTriggerNode": NodeTypeSpec(
        type="PreCheckoutTriggerNode",
        category="trigger",
        description=(
            "To'lov tasdiqlash (pre_checkout_query) trigger. filter: any/equals/"
            "starts_with/contains (invoice payload bo'yicha), value: filter qiymati. "
            "autoAnswer (default true) — engine avtomatik ok=true javob beradi "
            "(10s ichida javob bo'lmasa to'lov yiqiladi). Doim global. Context: "
            "pre_checkout_payload, pre_checkout_total_amount, pre_checkout_currency."
        ),
        is_trigger=True,
    ),
    "SuccessfulPaymentTriggerNode": NodeTypeSpec(
        type="SuccessfulPaymentTriggerNode",
        category="trigger",
        description=(
            "Muvaffaqiyatli to'lov trigger. filter/value — payload bo'yicha "
            "(any/equals/starts_with/contains). Doim global. Context: payment_payload, "
            "payment_total_amount, payment_currency, payment_telegram_charge_id."
        ),
        is_trigger=True,
    ),
    "PaymentStatusTriggerNode": NodeTypeSpec(
        type="PaymentStatusTriggerNode",
        category="trigger",
        description=(
            "Botspace Payment Link status trigger. AI payment_link tool order yaratganda "
            "tool natijasida order_id qaytadi; backend to'lov callback yuborganda shu trigger "
            "ishga tushadi. status: paid/failed/any, orderId/provider ixtiyoriy filter. "
            "Context: payment_order_id, payment_status, payment_amount, payment_net, "
            "payment_commission, payment_comment, payment_provider; aliaslar: order_id, "
            "status, amount, provider."
        ),
        optional_data=["status", "orderId", "provider", "global"],
        is_trigger=True,
    ),
    "ChatJoinRequestTriggerNode": NodeTypeSpec(
        type="ChatJoinRequestTriggerNode",
        category="trigger",
        description=(
            "Guruh/kanalga qo'shilish so'rovi trigger. Maydonsiz, doim global. "
            "Context: join_request_chat_id, join_request_user_id, first_name. "
            "Tasdiqlash/rad etish: ApproveChatJoinRequestNode/DeclineChatJoinRequestNode "
            "(chatId/userId bo'sh qoldirilsa context'dan oladi)."
        ),
        is_trigger=True,
    ),
    # ── Senders ────────────────────────────────────────────────────────
    "SendTextMessageNode": NodeTypeSpec(
        type="SendTextMessageNode",
        category="send",
        description=(
            "Text xabar yuborish. keyboard.active: 'inline'/'reply'/'none'. "
            "keyboard.inline: [[{text,type:'callback'/'url'/'web_app',value}]] (web_app value=https URL). "
            "keyboard.reply: [[{id:'btn-xxx',text,type:'reply'/'request_contact'/'request_location'}]]. "
            "recipients: {sendToSender:bool, additionalIds:[chatId]}. "
            "{{state.key}} — saqlangan qiymat, {{first_name}} — user ismi. "
            "TUGMA ROUTING (yangi usul): har reply/inline tugma o'zidan edge chiqaradi. "
            "flow.connect da source_handle='target-handler-{nodeId}-{buttonText}' formatidan foydalaniladi. "
            "Alohida ReplyButtonTriggerNode KERAK EMAS."
        ),
        required_data=["messageText"],
        optional_data=["remove_keyboard", "parseMode", "keyboard", "recipients", "disableWebPagePreview"],
    ),
    "SendPhotoNode": NodeTypeSpec(
        type="SendPhotoNode",
        category="send",
        description="Rasm yuborish",
        required_data=["photo"],
        optional_data=["remove_keyboard", "caption", "parseMode", "keyboard"],
    ),
    "SendVideoNode": NodeTypeSpec(
        type="SendVideoNode",
        category="send",
        description="Video yuborish",
        required_data=["video"],
        optional_data=["remove_keyboard", "caption", "parseMode", "keyboard"],
    ),
    "SendAudioNode": NodeTypeSpec(
        type="SendAudioNode",
        category="send",
        description="Audio yuborish",
        required_data=["audio"],
        optional_data=["remove_keyboard", "caption", "title", "performer"],
    ),
    "SendFileNode": NodeTypeSpec(
        type="SendFileNode",
        category="send",
        description="Fayl yuborish",
        required_data=["file"],
        optional_data=["remove_keyboard", "caption", "parseMode"],
    ),
    "SendAnimationNode": NodeTypeSpec(
        type="SendAnimationNode",
        category="send",
        description="GIF/animation yuborish",
        required_data=["animation"],
        optional_data=["remove_keyboard"],
    ),
    "SendVoiceNode": NodeTypeSpec(
        type="SendVoiceNode",
        category="send",
        description="Voice xabar yuborish",
        required_data=["voice"],
        optional_data=["remove_keyboard"],
    ),
    "SendVideoNoteNode": NodeTypeSpec(
        type="SendVideoNoteNode",
        category="send",
        description="Video note (round) yuborish",
        required_data=["videoNote"],
        optional_data=["remove_keyboard"],
    ),
    "SendStickerNode": NodeTypeSpec(
        type="SendStickerNode",
        category="send",
        description="Sticker yuborish",
        required_data=["sticker"],
        optional_data=["remove_keyboard"],
    ),
    "SendLocationNode": NodeTypeSpec(
        type="SendLocationNode",
        category="send",
        description="Geolokatsiya yuborish",
        required_data=["latitude", "longitude"],
        optional_data=["remove_keyboard"],
    ),
    "SendVenueNode": NodeTypeSpec(
        type="SendVenueNode",
        category="send",
        description="Venue (joy) yuborish",
        required_data=["latitude", "longitude", "title", "address"],
        optional_data=["remove_keyboard"],
    ),
    "SendContactNode": NodeTypeSpec(
        type="SendContactNode",
        category="send",
        description="Kontakt yuborish",
        required_data=["phoneNumber", "firstName"],
        optional_data=["remove_keyboard"],
    ),
    "SendPollNode": NodeTypeSpec(
        type="SendPollNode",
        category="send",
        description="Poll yuborish",
        required_data=["question", "options"],
        optional_data=["remove_keyboard"],
    ),
    "SendDiceNode": NodeTypeSpec(
        type="SendDiceNode",
        category="send",
        description="Dice/emoji yuborish",
        optional_data=["remove_keyboard", "emoji"],
    ),
    "SendInvoiceNode": NodeTypeSpec(
        type="SendInvoiceNode",
        category="send",
        description=(
            "Telegram to'lov invoice yuborish. title/description/payload majburiy "
            "({{var}} qo'llanadi). providerToken bo'sh = Telegram Stars; currency "
            "default 'XTR'. prices: [{label, amount}] (amount eng kichik birlik, "
            "XTR=stars soni). Ixtiyoriy: photoUrl, needName/needPhone/needEmail/"
            "needShippingAddress. To'lov oqimi: SendInvoice -> PreCheckoutTrigger "
            "(autoAnswer) -> SuccessfulPaymentTrigger."
        ),
        required_data=["title", "description", "payload", "prices"],
        optional_data=["providerToken", "currency", "photoUrl"],
    ),
    "SendMediaGroupNode": NodeTypeSpec(
        type="SendMediaGroupNode",
        category="send",
        description="Media group (album) yuborish",
        required_data=["media"],
        optional_data=["remove_keyboard"],
    ),
    "EditOrSendMessageNode": NodeTypeSpec(
        type="EditOrSendMessageNode",
        category="send",
        description="Mavjud bo'lsa edit qiladi, aks holda yangisini yuboradi",
        required_data=["messageText"],
        optional_data=["remove_keyboard", "targetMessageNodeId", "keyboard"],
    ),
    # ── Edit/Manage ────────────────────────────────────────────────────
    "EditMessageNode": NodeTypeSpec(
        type="EditMessageNode",
        category="edit",
        description=(
            "Avvalgi xabarni edit qilish. messageIdSource: 'this_message' (callback xabarini edit) "
            "yoki node ID. newText: yangi matn. keyboard: yangilangan tugmalar."
        ),
        required_data=["newText"],
        optional_data=["messageIdSource", "keyboard", "parseMode", "editType"],
    ),
    "DeleteMessageNode": NodeTypeSpec(
        type="DeleteMessageNode",
        category="edit",
        description="Xabarni o'chirish",
        optional_data=["targetMessageNodeId"],
    ),
    "ForwardMessageNode": NodeTypeSpec(
        type="ForwardMessageNode",
        category="edit",
        description="Xabarni forward qilish",
        required_data=["targetChatId"],
    ),
    "CopyMessageNode": NodeTypeSpec(
        type="CopyMessageNode",
        category="edit",
        description="Xabarni nusxalash",
        required_data=["targetChatId"],
    ),
    "PinMessageNode": NodeTypeSpec(type="PinMessageNode", category="edit", description="Xabarni pin qilish"),
    "UnpinMessageNode": NodeTypeSpec(type="UnpinMessageNode", category="edit", description="Pinni olib tashlash"),
    "UnpinAllMessagesNode": NodeTypeSpec(
        type="UnpinAllMessagesNode",
        category="edit",
        description="Barcha pinlarni olib tashlash",
    ),
    # ── Flow control ───────────────────────────────────────────────────
    "IfConditionNode": NodeTypeSpec(
        type="IfConditionNode",
        category="flow_control",
        description=(
            "Shart tekshirish. Oddiy: conditions[] + operator(AND/OR) → sourceHandle 'true'/'false'. "
            "Multi-branch: branches[{type,conditions,operator}] → sourceHandle 'branch_0','branch_1',..."
        ),
        required_data=["conditions"],
        optional_data=["branches", "operator"],
        source_handles=["true", "false"],
    ),
    "ElseIfNode": NodeTypeSpec(
        type="ElseIfNode",
        category="flow_control",
        description="Else-if branch (IfConditionNode'dan keyin)",
    ),
    "ElseNode": NodeTypeSpec(
        type="ElseNode",
        category="flow_control",
        description="Else branch",
    ),
    "DelayNode": NodeTypeSpec(
        type="DelayNode",
        category="flow_control",
        description="Sekundlarda kechiktirish",
        required_data=["seconds"],
    ),
    "SchedulerNode": NodeTypeSpec(
        type="SchedulerNode",
        category="flow_control",
        description="Kelajakdagi vaqtda flow ni davom ettirish",
        required_data=["scheduledAt"],
    ),
    "RandomNode": NodeTypeSpec(
        type="RandomNode",
        category="flow_control",
        description="Tasodifiy branch tanlash",
        required_data=["weights"],
    ),
    "ForLoopNode": NodeTypeSpec(
        type="ForLoopNode",
        category="flow_control",
        description="Massiv ustidan iteration. 'loop_item','loop_current_index' context vars.",
        required_data=["array"],
        source_handles=["loop-body", "no-items"],
    ),
    "ForLoopContinueNode": NodeTypeSpec(
        type="ForLoopContinueNode",
        category="flow_control",
        description="ForLoop ichidan continue (boshqa iteratsiyaga o'tish)",
    ),
    "VariableNode": NodeTypeSpec(
        type="VariableNode",
        category="data",
        description="O'zgaruvchi operatsiyasi (set/increment/decrement/append/remove)",
        required_data=["name", "operation"],
        optional_data=["value"],
    ),
    "StateNode": NodeTypeSpec(
        type="StateNode",
        category="data",
        description=(
            "Ma'lumotni state'ga saqlash. type: text/caption/callback/contact_phone_number/"
            "photo_file_id/video_file_id/audio_file_id/document_file_id/file_id/user_id/"
            "first_name/last_name/username/language_code/location_latitude/location_longitude/"
            "Custom. Custom bo'lsa customType maydonida expression yoziladi (masalan '{{inserted_id}}')."
        ),
        required_data=["key", "type"],
        optional_data=["customType"],
    ),
    # ── Collection (CMS) ───────────────────────────────────────────────
    "CollectionNode": NodeTypeSpec(
        type="CollectionNode",
        category="data",
        description=(
            "MongoDB collection'ga document qo'shish. fieldMappings: {db_field: state_key}. "
            "Saqlangandan keyin context'da '{{inserted_id}}' mavjud bo'ladi."
        ),
        required_data=["collection_name", "fieldMappings"],
    ),
    "LoadCollectionItemNode": NodeTypeSpec(
        type="LoadCollectionItemNode",
        category="data",
        description=(
            "Bitta document yuklash. collection.name, contextKey, "
            "filters: [{field, operator:'equals', valueSource:'state'/'static', stateKey, value}]. "
            "Topilsa context'da collection_{contextKey}_{field} mavjud."
        ),
        required_data=["collection", "contextKey", "filters"],
        source_handles=["found", "not_found"],
    ),
    "LoadCollectionListNode": NodeTypeSpec(
        type="LoadCollectionListNode",
        category="data",
        description="Documents ro'yxati (filter, sort, limit)",
        required_data=["collection_name"],
        optional_data=["filter", "sort", "limit"],
    ),
    "UpdateCollectionNode": NodeTypeSpec(
        type="UpdateCollectionNode",
        category="data",
        description="Document update",
        required_data=["collection_name", "filter", "update"],
    ),
    "DeleteCollectionNode": NodeTypeSpec(
        type="DeleteCollectionNode",
        category="data",
        description="Document delete",
        required_data=["collection_name", "filter"],
    ),
    # ── Integrations ───────────────────────────────────────────────────
    "HTTPRequestNode": NodeTypeSpec(
        type="HTTPRequestNode",
        category="integration",
        description="HTTP request (GET/POST/PUT/DELETE). SSRF protection enabled.",
        required_data=["url", "method"],
        optional_data=["headers", "body", "saveAs"],
    ),
    "CustomCodeNode": NodeTypeSpec(
        type="CustomCodeNode",
        category="integration",
        description=(
            "JavaScript kod ishga tushirish (Goja VM). Kod data.jsCode da saqlanadi "
            "(engine va UI shu kalitni o'qiydi). MongoDB/HTTP async API mavjud. "
            "data.outputs=[{name}] e'lon qilinsa, exit('name') bilan `output-{name}` "
            "source handle orqali tarmoqlanadi. data.terminal=true — node leaf "
            "(o'zi javob yuborib oqimni tugatadi), keyingi edge yo'qligi normal "
            "(validator 'dead-end' ogohlantirishini o'chiradi)."
        ),
        required_data=["jsCode"],
        optional_data=["outputs", "terminal"],
    ),
    "SendToAdminNode": NodeTypeSpec(
        type="SendToAdminNode",
        category="integration",
        description="Admin chatga xabar yuborish",
        required_data=["messageText"],
    ),
    "CheckMembershipNode": NodeTypeSpec(
        type="CheckMembershipNode",
        category="integration",
        description="Foydalanuvchi kanal/guruh a'zosi ekanligini tekshirish",
        required_data=["channels"],
        optional_data=["channelsCollection", "channelsField", "chatId"],
        source_handles=["is-member", "not-member"],
    ),
    "DownloadNode": NodeTypeSpec(
        type="DownloadNode",
        category="integration",
        description="Telegram file/photo'ni yuklab olib state'ga saqlash",
        required_data=["fileId", "saveAs"],
    ),
    "CronNode": NodeTypeSpec(
        type="CronNode",
        category="integration",
        description="Inline cron — periodic task in middle of flow",
        required_data=["cronExpression"],
    ),
    "ApproveChatJoinRequestNode": NodeTypeSpec(
        type="ApproveChatJoinRequestNode",
        category="integration",
        description=(
            "Guruh/kanalga qo'shilish so'rovini TASDIQLASH. Oddiy action, bitta chiqish. "
            "chatId/userId bo'sh bo'lsa ChatJoinRequestTriggerNode context'idan olinadi "
            "(join_request_chat_id/join_request_user_id). Odatda: ChatJoinRequestTriggerNode → "
            "(IfCondition) → ApproveChatJoinRequestNode."
        ),
        required_data=[],
        optional_data=["chatId", "userId"],
    ),
    "DeclineChatJoinRequestNode": NodeTypeSpec(
        type="DeclineChatJoinRequestNode",
        category="integration",
        description=(
            "Guruh/kanalga qo'shilish so'rovini RAD ETISH. Oddiy action, bitta chiqish. "
            "chatId/userId bo'sh bo'lsa ChatJoinRequestTriggerNode context'idan olinadi "
            "(join_request_chat_id/join_request_user_id). Odatda: ChatJoinRequestTriggerNode → "
            "(IfCondition) → DeclineChatJoinRequestNode."
        ),
        required_data=[],
        optional_data=["chatId", "userId"],
    ),
    "CreateInviteLinkNode": NodeTypeSpec(
        type="CreateInviteLinkNode",
        category="integration",
        description=(
            "Kanal/guruh (xususiy kanal ham) uchun join (invite) HAVOLA yaratadi — "
            "Telegram createChatInviteLink. Bot o'sha kanalda admin va 'invite via link' "
            "huquqiga ega bo'lishi shart. Yaratilgan havola context'ga `resultVar` "
            "(default `invite_link`) bilan yoziladi — keyin SendTextMessageNode'da "
            "{{invite_link}} sifatida yuboriladi. Oddiy action, bitta chiqish. "
            "Tipik foydalanish: to'lov o'tgach shaxsiy bir martalik (memberLimit=1) havola "
            "berish: PaymentNode → CreateInviteLinkNode → SendTextMessageNode({{invite_link}})."
        ),
        required_data=[],
        # chatId bo'sh bo'lsa joriy chat ishlatiladi (guruhda — o'sha guruh).
        optional_data=[
            "chatId",
            "name",
            "memberLimit",
            "expireDate",
            "createsJoinRequest",
            "resultVar",
        ],
    ),
    "PaymentNode": NodeTypeSpec(
        type="PaymentNode",
        category="integration",
        description=(
            "Botspace orqali to'lov. ⚠️ QURISHDAN OLDIN foydalanuvchidan to'lov "
            "tizimini SO'RA (ask_user select: 'click' yoki 'paycue') va javobni "
            "paymentType'ga ber. paymentType FAQAT 'click' (Click pay-link tugmasi) yoki "
            "'paycue' (Humo-karta — mijozga aniq summani kartaga o'tkazish ko'rsatmasi "
            "yuboriladi, tugma bo'lmaydi, avtomatik tasdiqlanadi) — boshqasini o'ylab topma. "
            "Egasi to'lov tizimi "
            "bilan o'zi integratsiya QILMAYDI. To'lov tasdiqlansa 'paid', aks holda 'failed'. "
            "amount: son yoki {{var}}. To'lovdan keyin context: payment_order_id, payment_amount, "
            "payment_net, payment_status, payment_comment, payment_provider. "
            "Eslatma: bu Telegram Stars EMAS — lokal kartalar (Click). Telegram Stars uchun "
            "alohida invoice (SendInvoiceNode) mexanizmi ishlatiladi."
        ),
        required_data=["amount"],
        optional_data=["paymentType", "comment", "buttonText"],
        source_handles=["paid", "failed"],
    ),
    "GoogleSheetsNode": NodeTypeSpec(
        type="GoogleSheetsNode",
        category="integration",
        description=(
            "Google Sheets (v4) — oauth2 credential orqali. credential_id type_key "
            "'google_sheets' bo'lgan credential'ga ishora qiladi (secret flow'da emas, "
            "engine runtime'da access_token'ga aylantiradi). operation: append/read/update/clear. "
            "spreadsheet_id majburiy. range to'liq 'Sheet1!A1:C10' yoki sheet alohida "
            "berilsa faqat 'A1:C10' (engine sheet+range birlashtiradi) — sheet yoki range'dan "
            "kamida biri bo'lishi shart. values JSON 2D massiv (append/update). Natija "
            "context[responseVariable] (default 'sheets_response'), "
            "context[responseVariable+'_status'], context['sheets_success'] ga yoziladi."
        ),
        required_data=["credential_id", "operation", "spreadsheet_id"],
        optional_data=["sheet", "range", "column", "values", "responseVariable"],
    ),
    "GoogleDriveNode": NodeTypeSpec(
        type="GoogleDriveNode",
        category="integration",
        description=(
            "Google Drive (v3) — oauth2 credential orqali. credential_id type_key "
            "'google_drive' bo'lgan credential'ga ishora qiladi. operation: list/create_folder/"
            "upload/delete. list uchun query (Drive q filtri); create_folder/upload uchun "
            "file_name; folder_id ota-papka; upload uchun content (matn) + mime_type "
            "(default 'text/plain'); delete uchun file_id. Natija context[responseVariable] "
            "(default 'drive_response'), _status, context['drive_success'] ga yoziladi."
        ),
        required_data=["credential_id", "operation"],
        optional_data=["query", "file_name", "folder_id", "content", "mime_type", "file_id", "responseVariable"],
    ),
    # ── Interaction ────────────────────────────────────────────────────
    "ChatActionNode": NodeTypeSpec(
        type="ChatActionNode",
        category="interaction",
        description="Typing indicator (typing/upload_photo/...)",
        required_data=["action"],
    ),
    "CallbackQueryAnswerNode": NodeTypeSpec(
        type="CallbackQueryAnswerNode",
        category="interaction",
        description="Callback query'ga javob (loading'ni o'chirish)",
        optional_data=["text", "showAlert"],
    ),
    "AnswerInlineQueryNode": NodeTypeSpec(
        type="AnswerInlineQueryNode",
        category="interaction",
        description="Inline query'ga natijalar qaytarish",
        required_data=["results"],
    ),
    # ── Sub-flow ───────────────────────────────────────────────────────
    "SubFlowNode": NodeTypeSpec(
        type="SubFlowNode",
        category="subflow",
        description="Boshqa flow ni ichki sifatida chaqirish",
        required_data=["subflowId"],
    ),
    # ── AI Agent (flow node + accessory parent) ────────────────────────
    "AIAgentNode": NodeTypeSpec(
        type="AIAgentNode",
        category="ai",
        description=(
            "LLM agent node. Bitta provider sub-node (acc:ai_model portiga) va ixtiyoriy "
            "tool sub-node'lar (acc:ai_tool portiga) ulanadi. systemPrompt + userInputSource → "
            "LLM tool-calling loop → natija outputKey context o'zgaruvchisiga yoziladi "
            "(standart: agent_output). Keyingi nodelar {{agent_output}} orqali foydalanadi. "
            "Bog'lash: flow.connect(source=<provider_id>, source_handle='source-default', "
            "target=<agent_id>, target_handle='acc:ai_model'). "
            "Tavsiya: OpenAI yoki Ollama provider'i bilan boshlang. "
            "SUHBAT XOTIRASI: chatbot / suhbatdosh / savdo-agent uchun autoHistory=true "
            "qo'y — engine har chat uchun dialogni avtomatik eslaydi. Aks holda bot "
            "amneziyali bo'ladi (har xabarda oldingi gapni unutadi). "
            "TOOL ISHLAT: agent shunchaki matn yozmasin — real ma'lumot/harakat kerak "
            "bo'lsa acc:ai_tool'ga tool ula. Tool'siz agent uydirma javob berishi mumkin. "
            "TOOL TANLASH YO'RIQNOMASI: "
            "• dinamik DB ma'lumoti (katalog, buyurtma, foydalanuvchi) o'qish/yozish → "
            "AICollectionToolNode (read uchun operations=['read']); "
            "• tashqi REST API chaqirish → AIHTTPToolNode (natija status-aware: model "
            "4xx/5xx xatoni ko'radi); "
            "• murakkab mantiq/hisob-kitob yoki bir nechta amal birga → "
            "AICustomCodeToolNode (kuchli — Mongo yoz, Telegram yubor, http; ehtiyot bo'l); "
            "• hujjat/bilim bazasi bo'yicha semantic (RAG) qidiruv → AIKnowledgeToolNode, "
            "LEKIN faqat tashqi Qdrant URL + embedding kaliti aniq mavjud bo'lsa (built-in "
            "vector DB YO'Q; aks holda tool jimgina tushib qoladi — o'rniga collection/http "
            "ishlat). Masalan 'narxni ayt' → AICollectionToolNode (real katalog), 'buyurtma "
            "yarat' → collection create yoki custom_code; "
            "• joriy sana/vaqt yoki sana hisobi (bron/jadval/eslatma) → AIDateTimeToolNode "
            "(credential YO'Q); "
            "• Google Sheet o'qish/yozish (lookup, log, buyurtma jurnali) → "
            "AIGoogleSheetsToolNode (type_key='google_sheets' OAuth credential, dashboard'da "
            "Connect with Google); "
            "• AI dinamik to'lov havolasi yaratsin (sotuv-agent narx aytib hisob chiqarsin) → "
            "AIPaymentLinkToolNode (paymentType click|paycue — qurishdan oldin SO'RA, payme YO'Q). "
            "STRUCTURED JSON: agent JSON qaytarishi kerak bo'lsa systemPrompt'da 'JSON "
            "qaytar' DEMA (ishonchsiz) — `structuredOutput:true` + `jsonSchema` (schema "
            "string) qo'y. Engine model'ni provider darajasida shu schema'ga majburlaydi "
            "va parse qilingan obyektni outputKey'ga yozadi ({{agent_output.field}}). "
            "Faqat schema'siz erkin matn kerak bo'lsa returnMode='json' (best-effort parse). "
            "XATO SEMANTIKASI (engine): SUCCESS bo'lsa errorOutputKey TOZALANADI (oldingi "
            "xato qoldig'i qolmaydi); provider XATO bersa outputKey bo'sh, errorOutputKey "
            "saqlanadi va flow KEYINGI node'ga DAVOM etadi (graceful) — shuning uchun "
            "errorOutputKey ustidan IfConditionNode bilan branch qilib xato holatini "
            "alohida ko'rsatish mumkin."
        ),
        required_data=[],
        optional_data=[
            "systemPrompt", "userInputSource", "userInputText", "outputKey",
            "maxIterations", "temperature", "maxTokens", "returnMode",
            "structuredOutput", "jsonSchema",
            "historyKey", "autoHistory", "maxHistory", "errorOutputKey",
        ],
        source_handles=["source-default"],
        target_handles=["target-default"],
        accessory_target_handles=["acc:ai_model", "acc:ai_tool"],
    ),
    # ── AI Provider sub-nodes (accessory, attach to acc:ai_model) ─────
    "AIOpenAIProviderNode": NodeTypeSpec(
        type="AIOpenAIProviderNode",
        category="ai",
        description=(
            "OpenAI provider sub-node. AIAgentNode'ning acc:ai_model portiga ulanadi. "
            "provider='openai', default model='gpt-4o-mini'. KALIT: manual apiKey "
            "EMAS — credential_id (type_key 'openai') bering "
            "(credentials.list(type_key='openai')). Engine runtime'da credential'dan "
            "api_key/base_url'ga aylantiradi (baseURL SET QILMA). "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_model'."
        ),
        required_data=[],
        optional_data=["provider", "credential_id", "apiKey", "model", "temperature", "maxTokens", "extraHeaders", "extraParams"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIOllamaProviderNode": NodeTypeSpec(
        type="AIOllamaProviderNode",
        category="ai",
        description=(
            "Ollama provider sub-node. AIAgentNode'ning acc:ai_model portiga ulanadi. "
            "AFZAL (DEFAULT) AI provider — bot LLM ishlatsa shuni tanla. "
            "provider='ollama'. Endpoint (base_url) CREDENTIAL'dan keladi — baseURL SET QILMA "
            "(bo'sh qoldir; engine credential'dan oladi, bo'sh bo'lsa provider-default). "
            "AFZAL model='gemma4:31b-cloud' (cloud model) — model'ni ANIQ shunga sozla, bo'sh "
            "qoldirma (bo'sh qolsa engine eski fallback 'gpt-oss:120b'ni oladi). KALIT: "
            "credential_id (type_key 'ollama') bering (credentials.list(type_key='ollama')) — "
            "manual apiKey EMAS. Credential yo'q bo'lsa: foydalanuvchidan kalit so'rab "
            "credentials.create qiling YOKI muqobil providerni (openai / ai_studio) taklif qiling. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_model'."
        ),
        required_data=[],
        optional_data=["provider", "credential_id", "apiKey", "model", "temperature", "maxTokens", "extraHeaders", "extraParams"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIGeminiProviderNode": NodeTypeSpec(
        type="AIGeminiProviderNode",
        category="ai",
        description=(
            "Google Gemini provider sub-node. AIAgentNode'ning acc:ai_model portiga ulanadi. "
            "provider='gemini', default model='gemini-1.5-flash'. KALIT: credential_id "
            "(type_key 'ai_studio' — Google AI Studio) bering "
            "(credentials.list(type_key='ai_studio')) — manual apiKey EMAS. "
            "base_url credential'dan keladi — baseURL SET QILMA. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_model'."
        ),
        required_data=[],
        optional_data=["provider", "credential_id", "apiKey", "model", "temperature", "maxTokens", "extraHeaders", "extraParams"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIAnthropicProviderNode": NodeTypeSpec(
        type="AIAnthropicProviderNode",
        category="ai",
        description=(
            "Anthropic (Claude) provider sub-node. AIAgentNode'ning acc:ai_model portiga ulanadi. "
            "provider='anthropic', default model='claude-3-5-haiku-latest'. "
            "ESLATMA: registry'da 'anthropic' credential type'i YO'Q — bu provider "
            "hanuz manual apiKey ishlatadi (ask_user bilan haqiqiy kalit ol). "
            "credential_id qo'llab-quvvatlanmaydi. baseURL SET QILMA (default ishlatiladi). "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_model'."
        ),
        required_data=[],
        optional_data=["provider", "apiKey", "model", "temperature", "maxTokens", "extraHeaders", "extraParams"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    # ── AI Tool sub-nodes (accessory, attach to acc:ai_tool) ──────────
    "AIHTTPToolNode": NodeTypeSpec(
        type="AIHTTPToolNode",
        category="ai",
        description=(
            "HTTP request tool sub-node. AIAgentNode'ning acc:ai_tool portiga ulanadi. "
            "tool='http_request'. LLM urlTemplate/bodyTemplate'ni tool args bilan chaqiradi. "
            "QACHON: tashqi REST API'ga so'rov yuborish kerak bo'lganda. "
            "NATIJA STATUS-AWARE: model javobda HTTP status'ni ko'radi — non-2xx bo'lsa "
            "natijada status va ok=false bo'ladi (masalan {\"status\":404,\"ok\":false,...}), "
            "shuning uchun model 4xx/5xx xatoni muvaffaqiyatdan farqlay oladi. "
            "SSRF: ichki/private manzillar (localhost, 10/172/192.168, .internal) standart "
            "bloklanadi; ataylab ichki API kerak bo'lsa allowInternal=true yoki allowedHosts "
            "ro'yxatini ber. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_tool'."
        ),
        required_data=[],
        optional_data=[
            "tool", "toolName", "description", "parametersSchema",
            "method", "urlTemplate", "headers", "bodyTemplate", "timeout",
            "allowInternal", "allowedHosts",
        ],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AICollectionToolNode": NodeTypeSpec(
        type="AICollectionToolNode",
        category="ai",
        description=(
            "Collection CRUD tool sub-node. AIAgentNode'ning acc:ai_tool portiga ulanadi. "
            "tool='collection'. Bot collection ustida to'liq CRUD: create/read/update/delete. "
            "operations — ruxsat etilgan amallar allow-list'i (masalan faqat ['read']). "
            "searchFields — LLM filtrlay oladigan maydonlar (ixtiyoriy). FORMAT: OBYEKT "
            "MASSIVI, har element {\"field\": \"<maydon nomi>\", \"description\": \"<izoh>\"} "
            "(faqat `field` majburiy). Masalan "
            "[{\"field\": \"chat_id\", \"description\": \"foydalanuvchi id\"}, {\"field\": \"kind\"}]. "
            "ODDIY STRING MASSIVI EMAS ([\"chat_id\",\"kind\"] xato — har element dict bo'lishi "
            "kerak), kalitlar `name`/`type` EMAS — `field`/`description`. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_tool'."
        ),
        required_data=["collection"],
        optional_data=[
            "tool", "toolName", "description", "parametersSchema",
            "operations", "searchFields",
        ],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AICustomCodeToolNode": NodeTypeSpec(
        type="AICustomCodeToolNode",
        category="ai",
        description=(
            "Custom JavaScript tool sub-node. AIAgentNode'ning acc:ai_tool portiga ulanadi. "
            "tool='custom_code'. Goja sandbox'da ishlaydi, args + flow context mavjud. "
            "QACHON: murakkab mantiq/hisob yoki bir nechta amalni birga bajarish kerak "
            "bo'lganda (oddiy DB o'qish → collection, oddiy API → http afzal). "
            "DIQQAT: to'liq engine imkoniyatiga ega — Mongo yoz (insert/update/delete), "
            "Telegram yubor, http chaqir. Faqat kerakli amal uchun ehtiyot bilan kod yoz, "
            "ataylab bo'lmagan buzg'unchi DB amallaridan saqlan, cheksiz loop yozma "
            "(30s timeout bor). "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_tool'."
        ),
        required_data=[],
        optional_data=[
            "tool", "toolName", "description", "parametersSchema", "jsCode",
        ],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIKnowledgeToolNode": NodeTypeSpec(
        type="AIKnowledgeToolNode",
        category="ai",
        description=(
            "RAG knowledge retrieval tool sub-node. AIAgentNode'ning acc:ai_tool portiga ulanadi. "
            "tool='knowledge'. Bot collection'ini vector DB'ga (lazy) embed qiladi va agent "
            "semantic qidiruv orqali tegishli hujjatlarni olishi mumkin — agent uni "
            "oddiy matnli `query` bilan chaqiradi. "
            "QACHON: faqat hujjat/bilim bazasi bo'yicha semantic (RAG) qidiruv kerak bo'lsa. "
            "OG'IR INFRA — bu tool'ni FAQAT tashqi Qdrant URL va embedding provider/kaliti "
            "aniq mavjud bo'lganda qo'sh. Built-in vector DB YO'Q: vectorDbUrl bo'sh bo'lsa "
            "engine tool'ni qura olmaydi va JIMGINA tashlab ketadi (agent tool'siz qoladi). "
            "Shubha bo'lsa AIKnowledgeToolNode QO'SHMA — o'rniga AICollectionToolNode (read) "
            "yoki AIHTTPToolNode ishlat. "
            "vectorDbUrl: MAJBURIY — tashqi Qdrant URL (masalan https://qdrant.example:6333). "
            "embeddingProvider: openai|ollama|gemini (majburiy, ishlaydigan kalit bilan). "
            "embeddingModel + embeddingApiKey: majburiy (bo'sh model/kalit → tool ishlamaydi). "
            "topK: qaytariladigan hujjatlar soni (standart 5). "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_tool'."
        ),
        required_data=["collection"],
        optional_data=[
            "tool", "toolName",
            "textFields", "embeddingProvider", "embeddingModel",
            "embeddingApiKey", "embeddingBaseURL",
            "vectorDbUrl", "vectorDbApiKey", "topK",
        ],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIDateTimeToolNode": NodeTypeSpec(
        type="AIDateTimeToolNode",
        category="ai",
        description=(
            "Date/time tool sub-node. AIAgentNode'ning acc:ai_tool portiga ulanadi. "
            "tool='datetime'. Credential YO'Q. Agentga haqiqiy soat + sana hisobini beradi — "
            "u 'bugun'/'hozir'ni taxmin qilmaydi, sanalarni o'z 'boshida' hisoblamaydi. "
            "LLM runtime args: op (now|add|diff), amount, unit "
            "(minutes|hours|days|weeks|months|years), from, to (ISO). "
            "QACHON: bron/jadval/eslatma botlari joriy sana yoki '3 kundan keyin' / "
            "'X gacha necha kun' kerak bo'lganda. "
            "Ixtiyoriy data: tz (IANA, default UTC), format. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_tool'."
        ),
        required_data=[],
        optional_data=["tool", "toolName", "tz", "format"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIGoogleSheetsToolNode": NodeTypeSpec(
        type="AIGoogleSheetsToolNode",
        category="ai",
        description=(
            "Google Sheets tool sub-node. AIAgentNode'ning acc:ai_tool portiga ulanadi. "
            "tool='google_sheets'. Agent Google Sheet'ni o'qiydi/qator qo'shadi (lookup "
            "jadval, log, buyurtma jurnali). LLM runtime args: operation (read|append), "
            "spreadsheet_id, sheet, range, values. "
            "CREDENTIAL MAJBURIY: credential_id — type_key 'google_sheets' OAuth "
            "(credentials.list(type_key='google_sheets')). API orqali YARATIB BO'LMAYDI — "
            "foydalanuvchi dashboard'da 'Connect with Google' qiladi. "
            "Ixtiyoriy default'lar: spreadsheetId, sheet, operation. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_tool'."
        ),
        required_data=["credential_id"],
        optional_data=["tool", "toolName", "spreadsheetId", "sheet", "operation"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIPaymentLinkToolNode": NodeTypeSpec(
        type="AIPaymentLinkToolNode",
        category="ai",
        description=(
            "Dinamik to'lov-havola tool sub-node. AIAgentNode'ning acc:ai_tool portiga "
            "ulanadi. tool='payment_link'. Agent uchib to'lov havolasi yaratadi (sotuv-agent "
            "narx aytib so'ng hisob chiqaradi). LLM runtime args: amount (MAJBURIY), "
            "description. paymentType qurish vaqtida belgilanadi, LLM tanlamaydi. "
            "QURISH QOIDASI (PaymentNode kabi): qo'shishdan oldin foydalanuvchidan provayderni "
            "SO'RA — 'click' yoki 'paycue' FAQAT, 'payme' O'YLAB TOPMA. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_tool'."
        ),
        required_data=["paymentType"],
        optional_data=["tool", "toolName"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
}


def all_node_types() -> list[NodeTypeSpec]:
    return list(NODE_TYPES.values())


def get_node_spec(type_name: str) -> NodeTypeSpec | None:
    return NODE_TYPES.get(type_name)


def is_trigger(type_name: str) -> bool:
    spec = NODE_TYPES.get(type_name)
    return spec.is_trigger if spec else False


def summary_for_llm() -> str:
    """Brief catalog dump suitable for system prompt."""
    lines: list[str] = []
    categories: dict[str, list[NodeTypeSpec]] = {}
    for spec in NODE_TYPES.values():
        categories.setdefault(spec.category, []).append(spec)
    for cat in ("trigger", "send", "edit", "flow_control", "data", "integration", "interaction", "subflow", "ai"):
        items = categories.get(cat, [])
        if not items:
            continue
        lines.append(f"\n[{cat.upper()}]")
        for spec in items:
            req = ", ".join(spec.required_data) if spec.required_data else "—"
            lines.append(f"  {spec.type}: {spec.description}  required: {req}")
    return "\n".join(lines)


# Stable category order — must not change between calls (byte-identical output).
_CATEGORY_ORDER = ("trigger", "send", "edit", "flow_control", "data", "integration", "interaction", "subflow", "ai")


def render_compact_catalog() -> str:
    """Deterministic, compact node-type catalog for the cached system-prompt prefix.

    Lists every node type with: type, category, 1-line description, is_trigger flag,
    and a short source_handles summary.  NO full per-type field schemas (those stay
    in the flow.list_node_types tool).

    Ordering is fixed (category order then insertion order within each category).
    No timestamps, no random or locale-dependent content — output is byte-identical
    across calls.
    """
    # Group by category in stable insertion order.
    buckets: dict[str, list[NodeTypeSpec]] = {cat: [] for cat in _CATEGORY_ORDER}
    for spec in NODE_TYPES.values():
        if spec.category in buckets:
            buckets[spec.category].append(spec)

    lines: list[str] = ["## Node tiplari kataloqi\n"]
    for cat in _CATEGORY_ORDER:
        items = buckets[cat]
        if not items:
            continue
        lines.append(f"### {cat.upper()}\n")
        for spec in items:
            trigger_flag = " [trigger]" if spec.is_trigger else ""
            accessory_flag = " [accessory]" if spec.is_accessory else ""
            handles = spec.source_handles
            if len(handles) == 1 and handles[0] == "source-handler":
                handles_str = ""
            else:
                handles_str = f"  handles: {', '.join(handles)}"
            lines.append(
                f"- **{spec.type}**{trigger_flag}{accessory_flag} "
                f"[{spec.category}]{handles_str}\n"
                f"  {spec.description}\n"
            )
    return "".join(lines)
