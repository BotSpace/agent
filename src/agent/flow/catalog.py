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
    # ── Senders ────────────────────────────────────────────────────────
    "SendTextMessageNode": NodeTypeSpec(
        type="SendTextMessageNode",
        category="send",
        description=(
            "Text xabar yuborish. keyboard.active: 'inline'/'reply'/'none'. "
            "keyboard.inline: [[{text,type:'callback'/'url',value}]]. "
            "keyboard.reply: [[{id:'btn-xxx',text,type:'reply'/'request_contact'/'request_location'}]]. "
            "recipients: {sendToSender:bool, additionalIds:[chatId]}. "
            "{{state.key}} — saqlangan qiymat, {{first_name}} — user ismi. "
            "TUGMA ROUTING (yangi usul): har reply/inline tugma o'zidan edge chiqaradi. "
            "flow.connect da source_handle='target-handler-{nodeId}-{buttonText}' formatidan foydalaniladi. "
            "Alohida ReplyButtonTriggerNode KERAK EMAS."
        ),
        required_data=["messageText"],
        optional_data=["parseMode", "keyboard", "recipients", "disableWebPagePreview"],
    ),
    "SendPhotoNode": NodeTypeSpec(
        type="SendPhotoNode",
        category="send",
        description="Rasm yuborish",
        required_data=["photo"],
        optional_data=["caption", "parseMode", "keyboard"],
    ),
    "SendVideoNode": NodeTypeSpec(
        type="SendVideoNode",
        category="send",
        description="Video yuborish",
        required_data=["video"],
        optional_data=["caption", "parseMode", "keyboard"],
    ),
    "SendAudioNode": NodeTypeSpec(
        type="SendAudioNode",
        category="send",
        description="Audio yuborish",
        required_data=["audio"],
        optional_data=["caption", "title", "performer"],
    ),
    "SendFileNode": NodeTypeSpec(
        type="SendFileNode",
        category="send",
        description="Fayl yuborish",
        required_data=["file"],
        optional_data=["caption", "parseMode"],
    ),
    "SendAnimationNode": NodeTypeSpec(
        type="SendAnimationNode",
        category="send",
        description="GIF/animation yuborish",
        required_data=["animation"],
    ),
    "SendVoiceNode": NodeTypeSpec(
        type="SendVoiceNode",
        category="send",
        description="Voice xabar yuborish",
        required_data=["voice"],
    ),
    "SendVideoNoteNode": NodeTypeSpec(
        type="SendVideoNoteNode",
        category="send",
        description="Video note (round) yuborish",
        required_data=["videoNote"],
    ),
    "SendStickerNode": NodeTypeSpec(
        type="SendStickerNode",
        category="send",
        description="Sticker yuborish",
        required_data=["sticker"],
    ),
    "SendLocationNode": NodeTypeSpec(
        type="SendLocationNode",
        category="send",
        description="Geolokatsiya yuborish",
        required_data=["latitude", "longitude"],
    ),
    "SendVenueNode": NodeTypeSpec(
        type="SendVenueNode",
        category="send",
        description="Venue (joy) yuborish",
        required_data=["latitude", "longitude", "title", "address"],
    ),
    "SendContactNode": NodeTypeSpec(
        type="SendContactNode",
        category="send",
        description="Kontakt yuborish",
        required_data=["phoneNumber", "firstName"],
    ),
    "SendPollNode": NodeTypeSpec(
        type="SendPollNode",
        category="send",
        description="Poll yuborish",
        required_data=["question", "options"],
    ),
    "SendDiceNode": NodeTypeSpec(
        type="SendDiceNode",
        category="send",
        description="Dice/emoji yuborish",
        optional_data=["emoji"],
    ),
    "SendMediaGroupNode": NodeTypeSpec(
        type="SendMediaGroupNode",
        category="send",
        description="Media group (album) yuborish",
        required_data=["media"],
    ),
    "EditOrSendMessageNode": NodeTypeSpec(
        type="EditOrSendMessageNode",
        category="send",
        description="Mavjud bo'lsa edit qiladi, aks holda yangisini yuboradi",
        required_data=["messageText"],
        optional_data=["targetMessageNodeId", "keyboard"],
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
            "source handle orqali tarmoqlanadi."
        ),
        required_data=["jsCode"],
        optional_data=["outputs"],
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
        required_data=["chatId"],
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
            "Xato: errorOutputKey (standart: agent_error). "
            "Bog'lash: flow.connect(source=<provider_id>, source_handle='source-default', "
            "target=<agent_id>, target_handle='acc:ai_model'). "
            "Tavsiya: OpenAI yoki Ollama provider'i bilan boshlang."
        ),
        required_data=[],
        optional_data=[
            "systemPrompt", "userInputSource", "userInputText", "outputKey",
            "maxIterations", "temperature", "maxTokens", "returnMode",
            "historyKey", "errorOutputKey",
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
            "provider='openai', default model='gpt-4o-mini'. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_model'."
        ),
        required_data=[],
        optional_data=["provider", "apiKey", "model", "baseURL", "temperature", "maxTokens", "extraHeaders", "extraParams"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIOllamaProviderNode": NodeTypeSpec(
        type="AIOllamaProviderNode",
        category="ai",
        description=(
            "Ollama provider sub-node. AIAgentNode'ning acc:ai_model portiga ulanadi. "
            "provider='ollama', default model='llama3.1', baseURL='http://localhost:11434'. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_model'."
        ),
        required_data=[],
        optional_data=["provider", "apiKey", "model", "baseURL", "temperature", "maxTokens", "extraHeaders", "extraParams"],
        source_handles=["source-default"],
        target_handles=[],
        is_accessory=True,
    ),
    "AIGeminiProviderNode": NodeTypeSpec(
        type="AIGeminiProviderNode",
        category="ai",
        description=(
            "Google Gemini provider sub-node. AIAgentNode'ning acc:ai_model portiga ulanadi. "
            "provider='gemini', default model='gemini-1.5-flash'. "
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_model'."
        ),
        required_data=[],
        optional_data=["provider", "apiKey", "model", "baseURL", "temperature", "maxTokens", "extraHeaders", "extraParams"],
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
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_model'."
        ),
        required_data=[],
        optional_data=["provider", "apiKey", "model", "baseURL", "temperature", "maxTokens", "extraHeaders", "extraParams"],
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
            "Edge: source=<this_id>, sourceHandle='source-default', "
            "target=<agent_id>, targetHandle='acc:ai_tool'."
        ),
        required_data=[],
        optional_data=[
            "tool", "toolName", "description", "parametersSchema",
            "method", "urlTemplate", "headers", "bodyTemplate", "timeout",
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
            "searchFields — LLM filtrlay oladigan maydonlar (ixtiyoriy). "
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
            "embeddingProvider: openai|ollama|gemini. "
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
