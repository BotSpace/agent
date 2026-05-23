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


NODE_TYPES: dict[str, NodeTypeSpec] = {
    # ── Triggers ────────────────────────────────────────────────────────
    "CommandTriggerNode": NodeTypeSpec(
        type="CommandTriggerNode",
        category="trigger",
        description="Telegram /command bilan boshlanadigan trigger",
        required_data=["command"],
        optional_data=["state"],
        is_trigger=True,
    ),
    "MessageTriggerNode": NodeTypeSpec(
        type="MessageTriggerNode",
        category="trigger",
        description="Foydalanuvchi xabar yuborganida ishga tushadi (text/photo/contact/location/file)",
        optional_data=["messageType", "state"],
        is_trigger=True,
    ),
    "CallbackQueryTriggerNode": NodeTypeSpec(
        type="CallbackQueryTriggerNode",
        category="trigger",
        description="Inline button callback bo'yicha trigger",
        required_data=["callbackData"],
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
        description="Reply keyboard button tomonidan trigger",
        required_data=["buttonText"],
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
        description="Text xabar yuborish, keyboard bilan",
        required_data=["messageText"],
        optional_data=["parseMode", "keyboard", "disableWebPagePreview", "replyToMessageId"],
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
        description="Avvalgi xabarni edit qilish",
        required_data=["messageText"],
        optional_data=["targetMessageNodeId"],
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
        description="Multi-branch shart. Har branch handle'i 'branch-0', 'branch-1', ...",
        required_data=["branches"],
        source_handles=["branch-0", "branch-1", "else"],
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
        description="Massiv ustidan iteration. 'item' va 'index' context variables.",
        required_data=["array"],
        source_handles=["iteration", "done"],
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
        description="Trigger ma'lumotini state'ga saqlash (text, callback, file_id, ...)",
        required_data=["key", "type"],
    ),
    # ── Collection (CMS) ───────────────────────────────────────────────
    "CollectionNode": NodeTypeSpec(
        type="CollectionNode",
        category="data",
        description="MongoDB collection ga document qo'shish",
        required_data=["collection_name", "fieldMappings"],
    ),
    "LoadCollectionItemNode": NodeTypeSpec(
        type="LoadCollectionItemNode",
        category="data",
        description="Bitta document yuklash (filter bo'yicha)",
        required_data=["collection_name", "filter"],
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
        description="JavaScript kod ishga tushirish (Goja VM). MongoDB/HTTP async API mavjud.",
        required_data=["code"],
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
        source_handles=["member", "not_member"],
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
    for cat in ("trigger", "send", "edit", "flow_control", "data", "integration", "interaction", "subflow"):
        items = categories.get(cat, [])
        if not items:
            continue
        lines.append(f"\n[{cat.upper()}]")
        for spec in items:
            req = ", ".join(spec.required_data) if spec.required_data else "—"
            lines.append(f"  {spec.type}: {spec.description}  required: {req}")
    return "\n".join(lines)
