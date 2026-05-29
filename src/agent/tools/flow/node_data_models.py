"""Per-node-type strict Pydantic data validators.

Har node turi uchun alohida `*Data` Pydantic model. `flow.add_node` chaqirilganda
NODE_DATA_MODELS dan kerakli model tanlanadi va `input.data` aniq tekshiriladi:
- Required field yo'q bo'lsa — qaysi field yo'qligi aniq aytiladi
- Tip noto'g'ri bo'lsa — kutilgan vs olingan tip aytiladi
- Enum noto'g'ri qiymat — ruxsat etilgan qiymatlar ko'rsatiladi
- Noma'lum field — ruxsat etilgan ro'yxat ko'rsatiladi

Engine catalog'idagi NodeTypeSpec.required_data / optional_data bilan
mos ravishda sinxron bo'lishi kerak.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictData(BaseModel):
    """`extra='forbid'` noma'lum fieldlarni rad etadi.

    `populate_by_name=True` — alias yoki original ismdan birini ishlatish mumkin
    (masalan `global` reserved word — `global_` field alias='global' bilan).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER (7)
# ═════════════════════════════════════════════════════════════════════════════


class CommandTriggerData(_StrictData):
    command: str = Field(description="Bot command, e.g. '/start' or '/help'.")
    global_: bool = Field(default=True, alias="global",
        description="True: fires anywhere. False: only while engine paused.")
    withArgs: bool = Field(default=False, description="Capture text after the command.")
    state: dict[str, str] | None = Field(default=None,
        description="Optional {key, type} to auto-save command args into state.")


class MessageTriggerData(_StrictData):
    type: Literal[
        "text", "photo", "video", "audio", "voice", "video_note",
        "document", "contact", "location", "sticker", "animation",
    ] = Field(description="Expected message media type.")
    filter: str = Field(default="any",
        description="'any' | 'starts_with' | 'equals' | regex.")
    context: dict[str, Any] | None = None
    global_: bool = Field(default=False, alias="global")
    state: dict[str, str] | None = Field(default=None,
        description="Auto-save: {key, type: 'text'|'phone_number'|'file_id'|...}.")


class CallbackQueryTriggerData(_StrictData):
    filter: Literal["any", "starts_with", "equals", "ends_with", "contains"] = "any"
    context: dict[str, Any] | None = Field(default=None,
        description="e.g. {value: 'accept:'} when filter='starts_with'.")
    global_: bool = Field(default=True, alias="global")
    state: dict[str, str] | None = None


class CallbackButtonTriggerData(_StrictData):
    pass  # No required/optional in catalog


class ReplyButtonTriggerData(_StrictData):
    selectedButton: str = Field(description="Exact reply button text to match.")
    catchNoButton: bool = Field(default=False,
        description="If True, also fires when user typed something other than a button.")


class CronTriggerData(_StrictData):
    cronExpression: str = Field(
        description="Cron expression (5 fields: m h d m w). E.g. '0 9 * * 1' = every Monday 9am.")


class ExternalWebhookTriggerData(_StrictData):
    provider: str = Field(description="External webhook provider identifier.")


# ═════════════════════════════════════════════════════════════════════════════
# SEND (16)
# ═════════════════════════════════════════════════════════════════════════════


_PARSE_MODE = Literal["HTML", "Markdown", "MarkdownV2"]


class _SendBase(_StrictData):
    """Common optional fields for most send_* nodes."""

    keyboard: dict[str, Any] | None = Field(default=None,
        description="{active:'reply'|'inline', reply:[[{id,text,type}...]] | inline:[[...]]}.")
    recipients: dict[str, Any] | None = Field(default=None,
        description="{sendToSender:bool, additionalIds:[chat_ids]} — admin/group broadcasts.")


class SendTextMessageData(_SendBase):
    messageText: str = Field(description="Message text. Supports {{var}} substitutions.")
    parseMode: _PARSE_MODE | None = None
    disableWebPagePreview: bool = False


class SendPhotoData(_SendBase):
    photo: str = Field(description="Photo URL or {{file_id}}.")
    caption: str | None = None
    parseMode: _PARSE_MODE | None = None


class SendVideoData(_SendBase):
    video: str = Field(description="Video URL or {{file_id}}.")
    caption: str | None = None
    parseMode: _PARSE_MODE | None = None


class SendAudioData(_SendBase):
    audio: str = Field(description="Audio URL or {{file_id}}.")
    caption: str | None = None
    title: str | None = None
    performer: str | None = None


class SendFileData(_SendBase):
    file: str = Field(description="Document/file URL or {{file_id}}.")
    caption: str | None = None
    parseMode: _PARSE_MODE | None = None


class SendAnimationData(_SendBase):
    animation: str = Field(description="GIF/animation URL or {{file_id}}.")


class SendVoiceData(_SendBase):
    voice: str = Field(description="Voice OGG URL or {{file_id}}.")


class SendVideoNoteData(_SendBase):
    videoNote: str = Field(description="Round video URL or {{file_id}}.")


class SendStickerData(_SendBase):
    sticker: str = Field(description="Sticker file_id or pack id.")


class SendLocationData(_SendBase):
    latitude: float
    longitude: float


class SendVenueData(_SendBase):
    latitude: float
    longitude: float
    title: str
    address: str


class SendContactData(_SendBase):
    phoneNumber: str
    firstName: str


class SendPollData(_SendBase):
    question: str
    options: list[str] = Field(description="2-10 poll option strings.")


class SendDiceData(_SendBase):
    emoji: Literal["🎲", "🎯", "🏀", "⚽", "🎰", "🎳"] | None = None


class SendMediaGroupData(_SendBase):
    media: list[dict[str, Any]] = Field(
        description="Array of {type, media, caption?}. 2-10 items.")


class EditOrSendMessageData(_SendBase):
    messageText: str
    targetMessageNodeId: str | None = Field(default=None,
        description="If set, edit that node's previous message. Else send new.")


# ═════════════════════════════════════════════════════════════════════════════
# EDIT (7)
# ═════════════════════════════════════════════════════════════════════════════


class EditMessageData(_StrictData):
    newText: str
    messageIdSource: Literal["this_message", "previous_node", "state_key"] | None = None
    keyboard: dict[str, Any] | None = None
    parseMode: _PARSE_MODE | None = None
    editType: Literal["text", "caption", "media"] | None = None


class DeleteMessageData(_StrictData):
    targetMessageNodeId: str | None = Field(default=None,
        description="Node whose sent message should be deleted. None = current callback message.")


class ForwardMessageData(_StrictData):
    targetChatId: str = Field(description="Destination chat id ('@channel' or '-100...').")


class CopyMessageData(_StrictData):
    targetChatId: str


class PinMessageData(_StrictData):
    pass


class UnpinMessageData(_StrictData):
    pass


class UnpinAllMessagesData(_StrictData):
    pass


# ═════════════════════════════════════════════════════════════════════════════
# FLOW_CONTROL (8)
# ═════════════════════════════════════════════════════════════════════════════


class IfConditionData(_StrictData):
    conditions: list[dict[str, Any]] = Field(
        description="[{left, operator, right}, ...]. Operators: equals, not_equals, gt, lt, contains.")
    branches: list[dict[str, Any]] | None = Field(default=None,
        description="Optional multi-branch: each {label, conditions:[...]}.")
    operator: Literal["and", "or"] = "and"


class ElseIfData(_StrictData):
    pass


class ElseData(_StrictData):
    pass


class DelayData(_StrictData):
    seconds: int = Field(ge=0, description="Wait time in seconds.")


class SchedulerData(_StrictData):
    scheduledAt: str = Field(description="ISO 8601 datetime or {{state_key}} expression.")


class RandomData(_StrictData):
    weights: list[int] = Field(min_length=2, description="Weight for each branch (option_0, option_1, ...).")


class ForLoopData(_StrictData):
    array: str = Field(description="Array expression like {{state.users}} or {{collection.X}}.")


class ForLoopContinueData(_StrictData):
    pass


# ═════════════════════════════════════════════════════════════════════════════
# DATA (7)
# ═════════════════════════════════════════════════════════════════════════════


class VariableData(_StrictData):
    name: str = Field(description="Variable name (no {{}} braces).")
    operation: Literal["set", "increment", "decrement", "append", "remove", "clear"]
    value: Any | None = Field(default=None,
        description="Value for set/increment/append. Skipped for clear.")


class StateData(_StrictData):
    key: str
    type: Literal[
        "text", "first_name", "last_name", "username", "user_id",
        "language_code", "Custom",
    ]
    customType: str | None = Field(default=None,
        description="Required when type='Custom'. Expression like {{inserted_id}}.")


class CollectionData(_StrictData):
    collection_name: str
    fieldMappings: dict[str, str] = Field(
        description="{collection_field: state_key}, e.g. {name:'user_name', phone:'phone'}.")


class LoadCollectionItemData(_StrictData):
    collection: dict[str, Any] = Field(description="{name: 'collection_name'}.")
    contextKey: str = Field(description="State key under which the loaded doc is stored.")
    filters: list[dict[str, Any]] = Field(
        description="[{field, operator, valueSource, value|stateKey}, ...].")


class LoadCollectionListData(_StrictData):
    collection_name: str
    filter: dict[str, Any] | None = None
    sort: dict[str, Any] | None = None
    limit: int | None = Field(default=None, ge=1)


class UpdateCollectionData(_StrictData):
    collection_name: str
    filter: dict[str, Any]
    update: dict[str, Any]


class DeleteCollectionData(_StrictData):
    collection_name: str
    filter: dict[str, Any]


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION (6)
# ═════════════════════════════════════════════════════════════════════════════


class HTTPRequestData(_StrictData):
    url: str = Field(description="Full URL. Supports {{var}}.")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    headers: dict[str, str] | None = None
    body: dict[str, Any] | str | None = None
    saveAs: str | None = Field(default=None,
        description="State key under which response.json is saved.")


class CustomCodeData(_StrictData):
    code: str = Field(
        description=(
            "JavaScript executed by Goja VM. See prompt Pattern 7 for the full "
            "runtime API (send_*, db_*, http_*, exit, set_context, ...)."
        )
    )
    outputs: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Optional named exits routed via exit('NAME'). Each becomes a "
            "source handle id='output-NAME'. Format: [{'name':'ok'}, ...]."
        ),
    )


class SendToAdminData(_StrictData):
    messageText: str = Field(description="Message to send to the admin/group chat.")


class CheckMembershipData(_StrictData):
    chatId: str = Field(description="Chat/channel id to check membership against.")


class DownloadData(_StrictData):
    fileId: str = Field(description="Telegram file_id or {{state.file_id}}.")
    saveAs: str = Field(description="State key for the local file path / URL.")


class CronData(_StrictData):
    cronExpression: str


# ═════════════════════════════════════════════════════════════════════════════
# INTERACTION (3)
# ═════════════════════════════════════════════════════════════════════════════


class ChatActionData(_StrictData):
    action: Literal[
        "typing", "upload_photo", "upload_video", "record_video", "upload_voice",
        "record_voice", "upload_document", "upload_video_note", "record_video_note",
        "find_location",
    ]


class CallbackQueryAnswerData(_StrictData):
    text: str | None = None
    showAlert: bool = False


class AnswerInlineQueryData(_StrictData):
    results: list[dict[str, Any]] = Field(description="Inline query result objects.")


# ═════════════════════════════════════════════════════════════════════════════
# SUBFLOW (1)
# ═════════════════════════════════════════════════════════════════════════════


class SubFlowData(_StrictData):
    subflowId: str = Field(description="Id of the subflow to invoke.")


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

NODE_DATA_MODELS: dict[str, type[_StrictData]] = {
    # trigger
    "CommandTriggerNode": CommandTriggerData,
    "MessageTriggerNode": MessageTriggerData,
    "CallbackQueryTriggerNode": CallbackQueryTriggerData,
    "CallbackButtonTriggerNode": CallbackButtonTriggerData,
    "ReplyButtonTriggerNode": ReplyButtonTriggerData,
    "CronTriggerNode": CronTriggerData,
    "ExternalWebhookTriggerNode": ExternalWebhookTriggerData,
    # send
    "SendTextMessageNode": SendTextMessageData,
    "SendPhotoNode": SendPhotoData,
    "SendVideoNode": SendVideoData,
    "SendAudioNode": SendAudioData,
    "SendFileNode": SendFileData,
    "SendAnimationNode": SendAnimationData,
    "SendVoiceNode": SendVoiceData,
    "SendVideoNoteNode": SendVideoNoteData,
    "SendStickerNode": SendStickerData,
    "SendLocationNode": SendLocationData,
    "SendVenueNode": SendVenueData,
    "SendContactNode": SendContactData,
    "SendPollNode": SendPollData,
    "SendDiceNode": SendDiceData,
    "SendMediaGroupNode": SendMediaGroupData,
    "EditOrSendMessageNode": EditOrSendMessageData,
    # edit
    "EditMessageNode": EditMessageData,
    "DeleteMessageNode": DeleteMessageData,
    "ForwardMessageNode": ForwardMessageData,
    "CopyMessageNode": CopyMessageData,
    "PinMessageNode": PinMessageData,
    "UnpinMessageNode": UnpinMessageData,
    "UnpinAllMessagesNode": UnpinAllMessagesData,
    # flow_control
    "IfConditionNode": IfConditionData,
    "ElseIfNode": ElseIfData,
    "ElseNode": ElseData,
    "DelayNode": DelayData,
    "SchedulerNode": SchedulerData,
    "RandomNode": RandomData,
    "ForLoopNode": ForLoopData,
    "ForLoopContinueNode": ForLoopContinueData,
    # data
    "VariableNode": VariableData,
    "StateNode": StateData,
    "CollectionNode": CollectionData,
    "LoadCollectionItemNode": LoadCollectionItemData,
    "LoadCollectionListNode": LoadCollectionListData,
    "UpdateCollectionNode": UpdateCollectionData,
    "DeleteCollectionNode": DeleteCollectionData,
    # integration
    "HTTPRequestNode": HTTPRequestData,
    "CustomCodeNode": CustomCodeData,
    "SendToAdminNode": SendToAdminData,
    "CheckMembershipNode": CheckMembershipData,
    "DownloadNode": DownloadData,
    "CronNode": CronData,
    # interaction
    "ChatActionNode": ChatActionData,
    "CallbackQueryAnswerNode": CallbackQueryAnswerData,
    "AnswerInlineQueryNode": AnswerInlineQueryData,
    # subflow
    "SubFlowNode": SubFlowData,
}


def get_data_model(node_type: str) -> type[_StrictData] | None:
    return NODE_DATA_MODELS.get(node_type)


def render_params_doc(node_type: str) -> str:
    """Markdown table of params for a single node (for tool responses)."""
    model = NODE_DATA_MODELS.get(node_type)
    if model is None:
        return f"_(no validator for `{node_type}`)_"
    rows: list[str] = []
    for name, info in model.model_fields.items():
        alias = info.alias or name
        annot = info.annotation
        type_str = _format_type(annot)
        req = "yes" if info.is_required() else "no"
        desc = (info.description or "").replace("\n", " ")
        rows.append(f"| `{alias}` | {type_str} | {req} | {desc} |")
    if not rows:
        return f"`{node_type}` — no data fields."
    head = "| field | type | required | description |\n|---|---|---|---|"
    return f"{head}\n" + "\n".join(rows)


def render_node_schema(node_type: str) -> dict[str, Any]:
    """JSON-schema-like dict for one node — used by flow.list_node_types."""
    model = NODE_DATA_MODELS.get(node_type)
    if model is None:
        return {"type": node_type, "fields": [], "validator": False}
    fields = []
    for name, info in model.model_fields.items():
        fields.append({
            "name": info.alias or name,
            "type": _format_type(info.annotation),
            "required": info.is_required(),
            "default": None if info.is_required() else _safe_default(info.default),
            "description": info.description or "",
        })
    return {"type": node_type, "fields": fields, "validator": True}


def render_summary_table() -> str:
    """Compact summary of ALL nodes for the flow-builder system prompt."""
    from ...flow.catalog import NODE_TYPES

    rows_by_cat: dict[str, list[str]] = {}
    for type_name, spec in NODE_TYPES.items():
        model = NODE_DATA_MODELS.get(type_name)
        if model is None:
            req = ", ".join(spec.required_data) or "—"
            opt = ", ".join(spec.optional_data) or "—"
        else:
            req_fields, opt_fields = [], []
            for fname, info in model.model_fields.items():
                disp = f"{info.alias or fname}"
                # type hint with enum values where applicable
                annot = _format_type(info.annotation)
                if "|" in annot and "[" not in annot:  # Literal enums
                    disp = f"{disp}({annot})"
                (req_fields if info.is_required() else opt_fields).append(disp)
            req = ", ".join(req_fields) or "—"
            opt = ", ".join(opt_fields) or "—"
        rows_by_cat.setdefault(spec.category, []).append(
            f"| `{type_name}` | **{req}** | {opt} |"
        )
    parts: list[str] = []
    for cat in ("trigger", "send", "edit", "flow_control", "data", "integration", "interaction", "subflow"):
        if cat not in rows_by_cat:
            continue
        parts.append(f"\n### {cat}")
        parts.append("| node type | required (with enums) | optional |")
        parts.append("|---|---|---|")
        parts.extend(rows_by_cat[cat])
    return "\n".join(parts)


def _safe_default(value: Any) -> Any:
    try:
        # Pydantic's PydanticUndefined → None
        if str(value).startswith("PydanticUndefined"):
            return None
        return value
    except Exception:  # noqa: BLE001
        return None


def _format_type(annot: Any) -> str:
    """Render a Pydantic field annotation as compact text for docs."""
    try:
        s = str(annot)
    except Exception:  # noqa: BLE001
        return "any"
    for prefix in ("typing.", "<class '", "'>", "agent.tools.flow.node_data_models."):
        s = s.replace(prefix, "")
    if "Literal[" in s:
        import re
        s = re.sub(
            r"Literal\[([^\]]+)\]",
            lambda m: m.group(1).replace("'", "").replace(", ", "|"),
            s,
        )
    return s
