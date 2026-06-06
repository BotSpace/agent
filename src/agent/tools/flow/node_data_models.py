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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


_PAYLOAD_FILTER = Literal["any", "equals", "starts_with", "contains"]


class PreCheckoutTriggerData(_StrictData):
    filter: _PAYLOAD_FILTER = Field(default="any",
        description="Invoice payload filter mode.")
    value: str | None = Field(default=None,
        description="Payload filter value (unused when filter='any').")
    autoAnswer: bool = Field(default=True,
        description="Auto answerPreCheckoutQuery(ok=true). Telegram cancels the "
        "payment if unanswered within 10s — keep True unless answering manually.")
    global_: bool = Field(default=True, alias="global",
        description="Always-global trigger.")


class SuccessfulPaymentTriggerData(_StrictData):
    filter: _PAYLOAD_FILTER = Field(default="any",
        description="Invoice payload filter mode.")
    value: str | None = Field(default=None,
        description="Payload filter value (unused when filter='any').")
    global_: bool = Field(default=True, alias="global",
        description="Always-global trigger.")


class ChatJoinRequestTriggerData(_StrictData):
    global_: bool = Field(default=True, alias="global",
        description="Always-global trigger; fires on any chat join request.")


# ═════════════════════════════════════════════════════════════════════════════
# SEND (16)
# ═════════════════════════════════════════════════════════════════════════════


_PARSE_MODE = Literal["HTML", "Markdown", "MarkdownV2"]


class _SendBase(_StrictData):
    """Common optional fields for most send_* nodes."""

    keyboard_kind: Literal["inline", "reply"] | None = Field(
        default=None,
        description="'inline' (default) or 'reply'. Set alongside rows.",
    )
    rows: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Button rows. EACH row is an object {columns: [button, ...]} — NOT a bare array. "
            "Button: {text, type:'callback', value:'cb_data'}. "
            "URL: {text, type:'url', value:'https://...'}. "
            "Contact/location: {text, type:'request_contact'} | {text, type:'request_location'}. "
            'Example: [{"columns":[{"text":"📚 Kurslar","type":"callback","value":"courses"}]}]'
        ),
    )
    recipients: dict[str, Any] | None = Field(default=None,
        description="{sendToSender:bool, additionalIds:[chat_ids]} — admin/group broadcasts.")
    remove_keyboard: bool | None = Field(
        default=None,
        description=(
            "true = remove the user's reply keyboard (Telegram ReplyKeyboardRemove) when "
            "sending this message. Only applies when the node itself has NO keyboard; if "
            "keyboard_kind/rows are set, the keyboard wins. Ignored on edit (EditOrSend "
            "edit branch) — Telegram only allows it on send."
        ),
    )


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


class InvoicePrice(_StrictData):
    label: str = Field(description="Price line label, e.g. 'Asosiy narx'.")
    amount: int = Field(description="Amount in smallest currency unit (XTR = stars count).")


class SendInvoiceData(_StrictData):
    title: str = Field(description="Product title. {{var}} supported.")
    description: str = Field(description="Product description.")
    payload: str = Field(
        description="Internal invoice id — match it in PreCheckout/SuccessfulPayment triggers.")
    providerToken: str | None = Field(default=None,
        description="Payment provider token. Empty/None = Telegram Stars (XTR).")
    currency: str = Field(default="XTR",
        description="ISO currency or 'XTR' for Telegram Stars.")
    prices: list[InvoicePrice] = Field(description="At least one price line.")
    photoUrl: str | None = Field(default=None, description="Optional product photo URL.")
    needName: bool = Field(default=False)
    needPhone: bool = Field(default=False)
    needEmail: bool = Field(default=False)
    needShippingAddress: bool = Field(default=False)


class EditOrSendMessageData(_SendBase):
    messageText: str
    parseMode: _PARSE_MODE | None = None
    targetMessageNodeId: str | None = Field(default=None,
        description="If set, edit that node's previous message. Else send new.")


# ═════════════════════════════════════════════════════════════════════════════
# EDIT (7)
# ═════════════════════════════════════════════════════════════════════════════


class EditMessageData(_StrictData):
    newText: str
    messageIdSource: Literal["this_message", "previous_node", "state_key"] | None = None
    keyboard_kind: Literal["inline", "reply"] | None = None
    rows: list[dict[str, Any]] | None = None  # [{columns:[btn,...]}, ...]
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
    jsCode: str = Field(
        description=(
            "JavaScript executed by Goja VM. Stored under data.jsCode — this exact "
            "key is read by the engine and rendered by the constructor UI. See prompt "
            "Pattern 7 for the full runtime API (send_*, db_*, http_*, exit, set_context, ...)."
        )
    )
    outputs: list[dict[str, str]] = Field(
        default_factory=lambda: [{"name": "default"}],
        description=(
            "Named exit branches routed via exit('NAME'). Each becomes a source "
            "handle id='output-NAME'. ALWAYS provide this. "
            "Single linear path: [{'name':'default'}]. "
            "Multiple branches: [{'name':'ok'}, {'name':'error'}] + exit('ok')/exit('error') in code. "
            "Edges MUST reference one of these names or the validator will reject them."
        ),
    )


class SendToAdminData(_StrictData):
    messageText: str = Field(description="Message to send to the admin/group chat.")


class MembershipChannel(_StrictData):
    value: str = Field(
        default="",
        description="Kanal @username (masalan '@my_channel') yoki bo'sh (id ishlatilsa).",
    )
    id: int | None = Field(
        default=None,
        description="Raqamli kanal id (masalan -1001234567890). value bo'lsa shart emas.",
    )


class CheckMembershipData(_StrictData):
    """Engine `channels: [{value|id}]` formatini o'qiydi.

    DIQQAT: `chatId` legacy alias — eski flow'lar uchun qabul qilinadi, lekin
    YANGI node'larda channels ishlating; channels/chatId ikkalasi ham bo'lmasa
    engine HAMMANI a'zo deb o'tkazib yuboradi.
    """

    channels: list[MembershipChannel] | None = Field(
        default=None,
        description="Tekshiriladigan kanallar ro'yxati: [{value: '@channel'}].",
    )
    channelsCollection: str | None = Field(
        default=None,
        description=(
            "Kanallarni DINAMIK olish: collection nomi (masalan 'channels'). "
            "Admin kanallarni collection orqali boshqaradi — flow'ni tahrirlamasdan. "
            "Har hujjatdagi channelsField (default 'value') ustuni @username yoki raqamli id."
        ),
    )
    channelsField: str | None = Field(
        default=None,
        description="channelsCollection hujjatlaridagi kanal ustuni nomi (default: 'value').",
    )
    chatId: str | None = Field(
        default=None,
        description="LEGACY alias (bitta kanal @username yoki id). Yangi flow'da channels ishlating.",
    )

    @model_validator(mode="after")
    def _at_least_one_channel(self) -> CheckMembershipData:
        if not self.channels and not self.channelsCollection and not self.chatId:
            raise ValueError(
                "CheckMembershipNode kamida bitta kanal manbai talab qiladi: "
                "channels: [{value: '@channel'}] yoki channelsCollection bering — "
                "aks holda engine HAMMANI a'zo deb o'tkazib yuboradi."
            )
        return self


class DownloadData(_StrictData):
    fileId: str = Field(description="Telegram file_id or {{state.file_id}}.")
    saveAs: str = Field(description="State key for the local file path / URL.")


class CronData(_StrictData):
    cronExpression: str


class ApproveChatJoinRequestData(_StrictData):
    """Guruh/kanalga qo'shilish so'rovini tasdiqlash.

    chatId/userId bo'sh bo'lsa, ChatJoinRequestTriggerNode bergan context'dan
    olinadi (join_request_chat_id / join_request_user_id).
    """

    chatId: str | int | None = Field(
        default=None,
        description="Chat id yoki {{var}}. Bo'sh bo'lsa context: join_request_chat_id.",
    )
    userId: str | int | None = Field(
        default=None,
        description="User id yoki {{var}}. Bo'sh bo'lsa context: join_request_user_id.",
    )


class DeclineChatJoinRequestData(_StrictData):
    """Guruh/kanalga qo'shilish so'rovini rad etish.

    chatId/userId bo'sh bo'lsa, ChatJoinRequestTriggerNode bergan context'dan
    olinadi (join_request_chat_id / join_request_user_id).
    """

    chatId: str | int | None = Field(
        default=None,
        description="Chat id yoki {{var}}. Bo'sh bo'lsa context: join_request_chat_id.",
    )
    userId: str | int | None = Field(
        default=None,
        description="User id yoki {{var}}. Bo'sh bo'lsa context: join_request_user_id.",
    )


class PaymentNodeData(_StrictData):
    """Botspace orqali to'lov (Click). Egasi to'lov tizimi bilan o'zi integratsiya qilmaydi.

    To'lov tasdiqlansa `paid`, aks holda `failed` source handle bo'yicha davom etadi.
    To'lovdan keyin context: payment_order_id, payment_amount, payment_net,
    payment_status, payment_comment, payment_provider.
    """

    paymentType: Literal["click"] = Field(
        default="click",
        description="To'lov provayderi. Hozircha faqat 'click' (kelajakda payme/uzum).",
    )
    amount: str | int = Field(
        description="To'lov summasi — son yoki {{var}} (masalan {{state.total}}).",
    )
    comment: str | None = Field(
        default=None,
        description="To'lov izohi. {{}} o'zgaruvchilar qo'llab-quvvatlanadi.",
    )
    buttonText: str = Field(
        default="💳 To'lash",
        description="Mijozga ko'rinadigan 'To'lash' tugmasi matni.",
    )


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
# AI (8) — AIAgentNode + 4 providers + 3 tools
# ═════════════════════════════════════════════════════════════════════════════


class AIAgentData(_StrictData):
    """Flow node: attach one provider to acc:ai_model and tools to acc:ai_tool."""

    systemPrompt: str = Field(default="", description="System prompt. Supports {{var}}/$var substitutions.")
    userInputSource: str = Field(default="message.text", description="Context path read for the user turn.")
    userInputText: str = Field(default="", description="Literal/templated user turn override; if non-empty, used instead of userInputSource.")
    outputKey: str = Field(default="agent_output", description="Context key the final text is written to. Read as {{agent_output}}.")
    maxIterations: int = Field(default=5, ge=1, le=10, description="Max tool-call loop iterations (hard cap 10).")
    temperature: float | None = Field(default=None, description="Optional per-agent temperature override.")
    maxTokens: int | None = Field(default=None, description="Optional per-agent response cap.")
    returnMode: Literal["text", "json"] = Field(default="text", description="'text' returns plain string; 'json' also stores parsed JSON under outputKey_json.")
    historyKey: str = Field(default="", description="Optional context key holding prior messages[] to seed the conversation.")
    errorOutputKey: str = Field(default="agent_error", description="Context key for error string on failure. Branch on this with IfConditionNode.")


class _AIProviderBase(_StrictData):
    """Shared fields for all provider sub-nodes (accessory, attach to acc:ai_model)."""

    apiKey: str = Field(default="", description="API key. May be {{env.OPENAI_KEY}}. Never logged.")
    model: str = Field(default="", description="Model id. Leave empty to use node-type default.")
    baseURL: str = Field(default="", description="Override endpoint URL. Leave empty for default.")
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    maxTokens: int | None = Field(default=None, description="Max response tokens.")
    extraHeaders: dict[str, str] = Field(default_factory=dict, description="Optional raw header map.")
    extraParams: dict[str, Any] = Field(default_factory=dict, description="Provider-specific passthrough (e.g. top_p).")


class AIOpenAIProviderData(_AIProviderBase):
    """OpenAI provider. provider='openai', default model='gpt-4o-mini', baseURL='https://api.openai.com/v1'."""

    provider: Literal["openai"] = "openai"


class AIOllamaProviderData(_AIProviderBase):
    """Ollama provider. provider='ollama', baseURL='https://ollama.com/v1' (cloud). Always ask user for model name."""

    provider: Literal["ollama"] = "ollama"
    baseURL: str = Field(default="https://ollama.com/v1", description="Ollama endpoint. Cloud: https://ollama.com/v1")


class AIGeminiProviderData(_AIProviderBase):
    """Gemini provider. provider='gemini', default model='gemini-1.5-flash'."""

    provider: Literal["gemini"] = "gemini"


class AIAnthropicProviderData(_AIProviderBase):
    """Anthropic provider. provider='anthropic', default model='claude-3-5-haiku-latest'."""

    provider: Literal["anthropic"] = "anthropic"


class _AIToolBase(_StrictData):
    """Shared fields for all tool sub-nodes (accessory, attach to acc:ai_tool)."""

    toolName: str = Field(default="", description="Function name given to the LLM (^[a-zA-Z0-9_]+$).")
    description: str = Field(default="", description="Natural-language tool description for the LLM.")
    parametersSchema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="JSON-Schema {type:'object', properties, required} the LLM fills.",
    )


class AIHTTPToolData(_AIToolBase):
    """HTTP request tool. tool='http_request'. urlTemplate/bodyTemplate templated with tool args."""

    tool: Literal["http_request"] = "http_request"
    method: str = Field(default="GET", description="HTTP method. '{{from_args}}' lets the LLM choose.")
    urlTemplate: str = Field(default="", description="URL template. Supports tool args, e.g. https://api.x/{{id}}.")
    headers: dict[str, str] = Field(default_factory=dict, description="Static request headers.")
    bodyTemplate: str = Field(default="", description="Request body template.")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds.")


class AICollectionToolData(_AIToolBase):
    """Collection CRUD tool. tool='collection'. Full CRUD on a bot collection.

    The LLM gets a single tool that can create/read/update/delete documents in
    `collection`, restricted to the `operations` allow-list. CROSS-STREAM
    CONTRACT with constructor + engine (AICollectionToolNode) — keep keys exact.
    """

    tool: Literal["collection"] = "collection"
    toolName: str = Field(default="collection", description="Function name given to the LLM (^[a-zA-Z0-9_]+$).")
    collection: str = Field(default="", description="Bot collection name the tool operates on.")
    operations: list[str] = Field(
        default_factory=lambda: ["create", "read", "update", "delete"],
        description="Allow-list of CRUD operations the LLM may perform on the collection.",
    )
    searchFields: list[dict[str, str]] | None = Field(
        default=None,
        description="Optional fields the LLM can filter/search by, e.g. [{'name':'email','type':'string'}].",
    )


class AICustomCodeToolData(_AIToolBase):
    """Custom JavaScript tool. tool='custom_code'. Goja sandbox; args + flow context available."""

    tool: Literal["custom_code"] = "custom_code"
    jsCode: str = Field(
        default="// return value becomes the tool result",
        description="JavaScript. return value (or exit(value)) is the tool result. args and flow context available.",
    )


class AIKnowledgeToolData(_StrictData):
    """RAG knowledge retrieval tool. tool='knowledge'.

    Embeds the bot collection into a vector DB (lazy, on first call) and lets the
    agent retrieve semantically relevant documents by passing a plain-text `query`.
    CROSS-STREAM CONTRACT with constructor + engine (AIKnowledgeToolNode) — keep keys exact.
    """

    tool: Literal["knowledge"] = "knowledge"
    toolName: str = Field(default="knowledge", description="Function name given to the LLM (^[a-zA-Z0-9_]+$).")
    collection: str = Field(description="Bot collection to index and search over.")
    textFields: list[str] | None = Field(
        default=None,
        description="Collection fields to embed as text. None = all string fields.",
    )
    embeddingProvider: str = Field(
        default="openai",
        description="Embedding provider: openai | ollama | gemini.",
    )
    embeddingModel: str = Field(
        default="text-embedding-3-small",
        description="Embedding model id for the chosen provider.",
    )
    embeddingApiKey: str = Field(default="", description="API key for the embedding provider. May be {{env.KEY}}.")
    embeddingBaseURL: str = Field(default="", description="Override endpoint URL for the embedding provider.")
    vectorDbUrl: str = Field(default="", description="Vector DB connection URL (e.g. qdrant/weaviate). Empty = built-in.")
    vectorDbApiKey: str = Field(default="", description="API key for the external vector DB.")
    topK: int = Field(default=5, ge=1, description="Number of top documents to return per query.")


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
    "PreCheckoutTriggerNode": PreCheckoutTriggerData,
    "SuccessfulPaymentTriggerNode": SuccessfulPaymentTriggerData,
    "ChatJoinRequestTriggerNode": ChatJoinRequestTriggerData,
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
    "SendInvoiceNode": SendInvoiceData,
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
    "ApproveChatJoinRequestNode": ApproveChatJoinRequestData,
    "DeclineChatJoinRequestNode": DeclineChatJoinRequestData,
    "PaymentNode": PaymentNodeData,
    # interaction
    "ChatActionNode": ChatActionData,
    "CallbackQueryAnswerNode": CallbackQueryAnswerData,
    "AnswerInlineQueryNode": AnswerInlineQueryData,
    # subflow
    "SubFlowNode": SubFlowData,
    # ai — agent node + providers + tools
    "AIAgentNode": AIAgentData,
    "AIOpenAIProviderNode": AIOpenAIProviderData,
    "AIOllamaProviderNode": AIOllamaProviderData,
    "AIGeminiProviderNode": AIGeminiProviderData,
    "AIAnthropicProviderNode": AIAnthropicProviderData,
    "AIHTTPToolNode": AIHTTPToolData,
    "AICollectionToolNode": AICollectionToolData,
    "AICustomCodeToolNode": AICustomCodeToolData,
    "AIKnowledgeToolNode": AIKnowledgeToolData,
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
    for cat in ("trigger", "send", "edit", "flow_control", "data", "integration", "interaction", "subflow", "ai"):
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
    except Exception:
        return None


def _format_type(annot: Any) -> str:
    """Render a Pydantic field annotation as compact text for docs."""
    try:
        s = str(annot)
    except Exception:
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
