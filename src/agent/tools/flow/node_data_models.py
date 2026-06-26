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


# Localizable text — foydalanuvchiga ko'rinadigan matn. Qiymat YO oddiy string,
# YOKI i18n marker obyekt: {"__i18n": true, "en": "Hello", "uz": "Salom",
# "ru": "Привет"}. Engine ishga tushganda markerni aktiv tilga "yassilaydi"
# (node data ichida HAR JOYDA ishlaydi). Ko'p tilli bot uchun marker ishlat.
# Aktiv til: public_lang → Telegram language_code → flow.defaultLanguage → "en"
# (faqat flow.languages ro'yxatidagi tillar tanlanadi). Batafsil: docs_guide('i18n').
LocalizedText = str | dict[str, Any]

_I18N_NOTE = (
    " i18n: oddiy string YOKI marker obyekt "
    '{"__i18n": true, "en": "...", "uz": "...", "ru": "..."} — '
    "ko'p tilli botda marker ishlat (flow.set_languages oldin chaqirilsin)."
)


# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER
# ═════════════════════════════════════════════════════════════════════════════


class CommandTriggerData(_StrictData):
    command: str = Field(description="Bot command, e.g. '/start' or '/help'.")
    global_: bool = Field(default=True, alias="global",
        description="True: fires anywhere. False: only while engine paused.")
    withArgs: bool = Field(default=False, description="Capture text after the command.")
    state: dict[str, str] | None = Field(default=None,
        description="Optional {key, type} to auto-save command args into public state (public_<key>, read as {{key}}).")


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
    """Telegram-native pre_checkout_query trigger — ONLY paired with SendInvoiceNode.

    Chain: SendInvoice -> PreCheckout (autoAnswer) -> SuccessfulPayment. NOT for
    PaymentNode (Click/Paycue is a separate system with paid/failed branches).
    """

    filter: _PAYLOAD_FILTER = Field(default="any",
        description="Invoice payload filter mode (match SendInvoice `payload`).")
    value: str | None = Field(default=None,
        description="Payload filter value (unused when filter='any').")
    autoAnswer: bool = Field(default=True,
        description="Auto answerPreCheckoutQuery(ok=true). Telegram cancels the "
        "payment if unanswered within 10s — keep True unless answering manually.")
    global_: bool = Field(default=True, alias="global",
        description="Always-global trigger.")


class SuccessfulPaymentTriggerData(_StrictData):
    """Telegram-native successful_payment trigger — ONLY paired with SendInvoiceNode.

    Fires when the Telegram payment actually goes through; deliver goods/access here.
    Chain: SendInvoice -> PreCheckout (autoAnswer) -> SuccessfulPayment (HERE). NOT for
    PaymentNode (Click/Paycue returns result via paid/failed branches, no trigger).
    """

    filter: _PAYLOAD_FILTER = Field(default="any",
        description="Invoice payload filter mode (match SendInvoice `payload`).")
    value: str | None = Field(default=None,
        description="Payload filter value (unused when filter='any').")
    global_: bool = Field(default=True, alias="global",
        description="Always-global trigger.")


class PaymentStatusTriggerData(_StrictData):
    status: Literal["any", "paid", "failed"] = Field(default="paid",
        description="Payment callback status to match.")
    orderId: str | None = Field(default=None,
        description="Optional exact Botspace payment order id filter.")
    provider: Literal["", "click", "paycue"] = Field(default="",
        description="Optional provider filter; empty means any.")
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
    messageText: LocalizedText = Field(
        description="Message text. Supports {{var}} substitutions." + _I18N_NOTE)
    parseMode: _PARSE_MODE | None = None
    disableWebPagePreview: bool = False


class SendPhotoData(_SendBase):
    photo: str = Field(description="Photo URL or {{file_id}}.")
    caption: LocalizedText | None = Field(default=None, description="Photo caption." + _I18N_NOTE)
    parseMode: _PARSE_MODE | None = None


class SendVideoData(_SendBase):
    video: str = Field(description="Video URL or {{file_id}}.")
    caption: LocalizedText | None = Field(default=None, description="Video caption." + _I18N_NOTE)
    parseMode: _PARSE_MODE | None = None


class SendAudioData(_SendBase):
    audio: str = Field(description="Audio URL or {{file_id}}.")
    caption: LocalizedText | None = Field(default=None, description="Audio caption." + _I18N_NOTE)
    title: str | None = None
    performer: str | None = None


class SendFileData(_SendBase):
    file: str = Field(description="Document/file URL or {{file_id}}.")
    caption: LocalizedText | None = Field(default=None, description="Document caption." + _I18N_NOTE)
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
    title: LocalizedText = Field(description="Venue title." + _I18N_NOTE)
    address: LocalizedText = Field(description="Venue address." + _I18N_NOTE)


class SendContactData(_SendBase):
    phoneNumber: str
    firstName: str


class SendPollData(_SendBase):
    question: LocalizedText = Field(description="Poll question." + _I18N_NOTE)
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
        description=(
            "Telegram payment provider token. Empty/None = Telegram Stars (XTR). "
            "NEVER invent or assume a provider (e.g. Payme/Click/Stripe) — a non-empty token "
            "only works if the BOT OWNER first connected that provider in @BotFather and gave "
            "you the exact token. If you don't have a real token, leave it EMPTY (Stars). "
            "For ready-to-use in-bot card payments, prefer PaymentNode (Botspace Click/Paycue) "
            "instead of a Telegram invoice."
        ))
    currency: str = Field(default="XTR",
        description="ISO currency or 'XTR' for Telegram Stars.")
    prices: list[InvoicePrice] = Field(description="At least one price line.")
    photoUrl: str | None = Field(default=None, description="Optional product photo URL.")
    needName: bool = Field(default=False)
    needPhone: bool = Field(default=False)
    needEmail: bool = Field(default=False)
    needShippingAddress: bool = Field(default=False)


class EditOrSendMessageData(_SendBase):
    messageText: LocalizedText = Field(description="Message text. {{var}} supported." + _I18N_NOTE)
    parseMode: _PARSE_MODE | None = None
    targetMessageNodeId: str | None = Field(default=None,
        description="If set, edit that node's previous message. Else send new.")


# ═════════════════════════════════════════════════════════════════════════════
# EDIT (7)
# ═════════════════════════════════════════════════════════════════════════════


class EditMessageData(_StrictData):
    newText: LocalizedText = Field(description="New message text." + _I18N_NOTE)
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
    array: str = Field(description="Array expression like {{users}} (or {{public.users}}) or {{collection.X}}.")


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
        description="{collection_field: var_key} (var_key — saqlangan ochiq o'zgaruvchi nomi, prefiksiz), e.g. {name:'user_name', phone:'phone'}.")


class LoadCollectionItemData(_StrictData):
    collection: dict[str, Any] = Field(description="{name: 'collection_name'}.")
    contextKey: str = Field(description="State key under which the loaded doc is stored.")
    filters: list[dict[str, Any]] = Field(
        description="[{field, operator, valueSource, value|stateKey}, ...].")


class LoadCollectionListData(_StrictData):
    # Field names map 1:1 to the engine's LoadCollectionListHandler data keys
    # (internal/engine/handlers/load_collection_list.go). The engine reads
    # `filters`/`sorts` (ARRAYS) and `contextKey` — the previous singular
    # `filter`/`sort` keys were SILENTLY IGNORED at runtime, and `contextKey`
    # was missing entirely so `extra='forbid'` rejected the (correct) field.
    collection_name: str = Field(
        description="Collection to load (engine also accepts collection={name}).")
    contextKey: str = Field(
        description=(
            "REQUIRED output key. Engine writes the loaded array to "
            "context[contextKey] (+ context[contextKey+'_found'] bool and "
            "context[contextKey+'_count'] int). Reference it in the next message "
            "as {{#each <contextKey>}}...{{/each}} / {{^<contextKey>}}bo'sh{{/<contextKey>}}. "
            "Engine alias: resultKey (same effect)."))
    filters: list[dict[str, Any]] | None = Field(
        default=None,
        description="[{field, operator, valueSource, value|stateKey}, ...]. Omit = load all.")
    sorts: list[dict[str, Any]] | None = Field(
        default=None,
        description="[{field, direction}], direction 'asc'|'desc'.")
    limit: int | None = Field(default=None, ge=1,
        description="Max items (engine default 100).")
    skip: int | None = Field(default=None, ge=0,
        description="Items to skip (pagination offset).")


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
            "Pattern 7 for the full runtime API (send_*, db_*, http_*, exit, set_context, ...). "
            "To READ state/context PREFER object-access: vars.<key> (public, maps to "
            "public_<key>; state.<key> is a legacy alias that still works), private.<key> "
            "(secret), and context.<key> (raw key), e.g. vars.user_phone, context.first_name "
            "(get_state/get_context also work, but object-access is shorter/cleaner). "
            "FORMAT CLEANLY — the engine stores your code verbatim (no auto-format), so "
            "write it readable on the FIRST try: one statement per line (never several "
            "joined by ';'), consistent 2-space indent for blocks, spaces around "
            "operators (a = 1, x === y — not a=1), meaningful variable names (movie, "
            "episodes — not m/e). Never emit minified or single-line code. See "
            "docs_guide(topic='custom_code') for a clean-format example to imitate."
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
    terminal: bool = Field(
        default=False,
        description=(
            "Mark this node as a TERMINAL leaf — it produces its own user-facing "
            "output (send_*/edit_* in jsCode) and ends the flow, so its declared "
            "outputs intentionally have NO downstream edges. Set true to silence "
            "the validator's 'unused output / dead-end' warning for leaf nodes."
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
    fileId: str = Field(description="Telegram file_id or {{file_id}} (saqlangan ochiq qiymat).")
    saveAs: str = Field(description="State key for the local file path / URL.")


class CronData(_StrictData):
    cronExpression: str


class _ChatJoinRequestActionData(_StrictData):
    """Guruh/kanalga qo'shilish so'roviga amal (tasdiqlash/rad etish) uchun umumiy maydonlar.

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


class ApproveChatJoinRequestData(_ChatJoinRequestActionData):
    """Guruh/kanalga qo'shilish so'rovini tasdiqlash."""


class DeclineChatJoinRequestData(_ChatJoinRequestActionData):
    """Guruh/kanalga qo'shilish so'rovini rad etish."""


class CreateInviteLinkData(_StrictData):
    """Kanal/guruh uchun join (invite) havola yaratish (createChatInviteLink).

    Bot maqsadli kanalda admin va "invite via link" huquqiga ega bo'lishi shart.
    Yaratilgan havola context'ga `resultVar` (default `invite_link`) bilan
    yoziladi; xato bo'lsa `invite_link_error`.
    """

    chatId: str | int | None = Field(
        default=None,
        description="Kanal/guruh id yoki {{var}}. Bo'sh bo'lsa joriy chat ishlatiladi.",
    )
    name: str | None = Field(
        default=None,
        description="Havola nomi (faqat admin ko'radi). Ixtiyoriy. {{var}} mumkin.",
    )
    memberLimit: int | str | None = Field(
        default=None,
        description="Havoladan necha kishi foydalana oladi (1..99999). Shaxsiy havola uchun 1.",
    )
    expireDate: int | str | None = Field(
        default=None,
        description="Amal qilish muddati — unix sekund. Ixtiyoriy.",
    )
    createsJoinRequest: bool | None = Field(
        default=None,
        description="True bo'lsa a'zolik admin tasdig'ini talab qiladi.",
    )
    resultVar: str | None = Field(
        default=None,
        description="Havola saqlanadigan context o'zgaruvchisi. Default: invite_link.",
    )


class PaymentNodeData(_StrictData):
    """Botspace orqali to'lov (Click). Egasi to'lov tizimi bilan o'zi integratsiya qilmaydi.

    To'lov tasdiqlansa `paid`, aks holda `failed` source handle bo'yicha davom etadi.
    To'lovdan keyin context: payment_order_id, payment_amount, payment_net,
    payment_status, payment_comment, payment_provider.
    """

    paymentType: Literal["click", "paycue"] = Field(
        default="click",
        description=(
            "To'lov provayderi — qurishdan OLDIN foydalanuvchidan SO'RA "
            "(ask_user select: 'click' yoki 'paycue'). FAQAT 'click' (Click pay-link) yoki "
            "'paycue' (Humo-karta — foydalanuvchi aniq summani kartaga o'tkazadi, avtomatik "
            "tasdiqlanadi). Boshqasini (payme, uzcard...) O'YLAB TOPMA."
        ),
    )
    amount: str | int = Field(
        description="To'lov summasi — son yoki {{var}} (masalan {{total}} yoki {{public.total}}).",
    )
    comment: str | None = Field(
        default=None,
        description="To'lov izohi. {{}} o'zgaruvchilar qo'llab-quvvatlanadi.",
    )
    buttonText: LocalizedText = Field(
        default="💳 To'lash",
        description="Mijozga ko'rinadigan 'To'lash' tugmasi matni." + _I18N_NOTE,
    )


_SHEETS_FILTER_OP = Literal[
    "eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "exists",
]


class SheetsFilterCondition(_StrictData):
    """GoogleSheetsNode read uchun BITTA in-memory filter sharti — STATIK (AI-slot YO'Q).

    Collection tool filteriga o'xshaydi, lekin BU node STATIK: builder field/op/value'ni
    o'zi yozadi (runtime AI to'ldirmaydi — shuning uchun `source`/`aiDescription` YO'Q).
    `value`da {{chat_id}}/{{<key>}} (saqlangan ochiq qiymat) placeholder ishlatish mumkin
    (engine runtime'da almashtiradi; legacy {{state_<key>}} ham ishlaydi).
    `field` — rows obyektidagi ustun nomi (Sheet 1-qator sarlavhasi).
    """

    field: str = Field(description="Filterlanadigan ustun nomi (Sheet 1-qator sarlavhasi).")
    op: _SHEETS_FILTER_OP = Field(
        default="eq",
        description="Taqqoslash operatori: eq|ne|gt|gte|lt|lte|in|nin|contains|exists.",
    )
    value: Any | None = Field(
        default=None,
        description=(
            "Taqqoslanadigan qiymat. {{chat_id}}/{{<key>}} (saqlangan ochiq qiymat) placeholder "
            "bo'lishi mumkin (engine runtime'da almashtiradi; legacy {{state_<key>}} ham). "
            "op='exists' uchun shart emas."
        ),
    )


class GoogleSheetsData(_StrictData):
    """Google Sheets (v4) integratsiyasi — oauth2 credential orqali.

    Credential FAQAT `credential_id` orqali bog'lanadi (type_key 'google_sheets');
    secret flow'da saqlanmaydi, engine runtime'da fresh access_token'ga aylantiradi.
    Natija context[responseVariable] (+ _status, sheets_success) ga yoziladi.

    READ JAVOB SHAKLI (MUHIM — matnda ishlatishda farqi bor):
    - context[responseVariable] (default 'sheets_response') = Google API'ning XOM
      javobi: {"majorDimension": "ROWS", "range": "Sheet1!A1:C10",
      "values": [["nom","narx","tavsif"], ["Olma","5000","..."], ...]}.
      `values` — 2D massiv: BIRINCHI qator sarlavha (kalitlar), qolganlari ma'lumot.
      Bu OBYEKT (nomli field yo'q, array emas) — `{{#each}}` bilan TO'G'RIDAN ISHLAMAYDI.
    - context[rowsVariable] (default '<responseVariable>_rows', ya'ni 'sheets_response_rows')
      = engine tomonidan tayyorlangan TOZA array-of-objects: sarlavha qatori kalit bo'lib
      ishlatiladi → [{"nom":"Olma","narx":"5000","tavsif":"..."}, ...].
      Matn/ro'yxat chiqarish uchun MANA SHUNI ishlat:
      `{{#each sheets_response_rows}}{{nom}} — {{narx}}{{/each}}`.
    QOIDA: ro'yxatni xabarda ko'rsatish uchun XOM responseVariable'ni `{{#each}}` bilan
    ISHLATMA; doim rowsVariable (toza rows) dan foydalan.
    """

    credential_id: int = Field(
        description="type_key 'google_sheets' bo'lgan oauth2 credential id'si.",
    )
    operation: Literal["append", "read", "update", "clear"] = Field(
        default="append",
        description="append: qator qo'shish | read: o'qish | update: yangilash | clear: tozalash.",
    )
    spreadsheet_id: str = Field(
        description="Spreadsheet ID (URL'dagi /d/<ID>/). {{var}} qo'llab-quvvatlanadi.",
    )
    sheet: str | None = Field(
        default=None,
        description=(
            "Sheet/tab nomi (masalan 'Sheet1'). range'da '!' bo'lmasa engine "
            "sheet + range'ni birlashtiradi → 'Sheet1!A1:C10'. {{var}} qo'llanadi."
        ),
    )
    range: str | None = Field(
        default=None,
        description=(
            "A1 range. To'liq 'Sheet1!A1:C10' yoki sheet alohida berilsa faqat "
            "'A1:C10'. sheet ham, range ham bo'sh bo'lmasligi kerak. {{var}} qo'llanadi."
        ),
    )
    column: str | None = Field(
        default=None,
        description="Ixtiyoriy ustun yordamchisi (UI dynamic_select). Engine o'qimaydi — range/sheet yetarli.",
    )
    values: str | None = Field(
        default=None,
        description="JSON 2D massiv, masalan [[\"a\",\"b\"],[\"c\",\"d\"]]. append/update uchun. {{var}} qo'llanadi.",
    )
    responseVariable: str = Field(
        default="sheets_response",
        description=(
            "XOM Google API javobi saqlanadigan context kaliti (default 'sheets_response'). "
            "read'da shakl {majorDimension, range, values} — values 2D massiv, 1-qator "
            "sarlavha. Bu OBYEKT, {{#each}} bilan ISHLAMAYDI; ro'yxat uchun rowsVariable'ni ishlat."
        ),
    )
    rowsVariable: str | None = Field(
        default=None,
        description=(
            "read uchun TOZA rows saqlanadigan context kaliti (ixtiyoriy; default "
            "'<responseVariable>_rows' = 'sheets_response_rows'). Engine values'ning 1-qatorini "
            "sarlavha (kalit) qilib array-of-objects yozadi: [{ustun_nomi: qiymat, ...}, ...]. "
            "Matnda `{{#each sheets_response_rows}}{{ustun_nomi}}{{/each}}` bilan ishlat — "
            "ro'yxat chiqarishda XOM responseVariable o'rniga ALBATTA SHUNI ishlat."
        ),
    )
    filterMatch: Literal["and", "or"] = Field(
        default="and",
        description=(
            "read uchun IN-MEMORY filter: filterConditions qanday birlashadi — "
            "'and' (HAMMA shart) yoki 'or' (KAMIDA bittasi). Default 'and'."
        ),
    )
    filterConditions: list[SheetsFilterCondition] | None = Field(
        default=None,
        description=(
            "read'da sheetdan olingan rowsni IN-MEMORY filterlash (STATIK, AI-slot YO'Q): "
            "[{field, op, value}]. field — ustun nomi (1-qator sarlavhasi), "
            "op ∈ eq|ne|gt|gte|lt|lte|in|nin|contains|exists, value'da {{chat_id}}/{{state_*}} "
            "ishlaydi. Masalan faqat bitta kategoriya: "
            "[{field:'category', op:'eq', value:'{{state_category}}'}]. None = filtersiz (hamma row). "
            "Filterlangan natija ham rowsVariable'ga yoziladi — {{#each}} bilan o'qiladi."
        ),
    )


class GoogleDriveData(_StrictData):
    """Google Drive (v3) integratsiyasi — oauth2 credential orqali.

    Credential FAQAT `credential_id` orqali bog'lanadi (type_key 'google_drive').
    Natija context[responseVariable] (+ _status, drive_success) ga yoziladi.
    """

    credential_id: int = Field(
        description="type_key 'google_drive' bo'lgan oauth2 credential id'si.",
    )
    operation: Literal["list", "create_folder", "upload", "delete"] = Field(
        default="list",
        description="list: ro'yxat | create_folder: papka | upload: matn fayl | delete: o'chirish.",
    )
    query: str | None = Field(
        default=None,
        description="list uchun Drive q filtri, masalan \"name contains 'report'\". {{var}} qo'llanadi.",
    )
    file_name: str | None = Field(
        default=None,
        description="create_folder / upload uchun fayl yoki papka nomi.",
    )
    folder_id: str | None = Field(
        default=None,
        description="Ota-papka id'si (ixtiyoriy).",
    )
    content: str | None = Field(
        default=None,
        description="upload uchun matn kontent (v1 = text/plain). {{var}} qo'llanadi.",
    )
    mime_type: str = Field(
        default="text/plain",
        description="upload uchun MIME tur (default 'text/plain').",
    )
    file_id: str | None = Field(
        default=None,
        description="delete uchun fayl id'si.",
    )
    responseVariable: str = Field(
        default="drive_response",
        description="Natija saqlanadigan context kaliti (default 'drive_response').",
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
    returnMode: Literal["text", "json"] = Field(default="text", description="'text' returns plain string; 'json' also stores parsed JSON under outputKey_json. NOTE: for guaranteed-valid JSON prefer structuredOutput+jsonSchema (below) — that constrains the model at the provider level; returnMode='json' only best-effort parses free text.")
    structuredOutput: bool = Field(
        default=False,
        description=(
            "Enable PROVIDER-LEVEL structured output. When true, supply `jsonSchema` "
            "and the engine forces the model to return JSON matching that schema; the "
            "parsed object is written to outputKey (read as {{outputKey.fieldName}}). "
            "USE THIS instead of telling the model 'return JSON' in systemPrompt — that "
            "is unreliable. Requires a provider that supports structured output "
            "(OpenAI/Gemini). On parse failure the raw text goes to outputKey and an "
            "error to errorOutputKey."
        ),
    )
    jsonSchema: str = Field(
        default="",
        description=(
            "JSON Schema as a STRING (the whole schema serialized), used ONLY when "
            "structuredOutput=true. Example: "
            "'{\"type\":\"object\",\"properties\":{\"intent\":{\"type\":\"string\"},"
            "\"amount\":{\"type\":\"number\"}},\"required\":[\"intent\"]}'. "
            "Invalid JSON is ignored (structured output silently disabled)."
        ),
    )
    historyKey: str = Field(default="", description="Optional context key holding prior messages[] to seed the conversation. Manual mode: builder assembles the [{role,content}] array. For a chat/conversational agent prefer autoHistory=true instead — the engine then manages history for you.")
    autoHistory: bool = Field(default=False, description="Engine-managed PER-CHAT conversation memory (keyed by node + chat_id, so every user keeps a SEPARATE history — never shared/global). When true the engine auto-saves/loads the dialog — NO manual historyKey array needed. SET THIS TRUE for any chatbot / conversational / sales-assistant agent; otherwise the bot is amnesiac (forgets every previous message).")
    maxHistory: int | None = Field(default=None, description="Max stored conversation messages for autoHistory (older ones dropped). Default 40 (~20 turns) when omitted. Lower to save tokens.")
    errorOutputKey: str = Field(default="agent_error", description="Context key for error string on failure. Branch on this with IfConditionNode.")


class _AIProviderBase(_StrictData):
    """Shared fields for all provider sub-nodes (accessory, attach to acc:ai_model)."""

    apiKey: str = Field(default="", description="API key. Never logged.")
    model: str = Field(default="", description="Model id. Leave empty to use node-type default.")
    baseURL: str = Field(
        default="",
        description=(
            "DO NOT SET. The endpoint comes from the CREDENTIAL (engine resolves it at "
            "runtime; empty → provider default). Field kept only for legacy/local-Ollama "
            "(localhost) flows — leave it EMPTY for every cloud provider."
        ),
    )
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    maxTokens: int | None = Field(default=None, description="Max response tokens.")
    extraHeaders: dict[str, str] = Field(default_factory=dict, description="Optional raw header map.")
    extraParams: dict[str, Any] = Field(default_factory=dict, description="Provider-specific passthrough (e.g. top_p).")


class _AICredentialProviderBase(_AIProviderBase):
    """Provider sub-nodes that PREFER a stored credential (openai/ollama/gemini).

    The panel UI no longer shows a manual API-key box for these — the user
    picks a saved credential and the node stores its `credential_id`. The engine
    resolves it to api_key/base_url at runtime (it OVERRIDES any apiKey). Set
    `credential_id` to the id from credentials.list (or the value the user gives).
    apiKey stays as an optional legacy fallback (e.g. local Ollama).
    """

    credential_id: int | None = Field(
        default=None,
        description=(
            "Stored credential id (PREFERRED). credentials.list(type_key=...) bilan "
            "topiladi yoki foydalanuvchidan so'raladi. Engine uni runtime'da "
            "api_key/base_url'ga aylantiradi (apiKey'ni override qiladi)."
        ),
    )


class AIOpenAIProviderData(_AICredentialProviderBase):
    """OpenAI provider. provider='openai', default model='gpt-4o-mini'.

    credential_id → type_key 'openai'. UI manual apiKey box'i yo'q — credential
    tanlang (yoki credential_id bering). credentials.list(type_key='openai').
    """

    provider: Literal["openai"] = "openai"


class AIOllamaProviderData(_AICredentialProviderBase):
    """Ollama provider. provider='ollama'. Endpoint comes from the CREDENTIAL.

    PREFERRED (default) AI provider. Set model EXPLICITLY to 'gemma4:31b-cloud'
    (cloud model); leaving model empty falls back to the engine default
    'gpt-oss:120b'. DO NOT set baseURL — the engine reads the endpoint from the
    credential (type_key 'ollama', empty → provider default ollama.com/v1).
    credentials.list(type_key='ollama').
    """

    provider: Literal["ollama"] = "ollama"


class AIGeminiProviderData(_AICredentialProviderBase):
    """Gemini provider. provider='gemini', default model='gemini-1.5-flash'.

    credential_id → type_key 'ai_studio' (Google AI Studio).
    credentials.list(type_key='ai_studio').
    """

    provider: Literal["gemini"] = "gemini"


class AIAnthropicProviderData(_AIProviderBase):
    """Anthropic provider. provider='anthropic', default model='claude-3-5-haiku-latest'.

    NOTE: no 'anthropic' credential type exists in the registry — this provider
    still uses a manual `apiKey` (ask_user). It is the ONLY provider without
    credential_id support.
    """

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
    """HTTP request tool. tool='http_request'. urlTemplate/bodyTemplate templated with tool args.

    RESULT IS STATUS-AWARE: the model sees the HTTP status in the tool result, so it
    can tell success from failure. On a non-2xx response the result includes the status
    and an ok=false flag (e.g. {"status": 404, "ok": false, "body": ...}); on 2xx it
    returns the body. Tell the model (via `description`) how to react to 4xx/5xx instead
    of treating every reply as success.

    SSRF GUARD: requests to internal/private addresses (localhost, 127.0.0.0/8,
    10/172.16/192.168, link-local, .internal, etc.) are BLOCKED by default. To reach an
    internal/private API on purpose, set allowInternal=true or list the exact hosts in
    allowedHosts (anything not listed is still SSRF-checked). Only opt in when the
    builder deliberately targets an internal service.
    """

    tool: Literal["http_request"] = "http_request"
    method: str = Field(default="GET", description="HTTP method. '{{from_args}}' lets the LLM choose.")
    urlTemplate: str = Field(default="", description="URL template. Supports tool args, e.g. https://api.x/{{id}}.")
    headers: dict[str, str] = Field(default_factory=dict, description="Static request headers.")
    bodyTemplate: str = Field(default="", description="Request body template.")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds.")
    allowInternal: bool = Field(
        default=False,
        description=(
            "Opt-in: allow requests to internal/private addresses (off by default; SSRF "
            "guard blocks them). Set true ONLY when intentionally calling an internal API."
        ),
    )
    allowedHosts: list[str] = Field(
        default_factory=list,
        description=(
            "Opt-in allow-list of hosts permitted to bypass the SSRF guard (e.g. "
            "['internal-api.svc']). Hosts not listed are still SSRF-checked. Use instead "
            "of allowInternal to whitelist specific internal hosts only."
        ),
    )


_FILTER_OP = Literal[
    "eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "exists",
]


class FilterCondition(_StrictData):
    """One TEMPLATE filter condition for read/update/delete.

    source='fixed' → the BUILDER pins `value` (a constant, or a flow placeholder
    like '{{chat_id}}'/'{{phone}}' (saved public var; legacy '{{state_phone}}' also
    works) resolved by the engine at runtime, type
    preserved). The runtime AI canNOT change it — use it to scope to the current
    user (user_id == {{chat_id}}). source='ai' → leave `value` empty and write an
    `aiDescription`; the RUNTIME AI fills the value per turn from that hint.
    """

    field: str = Field(description="Collection field name to filter on.")
    op: _FILTER_OP = Field(default="eq", description="Comparison operator.")
    source: Literal["fixed", "ai"] = Field(
        default="ai",
        description="'fixed' (builder pins value) | 'ai' (runtime AI fills the value).",
    )
    value: Any | None = Field(
        default=None,
        description="Fixed value (source='fixed'). May be a {{chat_id}}/{{<key>}} placeholder (saved public var; legacy {{state_<key>}} also works); raw type preserved.",
    )
    aiDescription: str | None = Field(
        default=None,
        description="Hint the runtime AI uses to fill the value (source='ai').",
    )


class FieldEntry(_StrictData):
    """One TEMPLATE field for create (`documentFields`) / update (`changeFields`).

    source='fixed' → builder pins `value` (constant or {{placeholder}}); source='ai'
    → runtime AI fills it from `aiDescription`.
    """

    field: str = Field(description="Collection field/document key to set.")
    source: Literal["fixed", "ai"] = Field(
        default="ai",
        description="'fixed' (builder pins value) | 'ai' (runtime AI fills the value).",
    )
    value: Any | None = Field(
        default=None,
        description="Fixed value (source='fixed'). May be a {{chat_id}}/{{<key>}} placeholder (saved public var; legacy {{state_<key>}} also works); raw type preserved.",
    )
    aiDescription: str | None = Field(
        default=None,
        description="Hint the runtime AI uses to fill the value (source='ai').",
    )


class AICollectionToolData(_AIToolBase):
    """Collection tool. tool='collection'. ONE operation per node, TEMPLATE-driven.

    CROSS-STREAM CONTRACT with constructor + engine (AICollectionToolNode) — keep
    keys exact.

    ───────────────────────────────────────────────────────────────────────────
    PREFERRED: TEMPLATE (HYBRID) MODEL — `operation` + per-field fixed/ai slots.
    ───────────────────────────────────────────────────────────────────────────
    The BUILDER (you, the flow-builder AI) sets ONE `operation`
    (create|read|update|delete) and a query SKELETON: for every field you decide
    whether it is FIXED (you pin the value — safe, the runtime AI cannot change it,
    e.g. user_id="{{chat_id}}" always scopes to the current user) or an AI-SLOT
    (you give an `aiDescription` and the RUNTIME AI fills the value per turn).

      operation='read'|'update'|'delete'  → `filterMatch` ('and'|'or') +
          `filterConditions: [{field, op, source, value?|aiDescription?}]`.
      operation='create'                  → `documentFields: [{field, source, ...}]`.
      operation='update'                  → ALSO `changeFields: [{field, source, ...}]`
          (the fields to SET; `filterConditions` selects WHICH docs).

    ONE operation per node. If the agent needs read + create (or create + update),
    attach SEPARATE AICollectionToolNodes to the SAME AIAgentNode — one per
    operation.

    FIXED vs AI:
      source='fixed' → BUILDER pins `value`. May be a constant OR a flow-context
        placeholder the engine resolves at runtime (raw type preserved):
          {{chat_id}} {{user_id}} {{first_name}} {{username}}
          {{<key>}} (asked/saved earlier; legacy {{state_<key>}} also works) {{collection_<key>}}
        Whole-placeholder value "{{chat_id}}" stays an int → user_id == {{chat_id}}
        matches an int field. This is how you safely lock a tool to the CURRENT
        user — the runtime AI cannot override a fixed slot.
      source='ai' → BUILDER leaves `value` empty and writes `aiDescription`; the
        RUNTIME AI (the bot's own agent) fills the value each turn from that hint.

    Examples (TEMPLATE — what YOU, the builder, write into node data):
      read this user's debts, AI picks the status:
        operation='read', filterMatch='and', filterConditions=[
          {field:'user_id', op:'eq', source:'fixed', value:'{{chat_id}}'},
          {field:'status',  op:'eq', source:'ai',
           aiDescription:'qaysi holatdagi qarzlar (masalan active)'}]
      create a debt for this user:
        operation='create', documentFields=[
          {field:'user_id', source:'fixed', value:'{{chat_id}}'},
          {field:'amount',  source:'ai', aiDescription:'qarz summasi'},
          {field:'text',    source:'ai', aiDescription:'qarz izohi'}]

    ───────────────────────────────────────────────────────────────────────────
    LEGACY (still accepted; engine falls back): `operations` allow-list +
    `searchFields`. When you do NOT set `operation`/`filterConditions`/
    `documentFields`/`changeFields`, the engine uses the old runtime-AI-builds-the-
    -whole-filter model: `operations` lists allowed CRUD verbs and `searchFields`
    lists filterable fields ([{field, description?}]). PREFER the TEMPLATE model
    above for new nodes — it lets you pin fixed slots (e.g. user_id) the runtime
    AI can't change.
    """

    tool: Literal["collection"] = "collection"
    toolName: str = Field(default="collection", description="Function name given to the LLM (^[a-zA-Z0-9_]+$).")
    collection: str = Field(default="", description="Bot collection name the tool operates on.")

    # ── TEMPLATE (hybrid) model — PREFERRED ──────────────────────────────────
    operation: Literal["create", "read", "update", "delete"] | None = Field(
        default=None,
        description=(
            "TEMPLATE model: the SINGLE CRUD operation this node performs. One "
            "operation per node — for read+create attach two collection tools to "
            "the same agent. Leave None to fall back to the legacy `operations` "
            "allow-list."
        ),
    )
    filterMatch: Literal["and", "or"] = Field(
        default="and",
        description="How filterConditions combine (read/update/delete).",
    )
    filterConditions: list[FilterCondition] | None = Field(
        default=None,
        description=(
            "TEMPLATE filter skeleton for read/update/delete: "
            "[{field, op, source:'fixed'|'ai', value?|aiDescription?}]. "
            "Fixed slots are pinned by you (e.g. user_id='{{chat_id}}'); ai slots "
            "are filled by the runtime AI from aiDescription."
        ),
    )
    documentFields: list[FieldEntry] | None = Field(
        default=None,
        description=(
            "TEMPLATE document skeleton for operation='create': "
            "[{field, source:'fixed'|'ai', value?|aiDescription?}]."
        ),
    )
    changeFields: list[FieldEntry] | None = Field(
        default=None,
        description=(
            "TEMPLATE changes skeleton for operation='update' (the fields to SET; "
            "filterConditions selects which docs): "
            "[{field, source:'fixed'|'ai', value?|aiDescription?}]."
        ),
    )

    # ── LEGACY model — backward-compat fallback (engine also falls back) ─────
    operations: list[str] = Field(
        default_factory=lambda: ["create", "read", "update", "delete"],
        description="LEGACY allow-list of CRUD operations (used when `operation` is unset).",
    )
    searchFields: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "LEGACY allow-list (used when the TEMPLATE `filterConditions` is unset): "
            "fields the LLM may filter/search by AT RUNTIME "
            "for read/update/delete. ARRAY OF OBJECTS — each item is a dict with "
            "EXACTLY these keys: `field` (the collection field name, REQUIRED) and "
            "`description` (human hint shown to the LLM, optional). "
            "Each listed field can be used by the LLM not just for EQUALITY but for "
            "comparison/range/membership/text-match — operators eq|ne|gt|gte|lt|lte|"
            "in|nin|contains|exists — and combined across fields with AND/OR. The "
            "engine builds the tool's filter schema from this list; the actual "
            "conditions (operator + value + AND/OR) are chosen by the LLM at runtime, "
            "NOT written here. Write a SHORT description per field hinting its type / "
            "valid values (e.g. status: 'active|inactive', amount: 'number') so the "
            "LLM picks sensible operators. "
            "Correct: [{\"field\": \"email\", \"description\": \"customer email\"}, "
            "{\"field\": \"amount\", \"description\": \"order total, number — gt/lt ok\"}, "
            "{\"field\": \"status\"}]. "
            "WRONG (rejected): a plain string array like [\"email\", \"status\"], "
            "or keys `name`/`type` instead of `field`/`description`."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_search_fields(cls, data: Any) -> Any:
        """Accept a plain string array and lenient key aliases for searchFields.

        Engine (AICollectionToolNode/mongoQueryTool) expects
        `[{"field": str, "description"?: str}]`. Models often emit
        `["chat_id", "kind"]` or `[{"name": "chat_id"}]`. We normalise those
        into the engine shape here so a well-intentioned call doesn't hard-fail
        on a key/shape mismatch — the validator below still rejects truly
        malformed items (e.g. dicts with no field name).
        """
        if not isinstance(data, dict):
            return data
        sf = data.get("searchFields")
        if not isinstance(sf, list):
            return data
        coerced: list[dict[str, str]] = []
        for item in sf:
            if isinstance(item, str):
                coerced.append({"field": item})
            elif isinstance(item, dict):
                field = item.get("field") or item.get("name")
                if not isinstance(field, str) or not field:
                    # Leave as-is; the field-level validator emits a clear error.
                    coerced.append(item)
                    continue
                entry: dict[str, str] = {"field": field}
                desc = item.get("description") or item.get("desc")
                if isinstance(desc, str) and desc:
                    entry["description"] = desc
                coerced.append(entry)
            else:
                coerced.append(item)
        data = dict(data)
        data["searchFields"] = coerced
        return data


class AICustomCodeToolData(_AIToolBase):
    """Custom JavaScript tool. tool='custom_code'. Goja sandbox; args + flow context available.

    POWERFUL — this tool has the FULL engine surface inside JS: MongoDB writes
    (insert/update/upsert/delete), Telegram sends (send_message/send_photo/...), and
    http_request. Treat it as a privileged action, not a scratchpad: write the minimum
    code for exactly the operation you need, never destructive DB ops you didn't intend,
    and avoid unbounded loops/recursion (there is a 30s execution timeout — a runaway
    loop just makes the tool call fail). Prefer the narrower collection/http tools when
    they suffice; reach for custom_code only for logic/computation they can't express.
    """

    tool: Literal["custom_code"] = "custom_code"
    jsCode: str = Field(
        default="// return value becomes the tool result",
        description=(
            "JavaScript. return value (or exit(value)) is the tool result. args and flow "
            "context available. Has full engine access (Mongo write, Telegram send, http) — "
            "write only the needed operation; avoid infinite loops (30s timeout). "
            "Format cleanly (stored verbatim): one statement per line, 2-space indent, "
            "spaces around operators, meaningful names — never minified/single-line. "
            "See docs_guide(topic='custom_code')."
        ),
    )


class AIKnowledgeToolData(_StrictData):
    """RAG knowledge retrieval tool. tool='knowledge'.

    Embeds the bot collection into a vector DB (lazy, on first call) and lets the
    agent retrieve semantically relevant documents by passing a plain-text `query`.
    CROSS-STREAM CONTRACT with constructor + engine (AIKnowledgeToolNode) — keep keys exact.

    HEAVY INFRA — do NOT add this tool unless an EXTERNAL vector DB (Qdrant) and an
    embedding provider/key are actually available and you have explicitly confirmed
    their values. There is NO built-in vector DB: if `vectorDbUrl` is empty the engine
    fails to build the tool and SILENTLY SKIPS it (the agent then has no tool and may
    hallucinate). When in doubt, prefer AICollectionToolNode (read) or AIHTTPToolNode
    for retrieval instead of this RAG tool.
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
        description=(
            "Embedding provider (REQUIRED): openai | ollama | gemini. Must match a "
            "provider you have a working key/endpoint for — the engine fails to build "
            "the tool if it cannot embed."
        ),
    )
    embeddingModel: str = Field(
        default="text-embedding-3-small",
        description=(
            "Embedding model id for the chosen provider (REQUIRED, must be non-empty — "
            "engine rejects an empty model)."
        ),
    )
    embeddingApiKey: str = Field(
        default="",
        description=(
            "API key for the embedding provider. May be {{env.KEY}}. REQUIRED for "
            "openai/gemini (and any keyed Ollama endpoint) — without a valid key "
            "embedding fails at first call and the tool returns errors."
        ),
    )
    embeddingBaseURL: str = Field(default="", description="Override endpoint URL for the embedding provider.")
    vectorDbUrl: str = Field(
        default="",
        description=(
            "REQUIRED — external vector DB (Qdrant) connection URL. There is NO built-in "
            "vector DB: leave this empty and the engine FAILS to build the tool and "
            "silently skips it. Always set a real reachable Qdrant URL "
            "(e.g. https://qdrant.example.com:6333). Do NOT add the knowledge tool at all "
            "unless you have this URL."
        ),
    )
    vectorDbApiKey: str = Field(default="", description="API key for the external vector DB (Qdrant), if it requires auth.")
    topK: int = Field(default=5, ge=1, description="Number of top documents to return per query.")


class AIDateTimeToolData(_StrictData):
    """Date/time helper tool. tool='datetime'. NO credential.

    Gives the agent a deterministic clock + date-math tool so it never guesses
    "today"/"now" or does arithmetic on dates in its head. The LLM calls it with
    runtime args (the engine builds the parametersSchema):
      op:    'now' (current date/time) | 'add' (shift a date) | 'diff' (difference)
      amount, unit ('minutes'|'hours'|'days'|'weeks'|'months'|'years') — for 'add'
      from, to  (ISO dates/datetimes) — 'add' uses `from`, 'diff' uses `from`+`to`
    Use for booking / scheduling / reminder bots that need the real current date,
    "3 days from now", or "how many days until X". CROSS-STREAM CONTRACT with
    engine (AIDateTimeToolNode) — keep `tool`/`tz`/`format` keys exact.
    """

    tool: Literal["datetime"] = "datetime"
    toolName: str = Field(default="datetime", description="Function name given to the LLM (^[a-zA-Z0-9_]+$).")
    tz: str = Field(
        default="UTC",
        description="IANA timezone the tool computes in (e.g. 'Asia/Tashkent'). Default 'UTC'.",
    )
    format: str = Field(
        default="",
        description="Optional output format hint (e.g. an strftime/layout string). Empty = engine default (ISO 8601).",
    )


class AIGoogleSheetsToolData(_StrictData):
    """Google Sheets agent tool. tool='google_sheets'. REQUIRES a `google_sheets` credential.

    Lets the agent VIEW (read, list_tabs) and MANAGE (append, update, clear) a
    Google Sheet (lookup tables, logs, order journals) on its own. operation is a
    RUNTIME arg the LLM picks — NOT static node config. The LLM calls it with runtime
    args (engine builds parametersSchema): operation, spreadsheet_id, sheet, range,
    values. Operations:
      - read        — read rows from a range -> {rows, count}.
      - append      — add row(s); values is a 2D array.
      - update      — write values into a range; values is a 2D array.
      - clear       — clear a range.
      - list_tabs   — list the spreadsheet's tab/sheet names.
    The optional `spreadsheetId`/`sheet`/`operation` data fields below are DEFAULTS
    the engine falls back to when the LLM omits the matching arg.
    CROSS-STREAM CONTRACT with engine (AIGoogleSheetsToolNode) — keep keys exact.

    CREDENTIAL: `credential_id` of a type_key='google_sheets' OAuth credential is
    REQUIRED. Find it with credentials.list(type_key='google_sheets'); if none
    exists it CANNOT be created via the API — the user must connect it in the
    dashboard ('Connect with Google').

    RESULT SHAPE: this is an AGENT TOOL — its result goes back to the LLM as a
    tool_result, NOT into a context variable for {{#each}} templates. On 'read' the
    tool returns the RAW Google API values (2D array: first row = header, rest = data,
    shape {majorDimension, range, values}). The LLM reasons over that array itself;
    no clean array-of-objects transform is applied for the tool path. (For TEMPLATE
    output use the GoogleSheetsNode read with rowsVariable instead — see GoogleSheetsData.)
    """

    tool: Literal["google_sheets"] = "google_sheets"
    toolName: str = Field(default="google_sheets", description="Function name given to the LLM (^[a-zA-Z0-9_]+$).")
    credential_id: int = Field(
        description="type_key 'google_sheets' OAuth credential id (REQUIRED). credentials.list(type_key='google_sheets').",
    )
    spreadsheetId: str = Field(
        default="",
        description="Default spreadsheet ID (URL /d/<ID>/) used when the LLM omits spreadsheet_id. Optional.",
    )
    sheet: str = Field(
        default="",
        description="Default sheet/tab name (e.g. 'Sheet1') used when the LLM omits sheet. Optional.",
    )
    operation: Literal["read", "append", "update", "clear", "list_tabs"] = Field(
        default="read",
        description=(
            "Default operation when the LLM omits it (the LLM normally picks at runtime): "
            "'read' | 'append' | 'update' | 'clear' | 'list_tabs'."
        ),
    )


class AIPaymentLinkToolData(_StrictData):
    """Dynamic payment-link tool. tool='payment_link'. paymentType REQUIRED.

    Lets the agent create a payment link on the fly (Click or Paycue) — e.g. a
    sales agent that quotes a price and then bills the user. The LLM calls it with
    runtime args (engine builds parametersSchema): amount (REQUIRED), description.
    The `paymentType` data field is fixed at build time and is NOT chosen by the
    LLM. CROSS-STREAM CONTRACT with engine (AIPaymentLinkToolNode) — keep keys exact.

    FIRE-AND-FORGET: returns only the payment LINK — there is NO payment confirmation
    (no callback / paid event). The bot never learns whether it was paid and cannot
    react. Use ONLY for the 'just show a payment link' case. If the bot must react to a
    successful payment (deliver goods/service, grant access, confirm an order), do NOT
    use this tool — use PaymentNode (Click/Paycue, paid/failed branches; the bot waits
    and branches) OR a Telegram invoice (SendInvoiceNode -> PreCheckoutTriggerNode
    [autoAnswer] -> SuccessfulPaymentTriggerNode).

    BUILD RULE (like PaymentNode): ASK the user which provider before adding —
    'click' or 'paycue' ONLY. Do NOT invent 'payme'. A merchant/credential may be
    required by engine/backend later; for now paymentType is assumed sufficient.
    """

    tool: Literal["payment_link"] = "payment_link"
    toolName: str = Field(default="payment_link", description="Function name given to the LLM (^[a-zA-Z0-9_]+$).")
    paymentType: Literal["click", "paycue"] = Field(
        description="Payment provider (REQUIRED — ASK the user first): 'click' or 'paycue'. No 'payme'.",
    )


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
    "PaymentStatusTriggerNode": PaymentStatusTriggerData,
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
    "CreateInviteLinkNode": CreateInviteLinkData,
    "PaymentNode": PaymentNodeData,
    "GoogleSheetsNode": GoogleSheetsData,
    "GoogleDriveNode": GoogleDriveData,
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
    "AIDateTimeToolNode": AIDateTimeToolData,
    "AIGoogleSheetsToolNode": AIGoogleSheetsToolData,
    "AIPaymentLinkToolNode": AIPaymentLinkToolData,
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
