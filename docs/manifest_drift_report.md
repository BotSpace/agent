# Drift hisoboti: qo'lyozma Pydantic ↔ manifest-kompilyatsiya

Manba=manifest (engine bilan mos). "- qo'lyozmada" = eskirgan/ko'chgan field.

## AIAgentNode
- qo'lyozmada(eskirgan): ['autoHistory', 'maxHistory', 'maxTokens', 'temperature', 'userInputSource']
! required: hand=[] → comp=['outputKey', 'systemPrompt', 'userInputText']

## AIAnthropicProviderNode
- qo'lyozmada(eskirgan): ['baseURL', 'extraHeaders', 'extraParams', 'provider']
! required: hand=[] → comp=['apiKey', 'model']

## AICollectionToolNode
- qo'lyozmada(eskirgan): ['changeFields', 'collection', 'documentFields', 'filterConditions', 'filterMatch', 'operation', 'operations', 'parametersSchema', 'searchFields', 'tool']
! required: hand=[] → comp=['description', 'toolName']

## AICustomCodeToolNode
- qo'lyozmada(eskirgan): ['tool']
! required: hand=[] → comp=['description', 'jsCode', 'parametersSchema', 'toolName']

## AIDateTimeToolNode
- qo'lyozmada(eskirgan): ['tool', 'toolName']
! required: hand=[] → comp=['format', 'tz']

## AIGeminiProviderNode
- qo'lyozmada(eskirgan): ['apiKey', 'extraHeaders', 'extraParams', 'provider']
! required: hand=[] → comp=['baseURL', 'credential_id', 'model']

## AIGoogleSheetsToolNode
- qo'lyozmada(eskirgan): ['tool', 'toolName']
! required: hand=['credential_id'] → comp=['credential_id', 'operation', 'sheet', 'spreadsheetId']

## AIHTTPToolNode
- qo'lyozmada(eskirgan): ['allowInternal', 'allowedHosts', 'headers', 'parametersSchema', 'tool']
! required: hand=[] → comp=['description', 'method', 'timeout', 'toolName', 'urlTemplate']

## AIKnowledgeToolNode
- qo'lyozmada(eskirgan): ['collection', 'embeddingApiKey', 'embeddingBaseURL', 'embeddingModel', 'embeddingProvider', 'textFields', 'tool', 'topK', 'vectorDbApiKey', 'vectorDbUrl']
+ manifestda(yangi): ['description']
! required: hand=['collection'] → comp=['description', 'toolName']

## AIOllamaProviderNode
- qo'lyozmada(eskirgan): ['apiKey', 'baseURL', 'extraHeaders', 'extraParams', 'provider']
! required: hand=[] → comp=['credential_id', 'model']

## AIOpenAIProviderNode
- qo'lyozmada(eskirgan): ['apiKey', 'baseURL', 'extraHeaders', 'extraParams', 'provider']
! required: hand=[] → comp=['credential_id', 'maxTokens', 'model', 'temperature']

## AIPaymentLinkToolNode
- qo'lyozmada(eskirgan): ['paramConfig', 'tool', 'toolName']

## AIPaymentStatusToolNode
- qo'lyozmada(eskirgan): ['paramConfig', 'tool', 'toolName']

## CallbackButtonTriggerNode
+ manifestda(yangi): ['global', 'private', 'save_state', 'state_key', 'state_type']
! required: hand=[] → comp=['global']

## CallbackQueryAnswerNode
- qo'lyozmada(eskirgan): ['showAlert', 'text']
+ manifestda(yangi): ['disableNotification', 'keyboard_kind', 'protectContent', 'rows']

## CallbackQueryTriggerNode
- qo'lyozmada(eskirgan): ['context', 'filter', 'state']
+ manifestda(yangi): ['private', 'save_state', 'state_key', 'state_type']
! required: hand=[] → comp=['global']

## CheckMembershipNode
- qo'lyozmada(eskirgan): ['channelsCollection', 'channelsField', 'chatId']
+ manifestda(yangi): ['inputType']
! required: hand=[] → comp=['channels', 'inputType']

## CollectionNode
- qo'lyozmada(eskirgan): ['fieldMappings']
! required: hand=['collection_name', 'fieldMappings'] → comp=['collection_name']

## CommandTriggerNode
- qo'lyozmada(eskirgan): ['state']
+ manifestda(yangi): ['private', 'save_state', 'state_key', 'state_type']

## CopyMessageNode
- qo'lyozmada(eskirgan): ['targetChatId']
! required: hand=['targetChatId'] → comp=[]

## CronTriggerNode
- qo'lyozmada(eskirgan): ['cronExpression']
+ manifestda(yangi): ['schedule']
! required: hand=['cronExpression'] → comp=['schedule']

## CustomCodeNode
- qo'lyozmada(eskirgan): ['terminal']

## DeleteMessageNode
- qo'lyozmada(eskirgan): ['targetMessageNodeId']
+ manifestda(yangi): ['deleteFromContext', 'targetNodeId']

## DownloadFileNode
- qo'lyozmada(eskirgan): ['messageType']

## EditMessageNode
- qo'lyozmada(eskirgan): ['editType']
+ manifestda(yangi): ['disableNotification', 'messageId', 'protectContent']
! required: hand=['newText'] → comp=['messageIdSource', 'newText']

## EditOrSendMessageNode
- qo'lyozmada(eskirgan): ['targetMessageNodeId']
+ manifestda(yangi): ['disableNotification', 'protectContent']

## ElseIfNode
+ manifestda(yangi): ['conditions', 'operator']
! required: hand=[] → comp=['conditions']

## ForLoopNode
- qo'lyozmada(eskirgan): ['array']
+ manifestda(yangi): ['dataKey', 'itemVariable']
! required: hand=['array'] → comp=['dataKey', 'itemVariable']

## FormNode
! required: hand=[] → comp=['items']

## ForwardMessageNode
- qo'lyozmada(eskirgan): ['targetChatId']
+ manifestda(yangi): ['recipientIds']
! required: hand=['targetChatId'] → comp=['recipientIds']

## GoogleDriveNode
- qo'lyozmada(eskirgan): ['mime_type']
! required: hand=['credential_id'] → comp=['credential_id', 'operation']

## GoogleSheetsNode
- qo'lyozmada(eskirgan): ['column']
! required: hand=['credential_id', 'spreadsheet_id'] → comp=['credential_id', 'operation', 'sheet', 'spreadsheet_id']

## HTTPRequestNode
- qo'lyozmada(eskirgan): ['saveAs']
+ manifestda(yangi): ['data', 'responseVariable', 'storeHeaders', 'timeout']
! required: hand=['method', 'url'] → comp=['method', 'responseVariable', 'url']

## IfConditionNode
- qo'lyozmada(eskirgan): ['branches']

## LoadCollectionItemNode
- qo'lyozmada(eskirgan): ['filters']
! required: hand=['collection', 'contextKey', 'filters'] → comp=['collection']

## LoadCollectionListNode
- qo'lyozmada(eskirgan): ['collection_name', 'contextKey', 'filters', 'limit', 'skip', 'sorts']
! required: hand=['collection_name', 'contextKey'] → comp=[]

## MessageTriggerNode
- qo'lyozmada(eskirgan): ['context', 'state']
+ manifestda(yangi): ['context.radius.km', 'context.radius.lat', 'context.radius.lng', 'context.size.max', 'context.size.min', 'context.value', 'private', 'save_state', 'state_key', 'state_type']
! required: hand=['type'] → comp=['global', 'type']

## PaymentNode
! required: hand=['amount'] → comp=['amount', 'comment', 'paymentType']

## PaymentStatusTriggerNode
! required: hand=[] → comp=['orderId', 'provider', 'status']

## PreCheckoutTriggerNode
! required: hand=[] → comp=['autoAnswer', 'filter']

## RandomNode
- qo'lyozmada(eskirgan): ['weights']
+ manifestda(yangi): ['disableNotification', 'keyboard_kind', 'protectContent', 'rows']
! required: hand=['weights'] → comp=[]

## ReplyButtonTriggerNode
- qo'lyozmada(eskirgan): ['catchNoButton', 'selectedButton']
+ manifestda(yangi): ['global', 'private', 'save_state', 'state_key', 'state_type']
! required: hand=['selectedButton'] → comp=['global']

## SendAnimationNode
- qo'lyozmada(eskirgan): ['animation', 'keyboard_kind', 'recipients', 'remove_keyboard', 'rows']
! required: hand=['animation'] → comp=[]

## SendAudioNode
- qo'lyozmada(eskirgan): ['audio', 'caption', 'performer', 'title']
+ manifestda(yangi): ['disableNotification', 'protectContent']
! required: hand=['audio'] → comp=[]

## SendContactNode
+ manifestda(yangi): ['disableNotification', 'lastName', 'protectContent', 'vcard']
! required: hand=['firstName', 'phoneNumber'] → comp=['firstName', 'lastName', 'phoneNumber', 'vcard']

## SendDiceNode
- qo'lyozmada(eskirgan): ['keyboard_kind', 'recipients', 'remove_keyboard', 'rows']
! required: hand=[] → comp=['emoji']

## SendFileNode
- qo'lyozmada(eskirgan): ['caption', 'file', 'parseMode']
+ manifestda(yangi): ['disableNotification', 'protectContent']
! required: hand=['file'] → comp=[]

## SendInvoiceNode
+ manifestda(yangi): ['recipients']
! required: hand=['description', 'payload', 'prices', 'title'] → comp=['currency', 'description', 'payload', 'prices', 'title']

## SendLocationNode
+ manifestda(yangi): ['disableNotification', 'protectContent']

## SendMediaGroupNode
- qo'lyozmada(eskirgan): ['media', 'recipients']
+ manifestda(yangi): ['disableNotification', 'mediaItems', 'protectContent']
! required: hand=['media'] → comp=['mediaItems']

## SendPhotoNode
- qo'lyozmada(eskirgan): ['caption', 'parseMode', 'photo']
+ manifestda(yangi): ['disableNotification', 'protectContent']
! required: hand=['photo'] → comp=[]

## SendPollNode
+ manifestda(yangi): ['allowsMultipleAnswers', 'correctOptionId', 'disableNotification', 'explanation', 'isAnonymous', 'openPeriod', 'protectContent', 'type']
! required: hand=['options', 'question'] → comp=['isAnonymous', 'options', 'question', 'type']

## SendStickerNode
- qo'lyozmada(eskirgan): ['keyboard_kind', 'recipients', 'remove_keyboard', 'rows', 'sticker']
+ manifestda(yangi): ['stickerFileId', 'stickerSourceType', 'stickerUrl']
! required: hand=['sticker'] → comp=['stickerSourceType']

## SendTextMessageNode
- qo'lyozmada(eskirgan): ['disableWebPagePreview']
+ manifestda(yangi): ['disableNotification', 'protectContent']

## SendToAdminNode
- qo'lyozmada(eskirgan): ['messageText']
! required: hand=['messageText'] → comp=[]

## SendVenueNode
- qo'lyozmada(eskirgan): ['keyboard_kind', 'remove_keyboard', 'rows']

## SendVideoNode
- qo'lyozmada(eskirgan): ['caption', 'parseMode', 'video']
+ manifestda(yangi): ['disableNotification', 'protectContent']
! required: hand=['video'] → comp=[]

## SendVideoNoteNode
- qo'lyozmada(eskirgan): ['keyboard_kind', 'recipients', 'remove_keyboard', 'rows', 'videoNote']
! required: hand=['videoNote'] → comp=[]

## SendVoiceNode
- qo'lyozmada(eskirgan): ['keyboard_kind', 'recipients', 'remove_keyboard', 'rows', 'voice']
! required: hand=['voice'] → comp=[]

## SetLanguageNode
! required: hand=[] → comp=['language']

## StateNode
- qo'lyozmada(eskirgan): ['customType', 'key', 'type']
! required: hand=['key', 'type'] → comp=[]

## SubFlowExitNode
! required: hand=[] → comp=['exitName']

## SubFlowNode
- qo'lyozmada(eskirgan): ['subflowId']
! required: hand=['subflowId'] → comp=[]

## SuccessfulPaymentTriggerNode
! required: hand=[] → comp=['filter']

## SwitchNode
! required: hand=['value'] → comp=['cases', 'default', 'value']

## UnpinMessageNode
+ manifestda(yangi): ['messageId', 'unpinAll']
! required: hand=[] → comp=['unpinAll']

## VariableNode
- qo'lyozmada(eskirgan): ['name']
+ manifestda(yangi): ['variableName']
! required: hand=['name', 'operation'] → comp=['operation', 'variableName']

## WebhookTriggerNode
! required: hand=[] → comp=['wait_response', 'webhook_url']

## Manifestsiz node'lar (qo'lyozmada bor, manifestda yo'q)
[]

---

## Reconciliation holati (Faza 2.4)

Qo'lyozma `node_data_models.py` ALLAQACHON o'chirilgan — `NODE_DATA_MODELS`
endi har bir modelni manifestdan `compile_node_model()` orqali kompilyatsiya
qiladi (`src/agent/tools/flow/node_data_models._CompiledRegistry`). Ya'ni
**manifest_compiler yagona haqiqat manbasi** va yuqoridagi "comp" ustuni =
kompilyatsiya natijasi = manifest. Shu sabab hisobotdagi drift'larning ko'pi
(yangi maydonlar, required farqlari) avtomatik bartaraf — kompilyatsiya
manifestning to'g'ri kalitlarini beradi.

Kompilyatorga qo'shilgan himoya renamelari (`compiler._NODE_KEY_RENAMES`) —
manifest regressiya qilsa ham engine-kanonik kalit chiqishi uchun:
- `VariableNode`: `name` → `variableName`
- `AIOpenAIProviderNode` / `AIGeminiProviderNode` / `AIOllamaProviderNode`:
  `apiKey` → `credential_id`

ATAYIN qamrab OLINMAGAN (drift emas, to'g'ri):
- `AIAnthropicProviderNode.apiKey` — Anthropic'ning credential type'i yo'q,
  u faqat `apiKey` ishlatadi (qarang `flow/validator.py` ai_provider tekshiruvi).

Qolgan farqlar (`- eskirgan` / `+ yangi` / required) — kod uchun emas, HISOBOT
o'zi eskirgan: ular endi manifestdan to'g'ridan-to'g'ri keladi va qo'lda
sinxronlashni talab qilmaydi. Yangi drift faqat manifest o'zgarganda paydo
bo'ladi va kompilyatsiya orqali avtomatik yopiladi.