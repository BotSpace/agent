"""Typed business-blueprint compiler for the common bot journeys.

The LLM chooses business intent and copy; this module owns the fragile graph
mechanics (trigger types, state capture, collection callbacks and mappings).
Keeping that boundary deterministic prevents a model from inventing manifest
fields or callback formats for the most common bots.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .schema import FlowDocument

BlueprintPattern = str
CaptureKind = Literal["text", "contact", "location"]


@dataclass(frozen=True)
class BlueprintDefinition:
    """One independently registerable blueprint kind.

    Future blueprint modules register one of these at import time. They do not
    need to edit BotBlueprint, the tool layer or the central dispatcher.
    """

    pattern: str
    title: str
    description: str
    compiler: Callable[[BotBlueprint], FlowDocument]
    test_generator: Callable[[BotBlueprint, dict[str, Any] | None], list[dict[str, Any]]]
    keywords: tuple[str, ...] = ()
    requires_destination: bool = False
    requires_fields: bool = False
    requires_catalog: bool = False
    requires_relation: bool = False
    version: int = 1
    field_presets: tuple[dict[str, Any], ...] = ()
    default_destination: str | None = None
    default_catalog: str | None = None
    default_relation_field: str | None = None
    options_model: type[BaseModel] | None = None


_BLUEPRINTS: dict[str, BlueprintDefinition] = {}


def register_blueprint(definition: BlueprintDefinition) -> None:
    """Register/replace a blueprint definition by stable pattern id."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", definition.pattern):
        raise ValueError(f"invalid blueprint pattern: {definition.pattern}")
    if definition.version < 1:
        raise ValueError("blueprint version must be positive")
    # Extension xatosini user buildining o'rtasida emas, servis ishga tushishida
    # ko'rsatamiz.
    for field in definition.field_presets:
        BlueprintField.model_validate(field)
    _BLUEPRINTS[definition.pattern] = definition


def get_blueprint_definition(pattern: str) -> BlueprintDefinition | None:
    return _BLUEPRINTS.get(pattern)


def list_blueprint_definitions() -> list[BlueprintDefinition]:
    return sorted(_BLUEPRINTS.values(), key=lambda item: item.pattern)


def suggest_blueprints(intent: str, *, limit: int = 3) -> list[tuple[BlueprintDefinition, int]]:
    """Rank registered patterns for a non-technical natural-language intent."""
    normalized = intent.casefold()
    ranked: list[tuple[BlueprintDefinition, int]] = []
    for definition in list_blueprint_definitions():
        score = sum(2 if " " in keyword else 1 for keyword in definition.keywords if keyword in normalized)
        if definition.pattern in normalized:
            score += 3
        if score:
            ranked.append((definition, score))
    ranked.sort(key=lambda item: (-item[1], item[0].pattern))
    return ranked[: max(1, limit)]


def discover_blueprint_modules() -> None:
    """Import every internal extension module from ``blueprint_plugins``.

    Adding a blueprint is intentionally append-only: create one module that
    calls ``register_blueprint``. No central import list or dispatcher edit is
    required, which keeps a large pattern library merge-friendly.
    """
    from . import blueprint_plugins

    prefix = blueprint_plugins.__name__ + "."
    for module in pkgutil.iter_modules(blueprint_plugins.__path__, prefix):
        importlib.import_module(module.name)


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    return value or "value"


class BlueprintField(BaseModel):
    key: str = Field(description="State key, snake_case; e.g. full_name or phone.")
    label: str = Field(description="Human label used in confirmation text.")
    prompt: str = Field(description="Question shown to the Telegram user.")
    kind: CaptureKind = "text"
    collection_field: str | None = Field(
        default=None,
        description="Destination collection field; defaults to key.",
    )

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        normalized = _slug(value)
        if normalized != value:
            raise ValueError(f"key must be snake_case; use `{normalized}`")
        return value


class BotBlueprint(BaseModel):
    pattern: BlueprintPattern = Field(
        description="Registered blueprint id; discover exact values with flow.list_blueprints."
    )
    title: str = Field(default="Yangi bot", min_length=1, max_length=100)
    intro: str = Field(default="Assalomu alaykum! 👋")
    fields: list[BlueprintField] = Field(default_factory=list, max_length=12)
    destination_collection: str | None = Field(
        default=None,
        description="Where captured lead/registration/order data is saved.",
    )
    catalog_collection: str | None = Field(
        default=None,
        description="Dynamic catalog source for catalog/registration/order.",
    )
    catalog_label_field: str = "name"
    catalog_secondary_field: str | None = None
    selection_state_key: str = "selected_item"
    relation_field: str | None = Field(
        default=None,
        description="Destination relation field receiving selected_item.id from full_data.",
    )
    include_confirmation: bool = True
    success_message: str = "✅ Ma'lumotlaringiz qabul qilindi."
    cancel_message: str = "Bekor qilindi. /start orqali qayta boshlashingiz mumkin."
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Blueprint extensioniga tegishli parametrlar. Plugin o'zining typed "
            "Pydantic modeli bilan tekshiradi; ixtiyoriy noma'lum options qabul qilinmaydi."
        ),
    )

    @model_validator(mode="after")
    def validate_contract(self) -> BotBlueprint:
        definition = get_blueprint_definition(self.pattern)
        if definition is None:
            available = ", ".join(item.pattern for item in list_blueprint_definitions())
            raise ValueError(f"unknown blueprint `{self.pattern}`; available: {available or '(none)'}")
        if definition.options_model is not None:
            self.options = definition.options_model.model_validate(self.options).model_dump()
        elif self.options:
            raise ValueError(f"{self.pattern} does not accept custom options")
        if definition.requires_destination and not self.destination_collection:
            raise ValueError(f"{self.pattern} requires destination_collection")
        if definition.requires_fields and not self.fields:
            raise ValueError(f"{self.pattern} requires at least one capture field")
        if definition.requires_catalog and not self.catalog_collection:
            raise ValueError(f"{self.pattern} requires catalog_collection")
        if definition.requires_relation and not self.relation_field:
            raise ValueError(f"{self.pattern} requires relation_field")
        if self.pattern == "feedback" and not self.fields:
            self.fields = [
                BlueprintField(
                    key="rating",
                    label="Baho",
                    prompt="Xizmatimizni 1 dan 5 gacha baholang:",
                )
            ]
        keys = [field.key for field in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError("field keys must be unique")
        return self


def _node(node_id: str, node_type: str, data: dict[str, Any], x: int, y: int = 0) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "data": data,
        "position": {"x": x, "y": y},
    }


def _edge(edge_id: str, source: str, target: str, source_handle: str | None = None) -> dict[str, Any]:
    edge: dict[str, Any] = {"id": edge_id, "source": source, "target": target}
    if source_handle:
        edge["sourceHandle"] = source_handle
    return edge


def _capture_nodes(field: BlueprintField, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    x = 640 + index * 640
    ask_data: dict[str, Any] = {"messageText": field.prompt, "parseMode": "None"}
    ask_type = "EditOrSendMessageNode"
    if field.kind == "contact":
        ask_type = "SendTextMessageNode"
        ask_data.update(
            {
                "keyboard_kind": "reply",
                "rows": [
                    {
                        "columns": [
                            {
                                "text": "📱 Telefon raqamni yuborish",
                                "type": "request_contact",
                            }
                        ]
                    }
                ],
            }
        )
    elif field.kind == "location":
        ask_type = "SendTextMessageNode"
        ask_data.update(
            {
                "keyboard_kind": "reply",
                "rows": [
                    {
                        "columns": [
                            {
                                "text": "📍 Joylashuvni yuborish",
                                "type": "request_location",
                            }
                        ]
                    }
                ],
            }
        )
    trigger_type = field.kind if field.kind in {"contact", "location"} else "text"
    state_type = {
        "contact": "phone_number",
        "location": "latitude",
        "text": "text",
    }[field.kind]
    ask = _node(f"ask-{field.key}", ask_type, ask_data, x)
    capture = _node(
        f"capture-{field.key}",
        "MessageTriggerNode",
        {
            "type": trigger_type,
            "filter": "any",
            "global": False,
            "state_key": field.key,
            "state_type": state_type,
        },
        x + 320,
    )
    return ask, capture


def _confirmation_text(blueprint: BotBlueprint) -> str:
    rows = ["Ma'lumotlarni tasdiqlaysizmi?", ""]
    if blueprint.pattern in {"registration", "order"}:
        rows.append(f"Tanlov: {{{{{blueprint.selection_state_key}.{blueprint.catalog_label_field}}}}}")
    rows.extend(f"{field.label}: {{{{{field.key}}}}}" for field in blueprint.fields)
    return "\n".join(rows)


def _data_flow(blueprint: BotBlueprint) -> FlowDocument:
    nodes: list[dict[str, Any]] = [
        _node("start", "CommandTriggerNode", {"command": "/start", "global": True}, 0)
    ]
    edges: list[dict[str, Any]] = []
    edge_no = 1
    save_source_handle: str | None = None

    if blueprint.pattern == "info":
        nodes.append(
            _node(
                "welcome",
                "EditOrSendMessageNode",
                {"messageText": blueprint.intro, "parseMode": "None", "rows": []},
                320,
            )
        )
        edges.append(_edge("e1", "start", "welcome"))
        return FlowDocument.model_validate({"nodes": nodes, "edges": edges})

    if blueprint.pattern in {"catalog", "registration", "order"}:
        collection = str(blueprint.catalog_collection)
        text = f"{{{{{blueprint.catalog_label_field}}}}}"
        if blueprint.catalog_secondary_field:
            text += f" — {{{{{blueprint.catalog_secondary_field}}}}}"
        nodes.append(
            _node(
                "catalog-menu",
                "EditOrSendMessageNode",
                {
                    "messageText": blueprint.intro,
                    "parseMode": "None",
                    "keyboard_kind": "inline",
                    "rows": [
                        {
                            "columns": [
                                {
                                    "collectionData": {
                                        "collection": collection,
                                        "field": blueprint.catalog_label_field,
                                        "text": text,
                                        "callbackDataPattern": f"collection:{collection}:{{id}}",
                                        "enablePagination": True,
                                        "layout": {"rows": 6, "columns": 1},
                                    }
                                }
                            ]
                        }
                    ],
                },
                320,
            )
        )
        edges.append(_edge(f"e{edge_no}", "start", "catalog-menu"))
        edge_no += 1
        nodes.append(
            _node(
                "catalog-selected",
                "CallbackQueryTriggerNode",
                {
                    "filter": "starts_with",
                    "context": {"value": f"collection:{collection}:"},
                    "global": True,
                    "state_key": blueprint.selection_state_key,
                    "state_type": "full_data",
                },
                640,
                -180,
            )
        )
        if blueprint.pattern == "catalog":
            details = [f"<b>{{{{{blueprint.selection_state_key}.{blueprint.catalog_label_field}}}}}</b>"]
            if blueprint.catalog_secondary_field:
                details.append(f"{{{{{blueprint.selection_state_key}.{blueprint.catalog_secondary_field}}}}}")
            nodes.append(
                _node(
                    "catalog-detail",
                    "EditOrSendMessageNode",
                    {"messageText": "\n".join(details), "parseMode": "HTML", "rows": []},
                    960,
                    -180,
                )
            )
            edges.append(_edge(f"e{edge_no}", "catalog-selected", "catalog-detail"))
            return FlowDocument.model_validate({"nodes": nodes, "edges": edges})
        previous = "catalog-selected"
    else:
        nodes.append(
            _node(
                "welcome",
                "EditOrSendMessageNode",
                {"messageText": blueprint.intro, "parseMode": "None", "rows": []},
                320,
            )
        )
        edges.append(_edge(f"e{edge_no}", "start", "welcome"))
        edge_no += 1
        previous = "welcome"

    for index, field in enumerate(blueprint.fields):
        ask, capture = _capture_nodes(field, index)
        nodes.extend([ask, capture])
        edges.append(_edge(f"e{edge_no}", previous, ask["id"]))
        edge_no += 1
        edges.append(_edge(f"e{edge_no}", ask["id"], capture["id"]))
        edge_no += 1
        previous = capture["id"]

    save_id = "save-result"
    if blueprint.include_confirmation:
        nodes.append(
            _node(
                "confirm",
                "EditOrSendMessageNode",
                {
                    "messageText": _confirmation_text(blueprint),
                    "parseMode": "None",
                    "keyboard_kind": "inline",
                    "rows": [
                        {
                            "columns": [
                                {"text": "✅ Tasdiqlash", "type": "callback", "value": "blueprint:confirm"},
                                {"text": "❌ Bekor qilish", "type": "callback", "value": "blueprint:cancel"},
                            ]
                        }
                    ],
                },
                640 + len(blueprint.fields) * 640,
            )
        )
        edges.append(_edge(f"e{edge_no}", previous, "confirm"))
        edge_no += 1
        nodes.append(
            _node(
                "cancelled",
                "EditOrSendMessageNode",
                {"messageText": blueprint.cancel_message, "parseMode": "None", "rows": []},
                960 + len(blueprint.fields) * 640,
                180,
            )
        )
        edges.append(
            _edge(
                f"e{edge_no}",
                "confirm",
                "cancelled",
                "target-handler-confirm-❌ Bekor qilish",
            )
        )
        edge_no += 1
        previous = "confirm"
        save_source_handle = "target-handler-confirm-✅ Tasdiqlash"

    mappings: dict[str, str] = {
        str(field.collection_field or field.key): field.key for field in blueprint.fields
    }
    if blueprint.pattern in {"registration", "order"} and blueprint.relation_field:
        # full_data normalizes Mongo `_id` to public `id` in runtime state.
        mappings[blueprint.relation_field] = f"{{{{{blueprint.selection_state_key}.id}}}}"
    nodes.append(
        _node(
            save_id,
            "CollectionNode",
            {
                "collection_name": blueprint.destination_collection,
                "fieldMappings": mappings,
            },
            960 + len(blueprint.fields) * 640,
            -180 if blueprint.include_confirmation else 0,
        )
    )
    nodes.append(
        _node(
            "success",
            "EditOrSendMessageNode",
            {"messageText": blueprint.success_message, "parseMode": "None", "rows": []},
            1280 + len(blueprint.fields) * 640,
            -180 if blueprint.include_confirmation else 0,
        )
    )
    edges.append(_edge(f"e{edge_no}", previous, save_id, save_source_handle))
    edge_no += 1
    edges.append(_edge(f"e{edge_no}", save_id, "success"))
    return FlowDocument.model_validate({"nodes": nodes, "edges": edges})


def compile_blueprint(blueprint: BotBlueprint) -> FlowDocument:
    """Compile a validated business blueprint into a deterministic flow."""
    definition = get_blueprint_definition(blueprint.pattern)
    if definition is None:  # model validation normally catches this
        raise ValueError(f"unknown blueprint `{blueprint.pattern}`")
    return definition.compiler(blueprint)


def compile_standard_blueprint(blueprint: BotBlueprint, *, base_pattern: str) -> FlowDocument:
    """Reuse a stable built-in graph shape from an extension module.

    Extensions remain independent registry entries while sharing the audited
    capture/catalog mechanics instead of copying node JSON.
    """
    return _data_flow(blueprint.model_copy(update={"pattern": base_pattern}))


def regression_steps(
    blueprint: BotBlueprint,
    catalog_document: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate a stable smoke/journey test for the compiled flow.

    Catalog-backed flows cannot know a document id before runtime, therefore
    their durable baseline verifies the dynamic entry screen. The agent can add
    a full seeded journey after the collection has at least one document.
    """
    steps: list[dict[str, Any]] = [
        {
            "type": "command",
            "text": "/start",
            "expect": {"contains": blueprint.intro[:40], "no_errors": True},
        }
    ]
    if blueprint.pattern == "info":
        return steps
    if blueprint.pattern in {"catalog", "registration", "order"}:
        doc = catalog_document if isinstance(catalog_document, dict) else None
        doc_id = doc.get("_id") or doc.get("id") if doc else None
        label = doc.get(blueprint.catalog_label_field) if doc else None
        if not doc_id or label is None:
            return steps
        steps[0]["expect"]["keyboard_contains"] = str(label)
        click: dict[str, Any] = {
            "type": "click",
            "data": f"collection:{blueprint.catalog_collection}:{doc_id}",
            "expect": {"no_errors": True},
        }
        if blueprint.pattern == "catalog":
            click["expect"]["contains"] = str(label)
            steps.append(click)
            return steps
        steps.append(click)
    for field in blueprint.fields:
        if field.kind == "contact":
            step: dict[str, Any] = {
                "type": "contact",
                "phone": "+998901234567",
                "first_name": "Test",
            }
        elif field.kind == "location":
            step = {"type": "location", "latitude": 41.2995, "longitude": 69.2401}
        else:
            step = {"type": "text", "text": f"Test {field.label}"}
        step["expect"] = {"no_errors": True}
        steps.append(step)
    if blueprint.include_confirmation:
        steps.append(
            {
                "type": "click",
                "data": "blueprint:confirm",
                "expect": {"contains": blueprint.success_message[:40], "no_errors": True},
            }
        )
    elif steps:
        steps[-1]["expect"] = {"contains": blueprint.success_message[:40], "no_errors": True}
    return steps


def _compile_feedback(blueprint: BotBlueprint) -> FlowDocument:
    return _data_flow(blueprint.model_copy(update={"pattern": "lead"}))


def _register_builtins() -> None:
    definitions = [
        BlueprintDefinition(
            "info",
            "Ma'lumot bot",
            "Salomlashish va statik ma'lumot oqimi.",
            _data_flow,
            regression_steps,
            keywords=("ma'lumot", "malumot", "vizitka", "menyu", "faq"),
        ),
        BlueprintDefinition(
            "lead",
            "Lead yig'ish",
            "Kontakt ma'lumotlarini collectionga saqlaydi.",
            _data_flow,
            regression_steps,
            keywords=("lead", "ariza", "ism va telefon", "bog'lan", "mijoz"),
            requires_destination=True,
            requires_fields=True,
            field_presets=(
                {"key": "full_name", "label": "Ism", "prompt": "Ismingizni kiriting:"},
                {
                    "key": "phone",
                    "label": "Telefon",
                    "prompt": "Telefon raqamingizni yuboring:",
                    "kind": "contact",
                },
            ),
            default_destination="leads",
        ),
        BlueprintDefinition(
            "registration",
            "Ro'yxatdan o'tish",
            "Dinamik tanlov va ro'yxatdan o'tish.",
            _data_flow,
            regression_steps,
            keywords=(
                "ro'yxat",
                "royxat",
                "registr",
                "kursga yoz",
                "kursni tanla",
                "qatnash",
            ),
            requires_destination=True,
            requires_fields=True,
            requires_catalog=True,
            requires_relation=True,
            field_presets=(
                {"key": "full_name", "label": "Ism", "prompt": "Ismingizni kiriting:"},
                {
                    "key": "phone",
                    "label": "Telefon",
                    "prompt": "Telefon raqamingizni yuboring:",
                    "kind": "contact",
                },
            ),
            default_destination="registrations",
            default_catalog="courses",
            default_relation_field="course_id",
        ),
        BlueprintDefinition(
            "catalog",
            "Dinamik katalog",
            "Collectiondan doim yangilanadigan katalog.",
            _data_flow,
            regression_steps,
            keywords=("katalog", "ro'yxatni ko'rsat", "kurslar", "mahsulotlar"),
            requires_catalog=True,
            default_catalog="catalog_items",
        ),
        BlueprintDefinition(
            "order",
            "Buyurtma",
            "Dinamik mahsulot tanlovi va buyurtma saqlash.",
            _data_flow,
            regression_steps,
            keywords=("buyurtma", "sotib", "zakaz", "mahsulot tanla"),
            requires_destination=True,
            requires_fields=True,
            requires_catalog=True,
            requires_relation=True,
            field_presets=(
                {"key": "full_name", "label": "Ism", "prompt": "Ismingizni kiriting:"},
                {
                    "key": "phone",
                    "label": "Telefon",
                    "prompt": "Telefon raqamingizni yuboring:",
                    "kind": "contact",
                },
                {
                    "key": "delivery_address",
                    "label": "Manzil",
                    "prompt": "Yetkazib berish manzilini kiriting:",
                },
            ),
            default_destination="orders",
            default_catalog="products",
            default_relation_field="product_id",
        ),
        BlueprintDefinition(
            "support",
            "Murojaat qabul qilish",
            "Support murojaatini collectionga saqlaydi.",
            _data_flow,
            regression_steps,
            keywords=("support", "yordam", "murojaat", "operator", "muammo"),
            requires_destination=True,
            requires_fields=True,
            field_presets=(
                {"key": "full_name", "label": "Ism", "prompt": "Ismingizni kiriting:"},
                {
                    "key": "phone",
                    "label": "Telefon",
                    "prompt": "Telefon raqamingizni yuboring:",
                    "kind": "contact",
                },
                {
                    "key": "message",
                    "label": "Murojaat",
                    "prompt": "Murojaatingizni yozing:",
                },
            ),
            default_destination="support_requests",
        ),
        BlueprintDefinition(
            "feedback",
            "Baho va fikr",
            "Foydalanuvchi bahosini collectionga saqlaydi.",
            _compile_feedback,
            regression_steps,
            keywords=("feedback", "baho", "fikr", "yulduz"),
            requires_destination=True,
            field_presets=(
                {"key": "rating", "label": "Baho", "prompt": "1 dan 5 gacha baholang:"},
                {"key": "comment", "label": "Fikr", "prompt": "Fikringizni yozing:"},
            ),
            default_destination="feedback",
        ),
    ]
    for definition in definitions:
        register_blueprint(definition)


_register_builtins()
discover_blueprint_modules()


__all__ = [
    "BlueprintDefinition",
    "BlueprintField",
    "BotBlueprint",
    "compile_blueprint",
    "compile_standard_blueprint",
    "discover_blueprint_modules",
    "get_blueprint_definition",
    "list_blueprint_definitions",
    "register_blueprint",
    "regression_steps",
    "suggest_blueprints",
]
