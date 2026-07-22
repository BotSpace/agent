"""Conservative, deterministic repairs for common historical flow mistakes."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .schema import FlowDocument

_STATE_TEMPLATE = re.compile(r"\{\{\s*state_([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_FULL_DATA_ID_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\._id\s*\}\}")
_NESTED_BARE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


def _walk(value: Any, visit) -> Any:
    if isinstance(value, list):
        return [_walk(item, visit) for item in value]
    if isinstance(value, dict):
        current = {key: _walk(item, visit) for key, item in value.items()}
        return visit(current)
    if isinstance(value, str):
        return visit(value)
    return value


def repair_flow(flow: FlowDocument) -> tuple[FlowDocument, list[str]]:
    """Apply only repairs whose intended canonical value is unambiguous."""
    raw = deepcopy(flow.model_dump())
    changes: list[str] = []

    # First pass: normalize state capture contracts so later template repair can
    # know which state keys really exist.
    state_keys: set[str] = set()
    full_data_keys: set[str] = set()
    for node in raw.get("nodes") or []:
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        legacy = data.get("state")
        if isinstance(legacy, dict) and legacy.get("key"):
            if not data.get("state_key"):
                data["state_key"] = legacy.get("key")
            if not data.get("state_type"):
                legacy_type = legacy.get("type")
                if legacy_type == "contact_phone_number":
                    legacy_type = "phone_number"
                data["state_type"] = legacy_type
            data.pop("state", None)
            changes.append(f"{node.get('id')}: legacy state → state_key/state_type")
        state_key = data.get("state_key")
        if isinstance(state_key, str) and state_key:
            state_keys.add(state_key)
            if data.get("state_type") == "full_data":
                full_data_keys.add(state_key)
        if node.get("type") == "StateNode" and isinstance(data.get("key"), str):
            state_keys.add(data["key"])

        if node.get("type") == "MessageTriggerNode" and data.get("type") == "any":
            state_type = data.get("state_type")
            if state_type in {
                "phone_number",
                "contact_phone_number",
                "contact_first_name",
                "contact_last_name",
            }:
                data["type"] = "contact"
                changes.append(f"{node.get('id')}: any trigger → contact")
            elif state_type in {"location", "latitude", "longitude"}:
                data["type"] = "location"
                changes.append(f"{node.get('id')}: any trigger → location")
        if node.get("type") == "CallbackQueryTriggerNode" and data.get("state_type") == "text":
            data["state_type"] = "callback"
            changes.append(f"{node.get('id')}: callback state_type text → callback")

    collection_prefixes: dict[str, str] = {}

    def visit_factory(node_id: str):
        def visit(value: Any) -> Any:
            if isinstance(value, str):
                replaced = _STATE_TEMPLATE.sub(
                    lambda match: (
                        "{{" + match.group(1) + "}}" if match.group(1) in state_keys else match.group(0)
                    ),
                    value,
                )
                if replaced != value:
                    changes.append(f"{node_id}: legacy state_ template tuzatildi")
                normalized_id = _FULL_DATA_ID_TEMPLATE.sub(
                    lambda match: (
                        "{{" + match.group(1) + ".id}}"
                        if match.group(1) in full_data_keys
                        else match.group(0)
                    ),
                    replaced,
                )
                if normalized_id != replaced:
                    changes.append(f"{node_id}: full_data ._id → .id")
                return normalized_id
            collection_data = value.get("collectionData") if isinstance(value, dict) else None
            if isinstance(collection_data, dict):
                collection = collection_data.get("collection")
                if isinstance(collection, str) and collection:
                    canonical = f"collection:{collection}:{{id}}"
                    if collection_data.get("callbackDataPattern") != canonical:
                        collection_data["callbackDataPattern"] = canonical
                        changes.append(f"{node_id}: collection callback canonical qilindi")
                    collection_prefixes[collection] = f"collection:{collection}:"
            button_type = (
                value.get("type") if isinstance(value, dict) and isinstance(value.get("text"), str) else None
            )
            if button_type == "contact":
                value["type"] = "request_contact"
                changes.append(f"{node_id}: contact button → request_contact")
            elif button_type == "location":
                value["type"] = "request_location"
                changes.append(f"{node_id}: location button → request_location")
            return value

        return visit

    for node in raw.get("nodes") or []:
        node_id = str(node.get("id") or "node")
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        node["data"] = _walk(data, visit_factory(node_id))

    # Matching collection triggers have one safe canonical contract.
    for node in raw.get("nodes") or []:
        if node.get("type") != "CallbackQueryTriggerNode":
            continue
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        context = data.get("context") if isinstance(data.get("context"), dict) else {}
        current = context.get("value")
        for collection, prefix in collection_prefixes.items():
            if isinstance(current, str) and collection in current and current != prefix:
                context["value"] = prefix
                data["context"] = context
                data["filter"] = "starts_with"
                data["global"] = True
                changes.append(f"{node.get('id')}: collection trigger canonical qilindi")
                break
            if current == prefix and not data.get("global"):
                data["global"] = True
                changes.append(f"{node.get('id')}: collection trigger global qilindi")

    # A dotted state source without braces is always stored literally by the
    # runtime; wrapping it is safe when its root is a produced state key.
    for node in raw.get("nodes") or []:
        if node.get("type") != "CollectionNode":
            continue
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        mappings = data.get("fieldMappings")
        if not isinstance(mappings, dict):
            continue
        for field, source in list(mappings.items()):
            if not isinstance(source, str) or not _NESTED_BARE.fullmatch(source):
                continue
            if source.split(".", 1)[0] in state_keys:
                root, nested = source.split(".", 1)
                if root in full_data_keys and nested == "_id":
                    nested = "id"
                mappings[field] = "{{" + root + "." + nested + "}}"
                changes.append(f"{node.get('id')}: {field} nested mapping templatega o'raldi")

    # Preserve order while removing duplicate messages from repeated templates.
    changes = list(dict.fromkeys(changes))
    return FlowDocument.model_validate(raw), changes


__all__ = ["repair_flow"]
