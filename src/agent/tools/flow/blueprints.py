"""High-level typed flow builders and blueprint transaction helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...flow.blueprint import (
    BotBlueprint,
    compile_blueprint,
    get_blueprint_definition,
    list_blueprint_definitions,
    suggest_blueprints,
)
from ...flow.diff import diff
from ...flow.schema import FlowDocument
from ...flow.validator import validate
from .._common import resolve_concrete_project_id
from ..base import Tool, ToolContext, ToolResult, tool


def _current(ctx: ToolContext) -> dict[str, Any]:
    raw = ctx.state.get("working_flow")
    return raw if isinstance(raw, dict) else {"nodes": [], "edges": []}


def _collection_contract(ctx: ToolContext) -> tuple[list[str], dict[str, dict[str, str]]]:
    names: list[str] = []
    schemas: dict[str, dict[str, str]] = {}
    raw = ctx.state.get("collections")
    for collection in raw if isinstance(raw, list) else []:
        if not isinstance(collection, dict) or not collection.get("name"):
            continue
        name = str(collection["name"])
        names.append(name)
        kinds = collection.get("field_kinds")
        schemas[name] = {
            str(key): str(value) for key, value in (kinds.items() if isinstance(kinds, dict) else [])
        }
    return names, schemas


def _business_summary(blueprint: BotBlueprint) -> list[str]:
    lines = [f"• Tur: {blueprint.pattern}", f"• Nomi: {blueprint.title}"]
    if blueprint.catalog_collection:
        lines.append(f"• Dinamik manba: {blueprint.catalog_collection}")
    if blueprint.destination_collection:
        lines.append(f"• Saqlash joyi: {blueprint.destination_collection}")
    if blueprint.fields:
        lines.append(
            "• Yig'iladigan ma'lumot: "
            + ", ".join(f"{field.label} ({field.kind})" for field in blueprint.fields)
        )
    lines.append("• Tasdiqlash: " + ("bor" if blueprint.include_confirmation else "yo'q"))
    return lines


class BlueprintPreviewInput(BaseModel):
    blueprint: BotBlueprint


class ListBlueprintsInput(BaseModel):
    pass


class SuggestBlueprintsInput(BaseModel):
    intent: str = Field(
        min_length=3,
        description="Foydalanuvchining aynan o'z so'zlari bilan yozilgan biznes talabi.",
    )
    limit: int = Field(default=3, ge=1, le=5)


@tool(
    name="flow.suggest_blueprints",
    description=(
        "Rank registered blueprints for an ordinary user's natural-language intent. "
        "Use before asking technical questions or manually choosing nodes."
    ),
    input_model=SuggestBlueprintsInput,
    core=True,
)
async def suggest_blueprint_tool(input: SuggestBlueprintsInput, ctx: ToolContext) -> ToolResult:
    matches = suggest_blueprints(input.intent, limit=input.limit)
    if not matches:
        return ToolResult.ok(
            "Aniq blueprint topilmadi — flow.list_blueprintsdan biznes maqsadga eng yaqinini "
            "tanlang yoki inkremental low-level build qiling.",
            data={"matches": []},
        )
    rows = [
        {
            "pattern": definition.pattern,
            "title": definition.title,
            "description": definition.description,
            "score": score,
        }
        for definition, score in matches
    ]
    return ToolResult.ok(
        "Mos blueprintlar:\n"
        + "\n".join(f"• `{row['pattern']}` — {row['title']} (moslik {row['score']})" for row in rows),
        data={"matches": rows},
    )


@tool(
    name="flow.list_blueprints",
    description=(
        "List every registered high-level bot blueprint and its requirements. "
        "New blueprint modules appear here automatically through the registry."
    ),
    input_model=ListBlueprintsInput,
    core=True,
)
async def list_blueprints(_: ListBlueprintsInput, ctx: ToolContext) -> ToolResult:
    definitions = list_blueprint_definitions()
    rows = []
    for definition in definitions:
        requirements = []
        if definition.requires_catalog:
            requirements.append("catalog")
        if definition.requires_destination:
            requirements.append("destination")
        if definition.requires_fields:
            requirements.append("fields")
        if definition.requires_relation:
            requirements.append("relation")
        rows.append(
            {
                "pattern": definition.pattern,
                "title": definition.title,
                "description": definition.description,
                "requirements": requirements,
            }
        )
    return ToolResult.ok(
        "Mavjud blueprintlar:\n"
        + "\n".join(
            f"• `{row['pattern']}` — {row['title']}: {row['description']}"
            + (f" [{', '.join(row['requirements'])}]" if row["requirements"] else "")
            for row in rows
        ),
        data={"blueprints": rows},
    )


@tool(
    name="flow.blueprint_preview",
    description=(
        "Turn a user's business intent into a typed, human-readable bot blueprint "
        "WITHOUT changing the flow. Use first for a new common bot; then call "
        "flow.build_blueprint with the same blueprint. This prevents free-form node JSON."
    ),
    input_model=BlueprintPreviewInput,
    core=True,
)
async def blueprint_preview(input: BlueprintPreviewInput, ctx: ToolContext) -> ToolResult:
    blueprint = input.blueprint
    flow = compile_blueprint(blueprint)
    names, schemas = _collection_contract(ctx)
    issues: list[str] = []
    for name in (blueprint.catalog_collection, blueprint.destination_collection):
        if name and names and name not in names:
            issues.append(f"Collection `{name}` mavjud emas — avval yaratilishi kerak.")
    if blueprint.destination_collection in schemas:
        schema = schemas[str(blueprint.destination_collection)]
        requested = {field.collection_field or field.key for field in blueprint.fields}
        if blueprint.relation_field:
            requested.add(blueprint.relation_field)
        missing = sorted(field for field in requested if schema and field not in schema)
        if missing:
            issues.append(f"`{blueprint.destination_collection}`da maydon yetishmaydi: {', '.join(missing)}.")
    if blueprint.catalog_collection in schemas and schemas[str(blueprint.catalog_collection)]:
        catalog_schema = schemas[str(blueprint.catalog_collection)]
        display_fields = [blueprint.catalog_label_field]
        if blueprint.catalog_secondary_field:
            display_fields.append(blueprint.catalog_secondary_field)
        missing_display = [field for field in display_fields if field not in catalog_schema]
        if missing_display:
            issues.append(
                f"`{blueprint.catalog_collection}`da ko'rsatish maydoni yo'q: "
                + ", ".join(missing_display)
                + "."
            )
    lines = ["Bot blueprint:", *_business_summary(blueprint)]
    lines.append(f"• Natija: {len(flow.nodes)} node, {len(flow.edges)} edge")
    if issues:
        lines.extend(["", "Qurishdan oldingi talablar:", *[f"- {item}" for item in issues]])
    else:
        lines.append("• Collection contract: tayyor")
    return ToolResult.ok(
        "\n".join(lines),
        data={
            "blueprint": blueprint.model_dump(),
            "nodes": len(flow.nodes),
            "edges": len(flow.edges),
            "requirements": issues,
            "ready": not issues,
        },
    )


class BuildBlueprintInput(BaseModel):
    blueprint: BotBlueprint
    replace_existing: bool = Field(
        default=False,
        description="Must be true to replace a non-empty flow. Existing flow is kept as a restorable draft snapshot.",
    )
    verify: bool = Field(default=True, description="Run the generated regression journey before applying.")
    save_regression_test: bool = Field(
        default=True,
        description="Persist the generated smoke/journey test for future flow.save regressions.",
    )


async def _progress(ctx: ToolContext, step: str, status: str, detail: str) -> None:
    if ctx.emit is not None:
        await ctx.emit(
            "workflow_progress",
            {"step": step, "status": status, "detail": detail},
        )


@tool(
    name="flow.build_blueprint",
    description=(
        "Compile any registered typed business blueprint into a complete deterministic flow. "
        "It validates collection contracts and graph semantics, generates a regression "
        "test, optionally simulates it, and only then applies the flow. Prefer this over "
        "manually adding nodes for common bots."
    ),
    input_model=BuildBlueprintInput,
    core=True,
)
async def build_blueprint(input: BuildBlueprintInput, ctx: ToolContext) -> ToolResult:
    previous = _current(ctx)
    if previous.get("nodes") and not input.replace_existing:
        return ToolResult.error(
            "Joriy flow bo'sh emas. Inkremental tahrir qiling yoki uni to'liq "
            "almashtirish aniq talab qilingan bo'lsa replace_existing=true bering."
        )

    blueprint = input.blueprint
    await _progress(ctx, "blueprint", "done", "Biznes talabi typed blueprintga aylantirildi")
    names, schemas = _collection_contract(ctx)
    required = [name for name in (blueprint.catalog_collection, blueprint.destination_collection) if name]
    missing_collections = sorted({name for name in required if names and name not in names})
    if missing_collections:
        await _progress(ctx, "collections", "failed", "Kerakli collection topilmadi")
        return ToolResult.error(
            "Blueprint uchun collection mavjud emas: "
            + ", ".join(f"`{name}`" for name in missing_collections)
            + ". Avval cms.create_collection bilan typed schema yarating."
        )

    destination = blueprint.destination_collection
    if destination and destination in schemas and schemas[destination]:
        schema = schemas[destination]
        requested = {field.collection_field or field.key for field in blueprint.fields}
        if blueprint.relation_field:
            requested.add(blueprint.relation_field)
        missing_fields = sorted(field for field in requested if field not in schema)
        if missing_fields:
            await _progress(ctx, "collections", "failed", "Collection schema mos emas")
            return ToolResult.error(
                f"`{destination}` collection schema'sida maydon yo'q: "
                + ", ".join(f"`{field}`" for field in missing_fields)
                + ". Avval cms.add_field bilan qo'shing."
            )
        if blueprint.relation_field and schema.get(blueprint.relation_field) != "relation":
            await _progress(ctx, "collections", "failed", "Relation maydoni noto'g'ri turda")
            return ToolResult.error(
                f"`{destination}.{blueprint.relation_field}` relation bo'lishi kerak, "
                f"hozir `{schema.get(blueprint.relation_field)}`. Eski fieldni tuzating."
            )
    catalog = blueprint.catalog_collection
    if catalog and catalog in schemas and schemas[catalog]:
        display_fields = [blueprint.catalog_label_field]
        if blueprint.catalog_secondary_field:
            display_fields.append(blueprint.catalog_secondary_field)
        missing_display = [field for field in display_fields if field not in schemas[catalog]]
        if missing_display:
            await _progress(ctx, "collections", "failed", "Catalog display field topilmadi")
            return ToolResult.error(
                f"`{catalog}` collectionda blueprint ko'rsatish maydoni yo'q: "
                + ", ".join(f"`{field}`" for field in missing_display)
                + ". catalog_label_field/catalog_secondary_fieldni real schema bilan moslang."
            )
    await _progress(ctx, "collections", "done", "Collection va field contract tekshirildi")

    flow = compile_blueprint(blueprint)
    result = validate(
        flow,
        known_collections=names if isinstance(ctx.state.get("collections"), list) else None,
        collection_schemas=schemas,
        module_nodes=ctx.state.get("module_nodes") or [],
    )
    if not result.ok:
        await _progress(ctx, "validation", "failed", "Semantic validation xato berdi")
        return ToolResult.error(
            "Compiler yaratgan flow validatsiyadan o'tmadi; oldingi flow saqlandi:\n" + result.render()
        )
    await _progress(ctx, "validation", "done", "Graph va node contractlar to'g'ri")

    definition = get_blueprint_definition(blueprint.pattern)
    if definition is None:  # BotBlueprint validation already guarantees this
        return ToolResult.error(f"Blueprint `{blueprint.pattern}` registryda topilmadi.")
    project_id = resolve_concrete_project_id(ctx)
    catalog_document: dict[str, Any] | None = None
    if project_id and blueprint.catalog_collection:
        try:
            from ...integrations.backend import BackendClient, get_backend_client

            backend = ctx.services.get("backend")
            if not isinstance(backend, BackendClient):
                backend = get_backend_client()
            page = await backend.cms_list_documents(
                project_id,
                blueprint.catalog_collection,
                limit=1,
                auth_claims=ctx.state.get("auth_claims"),
            )
            documents = page.get("results") if isinstance(page, dict) else None
            if isinstance(documents, list) and documents and isinstance(documents[0], dict):
                catalog_document = documents[0]
        except Exception:
            # Empty/unavailable catalog still gets a stable /start smoke test.
            catalog_document = None
    steps = definition.test_generator(blueprint, catalog_document)
    verification: dict[str, Any] = {"ran": False, "ok": True}
    if input.verify and project_id:
        from ...integrations.simulator import SimulatorUnavailableError
        from .power import _run_simulation

        temp_state = dict(ctx.state)
        temp_state["working_flow"] = flow.model_dump()
        temp_ctx = ToolContext(
            run_id=ctx.run_id,
            project_id=ctx.project_id,
            user_id=ctx.user_id,
            state=temp_state,
            emit=ctx.emit,
            services=ctx.services,
        )
        try:
            response = await _run_simulation(
                temp_ctx,
                flow=flow.model_dump(),
                steps=steps,
                seed_collections=True,
                mode="full",
                project_id=project_id,
            )
        except SimulatorUnavailableError:
            response = None
        if isinstance(response, dict):
            relation_saved: bool | None = None
            expected_relation = None
            if catalog_document and blueprint.relation_field and blueprint.destination_collection:
                expected_relation = catalog_document.get("_id") or catalog_document.get("id")
                after = response.get("collections_after")
                rows = after.get(blueprint.destination_collection) if isinstance(after, dict) else None
                relation_saved = bool(
                    expected_relation is not None
                    and isinstance(rows, list)
                    and rows
                    and isinstance(rows[-1], dict)
                    and str(rows[-1].get(blueprint.relation_field)) == str(expected_relation)
                )
            simulation_ok = bool(response.get("ok")) and relation_saved is not False
            verification = {
                "ran": True,
                "ok": simulation_ok,
                "relation_saved": relation_saved,
                "expected_relation": expected_relation,
                "errors": response.get("errors") or [],
            }
        else:
            verification = {"ran": False, "ok": True}
        if verification["ran"] and not verification["ok"]:
            await _progress(ctx, "simulation", "failed", "Generated journey o'tmadi")
            return ToolResult.error(
                "Generated flow simulator testidan o'tmadi; transaction bekor qilindi "
                "va oldingi flow o'zgarmadi.",
                data={"verification": verification, "steps": steps},
            )
        await _progress(
            ctx,
            "simulation",
            "done" if verification["ran"] else "skipped",
            "Journey muvaffaqiyatli" if verification["ran"] else "Simulator mavjud emas",
        )

    test_saved = False
    if input.save_regression_test and project_id:
        from ...mcp.flow_tests import get_flow_test_store

        test_saved = await get_flow_test_store().save(
            project_id=project_id,
            user_id=str(ctx.user_id) if ctx.user_id is not None else None,
            name=f"blueprint:{blueprint.pattern}",
            steps=steps,
            enabled=True,
        )
    await _progress(ctx, "apply", "done", "Flow va regression baseline tayyor")

    compiled = flow.model_dump()
    delta = diff(FlowDocument.model_validate(previous), flow)
    message = [
        f"✅ `{blueprint.title}` blueprint qurildi: {len(flow.nodes)} node, {len(flow.edges)} edge.",
        *_business_summary(blueprint),
        f"• Tekshiruv: {"o'tdi" if verification['ran'] else "semantic validation o'tdi"}",
        f"• Regression test: {'saqlandi' if test_saved else 'saqlanmadi'}",
        "• Oldingi flow: flow.discard_blueprint bilan shu run davomida tiklanadi.",
    ]
    return ToolResult.ok(
        "\n".join(message),
        data={
            "blueprint": blueprint.model_dump(),
            "nodes": len(flow.nodes),
            "edges": len(flow.edges),
            "verification": verification,
            "regression_test_saved": test_saved,
            "diff": {
                "added_nodes": delta.added_nodes,
                "removed_nodes": delta.removed_nodes,
                "modified_nodes": delta.modified_nodes,
            },
        },
        state_updates={
            "working_flow": compiled,
            "flow_dirty": True,
            "last_validation": {"ok": True, "issues": []},
            "active_blueprint": blueprint.model_dump(),
            "blueprint_previous_flow": previous,
            "blueprint_test_steps": steps,
        },
    )


class DiscardBlueprintInput(BaseModel):
    pass


@tool(
    name="flow.discard_blueprint",
    description="Restore the flow snapshot saved immediately before flow.build_blueprint in this run.",
    input_model=DiscardBlueprintInput,
)
async def discard_blueprint(_: DiscardBlueprintInput, ctx: ToolContext) -> ToolResult:
    previous = ctx.state.get("blueprint_previous_flow")
    if not isinstance(previous, dict):
        return ToolResult.error("Bu run ichida tiklanadigan blueprint snapshot yo'q.")
    return ToolResult.ok(
        f"Blueprint bekor qilindi; oldingi flow tiklandi "
        f"({len(previous.get('nodes') or [])} node, {len(previous.get('edges') or [])} edge).",
        state_updates={
            "working_flow": previous,
            "flow_dirty": True,
            "active_blueprint": None,
            "blueprint_previous_flow": None,
            "blueprint_test_steps": None,
        },
    )


def build_blueprint_tools() -> list[Tool]:
    return [
        list_blueprints,
        suggest_blueprint_tool,
        blueprint_preview,
        build_blueprint,
        discard_blueprint,
    ]


__all__ = ["build_blueprint_tools"]
