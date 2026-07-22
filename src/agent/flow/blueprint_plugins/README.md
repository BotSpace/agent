# Blueprint extension modules

Add one Python module per business pattern. The module must call
`register_blueprint(BlueprintDefinition(...))` at import time and provide:

- a stable `pattern` id;
- user-facing title and description;
- requirement flags;
- a positive schema `version` (increment when the public draft contract changes);
- `field_presets` and default collection/relation slugs so
  `flow.blueprint_draft` can produce a useful draft without hand-written JSON;
- optionally an `options_model: type[BaseModel]` for plugin-specific typed
  configuration (unknown options are rejected);
- a deterministic compiler returning `FlowDocument`;
- a regression-step generator.

Modules are discovered automatically. Do not add them to a central import list.
Malformed field presets fail at service startup, before a user flow is changed.
