# Blueprint extension modules

Add one Python module per business pattern. The module must call
`register_blueprint(BlueprintDefinition(...))` at import time and provide:

- a stable `pattern` id;
- user-facing title and description;
- requirement flags;
- a deterministic compiler returning `FlowDocument`;
- a regression-step generator.

Modules are discovered automatically. Do not add them to a central import list.
