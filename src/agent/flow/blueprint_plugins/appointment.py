"""Appointment/booking blueprint extension (auto-discovered)."""

from ..blueprint import (
    BlueprintDefinition,
    BotBlueprint,
    compile_standard_blueprint,
    register_blueprint,
    regression_steps,
)


def _compile(blueprint: BotBlueprint):
    return compile_standard_blueprint(blueprint, base_pattern="lead")


register_blueprint(
    BlueprintDefinition(
        pattern="appointment",
        title="Qabulga yozilish",
        description="Ism, aloqa, xizmat va vaqtni yig'ib appointment collectionga saqlaydi.",
        compiler=_compile,
        test_generator=regression_steps,
        keywords=("qabulga yoz", "uchrashuv", "appointment", "bron", "vaqt band"),
        requires_destination=True,
        requires_fields=True,
    )
)
