"""flow.simulate (va power tools) ctx'da project_id bo'lmasa ham explicit
project_id bilan ishlashi — dashboard sub-agent muammosi."""

from agent.tools.flow.power import FlowSimulateInput, _power_project_id


class _Ctx:
    def __init__(self, state=None, project_id=None):
        self.state = state or {}
        self.project_id = project_id


def test_explicit_project_id_wins_when_ctx_empty():
    ctx = _Ctx()  # ctx'da hech narsa yo'q (dashboard sub-agent holati)
    assert _power_project_id(ctx, "1217") == "1217"


def test_ctx_used_when_no_explicit():
    ctx = _Ctx(project_id="42")
    assert _power_project_id(ctx) == "42"


def test_simulate_input_accepts_project_id():
    inp = FlowSimulateInput.model_validate(
        {"project_id": "1217", "steps": [{"type": "command", "text": "/start"}]}
    )
    assert str(inp.project_id) == "1217"


def test_placeholder_project_ignored():
    ctx = _Ctx()
    assert _power_project_id(ctx, "global") is None
