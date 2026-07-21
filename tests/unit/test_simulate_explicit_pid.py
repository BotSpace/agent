"""flow.simulate (va power tools) ctx'da project_id bo'lmasa ham explicit
project_id bilan ishlashi — dashboard sub-agent muammosi."""

from agent.tools._common import resolve_concrete_project_id
from agent.tools.flow.power import FlowSimulateInput


class _Ctx:
    def __init__(self, state=None, project_id=None):
        self.state = state or {}
        self.project_id = project_id


def test_explicit_project_id_wins_when_ctx_empty():
    ctx = _Ctx()  # ctx'da hech narsa yo'q (dashboard sub-agent holati)
    assert resolve_concrete_project_id(ctx, "1217") == "1217"


def test_ctx_used_when_no_explicit():
    ctx = _Ctx(project_id="42")
    assert resolve_concrete_project_id(ctx) == "42"


def test_simulate_input_accepts_project_id():
    inp = FlowSimulateInput.model_validate(
        {"project_id": "1217", "steps": [{"type": "command", "text": "/start"}]}
    )
    assert str(inp.project_id) == "1217"


def test_placeholder_project_ignored():
    ctx = _Ctx()
    assert resolve_concrete_project_id(ctx, "global") is None


def test_partial_simulation_accepts_pinned_context_without_steps():
    inp = FlowSimulateInput.model_validate(
        {"project_id": "1217", "start_node_id": "send-result", "initial_context": {"name": "Ali"}}
    )
    assert inp.steps == []
    assert inp.start_node_id == "send-result"
    assert inp.initial_context == {"name": "Ali"}


def test_simulation_requires_steps_or_partial_node():
    try:
        FlowSimulateInput.model_validate({"project_id": "1217"})
    except ValueError:
        return
    raise AssertionError("empty simulation request must be rejected")
