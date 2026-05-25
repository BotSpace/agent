import pytest

from agent.botbuilder_ai import normalize_command, parse_commands
from agent.builder.session import build_from_commands


def test_normalize_command_aliases_engine_node():
    command = normalize_command(
        {"command": "add_node", "key": "check", "type": "CheckMembershipNode", "data": {"chatId": "-100"}}
    )

    assert command["action"] == "add_engine_node"
    assert command["node_type"] == "CheckMembershipNode"
    assert "type" not in command


def test_parse_commands_repairs_future_after_reference():
    commands = parse_commands(
        """
        {
          "commands": [
            {"action": "create_bot", "name": "Demo", "language": "uz"},
            {"action": "add_text", "key": "first", "text": "Birinchi", "after": "second"},
            {"action": "add_finish", "key": "second", "text": "Tamom"}
          ]
        }
        """
    )

    result = build_from_commands(commands)

    assert result.ok is True
    assert any(edge.source == "start" and edge.target == "first" for edge in result.spec.edges)
    assert any(edge.source == "first" and edge.target == "second" for edge in result.spec.edges)


def test_compiler_rejects_missing_engine_branch():
    commands = parse_commands(
        """
        {
          "commands": [
            {"action": "create_bot", "name": "Membership", "language": "uz"},
            {
              "action": "add_engine_node",
              "key": "member_check",
              "node_type": "CheckMembershipNode",
              "data": {"chatId": "-100123"},
              "after": "start"
            },
            {"action": "add_finish", "key": "done", "text": "OK", "after": "member_check"}
          ]
        }
        """
    )

    result = build_from_commands(commands)

    assert result.ok is False
    assert any(issue.code == "missing_branch" for issue in result.issues)


@pytest.mark.parametrize(
    ("operator", "normalized"),
    [
        ("==", "equals"),
        (">=", ">="),
        ("exists", "not_empty"),
    ],
)
def test_condition_operator_aliases(operator: str, normalized: str):
    command = normalize_command(
        {
            "action": "add_condition",
            "key": "check",
            "condition": {"variable": "age", "operator": operator, "value": 18},
        }
    )

    assert command["conditions"][0]["operator"] == normalized
