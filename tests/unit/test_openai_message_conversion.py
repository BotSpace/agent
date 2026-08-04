from agent.llm.base import Message, TextPart, ToolUsePart
from agent.llm.openai import OpenAIClient


def test_incomplete_parallel_tool_turn_drops_unanswered_call() -> None:
    messages = [
        Message.user_text("build"),
        Message(
            role="assistant",
            content=[
                ToolUsePart(id="call-1", name="flow.get", arguments={}),
                ToolUsePart(id="call-2", name="project.list", arguments={}),
            ],
        ),
        Message.tool_result(tool_use_id="call-1", content="ok"),
        Message.user_text("dynamic context"),
    ]

    converted = OpenAIClient._to_openai_messages(messages)

    assistant = next(message for message in converted if message["role"] == "assistant")
    assert [call["id"] for call in assistant["tool_calls"]] == ["call-1"]
    assert [message["tool_call_id"] for message in converted if message["role"] == "tool"] == ["call-1"]


def test_unanswered_tool_only_assistant_turn_is_removed() -> None:
    messages = [
        Message.user_text("build"),
        Message(
            role="assistant",
            content=[ToolUsePart(id="missing", name="flow.get", arguments={})],
        ),
        Message.user_text("continue"),
    ]

    converted = OpenAIClient._to_openai_messages(messages)

    assert [message["role"] for message in converted] == ["user", "user"]
    assert all("tool_calls" not in message for message in converted)


def test_orphan_tool_result_is_removed() -> None:
    messages = [
        Message.user_text("build"),
        Message.tool_result(tool_use_id="orphan", content="stale"),
        Message(role="assistant", content=[TextPart(text="done")]),
    ]

    converted = OpenAIClient._to_openai_messages(messages)

    assert [message["role"] for message in converted] == ["user", "assistant"]
