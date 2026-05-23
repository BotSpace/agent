"""Provider mappings should not drop content parts."""

from agent.llm.anthropic import AnthropicClient
from agent.llm.base import Message, TextPart, ToolResultPart, ToolUsePart
from agent.llm.openai import OpenAIClient


def test_anthropic_converts_tool_messages_to_user_role():
    msgs = [
        Message.user_text("hi"),
        Message(
            role="assistant",
            content=[
                TextPart(text="ok"),
                ToolUsePart(id="u1", name="x.do", arguments={"a": 1}),
            ],
        ),
        Message.tool_result(tool_use_id="u1", content="done"),
    ]
    system, out = AnthropicClient._to_anthropic_messages(msgs)
    assert system is None
    # user, assistant (with text + tool_use), then user (tool_result)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    types = [block["type"] for block in out[2]["content"]]
    assert "tool_result" in types


def test_openai_splits_tool_messages():
    msgs = [
        Message.system_text("sys"),
        Message.user_text("hi"),
        Message(
            role="assistant",
            content=[
                TextPart(text="ok"),
                ToolUsePart(id="u1", name="x.do", arguments={"a": 1}),
            ],
        ),
        Message.tool_result(tool_use_id="u1", content="done"),
    ]
    out = OpenAIClient._to_openai_messages(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "tool"]
    assistant_msg = out[2]
    assert "tool_calls" in assistant_msg
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "x.do"
