"""CheckMembershipNode sxemasi (manifest-driven).

Engine `channels` / `channelsCollection` / `chatId` ni o'qiydi (check_membership.go).
Manifest field-completeness'i shu kalitlarni ko'rsatishi kerak; "kamida bitta
kanal manbasi" qoidasi addendum (manifest ifoda eta olmaydi).
"""

import pytest

from agent.tools.flow.node_data_models import get_data_model

CheckMembership = get_data_model("CheckMembershipNode")


def test_channels_format_accepted():
    d = CheckMembership.model_validate({"channels": [{"value": "@my_channel"}]})
    assert d.model_dump(by_alias=True, exclude_none=True)["channels"] == [{"value": "@my_channel"}]


def test_channels_collection_accepted():
    d = CheckMembership.model_validate({"channelsCollection": "channels"})
    assert d.model_dump(by_alias=True, exclude_none=True)["channelsCollection"] == "channels"


def test_legacy_chat_id_still_accepted():
    d = CheckMembership.model_validate({"chatId": "@old_style"})
    assert d.model_dump(by_alias=True, exclude_none=True)["chatId"] == "@old_style"


def test_empty_rejected_with_actionable_error():
    with pytest.raises(Exception, match="kanal"):
        CheckMembership.model_validate({})
