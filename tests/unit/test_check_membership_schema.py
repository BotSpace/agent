"""CheckMembershipNode sxemasi: channels/channelsCollection majburiyligi.

Bug: docs/validator chatId o'rgatardi, engine esa channels o'qiydi —
node hammani a'zo deb o'tkazib yuborardi.
"""

import pytest
from pydantic import ValidationError

from agent.tools.flow.node_data_models import CheckMembershipData


def test_channels_format_accepted():
    d = CheckMembershipData(channels=[{"value": "@my_channel"}])
    assert d.channels[0].value == "@my_channel"


def test_channels_collection_accepted():
    d = CheckMembershipData(channelsCollection="channels")
    assert d.channelsCollection == "channels"


def test_legacy_chat_id_still_accepted():
    d = CheckMembershipData(chatId="@old_style")
    assert d.chatId == "@old_style"


def test_empty_rejected_with_actionable_error():
    with pytest.raises(ValidationError, match="kamida bitta kanal"):
        CheckMembershipData()
