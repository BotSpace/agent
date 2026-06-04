"""flow_simulate step type aliaslari va aniq xato xabari.

Real klient 'callback'/'callback_query'/'button' deb 7 marta taxmin qildi —
endi aliaslar avtomatik 'click'ka aylanadi, noma'lum tur esa darhol ro'yxat
bilan rad etiladi (engine'gacha bormaydi).
"""

import pytest
from pydantic import ValidationError

from agent.tools.flow.power import FlowSimulateStep


@pytest.mark.parametrize(
    "alias", ["callback", "callback_query", "callbackQuery", "button", "inline", "callback_data"]
)
def test_click_aliases_normalized(alias):
    step = FlowSimulateStep(type=alias, data="order")
    assert step.type == "click"


def test_message_alias_to_text():
    assert FlowSimulateStep(type="message", text="salom").type == "text"


def test_unknown_type_rejected_with_list():
    with pytest.raises(ValidationError, match="click"):
        FlowSimulateStep(type="tap", data="x")


def test_canonical_types_pass():
    for t in ("command", "text", "click", "click_text", "contact", "location", "advance_time"):
        FlowSimulateStep(type=t, text="x", data="y", phone="z", seconds=1)
