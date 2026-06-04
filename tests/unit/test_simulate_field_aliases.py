"""flow_simulate qadam MAYDON aliaslari + engine lat/lon serializatsiyasi."""

from agent.tools.flow.power import FlowSimulateStep


def test_command_field_alias():
    # Agent 'command':'/start' yozsa ham text'ga tushadi (round-trip yo'q).
    step = FlowSimulateStep(type="command", command="/start")
    assert step.text == "/start"
    assert step.model_dump(exclude_none=True, by_alias=True)["text"] == "/start"


def test_location_serializes_as_lat_lon():
    # Engine Step `lat`/`lon` json tagini kutadi — model latitude/longitude saqlaydi.
    step = FlowSimulateStep(type="location", latitude=41.31, longitude=69.24)
    dumped = step.model_dump(exclude_none=True, by_alias=True)
    assert dumped["lat"] == 41.31
    assert dumped["lon"] == 69.24
    assert "latitude" not in dumped


def test_location_accepts_lat_lon_input():
    step = FlowSimulateStep(type="location", lat=41.0, lon=69.0)
    assert step.latitude == 41.0 and step.longitude == 69.0


def test_phone_number_alias():
    step = FlowSimulateStep(type="contact", phone_number="+998901234567")
    assert step.phone == "+998901234567"
