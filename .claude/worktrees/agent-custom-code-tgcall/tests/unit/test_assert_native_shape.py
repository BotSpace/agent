"""Engine assert rejimining KOMPAKT javobi (steps'siz) to'g'ri o'qilishi.

Bug: _format_assert resp["steps"]dan qayta hisoblab, engine'ning native
{ok, passed, failed} javobida doim passed=0/failed=[] chiqarardi.
"""

from agent.tools.flow.power import _format_assert


def test_native_assert_shape_passed_counter():
    resp = {
        "ok": False,
        "passed": 2,
        "failed": [
            {
                "step": 2,
                "input": {"type": "command", "text": "/start"},
                "reason": 'contains "X" topilmadi',
                "errors": [],
            }
        ],
        "final_state": {"state": "", "context": {}},
    }
    text, structured = _format_assert(resp, 3)
    assert structured["passed"] == 2
    assert structured["ok"] is False
    assert len(structured["failed"]) == 1
    # engine 0-based -> 1-based
    assert structured["failed"][0]["step"] == 3
    assert "2/3 o'tdi" in text
    assert 'contains "X" topilmadi' in text


def test_native_assert_all_passed():
    resp = {"ok": True, "passed": 3, "failed": [], "final_state": {}}
    text, structured = _format_assert(resp, 3)
    assert structured["passed"] == 3
    assert structured["ok"] is True
    assert "3/3 o'tdi" in text


def test_full_shape_still_works():
    resp = {
        "ok": True,
        "steps": [
            {"input": {"type": "text", "text": "salom"}, "expect_passed": True, "errors": []},
            {
                "input": {"type": "click", "data": "x"},
                "expect_passed": False,
                "expect_failures": [{"reason": "topilmadi"}],
                "errors": [],
            },
        ],
        "final_state": {},
    }
    _, structured = _format_assert(resp, 2)
    assert structured["passed"] == 1
    assert len(structured["failed"]) == 1
