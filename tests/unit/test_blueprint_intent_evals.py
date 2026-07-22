import json
from pathlib import Path

from agent.flow.blueprint import suggest_blueprints


def test_ordinary_user_intents_choose_expected_blueprint():
    dataset = Path(__file__).parents[2] / "evals" / "ordinary_user_intents.json"
    cases = json.loads(dataset.read_text(encoding="utf-8"))
    failures = []
    for case in cases:
        matches = suggest_blueprints(case["intent"], limit=3)
        actual = matches[0][0].pattern if matches else None
        if actual != case["expected"]:
            failures.append(
                {
                    "intent": case["intent"],
                    "expected": case["expected"],
                    "actual": actual,
                }
            )
    assert failures == []
