"""Accuracy of the deterministic metrics on the labeled judge cases."""

import json
from pathlib import Path

from app.evaluators.normalized_match import NormalizedMatchEvaluator
from app.evaluators.numeric_match import NumericMatchEvaluator

CASES_PATH = Path(__file__).parent / "data" / "llm_judge_cases.json"


def load_cases() -> list[dict[str, str]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def wrong_decisions(evaluator, cases) -> list[str]:
    return [
        case["id"]
        for case in cases
        if evaluator.evaluate(
            expected_output=case["expected_output"],
            generated_output=case["generated_output"],
            input_text=case["question"],
        ).passed
        is not (case["expected_decision"] == "YES")
    ]


def test_normalized_match_accuracy_on_labeled_cases() -> None:
    cases = load_cases()

    wrong = wrong_decisions(NormalizedMatchEvaluator(), cases)

    assert len(cases) - len(wrong) >= 18, wrong


def test_normalized_match_known_false_positives() -> None:
    # The expected answer appears in these wrong answers ("Paris or Lyon",
    # "assume Paris, but it is actually Marseille"). If this changes, update
    # the README's metric accuracy table.
    cases = load_cases()
    evaluator = NormalizedMatchEvaluator()

    accepted_wrong = [
        case["id"]
        for case in cases
        if case["expected_decision"] == "NO"
        and evaluator.evaluate(
            expected_output=case["expected_output"],
            generated_output=case["generated_output"],
        ).passed
    ]

    assert accepted_wrong == ["wrong-6", "verbose-2"]


def test_numeric_match_is_correct_on_numeric_cases() -> None:
    evaluator = NumericMatchEvaluator()
    cases = []

    for case in load_cases():
        try:
            evaluator.validate_expected_output(case["expected_output"])
        except ValueError:
            continue

        cases.append(case)

    assert len(cases) == 7
    assert wrong_decisions(evaluator, cases) == []
