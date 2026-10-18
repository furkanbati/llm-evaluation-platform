"""Accuracy of the deterministic metrics on the labeled judge cases."""

import json
from pathlib import Path

import pytest

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


def test_token_f1_accuracy_on_labeled_cases() -> None:
    # Most correct answers here are full sentences, which lower precision,
    # so token_f1 accepts few of them; it suits short extractive answers.
    # Its false positive is a partial answer ("Red and blue").
    from app.evaluators.token_f1 import TokenF1Evaluator

    cases = load_cases()
    evaluator = TokenF1Evaluator()

    wrong = wrong_decisions(evaluator, cases)

    accepted_wrong = [
        case["id"]
        for case in cases
        if case["expected_decision"] == "NO"
        and evaluator.evaluate(
            expected_output=case["expected_output"],
            generated_output=case["generated_output"],
        ).passed
    ]

    assert len(cases) - len(wrong) >= 13, wrong
    assert accepted_wrong == ["wrong-7"]


def test_token_recall_accuracy_on_labeled_cases() -> None:
    # Full-sentence answers keep a high recall, so most correct answers
    # pass; wrong answers that contain the expected words pass too.
    from app.evaluators.token_f1 import TokenRecallEvaluator

    cases = load_cases()
    evaluator = TokenRecallEvaluator()

    wrong = wrong_decisions(evaluator, cases)

    accepted_wrong = [
        case["id"]
        for case in cases
        if case["expected_decision"] == "NO"
        and evaluator.evaluate(
            expected_output=case["expected_output"],
            generated_output=case["generated_output"],
        ).passed
    ]

    assert len(cases) - len(wrong) >= 18, wrong
    assert accepted_wrong == ["wrong-6", "wrong-7", "verbose-2"]


@pytest.mark.integration
def test_semantic_similarity_accuracy_on_labeled_cases() -> None:
    # Measured with nomic-embed-text and the default 0.73 threshold, which
    # was chosen on a separate calibration set: 15/23. Embeddings track the
    # topic rather than correctness, so wrong answers on the same topic
    # ("1944" for "1945") can pass. Allows a one-case margin.
    from app.config import (
        OLLAMA_BASE_URL,
        SEMANTIC_SIMILARITY_MODEL,
        SEMANTIC_SIMILARITY_THRESHOLD,
    )
    from app.evaluators.semantic_similarity import SemanticSimilarityEvaluator
    from app.services.ollama_embedding_client import OllamaEmbeddingClient

    evaluator = SemanticSimilarityEvaluator(
        embedding_client=OllamaEmbeddingClient(OLLAMA_BASE_URL),
        model_name=SEMANTIC_SIMILARITY_MODEL,
        threshold=SEMANTIC_SIMILARITY_THRESHOLD,
    )

    cases = load_cases()
    wrong = wrong_decisions(evaluator, cases)

    assert len(cases) - len(wrong) >= 14, wrong
