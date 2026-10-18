import pytest

from app.evaluators.token_f1 import TokenF1Evaluator, token_f1


@pytest.mark.parametrize(
    ("expected_output", "generated_output", "score"),
    [
        ("Paris", "Paris", 1.0),
        ("Paris", "paris!", 1.0),
        ("Paris", "London", 0.0),
        # precision 3/3, recall 3/4
        ("Red, yellow, and blue", "Red and blue", 6 / 7),
        # precision 1/5 ("the" is removed), recall 1/1
        ("Paris", "The capital of France is Paris.", 1 / 3),
        # precision 1/3, recall 1/2
        ("William Shakespeare", "Shakespeare wrote it.", 0.4),
        ("4", "The answer is four.", 0.0),
    ],
)
def test_token_f1_score(
    expected_output: str,
    generated_output: str,
    score: float,
) -> None:
    assert token_f1(expected_output, generated_output) == pytest.approx(score)


def test_token_f1_counts_repeated_tokens_once_per_match() -> None:
    # Only one "paris" in the expected output can be matched.
    assert token_f1("Paris", "Paris Paris") == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("expected_output", "generated_output", "score"),
    [
        ("The", "the", 1.0),
        ("The", "Paris", 0.0),
        ("Paris", "...", 0.0),
    ],
)
def test_token_f1_handles_empty_normalized_text(
    expected_output: str,
    generated_output: str,
    score: float,
) -> None:
    assert token_f1(expected_output, generated_output) == score


def test_token_f1_evaluator_passes_at_threshold() -> None:
    evaluator = TokenF1Evaluator()

    passing = evaluator.evaluate(
        expected_output="Red, yellow, and blue",
        generated_output="Red and blue",
    )
    failing = evaluator.evaluate(
        expected_output="Paris",
        generated_output="The capital of France is Paris.",
    )

    assert passing.metric_name == "token_f1"
    assert passing.passed is True
    assert failing.score == pytest.approx(1 / 3)
    assert failing.passed is False
