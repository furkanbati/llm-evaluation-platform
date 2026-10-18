import pytest

from app.evaluators.normalized_match import NormalizedMatchEvaluator, normalize_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The capital of France is Paris.", "capital of france is paris"),
        ("  PARIS!  ", "paris"),
        ("Red, yellow, and blue", "red yellow and blue"),
        ("An apple a day", "apple day"),
        ("4.0", "4.0"),
        ("1,000 apples", "1,000 apples"),
        ("End of sentence 4.", "end of sentence 4"),
    ],
)
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    ("expected_output", "generated_output"),
    [
        ("Paris", "paris"),
        ("Paris", "The capital of France is Paris."),
        ("William Shakespeare", "It was written by William Shakespeare."),
        ("Red, yellow, and blue", "red yellow and blue"),
        ("The", "the"),
    ],
)
def test_normalized_match_passes(
    expected_output: str,
    generated_output: str,
) -> None:
    result = NormalizedMatchEvaluator().evaluate(
        expected_output=expected_output,
        generated_output=generated_output,
    )

    assert result.metric_name == "normalized_match"
    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.parametrize(
    ("expected_output", "generated_output"),
    [
        ("Paris", "London"),
        ("Paris", "Parisian cuisine"),
        ("William Shakespeare", "Shakespeare"),
        ("4", "4.0"),
        ("1945", "1944"),
    ],
)
def test_normalized_match_fails(
    expected_output: str,
    generated_output: str,
) -> None:
    result = NormalizedMatchEvaluator().evaluate(
        expected_output=expected_output,
        generated_output=generated_output,
    )

    assert result.score == 0.0
    assert result.passed is False
