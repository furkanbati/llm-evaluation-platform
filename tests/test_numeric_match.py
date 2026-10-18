import pytest

from app.evaluators.numeric_match import NumericMatchEvaluator, extract_last_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4", 4.0),
        ("The answer is 4.", 4.0),
        ("2 + 2 = 4", 4.0),
        ("-3", -3.0),
        ("x = -0.25", -0.25),
        ("2-1", 1.0),
        ("1,000", 1000.0),
        ("1,234.5 units", 1234.5),
        ("50%", 50.0),
        ("1/2", 0.5),
        (".5", 0.5),
        ("The answer is four.", 4.0),
        ("Eight legs.", 8.0),
        ("4 is the one", 4.0),
        ("no number here", None),
        ("divide by 1/0", None),
    ],
)
def test_extract_last_number(text: str, expected: float | None) -> None:
    assert extract_last_number(text) == expected


@pytest.mark.parametrize(
    ("expected_output", "generated_output"),
    [
        ("4", "4"),
        ("4", "4.0"),
        ("4", "The answer is four."),
        ("4", "2 + 2 = 4"),
        ("1000", "It costs 1,000 dollars."),
        ("0.5", "1/2"),
        ("1945", "World War II ended in 1945."),
    ],
)
def test_numeric_match_passes(
    expected_output: str,
    generated_output: str,
) -> None:
    result = NumericMatchEvaluator().evaluate(
        expected_output=expected_output,
        generated_output=generated_output,
    )

    assert result.metric_name == "numeric_match"
    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.parametrize(
    ("expected_output", "generated_output"),
    [
        ("4", "5"),
        ("4", "The answer is 4, not 5."),
        ("1945", "World War II ended in 1944."),
        ("4", "I don't know."),
        ("0.333", "1/3"),
    ],
)
def test_numeric_match_fails(
    expected_output: str,
    generated_output: str,
) -> None:
    result = NumericMatchEvaluator().evaluate(
        expected_output=expected_output,
        generated_output=generated_output,
    )

    assert result.score == 0.0
    assert result.passed is False


def test_numeric_match_rejects_expected_output_without_number() -> None:
    with pytest.raises(ValueError, match="does not contain a number"):
        NumericMatchEvaluator().validate_expected_output("Paris")


def test_numeric_match_accepts_expected_output_with_number() -> None:
    NumericMatchEvaluator().validate_expected_output("1945")
