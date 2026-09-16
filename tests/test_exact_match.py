from app.evaluators.exact_match import ExactMatchEvaluator


def test_exact_match_success() -> None:
    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        expected_output="Paris",
        generated_output="Paris",
    )

    assert result.metric_name == "exact_match"
    assert result.score == 1.0
    assert result.passed is True


def test_exact_match_failure() -> None:
    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        expected_output="Paris",
        generated_output="London",
    )

    assert result.metric_name == "exact_match"
    assert result.score == 0.0
    assert result.passed is False


def test_exact_match_ignores_whitespace() -> None:
    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        expected_output="  Paris  ",
        generated_output="Paris",
    )

    assert result.metric_name == "exact_match"
    assert result.score == 1.0
    assert result.passed is True


def test_exact_match_is_case_sensitive() -> None:
    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        expected_output="Paris",
        generated_output="paris",
    )

    assert result.metric_name == "exact_match"
    assert result.score == 0.0
    assert result.passed is False